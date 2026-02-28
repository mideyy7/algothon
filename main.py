"""IMCity Trading Bot — main entry point.

Extends BaseBot with:
  - DataPipeline integration for fair-value estimates across all 8 markets
  - Harmonic tidal prediction (quant_models.py) to fill TIDE_SPOT / LON_ETF
    / LON_FLY when the settlement data isn't yet available
  - Cancel-and-replace market making around fair values
  - Risk / aggression scaling via risk_framework.py

Usage:
    export CMI_URL="http://<exchange-host>/"
    export CMI_USER="your_username"
    export CMI_PASS="your_password"
    export RAPID_API_KEY="your_rapidapi_key"   # needed for M5/M6/M7/M8

    python main.py
"""

import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

# ── Project root on sys.path ─────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from imc_template.bot_template import BaseBot, OrderBook, OrderRequest, Trade, Side
from data_pipeline import DataPipeline, get_market_window
from quant_models import fill_missing_estimates

# ── Risk framework ─────────────────────────────────────────────────────────
_RISK_AVAILABLE = False
try:
    import risk_framework
    _pnl_risk_level         = risk_framework.pnl_risk_level
    _prize_pressure         = risk_framework.prize_pressure
    _risk_budget_from_prize = risk_framework.risk_budget_from_prize
    _score_to_expected_pnl  = risk_framework.score_to_expected_pnl
    _DEFAULT_PRIZE_CURVE    = risk_framework.prize_curve
    _RISK_AVAILABLE = True
except Exception as _err:
    logging.warning("Risk framework unavailable (%s) — using fixed aggression=1.0", _err)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("imcbot")

# ── Exchange credentials (from environment or edit here for the competition) ─
CMI_URL  = os.environ.get("CMI_URL",  "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/")
USERNAME = os.environ.get("CMI_USER", "your_username_here")
PASSWORD = os.environ.get("CMI_PASS", "your_password_here")

# ── Market-making parameters ──────────────────────────────────────────────
BASE_QUOTE_SIZE  = 20      # contracts per side before aggression scaling
MAX_POSITION     = 95      # hard cap per product (±100 limit with 5-unit buffer)
MIN_REQUOTE_SECS = 1.0     # minimum seconds between requotes per product
SPREAD_PCT_EARLY = 0.002   # 0.2 % half-spread at window open
SPREAD_PCT_LATE  = 0.0005  # 0.05 % half-spread near settlement
PNL_HISTORY_LEN  = 200     # rolling PnL samples kept for risk calculation


# ── Bot ───────────────────────────────────────────────────────────────────────

class IMCBot(BaseBot):
    """IMCity trading bot: data-driven market maker across all 8 London markets.

    Architecture:
      - SSE stream (from BaseBot) fires on_orderbook() for every orderbook event.
      - on_orderbook() calls the pipeline (TTL-cached, so rarely hits the network),
        computes fair values via quant_models.fill_missing_estimates(), then
        cancel-and-replaces quotes around those fair values.
      - A background maintenance loop (every 30 s) syncs positions from the
        exchange and updates the aggression multiplier via the risk framework.
    """

    def __init__(self, cmi_url: str, username: str, password: str) -> None:
        super().__init__(cmi_url, username, password)

        # Data pipeline
        self._window_start, self._window_end = get_market_window()
        self.pipeline = DataPipeline(self._window_start, self._window_end)
        log.info("Market window: %s → %s", self._window_start, self._window_end)

        # Shared state (accessed from SSE thread + main thread)
        self._lock = threading.Lock()
        self._positions:   dict[str, int]            = {}
        self._fair_values: dict[str, Optional[float]] = {}

        # Per-product throttle: store monotonic time of last requote
        self._last_quote: dict[str, float] = {}

        # PnL history for risk calculation
        self._pnl_history: deque[float] = deque(maxlen=PNL_HISTORY_LEN)

        # Aggression multiplier set by risk framework (1.0 = neutral)
        self._aggression: float = 1.0

        # Prize curve (used by risk framework)
        self._prize_curve: "pd.DataFrame" = (
            _DEFAULT_PRIZE_CURVE if _RISK_AVAILABLE else pd.DataFrame()
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start the bot and block until KeyboardInterrupt."""
        log.info("Starting IMCBot as user '%s'", self.username)

        # Warm-up: sync products and positions before entering the SSE loop
        self._sync_positions()

        self.start()   # launches the SSE background thread
        log.info("SSE stream connected. Bot is live.")

        try:
            while True:
                self._periodic_maintenance()
                time.sleep(30)
        except KeyboardInterrupt:
            log.info("Shutting down …")
        finally:
            self.stop()
            log.info("Bot stopped.")

    # ── SSE callbacks ──────────────────────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook) -> None:
        """Called by the SSE thread on every orderbook update.

        Throttled per-product to MIN_REQUOTE_SECS to avoid hammering the
        exchange with cancels/replaces on every tick.
        """
        product = orderbook.product
        now = time.monotonic()

        if now - self._last_quote.get(product, 0.0) < MIN_REQUOTE_SECS:
            return  # too soon to requote this product

        try:
            # Refresh fair values (TTL-cached — usually instant)
            self._refresh_fair_values()

            fair_value = self._fair_values.get(product)
            if fair_value is None:
                return  # no price signal; skip silently

            with self._lock:
                position = self._positions.get(product, 0)

            self._place_quotes(product, fair_value, position, orderbook.tick_size)
            self._last_quote[product] = now

        except Exception:
            log.exception("on_orderbook error for %s", product)

    def on_trades(self, trade: Trade) -> None:
        """Called by the SSE thread on every market trade.

        Tracks our fills locally so position checks in on_orderbook() are
        low-latency (no extra exchange round-trip needed on each tick).
        """
        try:
            delta = 0
            if trade.buyer  == self.username:
                delta += trade.volume
            if trade.seller == self.username:
                delta -= trade.volume

            if delta != 0:
                with self._lock:
                    prev = self._positions.get(trade.product, 0)
                    self._positions[trade.product] = prev + delta
                log.info(
                    "Fill: %s %s %d @ %.2f  |  position now %+d",
                    trade.product,
                    "BUY" if delta > 0 else "SELL",
                    abs(delta),
                    trade.price,
                    self._positions[trade.product],
                )
        except Exception:
            log.exception("on_trades error")

    # ── Quoting logic ──────────────────────────────────────────────────────

    def _place_quotes(
        self,
        product: str,
        fair_value: float,
        position: int,
        tick_size: float,
    ) -> None:
        """Cancel stale quotes and replace with fresh bid/ask around fair_value.

        Spread narrows as the window progresses (less uncertainty near settlement).
        Order size is scaled by the current aggression multiplier and capped at
        the remaining position headroom so we never breach ±MAX_POSITION.
        """
        elapsed = self._elapsed_fraction()
        half_spread_pct = SPREAD_PCT_EARLY + (SPREAD_PCT_LATE - SPREAD_PCT_EARLY) * elapsed
        half_spread = max(fair_value * half_spread_pct, tick_size)

        bid = self._snap(fair_value - half_spread, tick_size)
        ask = self._snap(fair_value + half_spread, tick_size)

        if bid >= ask or bid <= 0:
            return

        size = max(1, int(BASE_QUOTE_SIZE * self._aggression))
        buy_headroom  = MAX_POSITION - position
        sell_headroom = MAX_POSITION + position

        orders: list[OrderRequest] = []
        if buy_headroom  > 0:
            orders.append(OrderRequest(product, bid, Side.BUY,  min(size, buy_headroom)))
        if sell_headroom > 0:
            orders.append(OrderRequest(product, ask, Side.SELL, min(size, sell_headroom)))

        if not orders:
            return

        # Cancel existing resting orders for this product, then place fresh ones
        existing = self.get_orders(product)
        if existing:
            cancel_threads = [
                threading.Thread(target=self.cancel_order, args=(o["id"],))
                for o in existing
            ]
            for t in cancel_threads:
                t.start()
            for t in cancel_threads:
                t.join()

        self.send_orders(orders)
        log.debug(
            "%s  bid=%.2f  ask=%.2f  size=%d  pos=%+d  aggression=%.2f",
            product, bid, ask, size, position, self._aggression,
        )

    # ── Fair value ─────────────────────────────────────────────────────────

    def _refresh_fair_values(self) -> None:
        """Fetch a pipeline snapshot (TTL-cached) and compute all fair values.

        Calls fill_missing_estimates() which overrides None values with the
        harmonic tidal prediction when settlement data is not yet available.
        """
        snap = self.pipeline.fetch_snapshot()
        fv   = fill_missing_estimates(snap)
        with self._lock:
            self._fair_values = fv

    # ── Risk management ────────────────────────────────────────────────────

    def _update_risk(self) -> None:
        """Fetch latest PnL, update history, recompute aggression multiplier."""
        if not _RISK_AVAILABLE:
            return
        try:
            pnl_data  = self.get_pnl()
            total_pnl = float(
                pnl_data.get("pnl")
                or pnl_data.get("totalPnl")
                or pnl_data.get("realizedPnl")
                or 0.0
            )
            self._pnl_history.append(total_pnl)

            if len(self._pnl_history) < 5:
                return  # not enough history yet

            # quantstats requires a DatetimeIndex for max_drawdown
            history = list(self._pnl_history)
            idx = pd.date_range(end=pd.Timestamp.now(), periods=len(history), freq="30s")
            pnl_series = pd.Series(history, index=idx)
            risk       = _pnl_risk_level(pnl_series)
            live       = self._get_live_ranking()
            pressure   = _prize_pressure(live)
            expected   = _score_to_expected_pnl(live, self._prize_curve)
            budget     = _risk_budget_from_prize(expected, risk, pressure)

            self._aggression = budget["aggression"]
            log.info(
                "Risk: pnl=%.2f  pressure=%.2f  aggression=%.2f  "
                "sharpe=%.2f  max_dd=%.4f",
                total_pnl, pressure, self._aggression,
                risk["sharpe"], risk["max_dd"],
            )
        except Exception:
            log.exception("Risk update failed — keeping aggression=%.2f", self._aggression)

    def _get_live_ranking(self) -> dict:
        """Fetch live leaderboard ranking; returns a safe fallback on any error."""
        try:
            import requests
            resp = requests.get(
                f"{self._cmi_url}/api/ranking",
                headers=self._auth_headers(),
                timeout=5,
            )
            if resp.ok:
                data = resp.json()
                return {
                    "rank":         data.get("rank", 9999),
                    "score":        float(data.get("score", 0.0)),
                    "leader_score": float(
                        data.get("leaderScore") or data.get("leader_score", 0.0)
                    ),
                    "metric": "pnl",
                }
        except Exception:
            pass
        return {"rank": 9999, "score": 0.0, "leader_score": 0.0, "metric": "pnl"}

    # ── Helpers ────────────────────────────────────────────────────────────

    def _sync_positions(self) -> None:
        """Authoritative position sync from the exchange (overwrites local cache)."""
        try:
            positions = self.get_positions()
            with self._lock:
                self._positions = positions
            log.info("Positions synced: %s", positions)
        except Exception:
            log.exception("Position sync failed")

    def _elapsed_fraction(self) -> float:
        """Fraction of the 24-hour market window that has elapsed (0.0 → 1.0)."""
        now     = datetime.now(self._window_start.tzinfo)
        total   = (self._window_end   - self._window_start).total_seconds()
        elapsed = (now - self._window_start).total_seconds()
        return float(np.clip(elapsed / total, 0.0, 1.0))

    @staticmethod
    def _snap(price: float, tick_size: float) -> float:
        """Round price to the nearest valid tick."""
        if tick_size <= 0:
            return price
        return round(round(price / tick_size) * tick_size, 10)

    def _periodic_maintenance(self) -> None:
        """Called every ~30 s from the main thread.

        Syncs positions from the exchange (authoritative) and recalculates
        the aggression multiplier based on latest PnL and ranking.
        """
        self._sync_positions()
        self._update_risk()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if USERNAME == "your_username_here" or PASSWORD == "your_password_here":
        print(
            "ERROR: Set CMI_USER and CMI_PASS environment variables before running.\n"
            "  export CMI_USER='<your_username>'\n"
            "  export CMI_PASS='<your_password>'\n"
            "  export CMI_URL='<exchange_url>'   # if different from default\n"
            "  export RAPID_API_KEY='<key>'      # for Markets 5/6/7/8\n"
        )
        sys.exit(1)

    bot = IMCBot(CMI_URL, USERNAME, PASSWORD)
    bot.run()
