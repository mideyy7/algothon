"""Statistical arbitrage strategies for IMCity trading.

Two complementary strategies that run alongside the passive market maker:

  A) VolatileOptionArb  — TIDE_SWING (Market 2) mean-reversion.
     Tracks the EWMA of the market mid and z-scores against a rolling window.
     Sells when the market overshots the EWMA, buys when it undershoots.
     Inventory skew widens the required edge when already loaded.

  B) ETFPackageArb      — LON_FLY (Market 8) vs LON_ETF (Market 7).
     Prices LON_FLY using a Monte Carlo simulation of the option package
     payoff under N(mu_ETF, sigma_ETF).  Trades the spread when the market
     deviates from model fair value and delta-hedges the ETF exposure.

Both strategies interface with IMCBot through BotExchangeAdapter, which
enforces the ±100 hard position limit as a final safety net on every order.
"""

import threading
import time
from typing import Callable, Optional

import numpy as np

from imc_template.bot_template import OrderRequest, Side

# ── MC cache constants ────────────────────────────────────────────────────────
_MC_CACHE_TTL      = 1.0   # seconds before recomputing fair value
_MC_CACHE_MID_TOL  = 1.0   # ETF mid must shift by this much to invalidate cache


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

class Rolling:
    """Fixed-capacity rolling window of float values."""

    def __init__(self, n: int = 80):
        self.n = n
        self._x: list[float] = []

    def add(self, v: float) -> None:
        self._x.append(float(v))
        if len(self._x) > self.n:
            self._x.pop(0)

    def mean(self) -> float:
        return float(np.mean(self._x)) if self._x else float("nan")

    def std(self) -> float:
        return float(np.std(self._x)) if len(self._x) > 5 else float("nan")

    def ready(self, min_n: int = 20) -> bool:
        return len(self._x) >= min_n

    def __len__(self) -> int:
        return len(self._x)


class EWMA:
    """Exponentially-weighted moving average."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self._v: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        self._v = x if self._v is None else (1.0 - self.alpha) * self._v + self.alpha * x
        return self._v

    @property
    def value(self) -> Optional[float]:
        return self._v


# ---------------------------------------------------------------------------
# Option payoff and Monte Carlo pricer
# ---------------------------------------------------------------------------

def etf_option_package_payoff(S: np.ndarray) -> np.ndarray:
    """Vectorised LON_FLY payoff for an array of ETF settlement values.

    Structure (from the competition spec):
        +2 × Put(K=6200)   +1 × Call(K=6200)
        −2 × Call(K=6600)  +3 × Call(K=7000)

    Payoff shape:
        S < 6200       : grows linearly downward   (net delta = −2 from two puts)
        6200 ≤ S < 6600: grows slowly upward        (delta = +1 from long call)
        6600 ≤ S < 7000: dips back downward         (delta = −1, short call dominates)
        S ≥ 7000       : grows steeply upward       (delta = +2, three calls dominating)
    """
    return (
        2.0 * np.maximum(0.0, 6200.0 - S)
        + 1.0 * np.maximum(0.0, S - 6200.0)
        - 2.0 * np.maximum(0.0, S - 6600.0)
        + 3.0 * np.maximum(0.0, S - 7000.0)
    )


def mc_fair_value(
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    mu: float,
    sigma: float,
    n_sims: int = 45_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte Carlo fair value E[payoff(S)] with S ~ N(mu, sigma).

    Uses antithetic variates for ~2× variance reduction at no extra cost:
    each pair (S, 2μ−S) shares the same uniform draw, halving the estimator
    variance relative to plain Monte Carlo with the same n_sims.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = max(1e-6, sigma)
    half  = n_sims // 2
    z     = rng.standard_normal(size=half)
    s_pos = mu + sigma * z
    s_neg = mu - sigma * z          # antithetic pair
    payoffs = (payoff_fn(s_pos) + payoff_fn(s_neg)) / 2.0
    return float(np.mean(payoffs))


def mc_delta(
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    mu: float,
    sigma: float,
    bump: float = 1.0,
    n_sims: int = 25_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte Carlo delta ∂E[payoff]/∂mu via central finite difference.

    Sharing the same set of draws for the up and down bumps eliminates
    simulation noise from the difference, giving a near-exact delta.
    """
    if rng is None:
        rng = np.random.default_rng()
    sigma = max(1e-6, sigma)
    sims  = rng.normal(mu, sigma, size=n_sims)
    up    = payoff_fn(sims + bump)
    dn    = payoff_fn(sims - bump)
    return float(np.mean((up - dn) / (2.0 * bump)))


# ---------------------------------------------------------------------------
# Exchange adapter — bridges arb strategies to IMCBot
# ---------------------------------------------------------------------------

class BotExchangeAdapter:
    """Maps the ExchangeAPI interface onto IMCBot's actual order infrastructure.

    Key responsibilities:
      - Maintain a live mid-price cache updated by IMCBot.on_orderbook().
      - Proxy get_position() to IMCBot._positions (under the bot's lock).
      - Enforce the ±100 hard position limit on EVERY order before submission.
        This is the last line of defence: even if an arb strategy mis-sizes,
        no order will push a product past the exchange hard cap.
    """

    HARD_LIMIT = 100

    def __init__(self, bot: "IMCBot") -> None:
        self._bot  = bot
        self._lock = threading.Lock()
        self._mids: dict[str, float] = {}

    def update_mid(self, product: str, mid: float) -> None:
        """Store the current mid price for a product (called on every tick)."""
        with self._lock:
            self._mids[product] = mid

    def get_mid(self, product: str) -> float:
        """Return the last known mid price, or NaN if never seen."""
        with self._lock:
            return self._mids.get(product, float("nan"))

    def get_position(self, product: str) -> int:
        """Return the current net position from IMCBot's position cache."""
        with self._bot._lock:
            return self._bot._positions.get(product, 0)

    def place_order(
        self,
        product: str,
        side: str,
        price: float,
        qty: int,
    ) -> bool:
        """Submit a single order, hard-capped at ±HARD_LIMIT.

        Returns True if an order was submitted, False if it was capped to 0.

        Thread-safety note: there is an inherent TOCTOU window between reading
        the position and submitting the order.  The 5-unit MM buffer (MAX_POSITION=95)
        and the strategy-level max_pos caps in each arb class provide the primary
        guard; this hard cap is the final backstop.
        """
        pos       = self.get_position(product)
        side_upper = side.upper()

        if side_upper == "BUY":
            headroom = max(0, self.HARD_LIMIT - pos)
        else:
            headroom = max(0, self.HARD_LIMIT + pos)

        safe_qty = min(int(qty), headroom)
        if safe_qty <= 0:
            return False

        cmi_side = Side.BUY if side_upper == "BUY" else Side.SELL
        self._bot.send_order(
            OrderRequest(product=product, price=price, side=cmi_side, volume=safe_qty)
        )
        return True


# ---------------------------------------------------------------------------
# Strategy A: VolatileOptionArb — TIDE_SWING mean-reversion
# ---------------------------------------------------------------------------

class VolatileOptionArb:
    """Mean-reversion stat arb on TIDE_SWING (Market 2).

    Algorithm:
      1. Maintain an EWMA of the market mid as the "fair" reference.
      2. Maintain a rolling window for z-score normalisation.
      3. Block trading when |z| > vol_block_z (avoids getting run over
         during genuine large moves).
      4. Trade when |mis| = |mid − fair| > effective_edge, where
         effective_edge is widened by inventory skew.

    ⚠️  Drift caveat: TIDE_SWING accumulates throughout the session, so both
    the EWMA and rolling mean will exhibit upward drift.  The EWMA adapts
    quickly (α=0.06 → ~16-obs memory), but early in the session the signal
    may be noisy.  Consider raising `edge` for the first 2 hours.

    Args:
        api:            BotExchangeAdapter tied to the live bot.
        product:        CMI symbol — should be "TIDE_SWING".
        fair_ema_alpha: EWMA smoothing factor.
        window:         Rolling window depth for z-score.
        edge:           Minimum mispricing (price units) required to trade.
        max_pos:        Strategy-level position allocation (≤ HARD_LIMIT).
        clip:           Max contracts per single order.
        vol_block_z:    |z| threshold above which trading is suspended.
        inv_skew:       Edge multiplier per unit of inventory fraction.
        cooldown:       Min seconds between consecutive orders.
    """

    def __init__(
        self,
        api:             BotExchangeAdapter,
        product:         str   = "TIDE_SWING",
        fair_ema_alpha:  float = 0.06,
        window:          int   = 90,
        edge:            float = 18.0,
        max_pos:         int   = 20,
        clip:            int   = 4,
        vol_block_z:     float = 2.8,
        inv_skew:        float = 0.35,
        cooldown:        float = 0.25,
    ):
        self.api         = api
        self.product     = product
        self.fair_ema    = EWMA(alpha=fair_ema_alpha)
        self.rolling     = Rolling(n=window)
        self.edge        = edge
        self.max_pos     = min(max_pos, BotExchangeAdapter.HARD_LIMIT)
        self.clip        = clip
        self.vol_block_z = vol_block_z
        self.inv_skew    = inv_skew
        self.cooldown    = cooldown
        self._last_trade_t: float = 0.0

    def step(self, mid: Optional[float] = None) -> dict:
        """Evaluate market conditions and optionally send one order.

        Args:
            mid: Current market mid price.  If None, fetched from the adapter.

        Returns:
            Status dict with keys: p_mid, fair, z, pos, effective_edge, mis, action.
        """
        p_mid = mid if mid is not None else self.api.get_mid(self.product)
        if np.isnan(p_mid):
            return {"action": "NO_DATA"}

        fair = self.fair_ema.update(p_mid)
        self.rolling.add(p_mid)
        pos  = self.api.get_position(self.product)

        if not self.rolling.ready():
            return {"p_mid": p_mid, "fair": fair, "z": float("nan"), "pos": pos, "action": "WARMUP"}

        mu = self.rolling.mean()
        sd = self.rolling.std()
        if np.isnan(sd) or sd < 1e-9:
            return {"p_mid": p_mid, "fair": fair, "z": float("nan"), "pos": pos, "action": "NO_VOL"}

        z = (p_mid - mu) / sd
        if abs(z) > self.vol_block_z:
            return {"p_mid": p_mid, "fair": fair, "z": z, "pos": pos, "action": "BLOCK_VOL"}

        # Inventory-skewed edge: widens proportionally as we accumulate inventory
        inv_frac       = 0.0 if self.max_pos == 0 else pos / self.max_pos
        effective_edge = self.edge * (1.0 + self.inv_skew * abs(inv_frac))

        mis       = p_mid - fair
        room_buy  = max(0, self.max_pos - pos)
        room_sell = max(0, self.max_pos + pos)

        now = time.monotonic()
        if now - self._last_trade_t < self.cooldown:
            return {
                "p_mid": p_mid, "fair": fair, "z": z, "pos": pos,
                "effective_edge": effective_edge, "mis": mis, "action": "COOLDOWN",
            }

        act = "HOLD"
        if mis > effective_edge and room_sell > 0:
            qty = int(min(self.clip, room_sell))
            if self.api.place_order(self.product, "SELL", p_mid, qty):
                self._last_trade_t = now
                act = f"SELL {qty}"
        elif mis < -effective_edge and room_buy > 0:
            qty = int(min(self.clip, room_buy))
            if self.api.place_order(self.product, "BUY", p_mid, qty):
                self._last_trade_t = now
                act = f"BUY {qty}"

        return {
            "p_mid": p_mid, "fair": fair, "z": z, "pos": pos,
            "effective_edge": effective_edge, "mis": mis, "action": act,
        }


# ---------------------------------------------------------------------------
# Strategy B: ETFPackageArb — LON_ETF vs LON_FLY Monte Carlo pricer
# ---------------------------------------------------------------------------

class ETFPackageArb:
    """Model-based arb between LON_ETF (Market 7) and LON_FLY (Market 8).

    Algorithm:
      1. Track ETF mid via EWMA (mu) and rolling std (sigma).
      2. Price LON_FLY fair value = MC E[payoff(ETF_settlement)] under
         ETF_settlement ~ N(mu, sigma).  Results are cached for 1 second
         to avoid running 45k sims on every SSE tick.
      3. Trade LON_FLY when |market_mid − fair| > edge.
      4. Delta-hedge with LON_ETF: when selling LON_FLY, buy Δ units of
         LON_ETF (and vice versa).  Delta is computed via MC finite difference
         only when a trade fires (not on every tick).

    Delta sign analysis by ETF level:
        S < 6200  : delta ≈ −2  (puts dominate — short delta)
        6200–6600 : delta ≈ +1  (long call)
        6600–7000 : delta ≈ −1  (short call region)
        S > 7000  : delta ≈ +2  (three calls)
    The hedge direction therefore flips in the 6600–7000 range — the code
    handles this correctly via `hedge_qty` sign.

    Args:
        api:         BotExchangeAdapter instance.
        etf:         CMI symbol for LON_ETF.
        pack:        CMI symbol for LON_FLY.
        model_alpha: EWMA alpha for ETF fair-value tracking.
        edge:        Min mispricing (price units) before trading LON_FLY.
        max_pos:     Strategy-level position cap for LON_FLY.
        clip:        Max LON_FLY contracts per order.
        hedge_clip:  Max LON_ETF contracts per hedge.
        mc_sims:     Sims for fair-value calculation (uses antithetic variates).
        delta_sims:  Sims for delta calculation (only on trade events).
    """

    def __init__(
        self,
        api:          BotExchangeAdapter,
        etf:          str   = "LON_ETF",
        pack:         str   = "LON_FLY",
        model_alpha:  float = 0.05,
        edge:         float = 12.0,
        max_pos:      int   = 40,
        clip:         int   = 6,
        hedge_clip:   int   = 12,
        mc_sims:      int   = 45_000,
        delta_sims:   int   = 25_000,
    ):
        self.api        = api
        self.etf        = etf
        self.pack       = pack
        self.edge       = edge
        self.max_pos    = min(max_pos, BotExchangeAdapter.HARD_LIMIT)
        self.clip       = clip
        self.hedge_clip = hedge_clip
        self.mc_sims    = mc_sims
        self.delta_sims = delta_sims
        self.rng        = np.random.default_rng(7)

        self.etf_model   = EWMA(alpha=model_alpha)
        self.etf_rolling = Rolling(n=120)

        # MC fair-value cache
        self._cache_mu:    Optional[float] = None
        self._cache_sigma: Optional[float] = None
        self._cache_fair:  Optional[float] = None
        self._cache_time:  float           = 0.0

    def _get_cached_fair(self, mu: float, sigma: float) -> float:
        """Return MC fair value from cache; recompute only when stale or inputs shifted."""
        now       = time.monotonic()
        stale     = (now - self._cache_time) > _MC_CACHE_TTL
        mu_moved  = self._cache_mu    is None or abs(mu    - self._cache_mu)    > _MC_CACHE_MID_TOL
        sig_moved = self._cache_sigma is None or abs(sigma - self._cache_sigma) > 1.0

        if stale or mu_moved or sig_moved:
            self._cache_fair  = mc_fair_value(
                etf_option_package_payoff, mu, sigma,
                n_sims=self.mc_sims, rng=self.rng,
            )
            self._cache_mu    = mu
            self._cache_sigma = sigma
            self._cache_time  = now

        return self._cache_fair  # type: ignore[return-value]

    def step(
        self,
        etf_mid:  Optional[float] = None,
        pack_mid: Optional[float] = None,
    ) -> dict:
        """Evaluate and potentially trade the ETF / package spread.

        Args:
            etf_mid:  Current LON_ETF mid.  Falls back to adapter cache.
            pack_mid: Current LON_FLY mid.  Falls back to adapter cache.

        Returns:
            Status dict with keys: etf_mid, mu, sigma, pack_mid, fair, mis,
            pack_pos, action.
        """
        if etf_mid  is None:
            etf_mid  = self.api.get_mid(self.etf)
        if pack_mid is None:
            pack_mid = self.api.get_mid(self.pack)

        if np.isnan(etf_mid) or np.isnan(pack_mid):
            return {"action": "NO_DATA"}

        mu = self.etf_model.update(etf_mid)
        self.etf_rolling.add(etf_mid)

        if not self.etf_rolling.ready():
            return {
                "etf_mid": etf_mid, "pack_mid": pack_mid,
                "fair": float("nan"), "mis": float("nan"), "action": "WARMUP",
            }

        sigma = self.etf_rolling.std()
        if np.isnan(sigma) or sigma < 1e-6:
            return {
                "etf_mid": etf_mid, "pack_mid": pack_mid,
                "fair": float("nan"), "mis": float("nan"), "action": "NO_VOL",
            }

        fair     = self._get_cached_fair(mu, sigma)
        mis      = pack_mid - fair
        pack_pos = self.api.get_position(self.pack)
        room_buy  = max(0, self.max_pos - pack_pos)
        room_sell = max(0, self.max_pos + pack_pos)

        act = "HOLD"

        if mis > self.edge and room_sell > 0:
            qty = int(min(self.clip, room_sell))
            if self.api.place_order(self.pack, "SELL", pack_mid, qty):
                delta     = mc_delta(etf_option_package_payoff, mu, sigma,
                                     n_sims=self.delta_sims, rng=self.rng)
                hedge_qty = int(np.clip(round(delta * qty), -self.hedge_clip, self.hedge_clip))
                if hedge_qty > 0:
                    self.api.place_order(self.etf, "BUY",  etf_mid,  hedge_qty)
                elif hedge_qty < 0:
                    self.api.place_order(self.etf, "SELL", etf_mid, -hedge_qty)
                act = f"SELL_PACK {qty} + HEDGE_ETF {hedge_qty:+d}"

        elif mis < -self.edge and room_buy > 0:
            qty = int(min(self.clip, room_buy))
            if self.api.place_order(self.pack, "BUY", pack_mid, qty):
                delta     = mc_delta(etf_option_package_payoff, mu, sigma,
                                     n_sims=self.delta_sims, rng=self.rng)
                hedge_qty = int(np.clip(round(delta * qty), -self.hedge_clip, self.hedge_clip))
                if hedge_qty > 0:
                    self.api.place_order(self.etf, "SELL", etf_mid,  hedge_qty)
                elif hedge_qty < 0:
                    self.api.place_order(self.etf, "BUY",  etf_mid, -hedge_qty)
                act = f"BUY_PACK {qty} + HEDGE_ETF {-hedge_qty:+d}"

        return {
            "etf_mid": etf_mid, "mu": mu, "sigma": sigma,
            "pack_mid": pack_mid, "fair": fair, "mis": mis,
            "pack_pos": pack_pos, "action": act,
        }
