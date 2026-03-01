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
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, List

import numpy as np

from imc_template.bot_template import OrderRequest, Side

@dataclass(frozen=True)
class BBO:
    mid: float
    best_bid: Optional[float]
    best_ask: Optional[float]

@dataclass(frozen=True)
class BasketLeg:
    product: str
    weight: int = 1

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


def solve_sigma_impl_bisect(
    mu: float,
    target_price: float,
    *,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 2500.0,
    tol_price: float = 1.0,
    max_iter: int = 22,
    mc_sims: int = 35_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Solve for sigma so that MC_fair_value(mu, sigma) ~ target_price using bisection.
    """
    if rng is None:
        rng = np.random.default_rng()

    if target_price <= 0:
        return float(sigma_lo)

    f_lo = mc_fair_value(etf_option_package_payoff, mu, sigma_lo, n_sims=mc_sims, rng=rng)
    f_hi = mc_fair_value(etf_option_package_payoff, mu, sigma_hi, n_sims=mc_sims, rng=rng)

    if target_price <= f_lo:
        return float(sigma_lo)
    if target_price >= f_hi:
        return float(sigma_hi)

    lo, hi = float(sigma_lo), float(sigma_hi)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        f_mid = mc_fair_value(etf_option_package_payoff, mu, mid, n_sims=mc_sims, rng=rng)

        if abs(f_mid - target_price) <= tol_price:
            return float(mid)

        if f_mid < target_price:
            lo = mid
        else:
            hi = mid

    return float(0.5 * (lo + hi))


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
        self._bbos: dict[str, BBO] = {}

    def update_bbo(self, product: str, bbo: BBO) -> None:
        """Store the current BBO for a product (called on every tick by main.py)."""
        with self._lock:
            self._mids[product] = bbo.mid
            self._bbos[product] = bbo

    def get_best_bid_ask(self, product: str) -> BBO:
        """Return the BBO, or nan if never seen."""
        with self._lock:
            return self._bbos.get(product, BBO(float("nan"), None, None))

    def update_mid(self, product: str, mid: float) -> None:
        """Store the current mid price for a product (fallback)."""
        with self._lock:
            self._mids[product] = mid
            if product not in self._bbos:
                self._bbos[product] = BBO(mid, None, None)

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
# Strategy B: ETFBasketArb — LON_ETF vs (TIDE_SPOT + WX_SPOT + LHR_COUNT)
# ---------------------------------------------------------------------------

class ETFBasketArb:
    """Classic identity arbitrage for IMCity (Market 7).

    Market 7 (LON_ETF) settles to Market 1 + Market 3 + Market 5.
    Trades deviations between the ETF and the live basket value.
    """

    def __init__(
        self,
        api: BotExchangeAdapter,
        *,
        etf: str = "LON_ETF",
        legs: Optional[List[BasketLeg]] = None,
        edge: float = 12.0,
        max_pos_etf: int = 60,
        max_pos_leg: int = 60,
        clip: int = 6,
        leg_clip: int = 6,
        cooldown: float = 0.30,
        price_mode: str = "CROSS",
    ) -> None:
        self.api = api
        self.etf = etf
        self.legs = legs or [
            BasketLeg("TIDE_SPOT"),
            BasketLeg("WX_SPOT"),
            BasketLeg("LHR_COUNT"),
        ]
        self.edge = float(edge)
        self.max_pos_etf = min(int(max_pos_etf), api.HARD_LIMIT)
        self.max_pos_leg = min(int(max_pos_leg), api.HARD_LIMIT)
        self.clip = int(clip)
        self.leg_clip = int(leg_clip)
        self.cooldown = float(cooldown)
        self.price_mode = price_mode.upper()
        if self.price_mode not in ("MID", "CROSS"):
            raise ValueError("price_mode must be MID or CROSS")
        self._last_t = 0.0

    def _mid_or_cross_price(self, product: str, side: str) -> float:
        bbo = self.api.get_best_bid_ask(product)
        if self.price_mode == "MID":
            return bbo.mid
        side_u = side.upper()
        if side_u == "BUY":
            return float(bbo.best_ask) if bbo.best_ask is not None else bbo.mid
        else:
            return float(bbo.best_bid) if bbo.best_bid is not None else bbo.mid

    def _basket_value(self) -> float:
        total = 0.0
        for leg in self.legs:
            mid = self.api.get_mid(leg.product)
            if np.isnan(mid):
                return float("nan")
            total += leg.weight * mid
        return float(total)

    def _room(self, product: str, cap: int) -> tuple[int, int, int]:
        pos = self.api.get_position(product)
        return pos, max(0, cap - pos), max(0, cap + pos)

    def step(self) -> Dict[str, Any]:
        now = time.monotonic()
        if now - self._last_t < self.cooldown:
            return {"action": "COOLDOWN"}

        etf_mid = self.api.get_mid(self.etf)
        basket_mid = self._basket_value()
        if np.isnan(etf_mid) or np.isnan(basket_mid):
            return {"action": "NO_DATA"}

        spread = float(etf_mid - basket_mid)
        etf_pos, etf_room_buy, etf_room_sell = self._room(self.etf, self.max_pos_etf)

        act = "HOLD"

        if spread > self.edge and etf_room_sell > 0:
            qty_etf = int(min(self.clip, etf_room_sell))

            q_caps = []
            for leg in self.legs:
                _, leg_room_buy, _ = self._room(leg.product, self.max_pos_leg)
                q_caps.append(int(min(self.leg_clip, leg_room_buy)))

            qty = int(min([qty_etf] + q_caps))
            if qty <= 0:
                return {"etf_mid": etf_mid, "basket_mid": basket_mid, "spread": spread, "action": "CAPPED"}

            ok = True
            for leg in self.legs:
                px = self._mid_or_cross_price(leg.product, "BUY")
                ok &= self.api.place_order(leg.product, "BUY", px, qty * abs(leg.weight))

            px_etf = self._mid_or_cross_price(self.etf, "SELL")
            ok &= self.api.place_order(self.etf, "SELL", px_etf, qty)

            if ok:
                self._last_t = now
                act = f"SELL_ETF {qty} / BUY_BASKET {qty}"

        elif spread < -self.edge and etf_room_buy > 0:
            qty_etf = int(min(self.clip, etf_room_buy))

            q_caps = []
            for leg in self.legs:
                _, _, leg_room_sell = self._room(leg.product, self.max_pos_leg)
                q_caps.append(int(min(self.leg_clip, leg_room_sell)))

            qty = int(min([qty_etf] + q_caps))
            if qty <= 0:
                return {"etf_mid": etf_mid, "basket_mid": basket_mid, "spread": spread, "action": "CAPPED"}

            ok = True
            px_etf = self._mid_or_cross_price(self.etf, "BUY")
            ok &= self.api.place_order(self.etf, "BUY", px_etf, qty)

            for leg in self.legs:
                px = self._mid_or_cross_price(leg.product, "SELL")
                ok &= self.api.place_order(leg.product, "SELL", px, qty * abs(leg.weight))

            if ok:
                self._last_t = now
                act = f"BUY_ETF {qty} / SELL_BASKET {qty}"

        return {
            "etf_mid": etf_mid, "basket_mid": basket_mid, "spread": spread,
            "etf_pos": etf_pos, "action": act,
        }


# ---------------------------------------------------------------------------
# Strategy C: ETFPackageImpliedVolArb — LON_FLY Implied Vol arb
# ---------------------------------------------------------------------------

class ETFPackageImpliedVolArb:
    """Market 8 implied-vol arb for IMCity."""
    
    def __init__(
        self,
        api: BotExchangeAdapter,
        *,
        etf: str = "LON_ETF",
        pack: str = "LON_FLY",
        model_alpha: float = 0.05,
        window: int = 120,
        vol_edge: float = 40.0,
        max_pos: int = 40,
        clip: int = 6,
        hedge_clip: int = 12,
        iv_mc_sims: int = 35_000,
        delta_sims: int = 25_000,
        bump: float = 1.0,
        cooldown: float = 0.35,
        price_mode: str = "CROSS",
    ) -> None:
        self.api = api
        self.etf = etf
        self.pack = pack

        self.mu_model = EWMA(alpha=model_alpha)
        self.etf_rolling = Rolling(n=window)

        self.vol_edge = float(vol_edge)
        self.max_pos = min(int(max_pos), api.HARD_LIMIT)
        self.clip = int(clip)
        self.hedge_clip = int(hedge_clip)

        self.iv_mc_sims = int(iv_mc_sims)
        self.delta_sims = int(delta_sims)
        self.bump = float(bump)

        self.cooldown = float(cooldown)
        self.price_mode = price_mode.upper()
        if self.price_mode not in ("MID", "CROSS"):
            raise ValueError("price_mode must be MID or CROSS")

        self.rng = np.random.default_rng(11)

        self._iv_cache_time = 0.0
        self._iv_cache_mu: Optional[float] = None
        self._iv_cache_pack: Optional[float] = None
        self._iv_cache_sigma: Optional[float] = None
        self._iv_ttl = 1.0
        self._mu_tol = 1.0
        self._pack_tol = 1.0

        self._last_t = 0.0

    def _price(self, product: str, side: str) -> float:
        bbo = self.api.get_best_bid_ask(product)
        if self.price_mode == "MID":
            return bbo.mid
        side_u = side.upper()
        if side_u == "BUY":
            return float(bbo.best_ask) if bbo.best_ask is not None else bbo.mid
        return float(bbo.best_bid) if bbo.best_bid is not None else bbo.mid

    def _sigma_impl_cached(self, mu: float, pack_mid: float) -> float:
        now = time.monotonic()
        stale = (now - self._iv_cache_time) > self._iv_ttl
        mu_moved = self._iv_cache_mu is None or abs(mu - self._iv_cache_mu) > self._mu_tol
        pack_moved = self._iv_cache_pack is None or abs(pack_mid - self._iv_cache_pack) > self._pack_tol

        if stale or mu_moved or pack_moved:
            sig = solve_sigma_impl_bisect(
                mu=mu,
                target_price=pack_mid,
                tol_price=1.2,
                mc_sims=self.iv_mc_sims,
                rng=self.rng,
            )
            self._iv_cache_mu = float(mu)
            self._iv_cache_pack = float(pack_mid)
            self._iv_cache_sigma = float(sig)
            self._iv_cache_time = now

        return float(self._iv_cache_sigma)

    def step(self) -> Dict[str, Any]:
        now = time.monotonic()
        if now - self._last_t < self.cooldown:
            return {"action": "COOLDOWN"}

        etf_mid = self.api.get_mid(self.etf)
        pack_mid = self.api.get_mid(self.pack)
        if np.isnan(etf_mid) or np.isnan(pack_mid):
            return {"action": "NO_DATA"}

        mu = self.mu_model.update(etf_mid)
        self.etf_rolling.add(etf_mid)

        if not self.etf_rolling.ready():
            return {"etf_mid": etf_mid, "pack_mid": pack_mid, "action": "WARMUP"}

        sigma_est = self.etf_rolling.std()
        if np.isnan(sigma_est) or sigma_est < 1e-6:
            return {"etf_mid": etf_mid, "pack_mid": pack_mid, "action": "NO_VOL"}

        sigma_impl = self._sigma_impl_cached(mu, pack_mid)
        vol_spread = float(sigma_impl - sigma_est)

        pack_pos = self.api.get_position(self.pack)
        room_buy = max(0, self.max_pos - pack_pos)
        room_sell = max(0, self.max_pos + pack_pos)

        act = "HOLD"

        if vol_spread > self.vol_edge and room_sell > 0:
            qty = int(min(self.clip, room_sell))
            px_pack = self._price(self.pack, "SELL")
            if self.api.place_order(self.pack, "SELL", px_pack, qty):
                delta = mc_delta(etf_option_package_payoff, mu, sigma_est, bump=self.bump, n_sims=self.delta_sims, rng=self.rng)
                hedge_qty = int(np.clip(round(delta * qty), -self.hedge_clip, self.hedge_clip))
                if hedge_qty > 0:
                    self.api.place_order(self.etf, "BUY", self._price(self.etf, "BUY"), hedge_qty)
                elif hedge_qty < 0:
                    self.api.place_order(self.etf, "SELL", self._price(self.etf, "SELL"), -hedge_qty)

                self._last_t = now
                act = f"SELL_PACK {qty} / IV_RICH {vol_spread:+.1f} / HEDGE_ETF {hedge_qty:+d}"

        elif vol_spread < -self.vol_edge and room_buy > 0:
            qty = int(min(self.clip, room_buy))
            px_pack = self._price(self.pack, "BUY")
            if self.api.place_order(self.pack, "BUY", px_pack, qty):
                delta = mc_delta(etf_option_package_payoff, mu, sigma_est, bump=self.bump, n_sims=self.delta_sims, rng=self.rng)
                hedge_qty = int(np.clip(round(delta * qty), -self.hedge_clip, self.hedge_clip))
                if hedge_qty > 0:
                    self.api.place_order(self.etf, "SELL", self._price(self.etf, "SELL"), hedge_qty)
                elif hedge_qty < 0:
                    self.api.place_order(self.etf, "BUY", self._price(self.etf, "BUY"), -hedge_qty)

                self._last_t = now
                act = f"BUY_PACK {qty} / IV_CHEAP {vol_spread:+.1f} / HEDGE_ETF {-hedge_qty:+d}"

        return {
            "etf_mid": etf_mid,
            "mu": mu,
            "sigma_est": sigma_est,
            "pack_mid": pack_mid,
            "sigma_impl": sigma_impl,
            "vol_spread": vol_spread,
            "pack_pos": pack_pos,
            "action": act,
        }


# ---------------------------------------------------------------------------
# Rolling helpers for regression-based stat arb (from new algo files)
# ---------------------------------------------------------------------------

class RollingRV:
    """Rolling realized volatility (log-return RMS) over the last n prices."""

    def __init__(self, n: int = 120):
        self._px: deque = deque(maxlen=n + 1)

    def update(self, p: float) -> float:
        self._px.append(float(p))
        if len(self._px) < 25:
            return float("nan")
        r = np.diff(np.log(np.array(self._px)))
        return float(np.sqrt(np.mean(r * r)))


class RollingRidge:
    """Rolling ridge regression: PACK ~ b0 + b1*ETF + b2*ETF² + b3*RV."""

    def __init__(self, window: int = 700, ridge: float = 5e-2):
        self._X: deque = deque(maxlen=window)
        self._y: deque = deque(maxlen=window)
        self._ridge = float(ridge)
        self.beta = np.zeros(4)

    def add(self, etf: float, rv: float, pack: float) -> None:
        if np.isnan(rv):
            return
        self._X.append(np.array([1.0, etf, etf * etf, rv], dtype=float))
        self._y.append(float(pack))

    def fit(self) -> np.ndarray:
        if len(self._y) < 120:
            return self.beta
        X = np.vstack(self._X)
        y = np.array(self._y)
        lam = self._ridge * np.eye(4)
        self.beta = np.linalg.solve(X.T @ X + lam, X.T @ y)
        return self.beta

    def predict(self, etf: float, rv: float) -> float:
        x = np.array([1.0, etf, etf * etf, rv], dtype=float)
        return float(x @ self.beta)

    def dprice_detf(self, etf: float) -> float:
        return float(self.beta[1] + 2.0 * self.beta[2] * etf)


class RollingZ:
    """Rolling z-score of a residual series."""

    def __init__(self, n: int = 350):
        self._r: deque = deque(maxlen=n)

    def add(self, v: float) -> None:
        self._r.append(float(v))

    def z(self, v: float) -> float:
        if len(self._r) < 80:
            return float("nan")
        arr = np.array(self._r, dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std()) + 1e-9
        return (v - mu) / sd


class RollingRidge3:
    """Rolling ridge regression (3-feature): Y ~ b0 + b1*X + b2*RV."""

    def __init__(self, window: int = 600, ridge: float = 5e-2):
        self._X: deque = deque(maxlen=window)
        self._y: deque = deque(maxlen=window)
        self._ridge = float(ridge)
        self.beta = np.zeros(3)

    def add(self, x: float, rv: float, y: float) -> None:
        if np.isnan(rv):
            return
        self._X.append(np.array([1.0, x, rv], dtype=float))
        self._y.append(float(y))

    def fit(self) -> np.ndarray:
        if len(self._y) < 120:
            return self.beta
        X = np.vstack(self._X)
        y = np.array(self._y)
        lam = self._ridge * np.eye(3)
        self.beta = np.linalg.solve(X.T @ X + lam, X.T @ y)
        return self.beta

    def predict(self, x: float, rv: float) -> float:
        v = np.array([1.0, x, rv], dtype=float)
        return float(v @ self.beta)

    def dy_dx(self) -> float:
        return float(self.beta[1])


# ---------------------------------------------------------------------------
# Strategy D: ETFPackStatArb — LON_ETF vs LON_FLY rolling ridge stat arb
# ---------------------------------------------------------------------------

class ETFPackStatArb:
    """Statistical arbitrage between LON_FLY (Market 8) and LON_ETF (Market 7).

    Uses rolling ridge regression (PACK ~ b0 + b1*ETF + b2*ETF² + b3*RV) to
    estimate fair value of the option package, then trades z-score deviations
    of the residual (market price − model price).

    Delta-hedges in ETF using the regression's dPACK/dETF evaluated at the
    current ETF level.

    Position limits:
        max_pack_pos, max_etf_pos — strategy-level soft caps.
        BotExchangeAdapter enforces ±100 as the final hard backstop.
    """

    def __init__(
        self,
        api: BotExchangeAdapter,
        etf: str = "LON_ETF",
        pack: str = "LON_FLY",
        max_pack_pos: int = 18,
        max_etf_pos: int = 35,
        clip_pack: int = 3,
        clip_etf: int = 6,
        z_enter: float = 2.5,
        z_exit: float = 0.6,
        cooldown: float = 0.15,
    ):
        self.api = api
        self.etf = etf
        self.pack = pack
        self.max_pack_pos = min(int(max_pack_pos), api.HARD_LIMIT)
        self.max_etf_pos = min(int(max_etf_pos), api.HARD_LIMIT)
        self.clip_pack = int(clip_pack)
        self.clip_etf = int(clip_etf)
        self.z_enter = float(z_enter)
        self.z_exit = float(z_exit)
        self.cooldown = float(cooldown)
        self._last_t = 0.0

        self._rv = RollingRV(n=120)
        self._reg = RollingRidge(window=700, ridge=5e-2)
        self._resid = RollingZ(n=350)

    def _flatten_pair(self, clip: int = 8) -> None:
        for p in [self.pack, self.etf]:
            pos = self.api.get_position(p)
            if pos == 0:
                continue
            bbo = self.api.get_best_bid_ask(p)
            q = int(min(abs(pos), clip))
            if pos > 0 and bbo.best_bid is not None:
                self.api.place_order(p, "SELL", bbo.best_bid, q)
            elif pos < 0 and bbo.best_ask is not None:
                self.api.place_order(p, "BUY", bbo.best_ask, q)

    def step(self) -> dict:
        now = time.monotonic()
        if now - self._last_t < self.cooldown:
            return {"act": "COOLDOWN"}

        etf_mid = self.api.get_mid(self.etf)
        pack_mid = self.api.get_mid(self.pack)
        if np.isnan(etf_mid) or np.isnan(pack_mid):
            return {"act": "NO_DATA"}

        bbo_p = self.api.get_best_bid_ask(self.pack)
        bbo_e = self.api.get_best_bid_ask(self.etf)
        bid_p, ask_p = bbo_p.best_bid, bbo_p.best_ask
        bid_e, ask_e = bbo_e.best_bid, bbo_e.best_ask

        rv = self._rv.update(etf_mid)
        if np.isnan(rv):
            return {"act": "WARMUP", "z": float("nan"), "resid": float("nan"), "fair": float("nan")}

        self._reg.add(etf_mid, rv, pack_mid)
        self._reg.fit()

        fair = self._reg.predict(etf_mid, rv)
        r = pack_mid - fair
        if not np.isnan(r):
            self._resid.add(r)
        z = self._resid.z(r)

        pack_pos = self.api.get_position(self.pack)
        etf_pos = self.api.get_position(self.etf)

        # Exit: flatten when spread has normalized
        if (not np.isnan(z)) and abs(z) < self.z_exit:
            if pack_pos != 0 or etf_pos != 0:
                self._flatten_pair(clip=8)
                self._last_t = now
                return {"act": "FLATTEN_EXIT", "z": z, "resid": r, "fair": fair}

        if np.isnan(z):
            return {"act": "WARMUP", "z": z, "resid": r, "fair": fair}

        dP_dE = self._reg.dprice_detf(etf_mid)

        # Residual high => pack rich => SELL pack + hedge ETF long
        if z > self.z_enter:
            room_sell_pack = max(0, self.max_pack_pos + pack_pos)
            if room_sell_pack > 0 and bid_p is not None:
                q_pack = int(min(self.clip_pack, room_sell_pack))
                self.api.place_order(self.pack, "SELL", bid_p, q_pack)

                hedge = int(np.clip(dP_dE * q_pack, -self.clip_etf, self.clip_etf))
                room_buy_etf  = max(0, self.max_etf_pos - etf_pos)
                room_sell_etf = max(0, self.max_etf_pos + etf_pos)
                if hedge > 0 and room_buy_etf > 0 and ask_e is not None:
                    self.api.place_order(self.etf, "BUY",  ask_e, int(min(hedge,  room_buy_etf)))
                elif hedge < 0 and room_sell_etf > 0 and bid_e is not None:
                    self.api.place_order(self.etf, "SELL", bid_e, int(min(-hedge, room_sell_etf)))

                self._last_t = now
                return {"act": f"SELL_PACK {q_pack} HEDGE {hedge}", "z": z, "resid": r, "fair": fair}

        # Residual low => pack cheap => BUY pack + hedge ETF short
        if z < -self.z_enter:
            room_buy_pack = max(0, self.max_pack_pos - pack_pos)
            if room_buy_pack > 0 and ask_p is not None:
                q_pack = int(min(self.clip_pack, room_buy_pack))
                self.api.place_order(self.pack, "BUY", ask_p, q_pack)

                hedge = int(np.clip(dP_dE * q_pack, -self.clip_etf, self.clip_etf))
                room_buy_etf  = max(0, self.max_etf_pos - etf_pos)
                room_sell_etf = max(0, self.max_etf_pos + etf_pos)
                if hedge > 0 and room_sell_etf > 0 and bid_e is not None:
                    self.api.place_order(self.etf, "SELL", bid_e, int(min(hedge,  room_sell_etf)))
                elif hedge < 0 and room_buy_etf > 0 and ask_e is not None:
                    self.api.place_order(self.etf, "BUY",  ask_e, int(min(-hedge, room_buy_etf)))

                self._last_t = now
                return {"act": f"BUY_PACK {q_pack} HEDGE {hedge}", "z": z, "resid": r, "fair": fair}

        return {"act": "HOLD", "z": z, "resid": r, "fair": fair}


# ---------------------------------------------------------------------------
# Strategy E: M2vsM1StatArb — TIDE_SWING (M2) vs TIDE_SPOT (M1) stat arb
# ---------------------------------------------------------------------------

class M2vsM1StatArb:
    """Statistical arbitrage: TIDE_SWING (Market 2) vs TIDE_SPOT (Market 1).

    Uses rolling ridge regression (M2 ~ b0 + b1*M1 + b2*RV) to estimate the
    fair value of TIDE_SWING relative to current tidal height, then trades
    z-score deviations of the residual.

    TIDE_SPOT is used both as the regression signal and as a delta hedge.

    Position limits:
        max_m2_pos, max_m1_pos — strategy-level soft caps.
        BotExchangeAdapter enforces ±100 as the final hard backstop.
    """

    def __init__(
        self,
        api: BotExchangeAdapter,
        m1: str = "TIDE_SPOT",
        m2: str = "TIDE_SWING",
        max_m2_pos: int = 25,
        max_m1_pos: int = 60,
        clip_m2: int = 4,
        clip_m1: int = 8,
        z_enter: float = 2.3,
        z_exit: float = 0.6,
        cooldown: float = 0.2,
    ):
        self.api = api
        self.m1 = m1
        self.m2 = m2
        self.max_m2_pos = min(int(max_m2_pos), api.HARD_LIMIT)
        self.max_m1_pos = min(int(max_m1_pos), api.HARD_LIMIT)
        self.clip_m2 = int(clip_m2)
        self.clip_m1 = int(clip_m1)
        self.z_enter = float(z_enter)
        self.z_exit = float(z_exit)
        self.cooldown = float(cooldown)
        self._last_t = 0.0

        self._rv = RollingRV(n=120)
        self._reg = RollingRidge3(window=650, ridge=5e-2)
        self._res = Rolling(n=350)

    def _flatten(self, clip: int = 10) -> None:
        for p in [self.m2, self.m1]:
            pos = self.api.get_position(p)
            if pos == 0:
                continue
            bbo = self.api.get_best_bid_ask(p)
            q = int(min(abs(pos), clip))
            if pos > 0 and bbo.best_bid is not None:
                self.api.place_order(p, "SELL", bbo.best_bid, q)
            elif pos < 0 and bbo.best_ask is not None:
                self.api.place_order(p, "BUY", bbo.best_ask, q)

    def step(self) -> dict:
        now = time.monotonic()
        if now - self._last_t < self.cooldown:
            return {"act": "COOLDOWN"}

        m1_mid = self.api.get_mid(self.m1)
        m2_mid = self.api.get_mid(self.m2)
        if np.isnan(m1_mid) or np.isnan(m2_mid):
            return {"act": "NO_DATA"}

        bbo1 = self.api.get_best_bid_ask(self.m1)
        bbo2 = self.api.get_best_bid_ask(self.m2)
        bid1, ask1 = bbo1.best_bid, bbo1.best_ask
        bid2, ask2 = bbo2.best_bid, bbo2.best_ask

        rv = self._rv.update(m1_mid)
        if np.isnan(rv):
            return {"act": "WARMUP", "z": float("nan"), "resid": float("nan"), "fair": float("nan")}

        self._reg.add(m1_mid, rv, m2_mid)
        self._reg.fit()

        fair = self._reg.predict(m1_mid, rv)
        resid = m2_mid - fair
        if not np.isnan(resid):
            self._res.add(resid)

        mu = self._res.mean()
        sd = self._res.std()
        z = (resid - mu) / sd if (not np.isnan(sd) and sd > 1e-9) else float("nan")

        pos2 = self.api.get_position(self.m2)
        pos1 = self.api.get_position(self.m1)

        # Exit when normalized: flatten both legs
        if (not np.isnan(z)) and abs(z) < self.z_exit and (pos2 != 0 or pos1 != 0):
            self._flatten(clip=10)
            self._last_t = now
            return {"act": "FLATTEN_EXIT", "z": z, "resid": resid, "fair": fair}

        if np.isnan(z):
            return {"act": "WARMUP", "z": z, "resid": resid, "fair": fair}

        beta = self._reg.dy_dx()

        # z high => M2 rich => SELL M2, hedge with M1
        if z > self.z_enter:
            room_sell2 = max(0, self.max_m2_pos + pos2)
            if room_sell2 > 0 and bid2 is not None:
                q2 = int(min(self.clip_m2, room_sell2))
                self.api.place_order(self.m2, "SELL", bid2, q2)

                hedge = int(np.clip(beta * q2, -self.clip_m1, self.clip_m1))
                room_buy1  = max(0, self.max_m1_pos - pos1)
                room_sell1 = max(0, self.max_m1_pos + pos1)
                if hedge > 0 and room_buy1 > 0 and ask1 is not None:
                    self.api.place_order(self.m1, "BUY",  ask1, int(min(hedge,  room_buy1)))
                elif hedge < 0 and room_sell1 > 0 and bid1 is not None:
                    self.api.place_order(self.m1, "SELL", bid1, int(min(-hedge, room_sell1)))

                self._last_t = now
                return {"act": f"SELL_M2 {q2} HEDGE {hedge}", "z": z, "beta": beta}

        # z low => M2 cheap => BUY M2, hedge with M1
        if z < -self.z_enter:
            room_buy2 = max(0, self.max_m2_pos - pos2)
            if room_buy2 > 0 and ask2 is not None:
                q2 = int(min(self.clip_m2, room_buy2))
                self.api.place_order(self.m2, "BUY", ask2, q2)

                hedge = int(np.clip(beta * q2, -self.clip_m1, self.clip_m1))
                room_buy1  = max(0, self.max_m1_pos - pos1)
                room_sell1 = max(0, self.max_m1_pos + pos1)
                if hedge > 0 and room_sell1 > 0 and bid1 is not None:
                    self.api.place_order(self.m1, "SELL", bid1, int(min(hedge,  room_sell1)))
                elif hedge < 0 and room_buy1 > 0 and ask1 is not None:
                    self.api.place_order(self.m1, "BUY",  ask1, int(min(-hedge, room_buy1)))

                self._last_t = now
                return {"act": f"BUY_M2 {q2} HEDGE {hedge}", "z": z, "beta": beta}

        return {"act": "HOLD", "z": z, "resid": resid, "fair": fair}


# ---------------------------------------------------------------------------
# Strategy F: UpDipBuyer — EWMA mean-reversion dip buyer
# ---------------------------------------------------------------------------

class UpDipBuyer:
    """Mean-reversion dip buyer: enters long when price falls below its EWMA.

    Suited for markets with structural upward drift (TIDE_SPOT at high tide,
    WX_SPOT, WX_SUM) where sharp dislocations below the moving average
    represent transient buying opportunities.

    Exits by selling when the price reverts back near the EWMA.

    Position limits:
        max_pos — strategy-level soft cap.
        BotExchangeAdapter enforces ±100 as the final hard backstop.
    """

    def __init__(
        self,
        api: BotExchangeAdapter,
        product: str,
        max_pos: int = 50,
        clip: int = 8,
        z_enter: float = 1.6,
        z_take: float = 0.3,
        cooldown: float = 0.2,
    ):
        self.api = api
        self.product = product
        self.max_pos = min(int(max_pos), api.HARD_LIMIT)
        self.clip = int(clip)
        self.z_enter = float(z_enter)
        self.z_take = float(z_take)
        self.cooldown = float(cooldown)
        self._last_t = 0.0
        self._ema = EWMA(alpha=0.06)
        self._dev = Rolling(n=220)

    def step(self) -> dict:
        now = time.monotonic()
        if now - self._last_t < self.cooldown:
            return {"p": self.product, "act": "COOLDOWN"}

        mid = self.api.get_mid(self.product)
        if np.isnan(mid):
            return {"p": self.product, "act": "NO_DATA"}

        bbo = self.api.get_best_bid_ask(self.product)
        bid, ask = bbo.best_bid, bbo.best_ask
        pos = self.api.get_position(self.product)

        m = self._ema.update(mid)
        d = mid - m
        self._dev.add(d)
        sd = self._dev.std()
        z = d / sd if (not np.isnan(sd) and sd > 1e-9) else float("nan")

        if np.isnan(z):
            return {"p": self.product, "act": "WARMUP", "z": z}

        # Buy dips: mid << EWMA
        if z < -self.z_enter:
            room_buy = max(0, self.max_pos - pos)
            if room_buy > 0 and ask is not None:
                q = int(min(self.clip, room_buy))
                if self.api.place_order(self.product, "BUY", ask, q):
                    self._last_t = now
                    return {"p": self.product, "act": f"BUY {q}", "z": z}

        # Take profit: price has reverted back toward EWMA
        if z > -self.z_take and pos > 0:
            if bid is not None:
                q = int(min(self.clip, pos))
                if self.api.place_order(self.product, "SELL", bid, q):
                    self._last_t = now
                    return {"p": self.product, "act": f"TAKE {q}", "z": z}

        return {"p": self.product, "act": "HOLD", "z": z}


# ---------------------------------------------------------------------------
# Pair-trading helpers (new2.py — M5 / M6 stat arb)
# ---------------------------------------------------------------------------

class RollingOLS2:
    """Rolling OLS: y ~ a + b·x  (closed-form 2-parameter regression).

    Requires at least 80 observations before fitting; returns last fitted
    coefficients on early calls so callers always get a valid (a, b) pair.
    """

    def __init__(self, window: int = 500):
        self._window = window
        self._x: deque = deque(maxlen=window)
        self._y: deque = deque(maxlen=window)
        self.a: float = 0.0
        self.b: float = 1.0

    def add(self, x: float, y: float) -> None:
        self._x.append(float(x))
        self._y.append(float(y))

    def fit(self) -> tuple[float, float]:
        """Fit OLS on the rolling window and return (intercept, slope)."""
        if len(self._x) < 80:
            return self.a, self.b
        x = np.array(self._x, dtype=float)
        y = np.array(self._y, dtype=float)
        mx  = float(x.mean())
        my  = float(y.mean())
        vx  = float(((x - mx) ** 2).mean()) + 1e-12   # regularise against flat x
        cov = float(((x - mx) * (y - my)).mean())
        self.b = cov / vx
        self.a = my - self.b * mx
        return self.a, self.b

    def predict(self, x: float) -> float:
        return float(self.a + self.b * float(x))


class RollingResidZ:
    """Rolling z-score of a sequence of residuals.

    Returns NaN until at least 80 observations have been accumulated.
    """

    def __init__(self, n: int = 300):
        self._r: deque = deque(maxlen=n)

    def add(self, v: float) -> None:
        self._r.append(float(v))

    def z(self, v: float) -> float:
        if len(self._r) < 80:
            return float("nan")
        arr = np.array(self._r, dtype=float)
        mu  = float(arr.mean())
        sd  = float(arr.std()) + 1e-9
        return (float(v) - mu) / sd


class PairTradingArb:
    """Rolling OLS pair-trading stat arb between two correlated products.

    Model:  pB ≈ a + β·pA
    Signal: residual r = pB_mid − (a + β·pA_mid), normalised to z-score.

    z > +z_enter  → pB is rich vs pA  → SELL pB,  BUY β-weighted pA
    z < −z_enter  → pB is cheap vs pA → BUY  pB, SELL β-weighted pA
    |z| < z_exit  → residual has decayed → flatten

    Position limits (max_pos_A / max_pos_B) are in addition to the hard
    ±100 enforced by BotExchangeAdapter.HARD_LIMIT.

    Designed for Markets 5 & 6: pA = LHR_COUNT, pB = LHR_INDEX.
    Both are derived from the same underlying flight data and therefore
    share a latent factor that this OLS regression exploits.
    """

    def __init__(
        self,
        api,
        pA: str,
        pB: str,
        max_pos_A: int = 70,
        max_pos_B: int = 70,
        clip_A: int = 12,
        clip_B: int = 12,
        z_enter: float = 2.2,
        z_exit: float = 0.6,
        cooldown: float = 0.2,
        ols_window: int = 600,
        resid_window: int = 350,
        hedge_clip: int = 12,
    ):
        self.api = api
        self.pA = pA
        self.pB = pB
        self.max_pos_A  = max_pos_A
        self.max_pos_B  = max_pos_B
        self.clip_A     = clip_A
        self.clip_B     = clip_B
        self.hedge_clip = hedge_clip
        self.z_enter    = z_enter
        self.z_exit     = z_exit
        self.cooldown   = cooldown
        self._last_t    = 0.0
        self._ols = RollingOLS2(window=ols_window)
        self._rz  = RollingResidZ(n=resid_window)

    # ------------------------------------------------------------------

    def _flatten(self, clip: int = 15) -> None:
        """Market-order both legs towards flat."""
        for p in (self.pA, self.pB):
            pos = self.api.get_position(p)
            if pos == 0:
                continue
            bbo = self.api.get_best_bid_ask(p)
            if bbo is None:
                continue
            q = int(min(abs(pos), clip))
            if pos > 0 and bbo.best_bid is not None:
                self.api.place_order(p, "SELL", bbo.best_bid, q)
            elif pos < 0 and bbo.best_ask is not None:
                self.api.place_order(p, "BUY", bbo.best_ask, q)

    def step(self) -> dict:
        """Run one tick of the pair-trading strategy.

        Returns a dict with keys: pair, act, z, beta.
        """
        a_mid = self.api.get_mid(self.pA)
        b_mid = self.api.get_mid(self.pB)
        if a_mid is None or b_mid is None or np.isnan(a_mid) or np.isnan(b_mid):
            return {"pair": f"{self.pA}/{self.pB}", "act": "NO_DATA", "z": float("nan"), "beta": 0.0}

        bbo_a = self.api.get_best_bid_ask(self.pA)
        bbo_b = self.api.get_best_bid_ask(self.pB)
        if bbo_a is None or bbo_b is None:
            return {"pair": f"{self.pA}/{self.pB}", "act": "NO_DATA", "z": float("nan"), "beta": 0.0}

        bid_a, ask_a = bbo_a.best_bid, bbo_a.best_ask
        bid_b, ask_b = bbo_b.best_bid, bbo_b.best_ask

        # Feed regression
        self._ols.add(a_mid, b_mid)
        _a, beta = self._ols.fit()

        fair_b = self._ols.predict(a_mid)
        resid  = b_mid - fair_b
        self._rz.add(resid)
        z = self._rz.z(resid)

        pos_a = self.api.get_position(self.pA)
        pos_b = self.api.get_position(self.pB)

        now = time.time()
        if now - self._last_t < self.cooldown:
            return {"pair": f"{self.pA}/{self.pB}", "act": "COOLDOWN", "z": z, "beta": beta}

        # WARMUP — not enough history yet
        if np.isnan(z):
            return {"pair": f"{self.pA}/{self.pB}", "act": "WARMUP", "z": z, "beta": beta}

        # EXIT — residual has mean-reverted
        if abs(z) < self.z_exit and (pos_a != 0 or pos_b != 0):
            self._flatten(clip=12)
            self._last_t = now
            return {"pair": f"{self.pA}/{self.pB}", "act": "FLATTEN_EXIT", "z": z, "beta": beta}

        # Hedge quantity in A per 1 unit of B
        hedge_a = int(np.clip(round(beta), -self.hedge_clip, self.hedge_clip))
        if hedge_a == 0:
            hedge_a = 1   # minimum hedge of 1 unit

        room_buy_a  = max(0, self.max_pos_A - pos_a)
        room_sell_a = max(0, self.max_pos_A + pos_a)
        room_buy_b  = max(0, self.max_pos_B - pos_b)
        room_sell_b = max(0, self.max_pos_B + pos_b)

        # z high → pB rich → SELL pB, BUY hedged pA
        if z > self.z_enter and room_sell_b > 0 and bid_b is not None:
            q_b = int(min(self.clip_B, room_sell_b))
            self.api.place_order(self.pB, "SELL", bid_b, q_b)

            q_a = int(min(self.clip_A, room_buy_a, abs(hedge_a) * q_b))
            if q_a > 0 and ask_a is not None:
                self.api.place_order(self.pA, "BUY", ask_a, q_a)

            self._last_t = now
            return {"pair": f"{self.pA}/{self.pB}", "act": f"SELL_B {q_b} BUY_A {q_a}", "z": z, "beta": beta}

        # z low → pB cheap → BUY pB, SELL hedged pA
        if z < -self.z_enter and room_buy_b > 0 and ask_b is not None:
            q_b = int(min(self.clip_B, room_buy_b))
            self.api.place_order(self.pB, "BUY", ask_b, q_b)

            q_a = int(min(self.clip_A, room_sell_a, abs(hedge_a) * q_b))
            if q_a > 0 and bid_a is not None:
                self.api.place_order(self.pA, "SELL", bid_a, q_a)

            self._last_t = now
            return {"pair": f"{self.pA}/{self.pB}", "act": f"BUY_B {q_b} SELL_A {q_a}", "z": z, "beta": beta}

        return {"pair": f"{self.pA}/{self.pB}", "act": "HOLD", "z": z, "beta": beta}


# ---------------------------------------------------------------------------
# Regime Directional Algorithm (Momentum & Trend Capturing)
# ---------------------------------------------------------------------------

class RegimeEWMA:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.mu: Optional[float] = None
        self.var: Optional[float] = None

    def update(self, x: float) -> tuple[float, float]:
        x = float(x)
        if self.mu is None or self.var is None:
            self.mu = x
            self.var = 1.0
            return self.mu, float(np.sqrt(self.var))
        dx = x - self.mu
        self.mu = (1 - self.alpha) * self.mu + self.alpha * x
        self.var = (1 - self.alpha) * self.var + self.alpha * (dx * dx)
        return self.mu, float(np.sqrt(self.var))


class RegimeDirectional:
    """Takes aggressively scaled position sizing leaning into structural trends."""
    
    def __init__(
        self,
        api: BotExchangeAdapter,
        product: str,
        direction: str,
        max_pos: int,
        clip: int,
        edge: float,
        alpha: float = 0.06
    ) -> None:
        self.api = api
        self.product = product
        self.direction = direction   # "UP" or "DOWN"
        self.max_pos = min(int(max_pos), api.HARD_LIMIT)
        self.clip = int(clip)
        self.edge = float(edge)
        self.model = RegimeEWMA(alpha=alpha)

    def step(self) -> Dict[str, Any]:
        bbo = self.api.get_best_bid_ask(self.product)
        if not bbo or bbo.mid is None or np.isnan(bbo.mid):
            return {"action": "NO_DATA"}

        mid = float(bbo.mid)
        mu, sig = self.model.update(mid)

        pos = self.api.get_position(self.product)
        bid = bbo.best_bid
        ask = bbo.best_ask
        
        if bid is None or ask is None:
            return {"action": "NO_DATA"}

        if self.direction == "DOWN":
            target = -self.max_pos
            room = max(0, pos - target)
            if room > 0 and (mid > mu + self.edge * sig):
                qty = int(min(self.clip, room))
                if self.api.place_order(self.product, "SELL", bid, qty):
                    return {"action": f"SELL {qty}", "pos": pos, "mid": mid}

        if self.direction == "UP":
            target = self.max_pos
            room = max(0, target - pos)
            if room > 0 and (mid < mu - self.edge * sig):
                qty = int(min(self.clip, room))
                if self.api.place_order(self.product, "BUY", ask, qty):
                    return {"action": f"BUY {qty}", "pos": pos, "mid": mid}

        return {"action": "HOLD", "pos": pos, "mid": mid}
