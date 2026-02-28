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
