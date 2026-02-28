"""
ETFPackageImpliedVolArb: Market 8 implied-vol arb for IMCity.

- Market 8 (LON_FLY) is an option package on ETF settlement (Market 7 LON_ETF).
- We estimate mu via EWMA of ETF mid.
- We estimate "realised" sigma_est via rolling std of ETF mid.
- We compute sigma_impl by inverting the Monte Carlo price to match the market pack mid.
- Trade when sigma_impl - sigma_est exceeds +/- vol_edge.
- Delta-hedge the package exposure using the ETF.

This is an upgrade of the "fair-value vs market" approach: it isolates vol richness/cheapness.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any

import numpy as np

from exchange_adapter import CMIExchangeAdapter

# -------------------------
# rolling / ewma helpers
# -------------------------

class Rolling:
    def __init__(self, n: int = 120):
        self.n = int(n)
        self._x: list[float] = []

    def add(self, v: float) -> None:
        self._x.append(float(v))
        if len(self._x) > self.n:
            self._x.pop(0)

    def std(self) -> float:
        return float(np.std(self._x)) if len(self._x) > 5 else float("nan")

    def ready(self, min_n: int = 25) -> bool:
        return len(self._x) >= min_n


class EWMA:
    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self._v: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        self._v = x if self._v is None else (1.0 - self.alpha) * self._v + self.alpha * x
        return float(self._v)


# -------------------------
# payoff + MC pricer
# -------------------------

def etf_option_package_payoff(S: np.ndarray) -> np.ndarray:
    # +2 Put(6200) +1 Call(6200) −2 Call(6600) +3 Call(7000)
    return (
        2.0 * np.maximum(0.0, 6200.0 - S)
        + 1.0 * np.maximum(0.0, S - 6200.0)
        - 2.0 * np.maximum(0.0, S - 6600.0)
        + 3.0 * np.maximum(0.0, S - 7000.0)
    )


def mc_fair_value(mu: float, sigma: float, *, n_sims: int, rng: np.random.Generator) -> float:
    sigma = max(1e-6, float(sigma))
    half = int(n_sims) // 2
    z = rng.standard_normal(size=half)
    s_pos = float(mu) + sigma * z
    s_neg = float(mu) - sigma * z  # antithetic
    pay = (etf_option_package_payoff(s_pos) + etf_option_package_payoff(s_neg)) / 2.0
    return float(np.mean(pay))


def mc_delta(mu: float, sigma: float, *, bump: float, n_sims: int, rng: np.random.Generator) -> float:
    sigma = max(1e-6, float(sigma))
    sims = rng.normal(float(mu), sigma, size=int(n_sims))
    up = etf_option_package_payoff(sims + float(bump))
    dn = etf_option_package_payoff(sims - float(bump))
    return float(np.mean((up - dn) / (2.0 * float(bump))))


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

    MC noise exists, so keep tol_price reasonably loose.
    """
    if rng is None:
        rng = np.random.default_rng()

    if target_price <= 0:
        return float(sigma_lo)

    f_lo = mc_fair_value(mu, sigma_lo, n_sims=mc_sims, rng=rng)
    f_hi = mc_fair_value(mu, sigma_hi, n_sims=mc_sims, rng=rng)

    if target_price <= f_lo:
        return float(sigma_lo)
    if target_price >= f_hi:
        return float(sigma_hi)

    lo, hi = float(sigma_lo), float(sigma_hi)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        f_mid = mc_fair_value(mu, mid, n_sims=mc_sims, rng=rng)

        if abs(f_mid - target_price) <= tol_price:
            return float(mid)

        if f_mid < target_price:
            lo = mid
        else:
            hi = mid

    return float(0.5 * (lo + hi))


# -------------------------
# strategy
# -------------------------

class ETFPackageImpliedVolArb:
    def __init__(
        self,
        api: CMIExchangeAdapter,
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
        price_mode: str = "CROSS",  # MID or CROSS
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

        # implied vol cache
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

        # IV rich => sell pack
        if vol_spread > self.vol_edge and room_sell > 0:
            qty = int(min(self.clip, room_sell))
            px_pack = self._price(self.pack, "SELL")
            if self.api.place_order(self.pack, "SELL", px_pack, qty):
                delta = mc_delta(mu, sigma_est, bump=self.bump, n_sims=self.delta_sims, rng=self.rng)
                hedge_qty = int(np.clip(round(delta * qty), -self.hedge_clip, self.hedge_clip))
                if hedge_qty > 0:
                    self.api.place_order(self.etf, "BUY", self._price(self.etf, "BUY"), hedge_qty)
                elif hedge_qty < 0:
                    self.api.place_order(self.etf, "SELL", self._price(self.etf, "SELL"), -hedge_qty)

                self._last_t = now
                act = f"SELL_PACK {qty} / IV_RICH {vol_spread:+.1f} / HEDGE_ETF {hedge_qty:+d}"

        # IV cheap => buy pack
        elif vol_spread < -self.vol_edge and room_buy > 0:
            qty = int(min(self.clip, room_buy))
            px_pack = self._price(self.pack, "BUY")
            if self.api.place_order(self.pack, "BUY", px_pack, qty):
                delta = mc_delta(mu, sigma_est, bump=self.bump, n_sims=self.delta_sims, rng=self.rng)
                hedge_qty = int(np.clip(round(delta * qty), -self.hedge_clip, self.hedge_clip))
                # Buying pack gives +delta exposure; hedge with -delta in ETF
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
