"""
ETFBasketArb: Classic identity arbitrage for IMCity.

IMCity spec: Market 7 (LON_ETF) settles to Market 1 + Market 3 + Market 5.
This strategy trades deviations between the ETF and the live basket value.

You MUST configure the correct symbols for the basket legs for your round.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np

from exchange_adapter import CMIExchangeAdapter


@dataclass(frozen=True)
class BasketLeg:
    product: str
    weight: int = 1  # identity uses +1 weights


class ETFBasketArb:
    """
    Trade:
        spread = etf_mid - sum_i (w_i * mid_i)

    If spread > +edge:
        SELL ETF, BUY basket
    If spread < -edge:
        BUY ETF, SELL basket

    Execution:
      - Sends legs sequentially. This is a simple "package" send, NOT an
        atomic spread order. If your exchange supports order cancel/replace,
        you can improve legging risk management.

    IMPORTANT:
      - Using mid as the order price means you might not get filled.
        In production you'd use best bid/ask +/- tick. This class supports both:
          price_mode="MID" or price_mode="CROSS"
        where CROSS uses best ask for buys and best bid for sells.
    """

    def __init__(
        self,
        api: CMIExchangeAdapter,
        *,
        etf: str = "LON_ETF",
        legs: Optional[List[BasketLeg]] = None,
        edge: float = 10.0,
        max_pos_etf: int = 60,
        max_pos_leg: int = 60,
        clip: int = 6,
        leg_clip: int = 6,
        cooldown: float = 0.30,
        price_mode: str = "CROSS",  # "MID" or "CROSS"
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
        """
        MID: use mid price
        CROSS: BUY at best ask; SELL at best bid (more fill probability)
        """
        side_u = side.upper()
        bbo = self.api.get_best_bid_ask(product)
        if self.price_mode == "MID":
            return bbo.mid
        # CROSS
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

        # ETF rich => sell ETF, buy basket
        if spread > self.edge and etf_room_sell > 0:
            qty_etf = int(min(self.clip, etf_room_sell))

            # leg buy capacity
            q_caps = []
            for leg in self.legs:
                _, leg_room_buy, _ = self._room(leg.product, self.max_pos_leg)
                q_caps.append(int(min(self.leg_clip, leg_room_buy)))

            qty = int(min([qty_etf] + q_caps))
            if qty <= 0:
                return {"etf_mid": etf_mid, "basket_mid": basket_mid, "spread": spread, "action": "CAPPED"}

            ok = True
            # buy basket legs first (reduces risk of being naked short ETF)
            for leg in self.legs:
                px = self._mid_or_cross_price(leg.product, "BUY")
                ok &= self.api.place_order(leg.product, "BUY", px, qty * abs(leg.weight))

            px_etf = self._mid_or_cross_price(self.etf, "SELL")
            ok &= self.api.place_order(self.etf, "SELL", px_etf, qty)

            if ok:
                self._last_t = now
                act = f"SELL_ETF {qty} / BUY_BASKET {qty}"

        # ETF cheap => buy ETF, sell basket
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
            # buy ETF first (reduces risk of being naked short legs)
            px_etf = self._mid_or_cross_price(self.etf, "BUY")
            ok &= self.api.place_order(self.etf, "BUY", px_etf, qty)

            for leg in self.legs:
                px = self._mid_or_cross_price(leg.product, "SELL")
                ok &= self.api.place_order(leg.product, "SELL", px, qty * abs(leg.weight))

            if ok:
                self._last_t = now
                act = f"BUY_ETF {qty} / SELL_BASKET {qty}"

        return {
            "etf_mid": etf_mid,
            "basket_mid": basket_mid,
            "spread": spread,
            "etf_pos": etf_pos,
            "action": act,
        }
