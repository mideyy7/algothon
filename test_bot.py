"""Test suite for IMCBot — market making, fair value, risk, position tracking.

All CMI exchange HTTP calls are mocked so no network connection is required.
Run with:
    python -m pytest test_bot.py -v
    # or
    python test_bot.py
"""

import importlib.util
import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import zoneinfo

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_LONDON_TZ = zoneinfo.ZoneInfo("Europe/London")


# ── Shared test fixtures ───────────────────────────────────────────────────────

def _make_window():
    """Fixed Saturday→Sunday market window for deterministic tests."""
    start = datetime(2026, 2, 28, 12, 0, tzinfo=_LONDON_TZ)
    return start, start + timedelta(hours=24)


def _make_orderbook(
    product: str = "TIDE_SPOT",
    tick_size: float = 1.0,
    bid: float = 1400.0,
    ask: float = 1420.0,
):
    from imc_template.bot_template import OrderBook, Order
    return OrderBook(
        product=product,
        tick_size=tick_size,
        buy_orders=[Order(price=bid,  volume=50, own_volume=0)],
        sell_orders=[Order(price=ask, volume=50, own_volume=0)],
    )


def _make_trade(
    product: str = "TIDE_SPOT",
    buyer:   str = "testuser",
    seller:  str = "other",
    price:  float = 1410.0,
    volume:   int = 5,
):
    from imc_template.bot_template import Trade
    return Trade(
        timestamp="2026-02-28T12:30:00Z",
        product=product,
        buyer=buyer,
        seller=seller,
        volume=volume,
        price=price,
    )


def _make_thames_readings(
    n: int = 8,
    base_level: float = -1.41,
    amplitude: float = 0.5,
):
    """Generate synthetic sinusoidal tide readings matching the M2 period."""
    from data_pipeline.thames import ThamesReading
    T = _M2_T = 12.42 * 3600
    start = datetime(2026, 2, 28, 12, 0, tzinfo=_LONDON_TZ)
    readings = []
    for i in range(n):
        t = start + timedelta(minutes=15 * i)
        t_sec = i * 15 * 60
        level = base_level + amplitude * np.sin(2 * np.pi * t_sec / T)
        readings.append(ThamesReading(dt=t, value_maod=level))
    window_end = start + timedelta(hours=24)
    return readings, window_end


# ── Helper: create IMCBot with mocked auth ─────────────────────────────────────

def _make_bot(username: str = "testuser", positions: Optional[dict] = None):
    """Create an IMCBot with DataPipeline and auth_token mocked out."""
    with patch("main.DataPipeline"), \
         patch("main.get_market_window", return_value=_make_window()):
        from main import IMCBot
        bot = IMCBot("http://localhost/", username, "testpass")

    # Bypass the cached_property authenticator
    bot.__dict__["auth_token"] = "Bearer test-token"
    bot.username = username
    bot._positions = dict(positions) if positions else {}
    return bot


# ── quant_models: predict_tide_spot ───────────────────────────────────────────

class TestPredictTideSpot(unittest.TestCase):

    def test_returns_float_with_enough_readings(self):
        from quant_models import predict_tide_spot
        readings, window_end = _make_thames_readings(n=8)
        result = predict_tide_spot(readings, window_end)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)

    def test_returns_none_with_too_few_readings(self):
        from quant_models import predict_tide_spot
        readings, window_end = _make_thames_readings(n=3)   # below minimum of 4
        result = predict_tide_spot(readings, window_end)
        self.assertIsNone(result)

    def test_returns_none_for_empty_list(self):
        from quant_models import predict_tide_spot
        window_end = datetime(2026, 3, 1, 12, 0, tzinfo=_LONDON_TZ)
        result = predict_tide_spot([], window_end)
        self.assertIsNone(result)

    def test_prediction_is_in_realistic_range(self):
        """Thames tidal heights × 1000 should be roughly 100–3000 for London."""
        from quant_models import predict_tide_spot
        readings, window_end = _make_thames_readings(n=48)
        result = predict_tide_spot(readings, window_end)
        self.assertIsNotNone(result)
        self.assertGreater(result, 50.0)    # unrealistically low
        self.assertLess(result, 10_000.0)   # unrealistically high

    def test_prediction_recovers_known_amplitude(self):
        """Model should roughly recover the true value for clean sinusoidal data."""
        from quant_models import predict_tide_spot
        from data_pipeline.thames import ThamesReading
        T = 12.42 * 3600
        start      = datetime(2026, 2, 28, 12, 0, tzinfo=_LONDON_TZ)
        window_end = start + timedelta(hours=24)
        # True tidal level at Sunday noon (t = 24 h = 86400 s):
        t_target = 24 * 3600
        true_level = -1.41 + 0.8 * np.sin(2 * np.pi * t_target / T)
        true_spot  = abs(true_level) * 1000.0

        # Build 48 readings of perfect sinusoidal data
        readings = []
        for i in range(48):
            t_sec = i * 15 * 60
            level = -1.41 + 0.8 * np.sin(2 * np.pi * t_sec / T)
            readings.append(ThamesReading(
                dt=start + timedelta(minutes=15 * i),
                value_maod=level,
            ))

        result = predict_tide_spot(readings, window_end)
        self.assertIsNotNone(result)
        # Should be within 10 % of the true value
        self.assertAlmostEqual(result, true_spot, delta=true_spot * 0.10)


# ── quant_models: prediction_confidence ───────────────────────────────────────

class TestPredictionConfidence(unittest.TestCase):

    def test_confidence_increases_with_elapsed(self):
        from quant_models import prediction_confidence
        c0   = prediction_confidence(0.0)
        c50  = prediction_confidence(0.5)
        c100 = prediction_confidence(1.0)
        self.assertLess(c0, c50)
        self.assertLess(c50, c100)

    def test_confidence_within_unit_range(self):
        from quant_models import prediction_confidence
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            c = prediction_confidence(frac)
            self.assertGreaterEqual(c, 0.0, f"frac={frac}")
            self.assertLessEqual(c, 1.0, f"frac={frac}")


# ── quant_models: fill_missing_estimates ──────────────────────────────────────

class TestFillMissingEstimates(unittest.TestCase):

    def _make_snap(
        self,
        m1=None, m3=None, m5=None, m7=None, m8=None,
        thames_readings=None,
    ):
        from data_pipeline.pipeline import PipelineSnapshot
        from data_pipeline.settlement import MarketEstimates
        start, end = _make_window()
        return PipelineSnapshot(
            fetched_at=start,
            window_start=start,
            window_end=end,
            thames_readings=thames_readings or [],
            weather_readings=[],
            flight_intervals=[],
            estimates=MarketEstimates(
                m1_tide_spot=m1,
                m3_wx_spot=m3,
                m5_lhr_count=m5,
                m7_lon_etf=m7,
                m8_lon_fly=m8,
            ),
        )

    def test_none_tide_spot_overridden_with_enough_readings(self):
        from quant_models import fill_missing_estimates
        readings, _ = _make_thames_readings(n=8)
        snap = self._make_snap(m1=None, m3=957.0, m5=4690.0, thames_readings=readings)
        fv   = fill_missing_estimates(snap)
        self.assertIsNotNone(fv["TIDE_SPOT"])

    def test_none_lon_etf_derived_after_tide_spot_filled(self):
        from quant_models import fill_missing_estimates
        readings, _ = _make_thames_readings(n=8)
        snap = self._make_snap(m1=None, m3=957.0, m5=4690.0, thames_readings=readings)
        fv   = fill_missing_estimates(snap)
        self.assertIsNotNone(fv["LON_ETF"])
        self.assertIsNotNone(fv["LON_FLY"])

    def test_known_tide_spot_preserved(self):
        from quant_models import fill_missing_estimates
        snap = self._make_snap(m1=1410.0, m3=957.0, m5=4690.0)
        fv   = fill_missing_estimates(snap)
        self.assertEqual(fv["TIDE_SPOT"], 1410.0)

    def test_known_etf_preserved(self):
        from quant_models import fill_missing_estimates
        snap = self._make_snap(m1=1410.0, m3=957.0, m5=4690.0, m7=7057.0, m8=400.0)
        fv   = fill_missing_estimates(snap)
        self.assertEqual(fv["LON_ETF"],  7057.0)
        self.assertEqual(fv["LON_FLY"],  400.0)

    def test_no_readings_leaves_tide_spot_none(self):
        from quant_models import fill_missing_estimates
        snap = self._make_snap(m1=None, m3=957.0, m5=4690.0, thames_readings=[])
        fv   = fill_missing_estimates(snap)
        self.assertIsNone(fv["TIDE_SPOT"])

    def test_no_flights_leaves_lhr_markets_none(self):
        from quant_models import fill_missing_estimates
        snap = self._make_snap(m1=1410.0, m3=957.0, m5=None)   # no flights
        fv   = fill_missing_estimates(snap)
        self.assertIsNone(fv["LHR_COUNT"])
        self.assertIsNone(fv["LON_ETF"])    # can't derive without M5


# ── IMCBot._snap ───────────────────────────────────────────────────────────────

class TestSnap(unittest.TestCase):

    def setUp(self):
        from main import IMCBot
        self.snap = IMCBot._snap

    def test_rounds_down(self):
        self.assertAlmostEqual(self.snap(1410.3, 1.0), 1410.0)

    def test_rounds_up(self):
        self.assertAlmostEqual(self.snap(1410.7, 1.0), 1411.0)

    def test_fractional_tick(self):
        self.assertAlmostEqual(self.snap(1410.3, 0.5), 1410.5)

    def test_zero_tick_returns_price(self):
        self.assertEqual(self.snap(1234.5, 0), 1234.5)

    def test_negative_tick_returns_price(self):
        self.assertEqual(self.snap(1234.5, -1.0), 1234.5)


# ── IMCBot._elapsed_fraction ──────────────────────────────────────────────────

class TestElapsedFraction(unittest.TestCase):

    def test_returns_value_in_unit_range(self):
        bot = _make_bot()
        frac = bot._elapsed_fraction()
        self.assertGreaterEqual(frac, 0.0)
        self.assertLessEqual(frac, 1.0)


# ── IMCBot.on_trades ──────────────────────────────────────────────────────────

class TestOnTrades(unittest.TestCase):

    def test_increments_position_on_our_buy(self):
        bot = _make_bot(positions={"TIDE_SPOT": 0})
        bot.on_trades(_make_trade(buyer="testuser", seller="other", volume=5))
        self.assertEqual(bot._positions["TIDE_SPOT"], 5)

    def test_decrements_position_on_our_sell(self):
        bot = _make_bot(positions={"TIDE_SPOT": 10})
        bot.on_trades(_make_trade(buyer="other", seller="testuser", volume=3))
        self.assertEqual(bot._positions["TIDE_SPOT"], 7)

    def test_ignores_other_traders(self):
        bot = _make_bot(positions={"TIDE_SPOT": 0})
        bot.on_trades(_make_trade(buyer="alice", seller="bob", volume=10))
        self.assertEqual(bot._positions.get("TIDE_SPOT", 0), 0)

    def test_new_product_initialised_from_zero(self):
        """A fill on a product not yet in _positions should be handled."""
        bot = _make_bot(positions={})
        bot.on_trades(_make_trade(product="WX_SPOT", buyer="testuser", volume=7))
        self.assertEqual(bot._positions["WX_SPOT"], 7)

    def test_position_goes_negative_on_sell(self):
        bot = _make_bot(positions={"TIDE_SPOT": 0})
        bot.on_trades(_make_trade(buyer="other", seller="testuser", volume=5))
        self.assertEqual(bot._positions["TIDE_SPOT"], -5)


# ── IMCBot.on_orderbook ───────────────────────────────────────────────────────

class TestOnOrderbook(unittest.TestCase):

    def _bot_with_fv(self, product="TIDE_SPOT", fv=1410.0, position=0):
        bot = _make_bot(positions={product: position})
        bot._fair_values = {product: fv}
        return bot

    def test_sends_bid_and_ask_when_fair_value_known(self):
        bot = _make_bot(positions={"TIDE_SPOT": 0})
        bot._fair_values = {"TIDE_SPOT": 1410.0}
        ob  = _make_orderbook("TIDE_SPOT", tick_size=1.0)

        with patch.object(bot, "_refresh_fair_values"), \
             patch.object(bot, "get_orders", return_value=[]), \
             patch.object(bot, "send_orders") as mock_send:

            bot._last_quote = {}
            bot.on_orderbook(ob)

            mock_send.assert_called_once()
            orders = mock_send.call_args[0][0]
            self.assertEqual(len(orders), 2)
            sides = {o.side for o in orders}
            self.assertIn(Side.BUY,  sides)
            self.assertIn(Side.SELL, sides)

    def test_skips_when_no_fair_value(self):
        bot = _make_bot()
        bot._fair_values = {}   # empty
        ob  = _make_orderbook()

        with patch.object(bot, "_refresh_fair_values"), \
             patch.object(bot, "send_orders") as mock_send:
            bot._last_quote = {}
            bot.on_orderbook(ob)
            mock_send.assert_not_called()

    def test_throttle_prevents_rapid_requote(self):
        bot = _make_bot(positions={"TIDE_SPOT": 0})
        bot._fair_values = {"TIDE_SPOT": 1410.0}
        ob  = _make_orderbook()

        with patch.object(bot, "_refresh_fair_values"), \
             patch.object(bot, "get_orders", return_value=[]), \
             patch.object(bot, "send_orders") as mock_send:
            # Simulate that we just quoted this product
            bot._last_quote = {"TIDE_SPOT": time.monotonic()}
            bot.on_orderbook(ob)
            mock_send.assert_not_called()

    def test_bid_strictly_below_ask(self):
        bot = _make_bot(positions={"TIDE_SPOT": 0})
        bot._fair_values = {"TIDE_SPOT": 1410.0}
        ob  = _make_orderbook("TIDE_SPOT", tick_size=1.0)

        with patch.object(bot, "_refresh_fair_values"), \
             patch.object(bot, "get_orders", return_value=[]), \
             patch.object(bot, "send_orders") as mock_send:
            bot._last_quote = {}
            bot.on_orderbook(ob)

            orders = mock_send.call_args[0][0]
            bids = [o for o in orders if o.side == Side.BUY]
            asks = [o for o in orders if o.side == Side.SELL]
            if bids and asks:
                self.assertLess(bids[0].price, asks[0].price)

    def test_buy_size_capped_at_max_position(self):
        """Near the long limit, buy volume must not push us past MAX_POSITION."""
        from main import MAX_POSITION
        pos = MAX_POSITION - 2
        bot = _make_bot(positions={"TIDE_SPOT": pos})
        bot._fair_values = {"TIDE_SPOT": 1410.0}
        ob  = _make_orderbook("TIDE_SPOT", tick_size=1.0)

        with patch.object(bot, "_refresh_fair_values"), \
             patch.object(bot, "get_orders", return_value=[]), \
             patch.object(bot, "send_orders") as mock_send:
            bot._last_quote = {}
            bot.on_orderbook(ob)

            orders = mock_send.call_args[0][0]
            buys = [o for o in orders if o.side == Side.BUY]
            if buys:
                self.assertLessEqual(buys[0].volume, 2)

    def test_no_buy_orders_at_max_position(self):
        """At MAX_POSITION long, no buy orders should be placed."""
        from main import MAX_POSITION
        bot = _make_bot(positions={"TIDE_SPOT": MAX_POSITION})
        bot._fair_values = {"TIDE_SPOT": 1410.0}
        ob  = _make_orderbook("TIDE_SPOT", tick_size=1.0)

        with patch.object(bot, "_refresh_fair_values"), \
             patch.object(bot, "get_orders", return_value=[]), \
             patch.object(bot, "send_orders") as mock_send:
            bot._last_quote = {}
            bot.on_orderbook(ob)

            if mock_send.called:
                orders = mock_send.call_args[0][0]
                buys = [o for o in orders if o.side == Side.BUY]
                self.assertEqual(len(buys), 0)

    def test_cancels_stale_orders_before_requoting(self):
        bot = _make_bot(positions={"TIDE_SPOT": 0})
        bot._fair_values = {"TIDE_SPOT": 1410.0}
        ob  = _make_orderbook("TIDE_SPOT", tick_size=1.0)
        stale = [{"id": "order-abc-123"}]

        with patch.object(bot, "_refresh_fair_values"), \
             patch.object(bot, "get_orders", return_value=stale), \
             patch.object(bot, "cancel_order") as mock_cancel, \
             patch.object(bot, "send_orders"):
            bot._last_quote = {}
            bot.on_orderbook(ob)
            mock_cancel.assert_called_once_with("order-abc-123")


# ── Risk framework integration ────────────────────────────────────────────────

def _load_risk_mod():
    import risk_framework
    return risk_framework


class TestRiskFramework(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = _load_risk_mod()
            cls.available = True
        except Exception as e:
            cls.available = False
            cls.skip_reason = str(e)

    def _skip_if_unavailable(self):
        if not self.available:
            self.skipTest(f"Risk framework unavailable: {self.skip_reason}")

    def test_pnl_risk_level_returns_required_keys(self):
        self._skip_if_unavailable()
        # quantstats >= 0.0.81 requires a DatetimeIndex
        idx = pd.date_range("2026-02-28 12:00", periods=10, freq="1min")
        series = pd.Series([0.0, 1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5], index=idx)
        result = self.mod.pnl_risk_level(series)
        for key in ("vol", "max_dd", "sharpe", "var_95"):
            self.assertIn(key, result, f"Key '{key}' missing from risk result")

    def test_prize_pressure_within_unit_range(self):
        self._skip_if_unavailable()
        live = {"rank": 5, "score": 80.0, "leader_score": 100.0, "metric": "pnl"}
        p = self.mod.prize_pressure(live)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_higher_prize_pressure_raises_aggression(self):
        self._skip_if_unavailable()
        risk = {"vol": 0.01, "max_dd": 0.05, "sharpe": 1.5, "var_95": 0.02}
        budget_low  = self.mod.risk_budget_from_prize(50.0, risk, pressure=0.1)
        budget_high = self.mod.risk_budget_from_prize(50.0, risk, pressure=0.9)
        self.assertGreater(
            budget_high["aggression"],
            budget_low["aggression"],
            "High pressure should produce higher aggression",
        )

    def test_aggression_within_configured_bounds(self):
        self._skip_if_unavailable()
        risk = {"vol": 0.01, "max_dd": 0.05, "sharpe": 1.5, "var_95": 0.02}
        for pressure in [0.0, 0.5, 1.0]:
            budget = self.mod.risk_budget_from_prize(50.0, risk, pressure=pressure)
            self.assertGreaterEqual(budget["aggression"], 0.1)   # practical floor
            self.assertLessEqual(budget["aggression"], 3.0)      # practical cap


# ── Side import smoke test ────────────────────────────────────────────────────

class TestBotTemplate(unittest.TestCase):

    def test_side_enum_values(self):
        from imc_template.bot_template import Side
        self.assertEqual(Side.BUY,  "BUY")
        self.assertEqual(Side.SELL, "SELL")

    def test_order_request_fields(self):
        from imc_template.bot_template import OrderRequest, Side
        req = OrderRequest("TIDE_SPOT", 1410.0, Side.BUY, 10)
        self.assertEqual(req.product, "TIDE_SPOT")
        self.assertEqual(req.price,   1410.0)
        self.assertEqual(req.side,    Side.BUY)
        self.assertEqual(req.volume,  10)


# ── Import the Side enum for use in assertions above ──────────────────────────
try:
    from imc_template.bot_template import Side
except ImportError:
    # Fallback if running in an environment without the module
    class Side:   # type: ignore[no-redef]
        BUY  = "BUY"
        SELL = "SELL"


if __name__ == "__main__":
    unittest.main(verbosity=2)
