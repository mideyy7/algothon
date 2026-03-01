"""Test suite for IMCBot — market making, fair value, risk, position tracking,
and stat-arb strategies (VolatileOptionArb, ETFPackageArb).

All CMI exchange HTTP calls are mocked so no network connection is required.
Run with:
    python -m pytest test_bot.py -v
    # or
    python test_bot.py
"""

import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch, call

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


def _make_orderbook_bid_only(product="TIDE_SPOT", bid=1410.0):
    from imc_template.bot_template import OrderBook, Order
    return OrderBook(
        product=product,
        tick_size=1.0,
        buy_orders=[Order(price=bid, volume=10, own_volume=0)],
        sell_orders=[],
    )


def _make_orderbook_empty(product="TIDE_SPOT"):
    from imc_template.bot_template import OrderBook
    return OrderBook(product=product, tick_size=1.0, buy_orders=[], sell_orders=[])


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
    bot._regime_traders = {}
    return bot


# ── Stat-arb shared helpers ────────────────────────────────────────────────────

def _make_oscillating_mids(n: int = 20, center: float = 100.0, amplitude: float = 5.0):
    """Return alternating [center+amplitude, center-amplitude, ...] pattern."""
    return [center + amplitude * (1 if i % 2 == 0 else -1) for i in range(n)]


def _make_ready_swing_arb(edge: float = 18.0, vol_block_z: float = 2.8):
    """VolatileOptionArb pre-warmed with 20 oscillating mids (mean=100, std=5).

    After warmup, api.place_order mock is reset and cooldown is cleared so
    each test starts from a clean slate.
    """
    from stat_arb import VolatileOptionArb
    api = MagicMock()
    api.get_position.return_value = 0
    api.place_order.return_value = True
    arb = VolatileOptionArb(
        api=api, product="TIDE_SWING",
        edge=edge, window=20, vol_block_z=vol_block_z,
    )
    for v in _make_oscillating_mids(20, 100.0, 5.0):
        arb.step(mid=v)
    api.place_order.reset_mock()
    arb._last_trade_t = 0.0
    return arb, api



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

        with patch.object(bot, "_run_arb"), \
             patch.object(bot, "_refresh_fair_values"), \
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

        with patch.object(bot, "_run_arb"), \
             patch.object(bot, "_refresh_fair_values"), \
             patch.object(bot, "send_orders") as mock_send:
            bot._last_quote = {}
            bot.on_orderbook(ob)
            mock_send.assert_not_called()

    def test_throttle_prevents_rapid_requote(self):
        bot = _make_bot(positions={"TIDE_SPOT": 0})
        bot._fair_values = {"TIDE_SPOT": 1410.0}
        ob  = _make_orderbook()

        with patch.object(bot, "_run_arb"), \
             patch.object(bot, "_refresh_fair_values"), \
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

        with patch.object(bot, "_run_arb"), \
             patch.object(bot, "_refresh_fair_values"), \
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

        with patch.object(bot, "_run_arb"), \
             patch.object(bot, "_refresh_fair_values"), \
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

        with patch.object(bot, "_run_arb"), \
             patch.object(bot, "_refresh_fair_values"), \
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

        with patch.object(bot, "_run_arb"), \
             patch.object(bot, "_refresh_fair_values"), \
             patch.object(bot, "get_orders", return_value=stale), \
             patch.object(bot, "cancel_order") as mock_cancel, \
             patch.object(bot, "send_orders"):
            bot._last_quote = {}
            bot.on_orderbook(ob)
            mock_cancel.assert_called_once_with("order-abc-123")


# ── IMCBot._compute_bbo ───────────────────────────────────────────────────────

class TestComputeBBO(unittest.TestCase):

    def setUp(self):
        from main import IMCBot
        self.compute_bbo = IMCBot._compute_bbo

    def test_bbo_with_both_sides(self):
        ob = _make_orderbook(bid=1400.0, ask=1420.0)
        self.assertAlmostEqual(self.compute_bbo(ob).mid, 1410.0)

    def test_bbo_bid_only(self):
        ob = _make_orderbook_bid_only(bid=1410.0)
        self.assertAlmostEqual(self.compute_bbo(ob).mid, 1410.0)

    def test_bbo_ask_only(self):
        from imc_template.bot_template import OrderBook, Order
        ob = OrderBook(
            product="TIDE_SPOT", tick_size=1.0,
            buy_orders=[],
            sell_orders=[Order(price=1420.0, volume=10, own_volume=0)],
        )
        self.assertAlmostEqual(self.compute_bbo(ob).mid, 1420.0)

    def test_bbo_empty_orderbook_returns_nan(self):
        import math
        ob = _make_orderbook_empty()
        self.assertTrue(math.isnan(self.compute_bbo(ob).mid))


# ── IMCBot._run_arb dispatcher ────────────────────────────────────────────────

class TestRunArb(unittest.TestCase):

    def test_tide_swing_dispatches_to_arb(self):
        bot = _make_bot()
        with patch.object(bot._tide_swing_arb, "step", return_value={"action": "HOLD"}) as mock_step:
            bot._run_arb("TIDE_SWING", 100.0)
        mock_step.assert_called_once_with(mid=100.0)

    def test_tide_swing_not_dispatched_when_mid_none(self):
        bot = _make_bot()
        with patch.object(bot._tide_swing_arb, "step") as mock_step:
            bot._run_arb("TIDE_SWING", None)
        mock_step.assert_not_called()

    def test_tide_spot_not_dispatched_to_any_arb(self):
        bot = _make_bot()
        with patch.object(bot._tide_swing_arb, "step") as mock_swing, \
             patch.object(bot._etf_package_arb, "step") as mock_etf:
            bot._run_arb("TIDE_SPOT", 1400.0)
        mock_swing.assert_not_called()
        mock_etf.assert_not_called()

    def test_lon_etf_dispatches(self):
        bot = _make_bot()
        with patch.object(bot._etf_package_arb, "step", return_value={"action": "HOLD"}) as mock_pack, \
             patch.object(bot._etf_basket_arb, "step", return_value={"action": "HOLD"}) as mock_basket:
            bot._run_arb("LON_ETF", 6500.0)
        mock_pack.assert_called_once()
        mock_basket.assert_called_once()


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


# =============================================================================
# Stat-arb unit tests
# =============================================================================

# ── Rolling window ────────────────────────────────────────────────────────────

class TestRolling(unittest.TestCase):

    def test_empty_mean_is_nan(self):
        from stat_arb import Rolling
        r = Rolling(n=10)
        self.assertTrue(np.isnan(r.mean()))

    def test_empty_std_is_nan(self):
        from stat_arb import Rolling
        r = Rolling(n=10)
        self.assertTrue(np.isnan(r.std()))

    def test_not_ready_until_min_n(self):
        from stat_arb import Rolling
        r = Rolling(n=10)
        for i in range(19):
            r.add(float(i))
        self.assertFalse(r.ready(min_n=20))

    def test_ready_at_min_n(self):
        from stat_arb import Rolling
        r = Rolling(n=30)
        for i in range(20):
            r.add(float(i))
        self.assertTrue(r.ready(min_n=20))

    def test_mean_correct(self):
        from stat_arb import Rolling
        r = Rolling(n=10)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            r.add(v)
        self.assertAlmostEqual(r.mean(), 3.0)

    def test_std_correct(self):
        from stat_arb import Rolling
        r = Rolling(n=20)
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]:
            r.add(v)
        self.assertAlmostEqual(r.std(), np.std([2, 4, 4, 4, 5, 5, 7, 9]), places=10)

    def test_capacity_capped_at_n(self):
        from stat_arb import Rolling
        r = Rolling(n=5)
        for i in range(20):
            r.add(float(i))
        self.assertEqual(len(r), 5)
        # Only the last 5 values should remain
        self.assertAlmostEqual(r.mean(), np.mean([15.0, 16.0, 17.0, 18.0, 19.0]))


# ── EWMA ──────────────────────────────────────────────────────────────────────

class TestEWMA(unittest.TestCase):

    def test_value_is_none_before_first_update(self):
        from stat_arb import EWMA
        e = EWMA(alpha=0.1)
        self.assertIsNone(e.value)

    def test_first_update_sets_value_to_x(self):
        from stat_arb import EWMA
        e = EWMA(alpha=0.1)
        result = e.update(42.0)
        self.assertAlmostEqual(result, 42.0)
        self.assertAlmostEqual(e.value, 42.0)

    def test_updates_exponentially(self):
        from stat_arb import EWMA
        e = EWMA(alpha=0.5)
        e.update(100.0)                     # EWMA = 100
        result = e.update(0.0)              # EWMA = 0.5*100 + 0.5*0 = 50
        self.assertAlmostEqual(result, 50.0)

    def test_converges_to_constant_series(self):
        from stat_arb import EWMA
        e = EWMA(alpha=0.2)
        for _ in range(200):
            e.update(77.0)
        self.assertAlmostEqual(e.value, 77.0, places=5)


# ── Option payoff and Monte Carlo ─────────────────────────────────────────────

class TestPayoffAndMC(unittest.TestCase):

    def test_payoff_at_6200_is_zero(self):
        from stat_arb import etf_option_package_payoff
        result = etf_option_package_payoff(np.array([6200.0]))
        self.assertAlmostEqual(result[0], 0.0)

    def test_payoff_below_6200_equals_two_times_put(self):
        """Below 6200: only the two puts pay off → 2*(K-S)."""
        from stat_arb import etf_option_package_payoff
        s = 6000.0
        result = etf_option_package_payoff(np.array([s]))
        self.assertAlmostEqual(result[0], 2.0 * (6200.0 - s))  # 400.0

    def test_payoff_between_6200_and_6600(self):
        """Between 6200–6600: only the long call(6200) pays → S - 6200."""
        from stat_arb import etf_option_package_payoff
        s = 6400.0
        result = etf_option_package_payoff(np.array([s]))
        self.assertAlmostEqual(result[0], s - 6200.0)  # 200.0

    def test_payoff_between_6600_and_7000(self):
        """6600–7000: long call(6200) − 2*call(6600); net = 200 for all S in range."""
        from stat_arb import etf_option_package_payoff
        s = 6800.0
        # +1*(6800-6200) − 2*(6800-6600) = 600 − 400 = 200
        result = etf_option_package_payoff(np.array([s]))
        self.assertAlmostEqual(result[0], 200.0)

    def test_payoff_above_7000(self):
        """Above 7000: all three calls pay."""
        from stat_arb import etf_option_package_payoff
        s = 7200.0
        # +1*(7200-6200) −2*(7200-6600) +3*(7200-7000) = 1000−1200+600 = 400
        result = etf_option_package_payoff(np.array([s]))
        self.assertAlmostEqual(result[0], 400.0)

    def test_mc_fair_value_near_deterministic_for_tiny_sigma(self):
        """With sigma → 0, E[payoff(S)] ≈ payoff(mu)."""
        from stat_arb import mc_fair_value, etf_option_package_payoff
        rng = np.random.default_rng(0)
        # payoff(6400) = 6400 - 6200 = 200
        fv = mc_fair_value(
            etf_option_package_payoff, mu=6400.0, sigma=1e-3,
            n_sims=5_000, rng=rng,
        )
        self.assertAlmostEqual(fv, 200.0, delta=1.0)

    def test_mc_delta_negative_below_6200(self):
        """Below 6200, puts dominate → ∂E/∂mu < 0 (delta ≈ −2)."""
        from stat_arb import mc_delta, etf_option_package_payoff
        rng = np.random.default_rng(2)
        delta = mc_delta(etf_option_package_payoff, mu=5800.0, sigma=50.0,
                         n_sims=20_000, rng=rng)
        self.assertLess(delta, 0.0)

    def test_mc_delta_positive_between_6200_and_6600(self):
        """In [6200, 6600], long call dominates → delta ≈ +1."""
        from stat_arb import mc_delta, etf_option_package_payoff
        rng = np.random.default_rng(3)
        delta = mc_delta(etf_option_package_payoff, mu=6400.0, sigma=50.0,
                         n_sims=20_000, rng=rng)
        self.assertGreater(delta, 0.0)


# ── BotExchangeAdapter ────────────────────────────────────────────────────────

class TestBotExchangeAdapter(unittest.TestCase):

    def setUp(self):
        self.bot = _make_bot()
        self.adapter = self.bot._exchange_adapter

    def test_update_and_get_mid(self):
        self.adapter.update_mid("TIDE_SPOT", 1410.0)
        self.assertAlmostEqual(self.adapter.get_mid("TIDE_SPOT"), 1410.0)

    def test_get_mid_unknown_product_returns_nan(self):
        mid = self.adapter.get_mid("UNKNOWN_PRODUCT")
        self.assertTrue(np.isnan(mid))

    def test_get_position_reads_from_bot(self):
        self.bot._positions["TIDE_SPOT"] = 42
        self.assertEqual(self.adapter.get_position("TIDE_SPOT"), 42)

    def test_get_position_defaults_to_zero(self):
        self.assertEqual(self.adapter.get_position("NEVER_SEEN"), 0)

    def test_place_order_submits_when_within_limit(self):
        self.bot._positions["TIDE_SPOT"] = 0
        with patch.object(self.bot, "send_order") as mock_send:
            result = self.adapter.place_order("TIDE_SPOT", "BUY", 1410.0, 10)
        self.assertTrue(result)
        mock_send.assert_called_once()

    def test_place_order_caps_buy_qty_near_limit(self):
        """Position=95, request=20 → only 5 should be submitted."""
        self.bot._positions["TIDE_SPOT"] = 95
        with patch.object(self.bot, "send_order") as mock_send:
            result = self.adapter.place_order("TIDE_SPOT", "BUY", 1410.0, 20)
        self.assertTrue(result)
        submitted_volume = mock_send.call_args[0][0].volume
        self.assertEqual(submitted_volume, 5)

    def test_place_order_caps_sell_qty_near_limit(self):
        """Position=−95, request=20 → only 5 should be submitted."""
        self.bot._positions["TIDE_SPOT"] = -95
        with patch.object(self.bot, "send_order") as mock_send:
            result = self.adapter.place_order("TIDE_SPOT", "SELL", 1410.0, 20)
        self.assertTrue(result)
        submitted_volume = mock_send.call_args[0][0].volume
        self.assertEqual(submitted_volume, 5)

    def test_place_order_returns_false_at_max_long(self):
        """Position=100: buying more is impossible → False, no send_order."""
        self.bot._positions["TIDE_SPOT"] = 100
        with patch.object(self.bot, "send_order") as mock_send:
            result = self.adapter.place_order("TIDE_SPOT", "BUY", 1410.0, 5)
        self.assertFalse(result)
        mock_send.assert_not_called()

    def test_place_order_returns_false_at_max_short(self):
        """Position=−100: selling more is impossible → False, no send_order."""
        self.bot._positions["TIDE_SPOT"] = -100
        with patch.object(self.bot, "send_order") as mock_send:
            result = self.adapter.place_order("TIDE_SPOT", "SELL", 1410.0, 5)
        self.assertFalse(result)
        mock_send.assert_not_called()


# ── VolatileOptionArb (TIDE_SWING mean-reversion) ────────────────────────────

class TestVolatileOptionArb(unittest.TestCase):
    """Test VolatileOptionArb with alternating mids (mean=100, std=5) for warmup."""

    def test_warmup_before_rolling_ready(self):
        from stat_arb import VolatileOptionArb
        api = MagicMock()
        api.get_position.return_value = 0
        arb = VolatileOptionArb(api=api, product="TIDE_SWING", window=20)
        for _ in range(5):
            result = arb.step(mid=100.0)
        self.assertEqual(result["action"], "WARMUP")

    def test_no_vol_when_all_mids_identical(self):
        """Identical mid prices → std=0 → NO_VOL action."""
        from stat_arb import VolatileOptionArb
        api = MagicMock()
        api.get_position.return_value = 0
        arb = VolatileOptionArb(api=api, product="TIDE_SWING", window=20)
        for _ in range(20):
            result = arb.step(mid=100.0)
        self.assertEqual(result["action"], "NO_VOL")

    def test_hold_within_edge(self):
        """Small mispricing (< edge) should yield HOLD."""
        # edge=18.0: mis ≈ 1 at mid=101 → HOLD
        arb, api = _make_ready_swing_arb(edge=18.0)
        result = arb.step(mid=101.0)
        self.assertEqual(result["action"], "HOLD")

    def test_sell_when_overpriced(self):
        """Mid well above EWMA (beyond edge, below vol_block_z) → SELL."""
        # After warmup: rolling_mean=100, rolling_std=5
        # mid=112: z=2.4<2.8 (no block), mis≈11>edge=2 → SELL
        arb, api = _make_ready_swing_arb(edge=2.0)
        result = arb.step(mid=112.0)
        self.assertTrue(result["action"].startswith("SELL"),
                        f"Expected SELL action, got: {result['action']!r}")
        call_args = api.place_order.call_args[0]
        self.assertEqual(call_args[1].upper(), "SELL")

    def test_buy_when_underpriced(self):
        """Mid well below EWMA (beyond edge, below vol_block_z) → BUY."""
        # mid=88: z=-2.4, abs=2.4<2.8 (no block), mis≈−11<−edge=−2 → BUY
        arb, api = _make_ready_swing_arb(edge=2.0)
        result = arb.step(mid=88.0)
        self.assertTrue(result["action"].startswith("BUY"),
                        f"Expected BUY action, got: {result['action']!r}")
        call_args = api.place_order.call_args[0]
        self.assertEqual(call_args[1].upper(), "BUY")

    def test_vol_block_on_extreme_z(self):
        """Extreme z-score (|z| > vol_block_z) suspends trading → BLOCK_VOL."""
        # mid=120: z=(120-100)/5=4.0 > vol_block_z=2.8 → BLOCK_VOL
        arb, api = _make_ready_swing_arb()
        result = arb.step(mid=120.0)
        self.assertEqual(result["action"], "BLOCK_VOL")
        api.place_order.assert_not_called()

    def test_cooldown_prevents_back_to_back_orders(self):
        """Two rapid step() calls: first fires, second returns COOLDOWN."""
        arb, api = _make_ready_swing_arb(edge=2.0, vol_block_z=4.0)
        # Use a long cooldown so the second call is guaranteed within it
        arb.cooldown = 60.0
        arb.step(mid=112.0)   # should fire a trade
        result = arb.step(mid=112.0)
        self.assertEqual(result["action"], "COOLDOWN")

    def test_inventory_skew_widens_effective_edge(self):
        """With a loaded position, effective_edge should exceed the base edge."""
        from stat_arb import VolatileOptionArb
        api = MagicMock()
        api.get_position.return_value = 15   # near max_pos=20
        api.place_order.return_value = True
        arb = VolatileOptionArb(api=api, product="TIDE_SWING",
                                edge=5.0, window=20, max_pos=20)
        for v in _make_oscillating_mids(20, 100.0, 5.0):
            arb.step(mid=v)
        arb._last_trade_t = 0.0
        result = arb.step(mid=101.0)
        if "effective_edge" in result:
            self.assertGreater(result["effective_edge"], 5.0,
                               "Inventory skew should widen effective edge")


# ── ETF Arbitrage Frameworks ─────────────────────────────────────────────────

class TestNewETFStrategies(unittest.TestCase):

    def test_basket_arb_initialises(self):
        from stat_arb import ETFBasketArb
        api = MagicMock()
        api.HARD_LIMIT = 100
        arb = ETFBasketArb(api=api)
        self.assertIsNotNone(arb)
        self.assertEqual(len(arb.legs), 3)

    def test_implied_vol_arb_initialises(self):
        from stat_arb import ETFPackageImpliedVolArb
        api = MagicMock()
        api.HARD_LIMIT = 100
        arb = ETFPackageImpliedVolArb(api=api)
        self.assertIsNotNone(arb)

    def test_basket_arb_no_data_returns_early(self):
        from stat_arb import ETFBasketArb
        api = MagicMock()
        api.HARD_LIMIT = 100
        api.get_mid.return_value = float("nan")
        arb = ETFBasketArb(api=api)
        result = arb.step()
        self.assertEqual(result["action"], "NO_DATA")


# ── Position compliance (explicit ±100 boundary tests) ───────────────────────

class TestPositionCompliance(unittest.TestCase):
    """BotExchangeAdapter must never allow a net position beyond ±100."""

    def _adapter(self, product, pos):
        bot = _make_bot(positions={product: pos})
        return bot._exchange_adapter, bot

    def test_buy_at_99_submits_exactly_1(self):
        adapter, bot = self._adapter("TEST", 99)
        with patch.object(bot, "send_order") as mock_send:
            result = adapter.place_order("TEST", "BUY", 100.0, 20)
        self.assertTrue(result)
        self.assertEqual(mock_send.call_args[0][0].volume, 1)

    def test_sell_at_minus_99_submits_exactly_1(self):
        adapter, bot = self._adapter("TEST", -99)
        with patch.object(bot, "send_order") as mock_send:
            result = adapter.place_order("TEST", "SELL", 100.0, 20)
        self.assertTrue(result)
        self.assertEqual(mock_send.call_args[0][0].volume, 1)

    def test_buy_blocked_at_hard_limit_100(self):
        adapter, bot = self._adapter("TEST", 100)
        with patch.object(bot, "send_order") as mock_send:
            result = adapter.place_order("TEST", "BUY", 100.0, 1)
        self.assertFalse(result)
        mock_send.assert_not_called()

    def test_sell_blocked_at_hard_limit_minus_100(self):
        adapter, bot = self._adapter("TEST", -100)
        with patch.object(bot, "send_order") as mock_send:
            result = adapter.place_order("TEST", "SELL", 100.0, 1)
        self.assertFalse(result)
        mock_send.assert_not_called()

    def test_buy_and_sell_correct_at_zero_position(self):
        """At flat position: full qty allowed for both sides (up to HARD_LIMIT)."""
        from stat_arb import BotExchangeAdapter
        adapter, bot = self._adapter("TEST", 0)
        with patch.object(bot, "send_order") as mock_send:
            result_buy  = adapter.place_order("TEST", "BUY",  100.0, BotExchangeAdapter.HARD_LIMIT)
        self.assertTrue(result_buy)
        self.assertEqual(mock_send.call_args[0][0].volume, BotExchangeAdapter.HARD_LIMIT)


# ── Monte Carlo timing ────────────────────────────────────────────────────────

class TestMCTiming(unittest.TestCase):
    """MC functions must complete fast enough not to block the SSE thread."""

    def test_mc_fair_value_45k_sims_under_50ms(self):
        from stat_arb import mc_fair_value, etf_option_package_payoff
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        mc_fair_value(etf_option_package_payoff, mu=6500.0, sigma=200.0,
                      n_sims=45_000, rng=rng)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 50.0,
                        f"mc_fair_value took {elapsed_ms:.1f} ms (limit: 50 ms)")

    def test_mc_delta_25k_sims_under_30ms(self):
        from stat_arb import mc_delta, etf_option_package_payoff
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        mc_delta(etf_option_package_payoff, mu=6500.0, sigma=200.0,
                 n_sims=25_000, rng=rng)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertLess(elapsed_ms, 30.0,
                        f"mc_delta took {elapsed_ms:.1f} ms (limit: 30 ms)")


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
