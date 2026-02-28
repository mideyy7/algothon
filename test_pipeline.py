"""Tests for the IMCity data pipeline.

Covers:
  1. Unit tests for every settlement calculation using PDF examples.
  2. Integration tests for fetchers using mocked HTTP responses
     (no real network calls, no API keys needed).
  3. Full pipeline test assembling a complete snapshot from mock data.

Run with:
    python -m pytest test_pipeline.py -v
  or:
    python test_pipeline.py
"""

import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
import zoneinfo

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_LONDON_TZ = zoneinfo.ZoneInfo("Europe/London")


def _london(year, month, day, hour=0, minute=0):
    """Shorthand for a timezone-aware London datetime."""
    return datetime(year, month, day, hour, minute, tzinfo=_LONDON_TZ)


# -----------------------------------------------------------------------
# 1. Thames settlement calculations
# -----------------------------------------------------------------------

class TestThamesSettlement(unittest.TestCase):

    def _make_reading(self, dt: datetime, value: float):
        from data_pipeline.thames import ThamesReading
        return ThamesReading(dt=dt, value_maod=value)

    # --- Market 1 ---

    def test_tide_spot_from_pdf_example(self):
        """PDF example: reading of -1.41 m → settlement 1410."""
        from data_pipeline.thames import compute_tide_spot
        settlement_dt = _london(2025, 11, 20, 12, 0)
        readings = [self._make_reading(_london(2025, 11, 20, 12, 0), -1.41)]
        result = compute_tide_spot(readings, settlement_dt)
        self.assertAlmostEqual(result, 1410.0, places=2)

    def test_tide_spot_positive_value(self):
        """Positive reading: 1.31 m → settlement 1310."""
        from data_pipeline.thames import compute_tide_spot
        settlement_dt = _london(2025, 11, 20, 12, 0)
        readings = [self._make_reading(_london(2025, 11, 20, 12, 0), 1.31)]
        result = compute_tide_spot(readings, settlement_dt)
        self.assertAlmostEqual(result, 1310.0, places=2)

    def test_tide_spot_picks_nearest_reading(self):
        """Should select the reading closest in time to the settlement target."""
        from data_pipeline.thames import compute_tide_spot
        settlement_dt = _london(2025, 11, 20, 12, 0)
        readings = [
            self._make_reading(_london(2025, 11, 20, 11, 45), -1.42),
            self._make_reading(_london(2025, 11, 20, 12,  0), -1.41),   # closest
            self._make_reading(_london(2025, 11, 20, 12, 15), -1.40),
        ]
        result = compute_tide_spot(readings, settlement_dt)
        self.assertAlmostEqual(result, 1410.0, places=2)

    def test_tide_spot_returns_none_if_no_readings(self):
        from data_pipeline.thames import compute_tide_spot
        result = compute_tide_spot([], _london(2025, 11, 20, 12, 0))
        self.assertIsNone(result)

    def test_tide_spot_returns_none_outside_tolerance(self):
        """Reading >20 minutes from target → None."""
        from data_pipeline.thames import compute_tide_spot
        settlement_dt = _london(2025, 11, 20, 12, 0)
        readings = [self._make_reading(_london(2025, 11, 20, 11, 30), -1.50)]  # 30 min away
        result = compute_tide_spot(readings, settlement_dt, tolerance_minutes=20)
        self.assertIsNone(result)

    # --- Market 2 strangle ---

    def test_strangle_below_put_strike(self):
        """diff=0.09 → put ITM: max(0, 0.20−0.09) = 0.11 (PDF example)."""
        from data_pipeline.thames import _strangle_payoff
        self.assertAlmostEqual(_strangle_payoff(0.09), 0.11, places=5)

    def test_strangle_above_call_strike(self):
        """diff=0.33 → call ITM: max(0, 0.33−0.25) = 0.08 (PDF example)."""
        from data_pipeline.thames import _strangle_payoff
        self.assertAlmostEqual(_strangle_payoff(0.33), 0.08, places=5)

    def test_strangle_between_strikes(self):
        """diff=0.21 → both options OTM → payoff = 0 (PDF example)."""
        from data_pipeline.thames import _strangle_payoff
        self.assertAlmostEqual(_strangle_payoff(0.21), 0.00, places=5)

    def test_strangle_at_put_strike(self):
        """diff=0.20 exactly → put payoff = 0, call payoff = 0 → 0."""
        from data_pipeline.thames import _strangle_payoff
        self.assertAlmostEqual(_strangle_payoff(0.20), 0.00, places=5)

    def test_strangle_at_call_strike(self):
        """diff=0.25 exactly → put payoff = 0, call payoff = 0 → 0."""
        from data_pipeline.thames import _strangle_payoff
        self.assertAlmostEqual(_strangle_payoff(0.25), 0.00, places=5)

    def test_tide_swing_sum_from_pdf_example(self):
        """Verify sum matches PDF: diffs 0.09 and 0.33 → (0.11+0.08)×100 = 19."""
        from data_pipeline.thames import compute_tide_swing
        start = _london(2025, 11, 20, 8, 0)
        end   = _london(2025, 11, 20, 9, 30)
        readings = [
            self._make_reading(_london(2025, 11, 20, 8,  0), 1.00),
            self._make_reading(_london(2025, 11, 20, 8, 15), 1.33),  # diff 0.33
            self._make_reading(_london(2025, 11, 20, 8, 30), 1.24),  # diff 0.09
        ]
        result = compute_tide_swing(readings, start, end)
        # 0.08 + 0.11 = 0.19 × 100 = 19.0
        self.assertAlmostEqual(result, 19.0, places=4)

    def test_tide_swing_gap_skipped(self):
        """A gap larger than max_gap_minutes should not be included in the sum."""
        from data_pipeline.thames import compute_tide_swing
        start = _london(2025, 11, 20, 8, 0)
        end   = _london(2025, 11, 20, 10, 0)
        readings = [
            self._make_reading(_london(2025, 11, 20, 8,  0), 1.00),
            # 60-minute gap — should be skipped
            self._make_reading(_london(2025, 11, 20, 9,  0), 2.00),  # diff=1.00 (huge)
            self._make_reading(_london(2025, 11, 20, 9, 15), 1.91),  # diff=0.09
        ]
        result = compute_tide_swing(readings, start, end, max_gap_minutes=20)
        # Only the second pair (diff=0.09) contributes: 0.11 × 100 = 11.0
        self.assertAlmostEqual(result, 11.0, places=4)

    def test_tide_swing_projection(self):
        """At 50% of window, running sum × 2 = projected total."""
        from data_pipeline.thames import project_tide_swing
        result = project_tide_swing(current_sum=500.0, readings_in_window=48, expected_total_readings=96)
        self.assertAlmostEqual(result, 1000.0, places=2)

    def test_tide_swing_projection_at_100_percent(self):
        """At 100% of window, projection should not exceed the actual total."""
        from data_pipeline.thames import project_tide_swing
        result = project_tide_swing(current_sum=500.0, readings_in_window=96, expected_total_readings=96)
        self.assertAlmostEqual(result, 500.0, places=2)


# -----------------------------------------------------------------------
# 2. Weather settlement calculations
# -----------------------------------------------------------------------

class TestWeatherSettlement(unittest.TestCase):

    def _make_reading(self, dt: datetime, temp_f: float, humidity: int):
        from data_pipeline.weather import WeatherReading
        return WeatherReading(dt=dt, temp_f=temp_f, humidity=humidity)

    # --- Market 3 ---

    def test_wx_spot_from_pdf_example(self):
        """PDF example: T=33°F, H=29% → settlement 957."""
        from data_pipeline.weather import compute_wx_spot
        settlement_dt = _london(2025, 11, 20, 12, 0)
        readings = [self._make_reading(_london(2025, 11, 20, 12, 0), 33.0, 29)]
        result = compute_wx_spot(readings, settlement_dt)
        self.assertAlmostEqual(result, 957.0, places=2)

    def test_wx_spot_rounds_temperature(self):
        """Temperature is rounded before multiplication: 33.4°F rounds to 33."""
        from data_pipeline.weather import compute_wx_spot
        settlement_dt = _london(2025, 11, 20, 12, 0)
        readings = [self._make_reading(_london(2025, 11, 20, 12, 0), 33.4, 29)]
        result = compute_wx_spot(readings, settlement_dt)
        self.assertAlmostEqual(result, 957.0, places=2)  # round(33.4)=33 → 33×29=957

    def test_wx_spot_rounds_temperature_up(self):
        """33.6°F rounds to 34: 34 × 29 = 986."""
        from data_pipeline.weather import compute_wx_spot
        settlement_dt = _london(2025, 11, 20, 12, 0)
        readings = [self._make_reading(_london(2025, 11, 20, 12, 0), 33.6, 29)]
        result = compute_wx_spot(readings, settlement_dt)
        self.assertAlmostEqual(result, 986.0, places=2)

    def test_wx_spot_returns_none_if_no_readings(self):
        from data_pipeline.weather import compute_wx_spot
        result = compute_wx_spot([], _london(2025, 11, 20, 12, 0))
        self.assertIsNone(result)

    # --- Market 4 ---

    def test_wx_sum_from_pdf_example(self):
        """PDF: (33×29)/100 + (30×30)/100 + (30×90)/100 = 9.57 + 9.00 + 27.00 = 45.57."""
        from data_pipeline.weather import compute_wx_sum
        start = _london(2025, 11, 20, 11, 30)
        end   = _london(2025, 11, 20, 12,  0)
        readings = [
            self._make_reading(_london(2025, 11, 20, 11, 30), 30.0, 90),
            self._make_reading(_london(2025, 11, 20, 11, 45), 30.0, 30),
            self._make_reading(_london(2025, 11, 20, 12,  0), 33.0, 29),
        ]
        result = compute_wx_sum(readings, start, end)
        expected = (30 * 90 / 100) + (30 * 30 / 100) + (33 * 29 / 100)
        self.assertAlmostEqual(result, expected, places=4)

    def test_wx_sum_empty_returns_zero(self):
        from data_pipeline.weather import compute_wx_sum
        result = compute_wx_sum([], _london(2025, 11, 20, 12, 0), _london(2025, 11, 21, 12, 0))
        self.assertAlmostEqual(result, 0.0, places=4)


# -----------------------------------------------------------------------
# 3. Flights settlement calculations
# -----------------------------------------------------------------------

class TestFlightsSettlement(unittest.TestCase):

    def _make_interval(self, bucket_start: datetime, arrivals: int, departures: int):
        from data_pipeline.flights import FlightInterval
        return FlightInterval(bucket_start=bucket_start, arrivals=arrivals, departures=departures)

    # --- Market 5 ---

    def test_lhr_count_from_pdf_example(self):
        """PDF: (13+10+15) arrivals + (9+19+11) departures = 77."""
        from data_pipeline.flights import compute_lhr_count
        intervals = [
            self._make_interval(_london(2025, 11, 20, 10, 30), 15, 11),
            self._make_interval(_london(2025, 11, 20, 11,  0), 10, 19),
            self._make_interval(_london(2025, 11, 20, 11, 30), 13,  9),
        ]
        result = compute_lhr_count(intervals)
        self.assertEqual(result, 77)

    def test_lhr_count_zero(self):
        from data_pipeline.flights import compute_lhr_count
        self.assertEqual(compute_lhr_count([]), 0)

    # --- FlightInterval.metric ---

    def test_interval_metric_from_pdf_example(self):
        """PDF: 13 arr, 9 dep → 100*(13-9)/(13+9) = 100*4/22 ≈ 18.18."""
        interval = self._make_interval(_london(2025, 11, 20, 9, 30), 13, 9)
        self.assertAlmostEqual(interval.metric, 18.1818, places=3)

    def test_interval_metric_negative(self):
        """10 arr, 19 dep → 100*(10-19)/(10+19) = 100*(−9)/29 ≈ −31.03."""
        interval = self._make_interval(_london(2025, 11, 20, 9, 0), 10, 19)
        self.assertAlmostEqual(interval.metric, -31.0345, places=3)

    def test_interval_metric_zero_flights(self):
        """0 arrivals and 0 departures → metric = 0 (denominator clipped to 1)."""
        interval = self._make_interval(_london(2025, 11, 20, 3, 0), 0, 0)
        self.assertAlmostEqual(interval.metric, 0.0, places=5)

    # --- Market 6 ---

    def test_lhr_index_from_pdf_example(self):
        """PDF example: 18.18 − 31.03 + 15.38 = 2.53... → ABS ≈ 2.52+."""
        from data_pipeline.flights import compute_lhr_index
        intervals = [
            self._make_interval(_london(2025, 11, 20, 8, 30), 15, 11),   # ~15.38
            self._make_interval(_london(2025, 11, 20, 9,  0), 10, 19),   # ~-31.03
            self._make_interval(_london(2025, 11, 20, 9, 30), 13,  9),   # ~18.18
        ]
        result = compute_lhr_index(intervals)
        # Sum ≈ 15.38 − 31.03 + 18.18 = 2.53; ABS = 2.53
        self.assertGreater(result, 0.0)
        self.assertAlmostEqual(result, abs(15.384615 - 31.034483 + 18.181818), places=2)

    def test_lhr_index_abs_value(self):
        """Result is always non-negative."""
        from data_pipeline.flights import compute_lhr_index
        intervals = [
            self._make_interval(_london(2025, 11, 20, 9, 0), 5, 20),   # very negative metric
        ]
        result = compute_lhr_index(intervals)
        self.assertGreaterEqual(result, 0.0)


# -----------------------------------------------------------------------
# 4. Settlement: LON_ETF and LON_FLY
# -----------------------------------------------------------------------

class TestDerivedMarkets(unittest.TestCase):

    # --- Market 7 ---

    def test_lon_etf_pdf_example(self):
        """PDF: water=1000, T×H=1500, flights=500 → ETF=3000."""
        from data_pipeline.settlement import compute_lon_etf
        result = compute_lon_etf(1000.0, 1500.0, 500.0)
        self.assertAlmostEqual(result, 3000.0, places=2)

    def test_lon_etf_returns_none_on_missing_component(self):
        from data_pipeline.settlement import compute_lon_etf
        self.assertIsNone(compute_lon_etf(None, 1500.0, 500.0))
        self.assertIsNone(compute_lon_etf(1000.0, None, 500.0))
        self.assertIsNone(compute_lon_etf(1000.0, 1500.0, None))

    def test_lon_etf_abs_applied(self):
        """ETF uses ABS — components summing to a negative are made positive."""
        from data_pipeline.settlement import compute_lon_etf
        result = compute_lon_etf(-2000.0, 500.0, 300.0)
        self.assertAlmostEqual(result, 1200.0, places=2)

    # --- Market 8 ---

    def test_lon_fly_pdf_example(self):
        """PDF: ETF=6300 → 0 + 100 + 0 + 0 = 100."""
        from data_pipeline.settlement import compute_lon_fly
        result = compute_lon_fly(6300.0)
        self.assertAlmostEqual(result, 100.0, places=4)

    def test_lon_fly_below_all_strikes(self):
        """ETF=5000 (< 6200): both puts ITM, call OTM.
        2×max(0,6200−5000) + 1×0 − 0 + 0 = 2×1200 = 2400.
        But the long call at 6200 is OTM, so: 2×1200 + 0 = 2400."""
        from data_pipeline.settlement import compute_lon_fly
        result = compute_lon_fly(5000.0)
        expected = 2 * max(0, 6200 - 5000)  # = 2400; call/short-call/long-call3 all 0
        self.assertAlmostEqual(result, expected, places=2)

    def test_lon_fly_between_6200_and_6600(self):
        """ETF=6400: puts expire worthless; 1 call(6200) ITM; calls(6600,7000) OTM.
        = 0 + max(0,6400−6200) − 0 + 0 = 200."""
        from data_pipeline.settlement import compute_lon_fly
        result = compute_lon_fly(6400.0)
        self.assertAlmostEqual(result, 200.0, places=2)

    def test_lon_fly_between_6600_and_7000(self):
        """ETF=6800: short 6600 calls start biting.
        = 0 + (6800−6200) − 2×(6800−6600) + 0
        = 600 − 400 = 200."""
        from data_pipeline.settlement import compute_lon_fly
        result = compute_lon_fly(6800.0)
        expected = (6800 - 6200) - 2 * (6800 - 6600)
        self.assertAlmostEqual(result, expected, places=2)

    def test_lon_fly_above_7000(self):
        """ETF=7500: all calls ITM.
        = 0 + (7500−6200) − 2×(7500−6600) + 3×(7500−7000)
        = 1300 − 1800 + 1500 = 1000."""
        from data_pipeline.settlement import compute_lon_fly
        result = compute_lon_fly(7500.0)
        expected = (7500 - 6200) - 2 * (7500 - 6600) + 3 * (7500 - 7000)
        self.assertAlmostEqual(result, expected, places=2)

    def test_lon_fly_returns_none_on_none_etf(self):
        from data_pipeline.settlement import compute_lon_fly
        self.assertIsNone(compute_lon_fly(None))


# -----------------------------------------------------------------------
# 5. Fetcher integration tests (mocked HTTP)
# -----------------------------------------------------------------------

class TestThamesFetcherMocked(unittest.TestCase):

    @patch("data_pipeline.thames.requests.get")
    def test_fetch_thames_parses_correctly(self, mock_get):
        """Verify the fetcher parses the EA API response format correctly."""
        from data_pipeline.thames import fetch_thames_readings

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "items": [
                {"dateTime": "2025-11-22T12:00:00Z", "value": -1.41},
                {"dateTime": "2025-11-22T11:45:00Z", "value": -1.42},
                {"dateTime": "2025-11-22T11:30:00Z", "value": -1.43},
            ]
        }
        mock_get.return_value = mock_resp

        since = datetime(2025, 11, 22, 11, 0, tzinfo=timezone.utc)
        readings = fetch_thames_readings(since=since)

        self.assertEqual(len(readings), 3)
        # Should be sorted ascending
        self.assertLess(readings[0].dt, readings[1].dt)
        self.assertLess(readings[1].dt, readings[2].dt)
        # Verify values
        self.assertAlmostEqual(readings[2].value_maod, -1.41, places=3)

    @patch("data_pipeline.thames.requests.get")
    def test_fetch_thames_handles_missing_fields(self, mock_get):
        """Items with missing dateTime or value should be silently skipped."""
        from data_pipeline.thames import fetch_thames_readings

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "items": [
                {"dateTime": "2025-11-22T12:00:00Z", "value": -1.41},
                {"dateTime": "2025-11-22T11:45:00Z"},              # missing value
                {"value": -1.43},                                   # missing dateTime
            ]
        }
        mock_get.return_value = mock_resp

        since = datetime(2025, 11, 22, 11, 0, tzinfo=timezone.utc)
        readings = fetch_thames_readings(since=since)
        self.assertEqual(len(readings), 1)


class TestWeatherFetcherMocked(unittest.TestCase):

    @patch("data_pipeline.weather.requests.get")
    def test_fetch_weather_parses_correctly(self, mock_get):
        """Verify the fetcher handles Open-Meteo's minutely_15 structure."""
        from data_pipeline.weather import fetch_weather_readings

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "minutely_15": {
                "time": [
                    "2025-11-22T12:00",
                    "2025-11-22T12:15",
                ],
                "temperature_2m": [33.0, 33.2],
                "relative_humidity_2m": [29, 30],
            }
        }
        mock_get.return_value = mock_resp

        start = _london(2025, 11, 22, 11, 59)
        end   = _london(2025, 11, 22, 12, 30)
        readings = fetch_weather_readings(start, end)

        self.assertEqual(len(readings), 2)
        self.assertAlmostEqual(readings[0].temp_f, 33.0, places=2)
        self.assertEqual(readings[0].humidity, 29)

    @patch("data_pipeline.weather.requests.get")
    def test_fetch_weather_filters_to_window(self, mock_get):
        """Readings outside [start, end] should be excluded."""
        from data_pipeline.weather import fetch_weather_readings

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "minutely_15": {
                "time": [
                    "2025-11-21T10:00",   # before start
                    "2025-11-22T12:00",   # in window
                    "2025-11-24T00:00",   # after end
                ],
                "temperature_2m": [30.0, 33.0, 35.0],
                "relative_humidity_2m": [80, 29, 60],
            }
        }
        mock_get.return_value = mock_resp

        start = _london(2025, 11, 22, 12, 0)
        end   = _london(2025, 11, 23, 12, 0)
        readings = fetch_weather_readings(start, end)

        self.assertEqual(len(readings), 1)
        self.assertAlmostEqual(readings[0].temp_f, 33.0, places=2)


class TestFlightsFetcherMocked(unittest.TestCase):

    def _make_flight_response(self, n_arrivals: int, n_departures: int, base_dt: str, start_hour: int = 12):
        """Build a fake AeroDataBox response.

        start_hour ensures all timestamps fall within the market window (≥12:00).
        Each flight gets a distinct minute within the window.
        """
        arrivals = [
            {"movement": {"scheduledTime": {
                "local": f"{base_dt} {start_hour + (i * 2 // 60):02d}:{(i * 2) % 60:02d}+00:00"
            }}}
            for i in range(n_arrivals)
        ]
        departures = [
            {"movement": {"scheduledTime": {
                "local": f"{base_dt} {start_hour + (i * 2 // 60):02d}:{(i * 2) % 60:02d}+00:00"
            }}}
            for i in range(n_departures)
        ]
        return {"arrivals": arrivals, "departures": departures}

    @patch("data_pipeline.flights.requests.get")
    def test_fetch_flight_intervals_counts_correctly(self, mock_get):
        """End-to-end: fetched flights should be bucketed and counted."""
        from data_pipeline.flights import fetch_flight_intervals, compute_lhr_count

        # Chunk 1: Saturday 12:00 → midnight — flights at 12:xx on 2025-11-22
        resp1 = MagicMock()
        resp1.raise_for_status = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = self._make_flight_response(10, 8, "2025-11-22", start_hour=12)

        # Chunk 2: midnight → Sunday 12:00 — flights at 00:xx on 2025-11-23
        resp2 = MagicMock()
        resp2.raise_for_status = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = self._make_flight_response(5, 7, "2025-11-23", start_hour=0)

        mock_get.side_effect = [resp1, resp2]

        start = _london(2025, 11, 22, 12, 0)
        end   = _london(2025, 11, 23, 12, 0)
        intervals = fetch_flight_intervals(start, end)

        total = compute_lhr_count(intervals)
        self.assertEqual(total, 10 + 8 + 5 + 7)  # 30

    @patch("data_pipeline.flights.requests.get")
    def test_fetch_flights_raises_on_bad_api_key(self, mock_get):
        """A 401 response should raise RuntimeError immediately (no retry)."""
        from data_pipeline.flights import fetch_flight_intervals

        mock_resp = MagicMock()
        mock_resp.status_code = 401

        mock_get.return_value = mock_resp

        with self.assertRaises(RuntimeError) as ctx:
            fetch_flight_intervals(_london(2025, 11, 22, 12, 0), _london(2025, 11, 23, 12, 0))

        self.assertIn("API key", str(ctx.exception))


# -----------------------------------------------------------------------
# 6. Full pipeline integration test
# -----------------------------------------------------------------------

class TestDataPipeline(unittest.TestCase):

    @patch("data_pipeline.pipeline.RAPIDAPI_KEY", "FAKE_KEY_FOR_TEST")
    @patch("data_pipeline.pipeline.fetch_flight_intervals")
    @patch("data_pipeline.pipeline.fetch_weather_readings")
    @patch("data_pipeline.pipeline.fetch_thames_readings")
    def test_full_pipeline_snapshot(self, mock_thames, mock_weather, mock_flights):
        """Full pipeline: mock all three fetchers and verify snapshot estimates."""
        from data_pipeline.thames import ThamesReading
        from data_pipeline.weather import WeatherReading
        from data_pipeline.flights import FlightInterval
        from data_pipeline.pipeline import DataPipeline

        settlement = _london(2025, 11, 23, 12, 0)
        start      = _london(2025, 11, 22, 12, 0)

        # Mock Thames: one reading at settlement time = -1.41m
        mock_thames.return_value = [
            ThamesReading(dt=settlement, value_maod=-1.41)
        ]

        # Mock Weather: one reading at settlement = 33°F, 29% humidity
        mock_weather.return_value = [
            WeatherReading(dt=settlement, temp_f=33.0, humidity=29)
        ]

        # Mock Flights: a single 30-min bucket with 300 arr, 200 dep
        mock_flights.return_value = [
            FlightInterval(bucket_start=_london(2025, 11, 22, 12, 0), arrivals=300, departures=200)
        ]

        pipeline = DataPipeline(window_start=start, window_end=settlement, skip_flights=False)
        snapshot = pipeline.fetch_snapshot(use_cache=False)

        est = snapshot.estimates

        # M1: ABS(-1.41) × 1000 = 1410
        self.assertAlmostEqual(est.m1_tide_spot, 1410.0, places=1)
        # M3: round(33) × 29 = 957
        self.assertAlmostEqual(est.m3_wx_spot, 957.0, places=1)
        # M5: 300 + 200 = 500
        self.assertIsNotNone(est.m5_lhr_count)
        # M7: ABS(1410 + 957 + 500) = 2867
        self.assertIsNotNone(est.m7_lon_etf)
        self.assertAlmostEqual(est.m7_lon_etf, 2867.0, places=0)
        # M8: all strikes (6200, 6600, 7000) above 2867 → only puts ITM
        # 2×max(0,6200−2867) + 1×max(0,2867−6200) = 2×3333 = 6666
        self.assertIsNotNone(est.m8_lon_fly)
        self.assertAlmostEqual(est.m8_lon_fly, 2 * (6200 - 2867), places=0)

        # No errors should be reported
        self.assertEqual(snapshot.errors, {})

    def test_pipeline_skip_flights_gives_none_for_derived(self):
        """Without flights, M5/M6/M7/M8 should all be None."""
        from data_pipeline.pipeline import DataPipeline

        start = _london(2025, 11, 22, 12, 0)
        end   = _london(2025, 11, 23, 12, 0)

        pipeline = DataPipeline(start, end, skip_flights=True)

        with patch("data_pipeline.pipeline.fetch_thames_readings") as mt, \
             patch("data_pipeline.pipeline.fetch_weather_readings") as mw:

            from data_pipeline.thames import ThamesReading
            from data_pipeline.weather import WeatherReading

            mt.return_value = [ThamesReading(dt=end, value_maod=-1.41)]
            mw.return_value = [WeatherReading(dt=end, temp_f=33.0, humidity=29)]

            snapshot = pipeline.fetch_snapshot(use_cache=False)

        est = snapshot.estimates
        self.assertIsNone(est.m5_lhr_count)
        self.assertIsNone(est.m6_lhr_index)
        self.assertIsNone(est.m7_lon_etf)
        self.assertIsNone(est.m8_lon_fly)

    def test_pipeline_caching(self):
        """Second fetch_snapshot call should reuse cached data (no new HTTP calls)."""
        from data_pipeline.pipeline import DataPipeline

        start = _london(2025, 11, 22, 12, 0)
        end   = _london(2025, 11, 23, 12, 0)
        pipeline = DataPipeline(start, end, skip_flights=True)

        with patch("data_pipeline.pipeline.fetch_thames_readings") as mt, \
             patch("data_pipeline.pipeline.fetch_weather_readings") as mw:

            from data_pipeline.thames import ThamesReading
            from data_pipeline.weather import WeatherReading

            mt.return_value = [ThamesReading(dt=end, value_maod=-1.41)]
            mw.return_value = [WeatherReading(dt=end, temp_f=33.0, humidity=29)]

            pipeline.fetch_snapshot(use_cache=True)
            pipeline.fetch_snapshot(use_cache=True)

        # Each fetcher should only have been called once (cache hit on second call)
        self.assertEqual(mt.call_count, 1)
        self.assertEqual(mw.call_count, 1)


# -----------------------------------------------------------------------
# 7. get_market_window helper
# -----------------------------------------------------------------------

class TestGetMarketWindow(unittest.TestCase):

    def test_window_is_24_hours(self):
        from data_pipeline.pipeline import get_market_window
        start, end = get_market_window()
        delta = end - start
        self.assertAlmostEqual(delta.total_seconds(), 24 * 3600, delta=60)

    def test_start_is_saturday_noon(self):
        from data_pipeline.pipeline import get_market_window
        start, _ = get_market_window()
        # weekday() 5 = Saturday
        self.assertEqual(start.weekday(), 5)
        self.assertEqual(start.hour, 12)
        self.assertEqual(start.minute, 0)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
