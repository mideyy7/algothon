"""Main data pipeline orchestrator.

Fetches all three data sources in parallel, caches results to avoid
hammering external APIs, and assembles a complete MarketEstimates snapshot.

Typical usage in your trading bot:

    from data_pipeline import DataPipeline, get_market_window

    start, end = get_market_window()
    pipeline = DataPipeline(start, end)

    # Inside your on_orderbook callback (or a polling loop):
    snapshot = pipeline.fetch_snapshot()
    fair_values = snapshot.estimates.as_dict()
    # e.g. {"TIDE_SPOT": 1410.0, "WX_SPOT": 957.0, ...}
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional
import zoneinfo

from .thames import (
    ThamesReading,
    fetch_thames_readings,
    compute_tide_spot,
    compute_tide_swing,
    project_tide_swing,
)
from .weather import (
    WeatherReading,
    fetch_weather_readings,
    compute_wx_spot,
    compute_wx_sum,
)
from .flights import (
    FlightInterval,
    fetch_flight_intervals,
    compute_lhr_count,
    compute_lhr_index,
    project_lhr_values,
)
from .settlement import (
    MarketEstimates,
    compute_lon_etf,
    compute_lon_fly,
)
from .config import (
    PIHUB_API_KEY,
    THAMES_CACHE_TTL,
    WEATHER_CACHE_TTL,
    FLIGHTS_CACHE_TTL,
    LONDON_TIMEZONE,
)

log = logging.getLogger(__name__)

_LONDON_TZ = zoneinfo.ZoneInfo(LONDON_TIMEZONE)


# ---------------------------------------------------------------------------
# Market window helper
# ---------------------------------------------------------------------------

def get_market_window() -> tuple[datetime, datetime]:
    """Return (start, end) for the current or most recent market window.

    The market runs Saturday 12:00 → Sunday 12:00 London time.
    If called mid-window the returned end is still Sunday noon (future).
    If called after the window has closed, returns the completed window.

    Returns:
        Tuple of timezone-aware datetimes in Europe/London.
    """
    now = datetime.now(_LONDON_TZ)
    # weekday(): Monday=0 … Saturday=5, Sunday=6
    days_since_saturday = (now.weekday() - 5) % 7
    last_saturday = now - timedelta(days=days_since_saturday)
    start = last_saturday.replace(hour=12, minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=24)   # Sunday noon
    return start, end


# ---------------------------------------------------------------------------
# Simple TTL cache
# ---------------------------------------------------------------------------

class _TTLCache:
    """Thread-safe time-to-live cache (stores one value per key)."""

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: float) -> None:
        self._store[key] = (value, time.monotonic() + ttl)


# ---------------------------------------------------------------------------
# Pipeline snapshot result
# ---------------------------------------------------------------------------

@dataclass
class PipelineSnapshot:
    """Complete data snapshot from one pipeline fetch.

    Attributes:
        fetched_at:       Wall-clock time when this snapshot was assembled.
        window_start:     Market window start datetime.
        window_end:       Market window end datetime.
        thames_readings:  Raw Thames readings for the window.
        weather_readings: Raw weather readings (hist + forecast) for the window.
        flight_intervals: Bucketed flight counts for the window.
        estimates:        Computed best estimates for all 8 market settlements.
        errors:           Dict of source → error message for any failed fetches.
    """
    fetched_at:       datetime
    window_start:     datetime
    window_end:       datetime
    thames_readings:  list[ThamesReading]       = field(default_factory=list)
    weather_readings: list[WeatherReading]      = field(default_factory=list)
    flight_intervals: list[FlightInterval]      = field(default_factory=list)
    estimates:        MarketEstimates           = field(default_factory=MarketEstimates)
    errors:           dict[str, str]            = field(default_factory=dict)
    
    @property
    def raw_api_data(self) -> dict[str, list[dict]]:
        """Returns the purely raw, uncalculated components as simple dictionaries
        for standard quantitative ingestion.
        """
        return {
            "THAMES_WATER_LEVELS": [
                {"timestamp": r.dt.isoformat(), "level": r.value_maod} 
                for r in self.thames_readings
            ],
            "WEATHER_FORECASTS": [
                {"timestamp": r.dt.isoformat(), "temp_f": r.temp_f, "humidity": r.humidity} 
                for r in self.weather_readings
            ],
            "FLIGHT_BUCKETS": [
                {"timestamp": r.bucket_start.isoformat(), "arrivals": r.arrivals, "departures": r.departures} 
                for r in self.flight_intervals
            ]
        }


# ---------------------------------------------------------------------------
# DataPipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """Fetches all London data sources in parallel and computes market estimates.

    Args:
        window_start: Market window start (Saturday 12:00 London time).
        window_end:   Market window end   (Sunday   12:00 London time).
                      Pass these from get_market_window().
        skip_flights: Set True to skip the AeroDataBox API entirely
                      (useful when RAPIDAPI_KEY is not yet configured).
    """

    def __init__(
        self,
        window_start: datetime,
        window_end: datetime,
        skip_flights: bool = False,
    ):
        self.window_start = window_start
        self.window_end   = window_end
        # Auto-skip flights only when the caller explicitly requests it.
        # PIHub may be a public endpoint; if auth IS required and the key is
        # absent the fetcher will raise a RuntimeError at fetch time with a
        # clear message rather than silently skipping.
        self.skip_flights = skip_flights

        if self.skip_flights:
            log.warning(
                "Flights data disabled — set skip_flights=False (default) to enable "
                "Markets 5 (LHR_COUNT), 6 (LHR_INDEX), and the ETF components. "
                "If PIHub requires auth, set PIHUB_API_KEY in your .env file."
            )

        self._cache = _TTLCache()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_snapshot(self, use_cache: bool = True) -> PipelineSnapshot:
        """Fetch all data sources in parallel and return a complete snapshot.

        Uses TTL caching so repeated calls within the cache TTL are free.
        Pass use_cache=False to force a fresh fetch (e.g., at settlement time).

        Returns:
            PipelineSnapshot with raw data and estimates for all 8 markets.
        """
        thames_data:  list[ThamesReading]  = []
        weather_data: list[WeatherReading] = []
        flight_data:  list[FlightInterval] = []
        errors: dict[str, str] = {}

        # --- Check cache first ---
        if use_cache:
            cached_thames  = self._cache.get("thames")
            cached_weather = self._cache.get("weather")
            cached_flights = self._cache.get("flights")
        else:
            cached_thames = cached_weather = cached_flights = None

        # --- Determine which sources need a fresh fetch ---
        tasks: dict[str, bool] = {
            "thames":  cached_thames  is None,
            "weather": cached_weather is None,
            "flights": cached_flights is None and not self.skip_flights,
        }

        # --- Parallel fetch for sources that need refreshing ---
        if any(tasks.values()):
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="pipeline") as ex:
                futures: dict[Future, str] = {}

                if tasks["thames"]:
                    futures[ex.submit(self._fetch_thames)] = "thames"
                if tasks["weather"]:
                    futures[ex.submit(self._fetch_weather)] = "weather"
                if tasks["flights"]:
                    futures[ex.submit(self._fetch_flights)] = "flights"

                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        result = future.result()
                        self._cache.set(source, result, self._ttl_for(source))
                        log.debug("Pipeline: %s fetch complete (%d items)", source, len(result))
                    except Exception as exc:
                        errors[source] = str(exc)
                        log.error("Pipeline: %s fetch failed: %s", source, exc)

        # --- Resolve from cache (fresh or previously cached) ---
        thames_data  = self._cache.get("thames")  or []
        weather_data = self._cache.get("weather") or []
        flight_data  = self._cache.get("flights") or []

        # --- Compute estimates ---
        estimates = self._compute_estimates(thames_data, weather_data, flight_data)

        return PipelineSnapshot(
            fetched_at=datetime.now(_LONDON_TZ),
            window_start=self.window_start,
            window_end=self.window_end,
            thames_readings=thames_data,
            weather_readings=weather_data,
            flight_intervals=flight_data,
            estimates=estimates,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internal fetch methods (each wraps one module's fetcher)
    # ------------------------------------------------------------------

    def _fetch_thames(self) -> list[ThamesReading]:
        return fetch_thames_readings(since=self.window_start, limit=200)

    def _fetch_weather(self) -> list[WeatherReading]:
        return fetch_weather_readings(
            start_dt=self.window_start,
            end_dt=self.window_end,
        )

    def _fetch_flights(self) -> list[FlightInterval]:
        return fetch_flight_intervals(
            start_dt=self.window_start,
            end_dt=self.window_end,
        )

    # ------------------------------------------------------------------
    # Estimate assembly
    # ------------------------------------------------------------------

    def _compute_estimates(
        self,
        thames: list[ThamesReading],
        weather: list[WeatherReading],
        flights: list[FlightInterval],
    ) -> MarketEstimates:
        """Combine raw data into best-estimate settlement prices."""
        now = datetime.now(_LONDON_TZ)
        elapsed = (now - self.window_start).total_seconds()
        total   = (self.window_end - self.window_start).total_seconds()
        elapsed_fraction = min(max(elapsed / total, 0.0), 1.0)

        # --- Markets 1 & 2 (Thames) ---
        m1 = compute_tide_spot(thames, self.window_end)
        m2_running = compute_tide_swing(thames, self.window_start, self.window_end)
        # Project M2 to the full 24-h total using what we have so far
        readings_in_window = len([r for r in thames
                                  if self.window_start <= r.dt <= self.window_end])
        m2 = project_tide_swing(m2_running, readings_in_window)

        # --- Markets 3 & 4 (Weather) ---
        # fetch_weather_readings already includes forecast data, so these are
        # full-window estimates even at the start of the trading session
        m3 = compute_wx_spot(weather, self.window_end)
        m4 = compute_wx_sum(weather, self.window_start, self.window_end)

        # --- Markets 5 & 6 (Flights) ---
        # AeroDataBox returns ALL flights for the full queried window (historical actuals
        # + future scheduled), so the count is already a full-window estimate from the
        # first fetch — no linear projection needed (unlike Thames which accumulates).
        if flights:
            m5: Optional[float] = float(compute_lhr_count(flights))
            m6: Optional[float] = compute_lhr_index(flights)
        else:
            m5 = None
            m6 = None

        # --- Markets 7 & 8 (Derived) ---
        m7 = compute_lon_etf(m1, m3, m5)
        m8 = compute_lon_fly(m7)

        # --- Compute real-time "Current Raw" for Quants ---
        # M1: Latest reading
        raw_m1 = abs(thames[-1].value_maod) * 1000 if thames else None
        
        # M2: The running sum without the forward projection
        raw_m2 = m2_running
        
        # M3 & M4: Since these use forecasts, the "raw" real-time is effectively the same as the estimate
        # but we provide it for completeness
        raw_m3 = m3
        raw_m4 = m4
        
        # M5: The running count without the forward projection
        raw_m5 = None
        raw_m6 = None
        if flights:
            raw_m5 = float(compute_lhr_count(flights))
            raw_m6 = float(compute_lhr_index(flights))
        
        # M7 & M8: Live derived values based on raw inputs
        raw_m7 = None
        raw_m8 = None
        if raw_m1 is not None and raw_m3 is not None and raw_m5 is not None:
            raw_m7 = abs(raw_m1 + raw_m3 + raw_m5)
            raw_m8 = compute_lon_fly(raw_m7)

        return MarketEstimates(
            m1_tide_spot=m1,
            m2_tide_swing=m2,
            m3_wx_spot=m3,
            m4_wx_sum=m4,
            m5_lhr_count=m5,
            m6_lhr_index=m6,
            m7_lon_etf=m7,
            m8_lon_fly=m8,
            window_elapsed_fraction=elapsed_fraction,
            current_raw_data={
                "TIDE_SPOT": raw_m1,
                "TIDE_SWING": raw_m2,
                "WX_SPOT": raw_m3,
                "WX_SUM": raw_m4,
                "LHR_COUNT": raw_m5,
                "LHR_INDEX": raw_m6,
                "LON_ETF": raw_m7,
                "LON_FLY": raw_m8
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ttl_for(source: str) -> float:
        return {
            "thames":  THAMES_CACHE_TTL,
            "weather": WEATHER_CACHE_TTL,
            "flights": FLIGHTS_CACHE_TTL,
        }.get(source, 60.0)
