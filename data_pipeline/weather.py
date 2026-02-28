"""London weather data — Markets 3 (WX_SPOT) and 4 (WX_SUM).

Data source: Open-Meteo API
  URL: https://api.open-meteo.com/v1/forecast
  Cost: FREE, no API key required.
  Update cadence: model updates roughly every hour; 15-min resolution.

KEY COMPETITIVE ADVANTAGE
  Open-Meteo provides FORECAST data — we can look at the predicted
  temperature and humidity at Sunday 12:00 *before that time arrives*.
  This gives an early, accurate estimate of WX_SPOT (Market 3).

Market 3 — WX_SPOT
  Settlement = round(Temperature_°F) × Humidity at Sunday 12:00 London time.
  Example: 33°F × 29% humidity → settles to 957.

Market 4 — WX_SUM
  Settlement = sum of (Temperature_°F × Humidity) / 100 for every
  15-minute interval in the 24-hour market window.
  Example: (33×29)/100 + (30×30)/100 + (30×90)/100 + … = 9.57 + 9.00 + 27 + …
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional
import zoneinfo

import requests

from .config import (
    LONDON_LAT, LONDON_LON, LONDON_TIMEZONE,
    REQUEST_TIMEOUT, http_session,
)

log = logging.getLogger(__name__)

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_LONDON_TZ = zoneinfo.ZoneInfo(LONDON_TIMEZONE)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeatherReading:
    """One 15-minute weather observation or forecast for London."""
    dt: datetime      # timezone-aware (Europe/London)
    temp_f: float     # temperature in Fahrenheit
    humidity: int     # relative humidity 0–100 %


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_weather_readings(
    start_dt: datetime,
    end_dt: datetime,
) -> list[WeatherReading]:
    """Fetch 15-minute temperature and humidity readings for the market window.

    Fetches both historical data (already elapsed portion of the window) AND
    forecast data (remaining portion), so we always have a full 24-h view.
    This enables early prediction of WX_SPOT before Sunday noon arrives.

    Args:
        start_dt: Market window start (Saturday 12:00 London time or UTC).
        end_dt:   Market window end   (Sunday   12:00 London time or UTC).

    Returns:
        List of WeatherReading sorted ascending by datetime (oldest first).
        Includes both past readings and forecast readings for the full window.

    Raises:
        RuntimeError: If every retry attempt fails.
    """
    # Ensure timezone-aware; treat naive as London time
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=_LONDON_TZ)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=_LONDON_TZ)

    # past_days=2 ensures we always capture Saturday data even if fetching Sunday
    params = {
        "latitude": LONDON_LAT,
        "longitude": LONDON_LON,
        "minutely_15": "temperature_2m,relative_humidity_2m",
        "temperature_unit": "fahrenheit",
        "timezone": LONDON_TIMEZONE,
        "past_days": 2,
        "forecast_days": 2,   # enough to cover Sunday noon
    }

    try:
        resp = http_session.get(_OPEN_METEO_URL, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        m15 = data.get("minutely_15", {})
        times = m15.get("time", [])
        temps = m15.get("temperature_2m", [])
        humids = m15.get("relative_humidity_2m", [])

        if not times:
            log.warning("Open-Meteo returned no minutely_15 data")
            return []

        readings: list[WeatherReading] = []
        for t_str, temp, humid in zip(times, temps, humids):
            if temp is None or humid is None:
                continue
            try:
                # Open-Meteo returns local datetime strings like "2025-11-22T14:15"
                dt = datetime.fromisoformat(t_str).replace(tzinfo=_LONDON_TZ)
                # Filter strictly to our market window
                if start_dt <= dt <= end_dt:
                    readings.append(WeatherReading(
                        dt=dt,
                        temp_f=float(temp),
                        humidity=int(round(humid)),
                    ))
            except (ValueError, TypeError) as exc:
                log.debug("Skipping malformed weather entry %s: %s", t_str, exc)

        readings.sort(key=lambda r: r.dt)
        log.debug("Weather: fetched %d readings for window", len(readings))
        return readings

    except requests.RequestException as exc:
        log.error("Weather API unreachable: %s", exc)
        raise RuntimeError(f"Open-Meteo API unreachable: {exc}")


# ---------------------------------------------------------------------------
# Market 3 — WX_SPOT
# ---------------------------------------------------------------------------

def compute_wx_spot(
    readings: list[WeatherReading],
    settlement_dt: datetime,
    tolerance_minutes: int = 20,
) -> Optional[float]:
    """Market 3: round(Temp_°F) × Humidity at settlement time.

    Because we fetch FORECAST data this function returns a valid estimate
    even hours before Sunday noon — giving early trading edge.

    Args:
        readings:          All WeatherReadings for the window (sorted asc).
                           Can include forecast entries for future timestamps.
        settlement_dt:     Target datetime (Sunday 12:00 London time).
        tolerance_minutes: Accept a reading up to this many minutes away.
    """
    if not readings:
        return None

    if settlement_dt.tzinfo is None:
        settlement_dt = settlement_dt.replace(tzinfo=_LONDON_TZ)

    best = min(readings, key=lambda r: abs((r.dt - settlement_dt).total_seconds()))
    delta_minutes = abs((best.dt - settlement_dt).total_seconds()) / 60.0

    if delta_minutes > tolerance_minutes:
        log.warning(
            "WX_SPOT: closest reading is %.1f min from settlement target "
            "(tolerance=%d min) — returning None",
            delta_minutes, tolerance_minutes,
        )
        return None

    return float(round(best.temp_f) * best.humidity)


# ---------------------------------------------------------------------------
# Market 4 — WX_SUM
# ---------------------------------------------------------------------------

def compute_wx_sum(
    readings: list[WeatherReading],
    start_dt: datetime,
    end_dt: datetime,
) -> float:
    """Market 4: sum of (Temp_°F × Humidity) / 100 over all 15-min intervals.

    Because we fetch FORECAST data this gives a projected full-window total
    even at the start of the trading session, not just a running partial sum.

    Args:
        readings:  WeatherReadings in [start_dt, end_dt] (sorted ascending).
                   Forecast entries count — they represent our best estimate
                   of what the temperature and humidity will be.
        start_dt:  Market window start.
        end_dt:    Market window end.

    Returns:
        Sum of contributions across all available (historical + forecast)
        readings in the window.
    """
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=_LONDON_TZ)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=_LONDON_TZ)

    window = [r for r in readings if start_dt <= r.dt <= end_dt]

    total = sum(round(r.temp_f) * r.humidity / 100.0 for r in window)
    log.debug("WX_SUM: %d intervals, running total=%.2f", len(window), total)
    return total


def project_wx_sum(
    readings_so_far: list[WeatherReading],
    start_dt: datetime,
    end_dt: datetime,
) -> float:
    """Estimate final WX_SUM settlement using forecast data.

    Since fetch_weather_readings() already includes forecast entries for the
    full window, compute_wx_sum() on those readings IS the projection.
    This helper exists as a clear alias and for documentation purposes.

    Returns:
        The projected final WX_SUM (same as compute_wx_sum with forecast data).
    """
    return compute_wx_sum(readings_so_far, start_dt, end_dt)
