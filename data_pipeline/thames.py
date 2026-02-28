"""Thames tidal level data — Markets 1 (TIDE_SPOT) and 2 (TIDE_SWING).

Data source: UK Environment Agency Flood Monitoring API
  URL: https://environment.data.gov.uk/flood-monitoring/...
  Cost: FREE, no API key required.
  Update cadence: every 15 minutes.

Market 1 — TIDE_SPOT
  Settlement = ABS(tidal height in MAOD) × 1000, at Sunday 12:00 London time.
  Example: reading of -1.41 m → settles to 1410.

Market 2 — TIDE_SWING
  Settlement = sum of strangle payoffs on every consecutive 15-min diff × 100.
  Strangle strikes: put at 0.20, call at 0.25 (on the absolute diff).
  Payoff per interval = max(0, 0.20 − diff) + max(0, diff − 0.25).
  Example: diff=0.09 → 0.11, diff=0.33 → 0.08, diff=0.21 → 0.00.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from .config import MAX_RETRIES, REQUEST_TIMEOUT, RETRY_BACKOFF

log = logging.getLogger(__name__)

_READINGS_URL = (
    "https://environment.data.gov.uk/flood-monitoring/id/measures/"
    "0006-level-tidal_level-i-15_min-mAOD/readings"
)

# Strangle option strikes for Market 2
_K_PUT = 0.20   # lower strike — put is ITM when diff < 0.20
_K_CALL = 0.25  # upper strike — call is ITM when diff > 0.25


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThamesReading:
    """One 15-minute Thames tidal reading."""
    dt: datetime        # timezone-aware UTC datetime
    value_maod: float   # metres above ordnance datum (can be negative)


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def fetch_thames_readings(
    since: datetime,
    limit: int = 200,
) -> list[ThamesReading]:
    """Fetch Thames tidal level readings from the Environment Agency API.

    Args:
        since: Fetch readings at or after this UTC datetime.
               Use the market window start (Saturday 12:00 London → UTC).
        limit: Maximum readings to fetch.
               96 readings cover exactly 24 h at 15-min intervals.
               200 gives comfortable headroom.

    Returns:
        List of ThamesReading sorted ascending by datetime (oldest first).

    Raises:
        RuntimeError: If every retry attempt fails.
    """
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)

    params = {
        "_sorted": "true",
        "_limit": limit,
        "since": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(_READINGS_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            items = resp.json().get("items", [])

            readings: list[ThamesReading] = []
            for item in items:
                dt_str = item.get("dateTime", "")
                value = item.get("value")
                if not dt_str or value is None:
                    continue
                try:
                    # API returns ISO 8601 with 'Z' suffix — normalise to +00:00
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    readings.append(ThamesReading(dt=dt, value_maod=float(value)))
                except (ValueError, TypeError) as exc:
                    log.debug("Skipping malformed Thames item %s: %s", item, exc)

            # The API returns in descending order; reverse to ascending
            readings.sort(key=lambda r: r.dt)
            log.debug("Thames: fetched %d readings since %s", len(readings), since)
            return readings

        except requests.RequestException as exc:
            log.warning(
                "Thames fetch attempt %d/%d failed: %s",
                attempt + 1, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))

    raise RuntimeError(f"Thames API unreachable after {MAX_RETRIES} attempts")


# ---------------------------------------------------------------------------
# Market 1 — TIDE_SPOT
# ---------------------------------------------------------------------------

def compute_tide_spot(
    readings: list[ThamesReading],
    settlement_dt: datetime,
    tolerance_minutes: int = 20,
) -> Optional[float]:
    """Market 1: ABS(tidal height in MAOD) × 1000 at settlement time.

    Finds the reading closest in time to settlement_dt.  Returns None if
    no reading exists within tolerance_minutes of the target time.

    Args:
        readings:          All Thames readings for the window (sorted asc).
        settlement_dt:     Target datetime (Sunday 12:00 London, as UTC).
        tolerance_minutes: Accept a reading up to this many minutes away.
    """
    if not readings:
        return None

    if settlement_dt.tzinfo is None:
        settlement_dt = settlement_dt.replace(tzinfo=timezone.utc)

    best = min(readings, key=lambda r: abs((r.dt - settlement_dt).total_seconds()))
    delta_minutes = abs((best.dt - settlement_dt).total_seconds()) / 60.0

    if delta_minutes > tolerance_minutes:
        log.warning(
            "TIDE_SPOT: closest reading is %.1f min from settlement target "
            "(tolerance=%d min) — returning None",
            delta_minutes, tolerance_minutes,
        )
        return None

    return abs(best.value_maod) * 1000.0


# ---------------------------------------------------------------------------
# Market 2 — TIDE_SWING
# ---------------------------------------------------------------------------

def _strangle_payoff(diff: float) -> float:
    """Long strangle payoff on a single absolute 15-min water-level change.

    Payoff = max(0, K_put − diff) + max(0, diff − K_call)
           = max(0, 0.20 − diff) + max(0, diff − 0.25)

    The strangle is zero (both options OTM) when 0.20 ≤ diff ≤ 0.25.
    """
    return max(0.0, _K_PUT - diff) + max(0.0, diff - _K_CALL)


def compute_tide_swing(
    readings: list[ThamesReading],
    start_dt: datetime,
    end_dt: datetime,
    max_gap_minutes: int = 20,
) -> float:
    """Market 2: cumulative strangle sum × 100 over the 24-hour window.

    For each consecutive pair of readings within [start_dt, end_dt]:
        diff    = abs(reading[t].value − reading[t−1].value)
        strangle = max(0, 0.20 − diff) + max(0, diff − 0.25)
    Settlement = sum(strangles) × 100.

    Args:
        readings:          All Thames readings (sorted ascending).
        start_dt:          Market window start (Saturday 12:00 UTC).
        end_dt:            Market window end   (Sunday   12:00 UTC).
        max_gap_minutes:   Skip a pair if the time gap exceeds this — avoids
                           inflating the diff across a data outage.

    Returns:
        Running strangle sum × 100.  During the window this is a partial
        total; at settlement time it is the final value.
    """
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=timezone.utc)

    window = [r for r in readings if start_dt <= r.dt <= end_dt]

    if len(window) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(window)):
        gap_minutes = (window[i].dt - window[i - 1].dt).total_seconds() / 60.0
        if gap_minutes > max_gap_minutes:
            # Gap too large — data probably missing; skip to avoid a spike
            log.debug(
                "TIDE_SWING: skipping pair with %.1f-min gap at %s",
                gap_minutes, window[i].dt,
            )
            continue
        diff = abs(window[i].value_maod - window[i - 1].value_maod)
        total += _strangle_payoff(diff)

    return total * 100.0


def project_tide_swing(
    current_sum: float,
    readings_in_window: int,
    expected_total_readings: int = 96,
) -> float:
    """Linear projection of Market 2 final settlement from partial data.

    If we are 40 % through the window and the running sum is X, the
    projected final value is X / 0.40 = 2.5 × X.

    Args:
        current_sum:              Running TIDE_SWING value so far.
        readings_in_window:       Number of 15-min readings processed.
        expected_total_readings:  96 for a full 24-h window (4 per hour × 24).
    """
    if readings_in_window <= 1:
        return current_sum
    fraction = min(readings_in_window / expected_total_readings, 1.0)
    return current_sum / fraction if fraction > 0 else current_sum
