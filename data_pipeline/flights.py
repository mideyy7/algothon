"""Heathrow airport flight data — Markets 5 (LHR_COUNT) and 6 (LHR_INDEX).

Data source: AeroDataBox via RapidAPI
  URL: https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR/...
  Cost: Requires a RapidAPI key. Free tier = 500 calls/month (plenty).
  Update cadence: near-real-time for actual departure/arrival times.

USER SETUP: Set RAPIDAPI_KEY in config.py before using this module.

The API supports a maximum 12-hour query window per request.
For the 24-hour market window we split into two 12-hour chunks.

Market 5 — LHR_COUNT
  Settlement = total arrivals + total departures at LHR over 24 hours.

Market 6 — LHR_INDEX
  For each 30-minute interval:
      metric = 100 × (arrivals − departures) / max(arrivals + departures, 1)
  Settlement = ABS(sum of all interval metrics).
  This measures the net directional imbalance of flights across the day.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import zoneinfo

import requests

from .config import (
    RAPIDAPI_KEY, RAPIDAPI_HOST,
    REQUEST_TIMEOUT,
    LONDON_TIMEZONE,
    http_session,
)

log = logging.getLogger(__name__)

_LONDON_TZ = zoneinfo.ZoneInfo(LONDON_TIMEZONE)
_FLIGHTS_URL = "https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR"

# AeroDataBox supports at most 12 h per request
_MAX_WINDOW_HOURS = 12

# In-memory cache for fully elapsed 12-hour flight chunks to save RapidAPI quota
_historical_chunk_cache: dict[tuple[datetime, datetime], tuple[list[datetime], list[datetime]]] = {}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FlightInterval:
    """Arrivals and departures in one 30-minute time bucket."""
    bucket_start: datetime   # timezone-aware (Europe/London), start of the 30-min slot
    arrivals: int = 0
    departures: int = 0

    @property
    def total(self) -> int:
        return self.arrivals + self.departures

    @property
    def metric(self) -> float:
        """Per-interval contribution to LHR_INDEX (before summing)."""
        denom = max(self.arrivals + self.departures, 1)
        return 100.0 * (self.arrivals - self.departures) / denom


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rapidapi_headers() -> dict[str, str]:
    return {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }


def _fetch_chunk(
    chunk_start: datetime,
    chunk_end: datetime,
) -> tuple[list[datetime], list[datetime]]:
    """Fetch raw arrival and departure datetimes for one ≤12-hour chunk.

    Returns:
        (arrival_dts, departure_dts) — lists of timezone-aware datetimes
        representing when each flight actually arrived/departed (or scheduled
        time if the actual time is not yet known).

    Raises:
        RuntimeError: If every retry attempt fails for this chunk.

    NOTE: If the AeroDataBox response structure differs from what is expected
    here, adjust the key paths inside the loop below. The relevant fields are:
      - For departures: flight["departure"]["scheduledTime"]["local"] or
                        flight["departure"]["actualTime"]["local"]
      - For arrivals:   flight["arrival"]["scheduledTime"]["local"] or
                        flight["arrival"]["actualTime"]["local"]
    """
    # AeroDataBox expects local London time in the URL path: YYYY-MM-DDTHH:mm
    fmt = "%Y-%m-%dT%H:%M"
    if chunk_start.tzinfo is None:
        chunk_start = chunk_start.replace(tzinfo=_LONDON_TZ)
    if chunk_end.tzinfo is None:
        chunk_end = chunk_end.replace(tzinfo=_LONDON_TZ)

    local_start = chunk_start.astimezone(_LONDON_TZ).strftime(fmt)
    local_end   = chunk_end.astimezone(_LONDON_TZ).strftime(fmt)
    url = f"{_FLIGHTS_URL}/{local_start}/{local_end}"

    # Check if historical chunk has already been fetched
    now = datetime.now(_LONDON_TZ)
    is_historical = chunk_end < now
    if is_historical and (chunk_start, chunk_end) in _historical_chunk_cache:
        log.debug("Flights chunk %s→%s loaded from historical cache", local_start, local_end)
        return _historical_chunk_cache[(chunk_start, chunk_end)]

    # Additional filters to exclude cargo, private jets, and codeshares
    # (these typically aren't counted in published airport statistics)
    params = {
        "withLeg": "false",
        "withCancelled": "false",
        "withCodeshared": "true",   # include codeshares — they do use airport slots
        "withCargo": "false",
        "withPrivate": "false",
        "withLocation": "false",
    }

    try:
        resp = http_session.get(
            url,
            params=params,
            headers=_rapidapi_headers(),
            timeout=REQUEST_TIMEOUT,
        )

        # 401/403 almost always means the API key is wrong or missing
        if resp.status_code in (401, 403):
            raise RuntimeError(
                "AeroDataBox API key rejected (HTTP %d). "
                "Check RAPIDAPI_KEY in data_pipeline/config.py." % resp.status_code
            )

        resp.raise_for_status()
        data = resp.json()

        arrival_dts: list[datetime] = []
        departure_dts: list[datetime] = []

        def _parse_flight_time(flight: dict) -> Optional[datetime]:
            """Extract the best available timestamp from a flight dict.

            AeroDataBox response structure (current):
              flight["movement"]["runwayTime"]["local"]   — actual wheels on/off runway
              flight["movement"]["revisedTime"]["local"]  — gate revised time
              flight["movement"]["scheduledTime"]["local"] — original schedule
            Both arrivals and departures use "movement" as the timing key.
            """
            leg = flight.get("movement", {})
            # Prefer actual runway time, then revised, then scheduled
            for time_key in ("runwayTime", "revisedTime", "scheduledTime"):
                t = leg.get(time_key, {})
                local_str = t.get("local")
                if local_str:
                    try:
                        # Strings look like "2026-02-28 12:00+00:00"
                        return datetime.fromisoformat(local_str).astimezone(_LONDON_TZ)
                    except ValueError:
                        pass
            return None

        for flight in data.get("departures", []):
            dt = _parse_flight_time(flight)
            if dt:
                departure_dts.append(dt)

        for flight in data.get("arrivals", []):
            dt = _parse_flight_time(flight)
            if dt:
                arrival_dts.append(dt)

        log.debug(
            "Flights chunk %s→%s: %d arrivals, %d departures",
            local_start, local_end, len(arrival_dts), len(departure_dts),
        )

        if is_historical:
            _historical_chunk_cache[(chunk_start, chunk_end)] = (arrival_dts, departure_dts)

        return arrival_dts, departure_dts

    except requests.RequestException as exc:
        log.error("Flights API unreachable: %s", exc)
        raise RuntimeError(f"AeroDataBox API unreachable: {exc}")


def _assign_to_buckets(
    dts: list[datetime],
    window_start: datetime,
    bucket_minutes: int,
    buckets: dict[datetime, FlightInterval],
    is_arrival: bool,
) -> None:
    """Increment the appropriate FlightInterval counter for each flight."""
    for dt in dts:
        # Round down to nearest bucket boundary
        offset_seconds = (dt - window_start).total_seconds()
        if offset_seconds < 0:
            continue
        bucket_index = int(offset_seconds // (bucket_minutes * 60))
        bucket_start = window_start + timedelta(minutes=bucket_minutes * bucket_index)
        if bucket_start not in buckets:
            buckets[bucket_start] = FlightInterval(bucket_start=bucket_start)
        if is_arrival:
            buckets[bucket_start].arrivals += 1
        else:
            buckets[bucket_start].departures += 1


# ---------------------------------------------------------------------------
# Public fetcher
# ---------------------------------------------------------------------------

def fetch_flight_intervals(
    start_dt: datetime,
    end_dt: datetime,
    bucket_minutes: int = 30,
) -> list[FlightInterval]:
    """Fetch Heathrow flight data and bucket into 30-minute intervals.

    Splits the 24-hour window into two 12-hour API calls to stay within
    AeroDataBox's per-request limit.

    Args:
        start_dt:       Market window start (Saturday 12:00 London or UTC).
        end_dt:         Market window end   (Sunday   12:00 London or UTC).
        bucket_minutes: Width of each time bucket in minutes (30 for Markets
                        5 and 6 as specified in the rules).

    Returns:
        List of FlightInterval sorted ascending by bucket_start, covering
        the full [start_dt, end_dt) window (even buckets with 0 flights).

    Raises:
        RuntimeError: If the API key is invalid or all retries fail.
    """
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=_LONDON_TZ)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=_LONDON_TZ)

    # Split 24-hour window into ≤12-hour chunks
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(hours=_MAX_WINDOW_HOURS), end_dt)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end

    # Pre-populate every bucket with zeros so totals are correct even if
    # AeroDataBox returns nothing for a quiet period
    buckets: dict[datetime, FlightInterval] = {}
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    for i in range(0, total_minutes, bucket_minutes):
        bs = start_dt + timedelta(minutes=i)
        buckets[bs] = FlightInterval(bucket_start=bs)

    all_arrivals: list[datetime] = []
    all_departures: list[datetime] = []

    for chunk_start, chunk_end in chunks:
        arrivals, departures = _fetch_chunk(chunk_start, chunk_end)
        all_arrivals.extend(arrivals)
        all_departures.extend(departures)

    _assign_to_buckets(all_arrivals,   start_dt, bucket_minutes, buckets, is_arrival=True)
    _assign_to_buckets(all_departures, start_dt, bucket_minutes, buckets, is_arrival=False)

    result = sorted(buckets.values(), key=lambda b: b.bucket_start)
    log.debug(
        "Flights: %d buckets, %d total arrivals, %d total departures",
        len(result),
        sum(b.arrivals for b in result),
        sum(b.departures for b in result),
    )
    return result


# ---------------------------------------------------------------------------
# Market 5 — LHR_COUNT
# ---------------------------------------------------------------------------

def compute_lhr_count(intervals: list[FlightInterval]) -> int:
    """Market 5: total arrivals + departures across the full 24-hour window."""
    return sum(b.total for b in intervals)


# ---------------------------------------------------------------------------
# Market 6 — LHR_INDEX
# ---------------------------------------------------------------------------

def compute_lhr_index(intervals: list[FlightInterval]) -> float:
    """Market 6: ABS(sum of per-interval directional metrics).

    Per interval: 100 × (arrivals − departures) / max(arrivals + departures, 1)
    Settlement  = ABS(sum of all interval metrics).
    """
    total = sum(b.metric for b in intervals)
    return abs(total)


def project_lhr_values(
    intervals: list[FlightInterval],
    elapsed_fraction: float,
) -> tuple[float, float]:
    """Linear projection of final LHR_COUNT and LHR_INDEX from partial data.

    Args:
        intervals:        FlightIntervals observed so far.
        elapsed_fraction: How far through the 24-hour window we are (0–1).

    Returns:
        (projected_lhr_count, projected_lhr_index)
    """
    if elapsed_fraction <= 0:
        return 0.0, 0.0
    frac = min(elapsed_fraction, 1.0)
    count = compute_lhr_count(intervals)
    index = compute_lhr_index(intervals)
    return count / frac, index / frac
