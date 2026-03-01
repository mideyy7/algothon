"""Heathrow airport flight data — Markets 5 (LHR_COUNT) and 6 (LHR_INDEX).

Data source: Heathrow PIHub (official Heathrow Airport data platform)
  Arrivals:   https://api-dp-prod.dp.heathrow.com/pihub/flights/arrivals
  Departures: https://api-dp-prod.dp.heathrow.com/pihub/flights/departures

Authentication
  If the endpoint requires an API key, set PIHUB_API_KEY in your .env file.
  Adjust _pihub_headers() below for the exact auth scheme (Bearer token,
  X-Api-Key header, or query-param token).

JSON schema
  The exact field names for the PIHub response are marked with:
      # *** PIHUB_SCHEMA: update "<field>" to the real JSON key ***
  Run a raw requests.get() against the endpoint and inspect the response to
  confirm the key names before going live.

Market 5 — LHR_COUNT
  Settlement = total arrivals + total departures at LHR over 24 hours.

Market 6 — LHR_INDEX
  For each 30-minute interval:
      metric = 100 × (arrivals − departures) / max(arrivals + departures, 1)
  Settlement = ABS(sum of all interval metrics).
  This measures the net directional imbalance of flights across the day.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional
import zoneinfo

import requests

from .config import (
    PIHUB_API_KEY,
    PIHUB_BASE_URL,
    REQUEST_TIMEOUT,
    LONDON_TIMEZONE,
    http_session,
)

log = logging.getLogger(__name__)

_LONDON_TZ = zoneinfo.ZoneInfo(LONDON_TIMEZONE)

# PIHub endpoint paths — appended to PIHUB_BASE_URL
_ARRIVALS_URL   = f"{PIHUB_BASE_URL}/arrivals"
_DEPARTURES_URL = f"{PIHUB_BASE_URL}/departures"

# In-memory cache for fully elapsed windows to avoid redundant API calls
_historical_cache: dict[tuple[datetime, datetime], tuple[list[datetime], list[datetime]]] = {}


# ---------------------------------------------------------------------------
# Data model  (unchanged — plugs directly into stat_arb.py solvers)
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
# Auth / headers
# ---------------------------------------------------------------------------

def _pihub_headers() -> dict[str, str]:
    """Build HTTP headers for PIHub requests.

    *** PIHUB_SCHEMA: adjust the auth scheme to match what PIHub actually uses ***

    Common patterns:
      Bearer token  → {"Authorization": f"Bearer {PIHUB_API_KEY}"}
      API key       → {"X-Api-Key": PIHUB_API_KEY}
      No auth       → return {}  (if the endpoint is public)
    """
    headers: dict[str, str] = {
        "Accept": "application/json",
    }
    if PIHUB_API_KEY:
        # *** PIHUB_SCHEMA: replace "Authorization" / "Bearer" with correct scheme ***
        headers["Authorization"] = f"Bearer {PIHUB_API_KEY}"
    return headers


# ---------------------------------------------------------------------------
# Timestamp parser — flexible across common aviation API schemas
# ---------------------------------------------------------------------------

def _parse_pihub_time(flight: dict) -> Optional[datetime]:
    """Extract the best available timestamp from a PIHub flight object.

    Tries fields in priority order (actual → estimated → scheduled) and
    across several common key-naming conventions.  Update the lists below
    if the real PIHub schema uses different names.

    *** PIHUB_SCHEMA: confirm field names by inspecting a live API response ***

    Expected nesting patterns tried:
      Flat:   flight["actualTime"], flight["scheduledArrivalTime"], …
      Nested: flight["timing"]["actual"], flight["times"]["scheduled"], …
    """
    # --- flat timestamp fields (try actual first, then estimated, then scheduled) ---
    flat_candidates: list[str] = [
        # *** PIHUB_SCHEMA: actual runway / gate time field ***
        "actualTime",
        "actualArrivalTime",       # arrival-specific variant
        "actualDepartureTime",     # departure-specific variant
        "actualLandingTime",
        "actualTakeoffTime",
        # *** PIHUB_SCHEMA: estimated time field ***
        "estimatedTime",
        "estimatedArrivalTime",
        "estimatedDepartureTime",
        "estimatedLandingTime",
        # *** PIHUB_SCHEMA: scheduled time field ***
        "scheduledTime",
        "scheduledArrivalTime",
        "scheduledDepartureTime",
        "scheduledLocalTime",
        # Camel-case variants sometimes used by UK gov / Heathrow APIs
        "scheduled_time",
        "actual_time",
        "estimated_time",
    ]

    for key in flat_candidates:
        raw = flight.get(key)
        if raw:
            try:
                return datetime.fromisoformat(str(raw)).astimezone(_LONDON_TZ)
            except (ValueError, TypeError):
                pass

    # --- nested timing objects ---
    # *** PIHUB_SCHEMA: adjust outer key ("timing", "times", "movement") ***
    for outer_key in ("timing", "times", "movement", "schedule"):
        block = flight.get(outer_key)
        if not isinstance(block, dict):
            continue
        for inner_key in ("actual", "estimated", "scheduled", "local"):
            raw = block.get(inner_key)
            if raw:
                try:
                    return datetime.fromisoformat(str(raw)).astimezone(_LONDON_TZ)
                except (ValueError, TypeError):
                    pass

    return None


# ---------------------------------------------------------------------------
# Core PIHub fetcher (one direction: arrivals OR departures)
# ---------------------------------------------------------------------------

def _fetch_pihub_direction(
    url: str,
    start_dt: datetime,
    end_dt: datetime,
    direction_label: str,
) -> list[datetime]:
    """Fetch all flight timestamps from one PIHub direction endpoint.

    Handles pagination automatically (stops when the API signals no more pages).

    Args:
        url:             Full URL for the arrivals or departures endpoint.
        start_dt:        Window start (timezone-aware).
        end_dt:          Window end   (timezone-aware).
        direction_label: "arrivals" or "departures" (used only for logging).

    Returns:
        List of timezone-aware datetimes for each flight in the window.

    Raises:
        RuntimeError: On HTTP 401/403 (auth failure) or persistent network errors.
    """
    # Format timestamps as UTC ISO-8601 strings for the query params.
    # *** PIHUB_SCHEMA: check whether PIHub expects UTC ("Z") or local London time ***
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    params: dict[str, object] = {
        # *** PIHUB_SCHEMA: replace "from" / "to" with actual param names ***
        # Common alternatives: "scheduledFrom"/"scheduledTo", "startDate"/"endDate",
        #   "fromDate"/"toDate", "date" (single day), "departureTime"/"arrivalTime"
        "from": start_dt.astimezone(timezone.utc).strftime(fmt),
        "to":   end_dt.astimezone(timezone.utc).strftime(fmt),
        # *** PIHUB_SCHEMA: add any mandatory params the API requires, e.g.: ***
        # "terminal": "all",
        # "type": "commercial",
    }

    all_times: list[datetime] = []
    page = 1

    while True:
        # *** PIHUB_SCHEMA: remove "page" param if the API doesn't paginate ***
        params["page"] = page

        try:
            resp = http_session.get(
                url,
                params=params,
                headers=_pihub_headers(),
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            log.error("PIHub %s fetch failed (network): %s", direction_label, exc)
            raise RuntimeError(f"PIHub {direction_label} unreachable: {exc}") from exc

        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"PIHub API authentication failed (HTTP {resp.status_code}) for "
                f"{direction_label}. Check PIHUB_API_KEY in your .env file."
            )

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            log.error("PIHub %s HTTP error: %s", direction_label, exc)
            raise RuntimeError(f"PIHub {direction_label} HTTP error: {exc}") from exc

        data = resp.json()

        # *** PIHUB_SCHEMA: adjust root key to match actual response structure ***
        # Try the most common envelope patterns in order:
        if isinstance(data, list):
            flights = data
        else:
            flights = (
                data.get("flights")
                or data.get("data")
                or data.get("items")
                or data.get("arrivals")      # arrivals endpoint self-key
                or data.get("departures")    # departures endpoint self-key
                or data.get("results")
                or []
            )

        if not flights:
            break  # empty page → stop pagination

        for flight in flights:
            dt = _parse_pihub_time(flight)
            if dt:
                all_times.append(dt)

        # --- Pagination detection ---
        # *** PIHUB_SCHEMA: adjust to match actual pagination envelope ***
        # Common patterns: "hasMore", "nextPage", "nextPageToken", "totalPages"
        if isinstance(data, dict):
            has_more = bool(
                data.get("hasMore")
                or data.get("nextPage")
                or data.get("nextPageToken")
                or (
                    isinstance(data.get("totalPages"), int)
                    and data["totalPages"] > page
                )
            )
        else:
            has_more = False  # bare list response → single page

        if not has_more:
            break

        page += 1

    log.debug(
        "PIHub %s: %d timestamps fetched (pages=%d)", direction_label, len(all_times), page
    )
    return all_times


# ---------------------------------------------------------------------------
# Combined arrivals + departures fetch with historical cache
# ---------------------------------------------------------------------------

def _fetch_window(
    start_dt: datetime,
    end_dt: datetime,
) -> tuple[list[datetime], list[datetime]]:
    """Fetch arrivals and departures for the given window from PIHub.

    Results for fully elapsed windows are cached in-process to avoid
    re-hitting the API when fetch_snapshot() is called repeatedly.

    Returns:
        (arrival_dts, departure_dts) — lists of timezone-aware datetimes.
    """
    now = datetime.now(_LONDON_TZ)
    is_historical = end_dt < now
    cache_key = (
        start_dt.astimezone(timezone.utc).replace(second=0, microsecond=0),
        end_dt.astimezone(timezone.utc).replace(second=0, microsecond=0),
    )

    if is_historical and cache_key in _historical_cache:
        log.debug("PIHub window %s→%s served from cache", start_dt, end_dt)
        return _historical_cache[cache_key]

    arrival_dts   = _fetch_pihub_direction(_ARRIVALS_URL,   start_dt, end_dt, "arrivals")
    departure_dts = _fetch_pihub_direction(_DEPARTURES_URL, start_dt, end_dt, "departures")

    if is_historical:
        _historical_cache[cache_key] = (arrival_dts, departure_dts)

    return arrival_dts, departure_dts


# ---------------------------------------------------------------------------
# Bucket assignment  (unchanged from original)
# ---------------------------------------------------------------------------

def _assign_to_buckets(
    dts: list[datetime],
    window_start: datetime,
    bucket_minutes: int,
    buckets: dict[datetime, FlightInterval],
    is_arrival: bool,
) -> None:
    """Increment the appropriate FlightInterval counter for each flight."""
    for dt in dts:
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
    """Fetch Heathrow flight data from PIHub and bucket into 30-minute intervals.

    Unlike the old AeroDataBox fetcher (which split into 12-hour chunks), PIHub
    is queried for the full window in one pass.  The historical-cache layer
    prevents redundant calls on repeated invocations.

    Args:
        start_dt:       Market window start (Saturday 12:00 London or UTC).
        end_dt:         Market window end   (Sunday   12:00 London or UTC).
        bucket_minutes: Width of each time bucket (30 for Markets 5 and 6).

    Returns:
        List of FlightInterval sorted ascending by bucket_start, covering
        the full [start_dt, end_dt) window (even buckets with 0 flights).

    Raises:
        RuntimeError: On authentication failure or persistent network errors.
    """
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=_LONDON_TZ)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=_LONDON_TZ)

    # Pre-populate every bucket with zeros so totals are correct even if
    # PIHub returns nothing for a quiet period
    buckets: dict[datetime, FlightInterval] = {}
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    for i in range(0, total_minutes, bucket_minutes):
        bs = start_dt + timedelta(minutes=i)
        buckets[bs] = FlightInterval(bucket_start=bs)

    arrival_dts, departure_dts = _fetch_window(start_dt, end_dt)

    _assign_to_buckets(arrival_dts,   start_dt, bucket_minutes, buckets, is_arrival=True)
    _assign_to_buckets(departure_dts, start_dt, bucket_minutes, buckets, is_arrival=False)

    result = sorted(buckets.values(), key=lambda b: b.bucket_start)
    log.debug(
        "PIHub: %d buckets, %d total arrivals, %d total departures",
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
    return abs(sum(b.metric for b in intervals))


# ---------------------------------------------------------------------------
# Projection helper (used by DataPipeline and export_flight_tables.py)
# ---------------------------------------------------------------------------

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
    return compute_lhr_count(intervals) / frac, compute_lhr_index(intervals) / frac
