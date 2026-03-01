#!/usr/bin/env python3
"""Export Heathrow flight data tables for quant analysis.

Fetches a 24-hour window from the PIHub API and writes two CSV files:

    lhr_count_table.csv   Market 5  (LHR_COUNT)  — one row per 30-min bucket
    lhr_index_table.csv   Market 6  (LHR_INDEX)  — one row per 30-min bucket

Projected Expected_Settlement values (extrapolated from the elapsed fraction
of the day) are written into the CSVs *and* printed to the terminal.

Usage:
    source .venv/bin/activate
    python export_flight_tables.py

    # Override the market window manually (useful for back-testing):
    python export_flight_tables.py --start "2026-03-01T12:00:00" --end "2026-03-02T12:00:00"

    # Save CSVs to a custom directory:
    python export_flight_tables.py --outdir /tmp/quant_output
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import zoneinfo

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed.  Run:  pip install pandas")
    sys.exit(1)

from data_pipeline import get_market_window
from data_pipeline.flights import (
    fetch_flight_intervals,
    compute_lhr_count,
    compute_lhr_index,
    project_lhr_values,
    FlightInterval,
)

_LONDON_TZ = zoneinfo.ZoneInfo("Europe/London")

# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_m5_table(
    intervals: list[FlightInterval],
    elapsed_fraction: float,
) -> tuple[pd.DataFrame, float]:
    """Build Table 1: Market 5 — LHR_COUNT.

    Columns:
        Date                  ISO-8601 string for the 30-min bucket start
        Arrival               Arriving flights in this bucket
        Departure             Departing flights in this bucket
        Expected_Settlement   Extrapolated full-day total (same scalar in every row)

    Returns:
        (DataFrame, projected_lhr_count)
    """
    proj_count, _ = project_lhr_values(intervals, elapsed_fraction)

    rows = [
        {
            "Date":                 ivl.bucket_start.isoformat(),
            "Arrival":              ivl.arrivals,
            "Departure":            ivl.departures,
            "Expected_Settlement":  round(proj_count, 1),
        }
        for ivl in intervals
    ]
    df = pd.DataFrame(rows, columns=["Date", "Arrival", "Departure", "Expected_Settlement"])
    return df, proj_count


def build_m6_table(
    intervals: list[FlightInterval],
    elapsed_fraction: float,
) -> tuple[pd.DataFrame, float]:
    """Build Table 2: Market 6 — LHR_INDEX.

    Columns:
        Date        ISO-8601 string for the 30-min bucket start
        Arrival     Arriving flights in this bucket
        Departure   Departing flights in this bucket
        Metric      100 × (arrivals − departures) / max(arrivals + departures, 1)

    Returns:
        (DataFrame, projected_lhr_index)
    """
    _, proj_index = project_lhr_values(intervals, elapsed_fraction)

    rows = [
        {
            "Date":      ivl.bucket_start.isoformat(),
            "Arrival":   ivl.arrivals,
            "Departure": ivl.departures,
            "Metric":    round(ivl.metric, 4),
        }
        for ivl in intervals
    ]
    df = pd.DataFrame(rows, columns=["Date", "Arrival", "Departure", "Metric"])
    return df, proj_index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export Heathrow PIHub flight tables.")
    parser.add_argument(
        "--start",
        default=None,
        help="Window start as ISO-8601 (e.g. 2026-03-01T12:00:00). "
             "Defaults to the most recent Saturday 12:00 London time.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Window end as ISO-8601 (e.g. 2026-03-02T12:00:00). "
             "Defaults to 24 h after --start.",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory to write CSV files into (default: current directory).",
    )
    args = parser.parse_args()

    # --- Resolve window ---
    if args.start:
        window_start = datetime.fromisoformat(args.start)
        if window_start.tzinfo is None:
            window_start = window_start.replace(tzinfo=_LONDON_TZ)
    else:
        window_start, _ = get_market_window()

    if args.end:
        window_end = datetime.fromisoformat(args.end)
        if window_end.tzinfo is None:
            window_end = window_end.replace(tzinfo=_LONDON_TZ)
    else:
        from datetime import timedelta
        window_end = window_start + timedelta(hours=24)

    # --- Elapsed fraction ---
    now = datetime.now(_LONDON_TZ)
    elapsed_secs = (now - window_start).total_seconds()
    total_secs   = (window_end - window_start).total_seconds()
    elapsed_fraction = max(0.0, min(elapsed_secs / total_secs, 1.0))

    # --- Banner ---
    print("=" * 60)
    print("  Heathrow PIHub — Flight Data Export")
    print("=" * 60)
    print(f"  Window start : {window_start.isoformat()}")
    print(f"  Window end   : {window_end.isoformat()}")
    print(f"  Elapsed      : {elapsed_fraction:.1%} of 24-hour window")
    print("-" * 60)
    print("  Fetching flight data from PIHub…")

    try:
        intervals = fetch_flight_intervals(start_dt=window_start, end_dt=window_end)
    except RuntimeError as exc:
        print(f"\n  ERROR: {exc}")
        print(
            "\n  Troubleshooting:\n"
            "    • Check that PIHUB_API_KEY is set in your .env file (if required).\n"
            "    • Inspect the raw API response:  curl <url>  to confirm field names.\n"
            "    • Refer to the *** PIHUB_SCHEMA *** comments in data_pipeline/flights.py."
        )
        sys.exit(1)

    if not intervals:
        print("  ERROR: No intervals returned — check the API and time-window parameters.")
        sys.exit(1)

    total_arr = sum(i.arrivals  for i in intervals)
    total_dep = sum(i.departures for i in intervals)

    print(f"  Fetched {len(intervals)} × 30-min buckets")
    print(f"  Observed so far  →  arrivals: {total_arr}, departures: {total_dep}")
    print()

    # --- Table 1: M5 / LHR_COUNT ---
    df_m5, proj_m5 = build_m5_table(intervals, elapsed_fraction)
    obs_total = total_arr + total_dep

    print("━" * 60)
    print("  TABLE 1  ·  Market 5  ·  LHR_COUNT")
    print("━" * 60)
    print(df_m5.to_string(index=False))
    print()
    print(f"  Observed total (arrivals + departures) : {obs_total}")
    print(f"  ★ Expected Settlement (extrapolated)   : {proj_m5:.1f}")
    print(f"    [= {obs_total} ÷ {elapsed_fraction:.4f} elapsed fraction]")

    # --- Table 2: M6 / LHR_INDEX ---
    running_index = compute_lhr_index(intervals)
    df_m6, proj_m6 = build_m6_table(intervals, elapsed_fraction)

    print()
    print("━" * 60)
    print("  TABLE 2  ·  Market 6  ·  LHR_INDEX")
    print("━" * 60)
    print(df_m6.to_string(index=False))
    print()
    print(f"  Running ABS(Σ metric)                  : {running_index:.4f}")
    print(f"  ★ Expected Settlement (extrapolated)   : {proj_m6:.4f}")
    print(f"    [= {running_index:.4f} ÷ {elapsed_fraction:.4f} elapsed fraction]")

    # --- Write CSVs ---
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    m5_path = outdir / "lhr_count_table.csv"
    m6_path = outdir / "lhr_index_table.csv"

    df_m5.to_csv(m5_path, index=False)
    df_m6.to_csv(m6_path, index=False)

    print()
    print("  CSV files written:")
    print(f"    → {m5_path.resolve()}")
    print(f"    → {m6_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
