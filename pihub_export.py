import os
import sys
import pandas as pd
import requests
from datetime import datetime, timedelta

# =====================================================================
# Heathrow PIHub Flight Data Exporter
# =====================================================================
# The Quant Researcher wants two tables for the 24h period:
# 1. Market 5 (LHR_COUNT): Date, Arrival, Departure
# 2. Market 6 (LHR_INDEX): Date, Arrival, Departure, Metric
#
# NOTE: The PIHub API requires authorization. You MUST feed the prompt 
# below to an LLM to get the exact JSON parsing loop and Auth headers 
# for your specific credentials.
# =====================================================================

def fetch_pihub_flights(target_date_str: str) -> tuple[list[dict], list[dict]]:
    """
    Fetches raw flight arrays from the API for a given date (YYYY-MM-DD).
    Returns (arrivals_list, departures_list).
    """
    # ⚠️ TODO: Paste the LLM's API implementation here!
    # Example structure:
    # headers = {"Ocp-Apim-Subscription-Key": "YOUR_KEY"}
    # arr = requests.get(f"https://api-dp-prod.dp.heathrow.com/pihub/flights/arrivals?date={target_date_str}", headers=headers).json()
    # dep = requests.get(f"https://api-dp-prod.dp.heathrow.com/pihub/flights/departures?date={target_date_str}", headers=headers).json()
    # return arr, dep
    
    print("WARNING: PIHub Fetcher is not implemented. Please use the LLM prompt to generate the API logic.")
    return [], []

def parse_flight_time(flight_obj: dict) -> datetime:
    """
    Extracts the datetime of the flight from the JSON object.
    """
    # ⚠️ TODO: Paste the LLM's JSON parsing logic here!
    # return datetime.fromisoformat(flight_obj['actualTime'])
    return datetime.now()


def generate_tables(target_date: datetime):
    date_str = target_date.strftime("%Y-%m-%d")
    raw_arrivals, raw_departures = fetch_pihub_flights(date_str)
    
    # Process Timestamps
    arr_times = [parse_flight_time(f) for f in raw_arrivals]
    dep_times = [parse_flight_time(f) for f in raw_departures]
    
    # ---------------------------------------------------------
    # Table 1: Market 5 (LHR_COUNT) -> Aggregate 24h Totals
    # ---------------------------------------------------------
    total_arr = len(arr_times)
    total_dep = len(dep_times)
    
    df_m5 = pd.DataFrame([{
        "Date": date_str,
        "Arrival": total_arr,
        "Departure": total_dep,
        "LHR_COUNT_Settlement": total_arr + total_dep
    }])
    
    print("\n" + "="*50)
    print("Market 5: LHR_COUNT (24h Aggregation)")
    print("="*50)
    print(df_m5.to_string(index=False))
    df_m5.to_csv(f"M5_LHR_COUNT_{date_str}.csv", index=False)
    
    # ---------------------------------------------------------
    # Table 2: Market 6 (LHR_INDEX) -> 30-min Interval Buckets
    # ---------------------------------------------------------
    # Setup 30 min buckets
    buckets = []
    cursor = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_cursor = cursor + timedelta(days=1)
    
    while cursor < end_cursor:
        buckets.append({"Time": cursor, "Arrivals": 0, "Departures": 0})
        cursor += timedelta(minutes=30)
        
    df_buckets = pd.DataFrame(buckets)
    df_buckets.set_index("Time", inplace=True)
    
    # Bin flights
    for t in arr_times:
        bucket = t.replace(minute=(t.minute // 30) * 30, second=0, microsecond=0)
        if bucket in df_buckets.index:
            df_buckets.loc[bucket, "Arrivals"] += 1
            
    for t in dep_times:
        bucket = t.replace(minute=(t.minute // 30) * 30, second=0, microsecond=0)
        if bucket in df_buckets.index:
            df_buckets.loc[bucket, "Departures"] += 1

    # Apply Metric Formula
    def calc_metric(row):
        denom = max(row["Arrivals"] + row["Departures"], 1)
        return 100 * ((row["Arrivals"] - row["Departures"]) / denom)

    df_buckets["Metric"] = df_buckets.apply(calc_metric, axis=1)
    
    # Format the requested output table
    df_m6 = df_buckets.reset_index().rename(columns={"Time": "Date", "Arrivals": "Arrival", "Departures": "Departure"})
    
    print("\n" + "="*50)
    print("Market 6: LHR_INDEX (Interval Progression)")
    print("="*50)
    print(df_m6.to_string(index=False))
    
    total_metric = abs(df_m6["Metric"].sum())
    print("\n-> Final LHR_INDEX Settlement (Absolute Sum):", round(total_metric, 4))
    
    df_m6.to_csv(f"M6_LHR_INDEX_{date_str}.csv", index=False)
    
    # ---------------------------------------------------------
    # Expected Value Projections
    # ---------------------------------------------------------
    now = datetime.now()
    if target_date.date() == now.date():
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed_seconds = (now - start_of_day).total_seconds()
        elapsed_fraction = elapsed_seconds / 86400.0
        
        if elapsed_fraction > 0 and elapsed_fraction <= 1.0:
            expected_m5 = (total_arr + total_dep) / elapsed_fraction
            expected_m6 = total_metric / elapsed_fraction
            
            print("\n" + "="*50)
            print(f"EXPECTED SETTLEMENT PROJECTIONS (Elapsed Day: {elapsed_fraction:.1%})")
            print("="*50)
            print(f"Market 5 (LHR_COUNT) Expected: {expected_m5:.1f}")
            print(f"Market 6 (LHR_INDEX) Expected: {expected_m6:.4f}")
            
            # Add to M5 dataframe output for completeness
            df_m5["Expected_Settlement"] = expected_m5
            df_m5.to_csv(f"M5_LHR_COUNT_{date_str}.csv", index=False)


if __name__ == "__main__":
    # Target today
    target = datetime.now()
    generate_tables(target)
