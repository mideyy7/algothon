import time
import os
from datetime import datetime
from data_pipeline import DataPipeline, get_market_window

def run_live_test():
    start, end = get_market_window()
    pipeline = DataPipeline(start, end)
    snap = pipeline.fetch_snapshot()
    qr_data = snap.estimates.as_dict(include_raw=True)
    
    for key, value in qr_data.items():
        if value is None:
            formatted_val = "None (Waiting/Lag)"
        else:
            formatted_val = f"{value}"
            
        # Highlight the raw real-time values for the Quants
        if key.startswith("RAW_"):
            print(f"  * {key:<14} : {formatted_val}")
        else:
            print(f"    {key:<14} : {formatted_val}")
            
    if snap.errors:
        print("\nERRORS Encountered:")
        for source, error in snap.errors.items():
            print(f"  {source}: {error}")
            
    print("\n" + "=" * 50)
    print("PURE RAW API ARRAYS REQUESTED BY RESEARCHERS:")
    raw_arrays = snap.raw_api_data
    
    print(f"\n[THAMES] Sample of raw tide array ({len(raw_arrays['THAMES_WATER_LEVELS'])} items):")
    for item in raw_arrays['THAMES_WATER_LEVELS'][:3]:  # Print first 3
        print(f"    {item}")
    
    print(f"\n[WEATHER] Sample of raw weather array ({len(raw_arrays['WEATHER_FORECASTS'])} items):")
    for item in raw_arrays['WEATHER_FORECASTS'][:3]:
        print(f"    {item}")
        
    print(f"\n[FLIGHTS] Sample of raw flight buckets ({len(raw_arrays['FLIGHT_BUCKETS'])} items):")
    if raw_arrays['FLIGHT_BUCKETS']:
        for item in raw_arrays['FLIGHT_BUCKETS'][:3]:
            print(f"    {item}")
    else:
        print("    No flight data fetched (Check Rate limit or wait for chunks to finish).")


if __name__ == "__main__":
    run_live_test()
