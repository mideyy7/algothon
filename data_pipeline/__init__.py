"""IMCity Data Pipeline.

Fetches real-time London data from three sources and computes running
settlement price estimates for all 8 markets.

Quick start:
    from data_pipeline import DataPipeline, get_market_window
    from data_pipeline.config import MARKET_PRODUCTS

    start, end = get_market_window()
    pipeline = DataPipeline(start, end)
    snapshot = pipeline.fetch_snapshot()
    print(snapshot.estimates)
"""

from .pipeline import DataPipeline, PipelineSnapshot
from .thames import ThamesReading, fetch_thames_readings, compute_tide_spot, compute_tide_swing
from .weather import WeatherReading, fetch_weather_readings, compute_wx_spot, compute_wx_sum
from .flights import FlightInterval, fetch_flight_intervals, compute_lhr_count, compute_lhr_index
from .settlement import compute_lon_etf, compute_lon_fly, MarketEstimates
from .pipeline import get_market_window

__all__ = [
    "DataPipeline",
    "PipelineSnapshot",
    "ThamesReading",
    "WeatherReading",
    "FlightInterval",
    "MarketEstimates",
    "fetch_thames_readings",
    "fetch_weather_readings",
    "fetch_flight_intervals",
    "compute_tide_spot",
    "compute_tide_swing",
    "compute_wx_spot",
    "compute_wx_sum",
    "compute_lhr_count",
    "compute_lhr_index",
    "compute_lon_etf",
    "compute_lon_fly",
    "get_market_window",
]
