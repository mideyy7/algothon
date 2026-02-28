import sys
from data_pipeline import DataPipeline, get_market_window
from quant_models import fill_missing_estimates

start, end = get_market_window()
pipeline = DataPipeline(start, end)
snap = pipeline.fetch_snapshot()
fv = fill_missing_estimates(snap)
print("Fair values after quant models:", fv)
