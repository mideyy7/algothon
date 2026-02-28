"""Predictive quant models for markets where settlement is in the future.

Key challenge: TIDE_SPOT, LON_ETF, and LON_FLY are None early in the trading
window because they depend on the tidal reading at Sunday 12:00.  This module
uses a harmonic (sinusoidal) model fitted to the available Thames readings to
predict the Sunday-noon tidal level so we can trade those markets from open.

Usage:
    from quant_models import fill_missing_estimates
    fv = fill_missing_estimates(snapshot)   # dict[str, Optional[float]]
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from data_pipeline.pipeline import PipelineSnapshot
from data_pipeline.thames import ThamesReading
from data_pipeline.settlement import compute_lon_etf, compute_lon_fly

log = logging.getLogger(__name__)

# Thames is dominated by the M2 semi-diurnal tidal constituent: T ≈ 12.42 h
_M2_PERIOD_SECONDS: float = 12.42 * 3600.0

# Require at least this many readings before we trust the harmonic fit
_MIN_READINGS_FOR_FIT: int = 4


# ---------------------------------------------------------------------------
# Harmonic fitting
# ---------------------------------------------------------------------------

def _fit_harmonic(
    t_seconds: np.ndarray,
    y: np.ndarray,
) -> Optional[np.ndarray]:
    """Fit  y = A·cos(2π·t/T) + B·sin(2π·t/T) + C  via ordinary least squares.

    Returns coefficient vector [A, B, C], or None if underdetermined.
    """
    if len(t_seconds) < 3:
        return None
    A_mat = np.column_stack([
        np.cos(2 * np.pi * t_seconds / _M2_PERIOD_SECONDS),
        np.sin(2 * np.pi * t_seconds / _M2_PERIOD_SECONDS),
        np.ones(len(t_seconds)),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
    return coeffs


# ---------------------------------------------------------------------------
# Public predictor
# ---------------------------------------------------------------------------

def predict_tide_spot(
    readings: list[ThamesReading],
    window_end_dt: datetime,
) -> Optional[float]:
    """Predict TIDE_SPOT at window_end_dt using a harmonic tide model.

    Fits a sinusoidal model (dominant M2 constituent, T ≈ 12.42 h) to the
    available Thames readings and extrapolates to Sunday 12:00 noon.

    Args:
        readings:      Thames readings so far, sorted ascending by datetime.
        window_end_dt: Settlement time (Sunday 12:00 London).

    Returns:
        Predicted  ABS(level_mAOD) × 1000,  or None if too few readings.
    """
    if len(readings) < _MIN_READINGS_FOR_FIT:
        log.debug(
            "TidePredictor: only %d readings (need %d) — skipping prediction",
            len(readings), _MIN_READINGS_FOR_FIT,
        )
        return None

    t0 = readings[0].dt
    t_sec = np.array([(r.dt - t0).total_seconds() for r in readings])
    y     = np.array([r.value_maod for r in readings])

    coeffs = _fit_harmonic(t_sec, y)
    if coeffs is None:
        return None

    t_target = (window_end_dt - t0).total_seconds()
    x_target = np.array([
        np.cos(2 * np.pi * t_target / _M2_PERIOD_SECONDS),
        np.sin(2 * np.pi * t_target / _M2_PERIOD_SECONDS),
        1.0,
    ])
    predicted_level = float(x_target @ coeffs)
    predicted_spot  = abs(predicted_level) * 1000.0

    log.debug(
        "TidePredictor: predicted level=%.3f mAOD → TIDE_SPOT=%.1f (target %s)",
        predicted_level, predicted_spot, window_end_dt,
    )
    return predicted_spot


def prediction_confidence(elapsed_fraction: float) -> float:
    """Return a model confidence score (0–1) for the harmonic prediction.

    Starts at 0.30 as soon as we have 4+ readings, then rises smoothly to
    0.95 as we approach settlement (more data constrains the fit).

    Args:
        elapsed_fraction: 0.0 (window open) → 1.0 (window close).
    """
    return float(np.clip(0.30 + 0.65 * elapsed_fraction, 0.0, 0.95))


# ---------------------------------------------------------------------------
# Main entry point used by the bot
# ---------------------------------------------------------------------------

def fill_missing_estimates(
    snap: PipelineSnapshot,
) -> dict[str, Optional[float]]:
    """Return a fair-value dict for all 8 markets, filling Nones with models.

    Algorithm:
      1. Start from the pipeline's formal settlement estimates (most accurate).
      2. If TIDE_SPOT is still None (early in window), override with the
         harmonic prediction from available Thames readings.
      3. If LON_ETF or LON_FLY are still None after step 2, re-derive them
         from the now-filled components.
      4. For markets that cannot be predicted (e.g., no flights data),
         leave them as None — the bot will simply skip quoting them.

    Args:
        snap: PipelineSnapshot from DataPipeline.fetch_snapshot().

    Returns:
        Dict keyed by CMI product symbol → estimated fair value.
        Values may still be None if data is truly unavailable.
    """
    fv: dict[str, Optional[float]] = snap.estimates.as_dict()

    # -- Step 1: fill TIDE_SPOT via harmonic model if still None -------------
    if fv["TIDE_SPOT"] is None and snap.thames_readings:
        predicted_spot = predict_tide_spot(snap.thames_readings, snap.window_end)
        if predicted_spot is not None:
            fv["TIDE_SPOT"] = predicted_spot
            log.info(
                "Fair value override: TIDE_SPOT=%.1f (harmonic model, %d readings)",
                predicted_spot, len(snap.thames_readings),
            )

    # -- Step 2: re-derive LON_ETF and LON_FLY if now possible ---------------
    if fv["LON_ETF"] is None:
        new_etf = compute_lon_etf(fv["TIDE_SPOT"], fv["WX_SPOT"], fv["LHR_COUNT"])
        if new_etf is not None:
            fv["LON_ETF"] = new_etf
            fv["LON_FLY"] = compute_lon_fly(new_etf)
            log.info(
                "Fair value override: LON_ETF=%.1f, LON_FLY=%.1f (derived)",
                new_etf, fv["LON_FLY"],
            )

    return fv
