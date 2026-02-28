"""Derived market settlement calculations — Markets 7 (LON_ETF) and 8 (LON_FLY).

Both markets are derived from the earlier markets rather than raw external data:

Market 7 — LON_ETF (spot)
  Settlement = ABS(Market1 + Market3 + Market5)
             = ABS(TIDE_SPOT + WX_SPOT + LHR_COUNT)
  Note: ABS is applied because each component can theoretically be negative,
  though in practice they are all positive (TIDE_SPOT uses ABS internally,
  WX_SPOT is T×H which should be positive, LHR_COUNT is a plain count).

Market 8 — LON_FLY (options structure on the ETF)
  An exotic option package written on the ETF settlement value S:
      +2 × Put  (strike 6200) → 2 × max(0, 6200 − S)
      +1 × Call (strike 6200) → 1 × max(0, S − 6200)
      −2 × Call (strike 6600) → −2 × max(0, S − 6600)
      +3 × Call (strike 7000) → 3 × max(0, S − 7000)

  Payoff shape:
    S < 6200  : payoff grows linearly downward (net delta = +1 from 2 puts − 1 call)
    6200≤S<6600: payoff = (S − 6200) from the long call → grows slowly
    6600≤S<7000: −2×(S−6600) overwhelms the long call → dips back down
    S ≥ 7000  : +3×(S−7000) − 2×(S−6600) + (S−6200) → grows steeply again
  So the structure profits strongly from very low or very high ETF values.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MarketEstimates:
    """Current best estimates for all 8 market settlement values.

    Fields are None if the underlying data has not been fetched yet
    (e.g., flights are None until RAPIDAPI_KEY is configured).

    Attributes:
        m1_tide_spot:  Market 1 — TIDE_SPOT
        m2_tide_swing: Market 2 — TIDE_SWING (running or projected)
        m3_wx_spot:    Market 3 — WX_SPOT (forecast-aware)
        m4_wx_sum:     Market 4 — WX_SUM  (forecast-aware)
        m5_lhr_count:  Market 5 — LHR_COUNT
        m6_lhr_index:  Market 6 — LHR_INDEX
        m7_lon_etf:    Market 7 — LON_ETF (derived)
        m8_lon_fly:    Market 8 — LON_FLY (derived)
        window_elapsed_fraction: 0.0 → 1.0, how far through the 24-h window.
        current_raw_data: A dict containing the latest raw values even if the 
                          true settlement is None (useful for Quants predicting models)
    """
    m1_tide_spot:  Optional[float] = None
    m2_tide_swing: Optional[float] = None
    m3_wx_spot:    Optional[float] = None
    m4_wx_sum:     Optional[float] = None
    m5_lhr_count:  Optional[float] = None
    m6_lhr_index:  Optional[float] = None
    m7_lon_etf:    Optional[float] = None
    m8_lon_fly:    Optional[float] = None
    window_elapsed_fraction: float = 0.0
    current_raw_data: dict[str, Optional[float]] = None  # type: ignore

    def as_dict(self, include_raw: bool = False) -> dict[str, Optional[float]]:
        """Return estimates keyed by CMI product name."""
        base = {
            "TIDE_SPOT":  self.m1_tide_spot,
            "TIDE_SWING": self.m2_tide_swing,
            "WX_SPOT":    self.m3_wx_spot,
            "WX_SUM":     self.m4_wx_sum,
            "LHR_COUNT":  self.m5_lhr_count,
            "LHR_INDEX":  self.m6_lhr_index,
            "LON_ETF":    self.m7_lon_etf,
            "LON_FLY":    self.m8_lon_fly,
        }
        if include_raw and self.current_raw_data:
            base.update({f"RAW_{k}": v for k, v in self.current_raw_data.items()})
        return base


# ---------------------------------------------------------------------------
# Market 7 — LON_ETF
# ---------------------------------------------------------------------------

def compute_lon_etf(
    m1_tide_spot: Optional[float],
    m3_wx_spot: Optional[float],
    m5_lhr_count: Optional[float],
) -> Optional[float]:
    """Market 7: ABS(TIDE_SPOT + WX_SPOT + LHR_COUNT).

    Returns None if any component is unavailable (e.g., flights not fetched).

    Args:
        m1_tide_spot:  Settlement estimate for Market 1.
        m3_wx_spot:    Settlement estimate for Market 3.
        m5_lhr_count:  Settlement estimate for Market 5.
    """
    if m1_tide_spot is None or m3_wx_spot is None or m5_lhr_count is None:
        return None
    return abs(m1_tide_spot + m3_wx_spot + m5_lhr_count)


# ---------------------------------------------------------------------------
# Market 8 — LON_FLY (option package)
# ---------------------------------------------------------------------------

def _call_payoff(strike: float, s: float) -> float:
    """Payoff of a long call: max(0, S − K)."""
    return max(0.0, s - strike)


def _put_payoff(strike: float, s: float) -> float:
    """Payoff of a long put: max(0, K − S)."""
    return max(0.0, strike - s)


def compute_lon_fly(etf_value: Optional[float]) -> Optional[float]:
    """Market 8: option package payoff on the ETF settlement value.

    Structure:
        +2 × Put(K=6200) + 1 × Call(K=6200) − 2 × Call(K=6600) + 3 × Call(K=7000)

    Args:
        etf_value: Estimated or actual LON_ETF settlement (Market 7).

    Returns:
        Option package payoff, or None if etf_value is None.

    Example (from the PDF):
        ETF = 6300:
          +2 × max(0, 6200−6300) = 0
          +1 × max(0, 6300−6200) = 100
          −2 × max(0, 6300−6600) = 0
          +3 × max(0, 6300−7000) = 0
          Total = 100
    """
    if etf_value is None:
        return None

    s = etf_value
    payoff = (
        2.0 * _put_payoff(6200, s)
        + 1.0 * _call_payoff(6200, s)
        - 2.0 * _call_payoff(6600, s)
        + 3.0 * _call_payoff(7000, s)
    )
    return payoff


def lon_fly_payoff_table(etf_values: list[float]) -> list[tuple[float, float]]:
    """Compute LON_FLY payoff for a range of ETF values.

    Useful for understanding the payoff shape and for pricing.

    Returns:
        List of (etf_value, payoff) tuples.
    """
    return [(s, compute_lon_fly(s)) for s in etf_values]  # type: ignore[arg-type]
