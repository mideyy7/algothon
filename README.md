# IMC Algothon 2026

**Team:** AAKK


## Overview

This repository is our submission for the [Imperial College AlgoSoc](https://algosoc.org) IMC Trading Hackathon (IMCity Challenge). The competition simulates a live exchange (CMI) where bots trade 8 London-data-derived financial products over a 24-hour market window (Saturday 12:00 → Sunday 12:00).


## The 8 Markets

| # | Product | Settlement |
|---|---------|------------|
| 1 | `TIDE_SPOT` | ABS(Thames tidal height MAOD) × 1000 at Sunday 12:00 |
| 2 | `TIDE_SWING` | Sum of strangle(0.2–0.25 strikes) on 15-min diffs × 100 over 24h |
| 3 | `WX_SPOT` | round(Temp_F) × Humidity at Sunday 12:00 |
| 4 | `WX_SUM` | Sum(Temp_F × Humidity / 100) over all 15-min intervals |
| 5 | `LHR_COUNT` | Total arrivals + departures at Heathrow over 24h |
| 6 | `LHR_INDEX` | ABS(Sum over 30-min: 100×(arrivals−dep)/max(arr+dep, 1)) |
| 7 | `LON_ETF` | ABS(M1 + M3 + M5) |
| 8 | `LON_FLY` | +2×P6200 +1×C6200 −2×C6600 +3×C7000 on ETF settlement |

---

## Algorithm Architecture (as described in `jump.pdf`)

The full algorithm design document is [`jump.pdf`](jump.pdf). Below is a summary of each strategy.

### A. Rolling Ridge Regression & Dynamic Delta-Hedging (Markets 1, 2, 7, 8)

**Classes:** `RollingRidge3`, `RollingRidge4`, `M2vsM1StatArb`, `ETFPackStatArb` — all in [`stat_arb.py`](stat_arb.py)

Instead of fixed ratios, we run continuous rolling ridge regressions over a 650-tick sliding window on the live orderbook:

- **M2 vs M1 (Tide Swing vs Spot):** Fits `M2 ~ b0 + b1(M1) + b2(RV)`. The partial derivative `β₁` gives the exact integer delta-hedge ratio — how many TIDE_SPOT units to buy/sell per TIDE_SWING position.
- **M8 vs M7 (Option Pack vs ETF):** Uses a 4-factor polynomial regression `PACK ~ b0 + b1(ETF) + b2(ETF²) + b3(RV)`, recalculating Greeks live and hedging the ETF leg accordingly.

Ridge penalty (`λ = 5×10⁻²`) prevents singular matrix inversions during thin-liquidity periods.

Entry signal: Z-score of the regression residual. Asymmetric execution — enter when |Z| > threshold, exit when residual mean-reverts.

### B. Rolling OLS Co-Integration Pairs Arb (Markets 5, 6)

**Classes:** `RollingOLS2`, `PairTradingArb` — in [`stat_arb.py`](stat_arb.py)

LHR_COUNT and LHR_INDEX are both derived from the same Heathrow PIHub dataset, making them structurally co-integrated. We model `M6 = α + β(M5)` using closed-form rolling OLS:

```
β = Cov(M5, M6) / Var(M5)
α = mean(M6) − β·mean(M5)
```

The spread is normalised to a Z-score over a 350-tick window. Positions are entered when `|Z| > 2.2` and sized relative to the live β constraint.

### C. EWMA Volatility-Adjusted Dip Buying (Markets 1, 3, 4)

**Classes:** `EWMA1`, `RollingStd`, `UpDipBuyer` — in [`stat_arb.py`](stat_arb.py)

For assets with hard physical boundaries or structural upward drift (e.g. TIDE_SPOT approaching high tide), symmetric mean-reversion is unsafe. Instead we run an asymmetric dip-buyer:

- Tracks an Exponentially Weighted Moving Average (α = 0.06)
- Measures the Z-score of deviation from the EWMA
- **Only buys** extreme downside dislocations (Z < −1.6), never shorts tops
- Rides the reversion back to EWMA baseline

### D. Multi-Leg Bottleneck Identity Arbitrage (Markets 1, 3, 5, 7)

**Classes:** `BasketLeg`, `ETFBasketArb` — in [`stat_arb.py`](stat_arb.py)

LON_ETF settles to `ABS(M1 + M3 + M5)`, so the ETF and its three component legs are linked by a hard identity. When the ETF trades away from the basket fair value, we trade the full package simultaneously:

```
qty = min(etf_room, leg1_room, leg2_room, leg3_room)   # bottleneck
```

The bottleneck constraint prevents partial ("legged-out") execution that would leave a naked unhedged position on any single leg.

### E. Harmonic Predictive Models (Markets 1, 3, 7, 8)

**Module:** [`quant_models.py`](quant_models.py)

Early in the 24-hour window, settlement values for spot markets are uncertain. We fit a sinusoidal (harmonic) model to available Thames tidal readings to predict the Sunday-noon level, providing fair-value estimates for TIDE_SPOT, LON_ETF, and LON_FLY from market open.

---

## Engineering Infrastructure

### Event-Driven Dispatcher ([`main.py`](main.py))

A tick-level execution loop routes live BBO (Best Bid/Offer) orderbook events to isolated strategy instances. Each strategy instance is fully independent — `LHR_COUNT` runs its Identity Basket Arb without interfering with `LHR_INDEX`'s Pairs Arb module.

### Risk Management (`BotExchangeAdapter` in [`main.py`](main.py))

- **Sub-position clipping:** Each strategy limits its injection per tick (e.g. `clip=12`), distributing liquidity rather than crossing the full book
- **Inventory caps:** Hard `max_pos` limits per product enforcing the ±100 exchange constraint
- **Cooldown throttles:** Prevents runaway execution loops on fast-moving markets
- **MM suppression:** Market-making is disabled for products assigned to directional strategies to prevent self-churn

### Data Pipelines ([`data_pipeline/`](data_pipeline/))

Three parallel live API integrations:

| Data | Source | Module |
|------|--------|--------|
| Thames tidal height (15-min) | UK Environment Agency | [`thames.py`](data_pipeline/thames.py) |
| Temperature & Humidity (15-min) | Open-Meteo (51.5074°N, −0.1278°E) | [`weather.py`](data_pipeline/weather.py) |
| Heathrow arrivals/departures (30-min) | Heathrow PIHub API | [`flights.py`](data_pipeline/flights.py) |

---

## Repository Structure

```
.
├── main.py                    # Bot dispatcher & risk adapter
├── stat_arb.py                # All strategy classes
├── quant_models.py            # Harmonic predictive models
├── data_pipeline/
│   ├── pipeline.py            # DataPipeline orchestrator
│   ├── thames.py              # Thames EA API + M1/M2 calc
│   ├── weather.py             # Open-Meteo API + M3/M4 calc
│   ├── flights.py             # PIHub API + M5/M6 calc
│   ├── settlement.py          # M7/M8 derived calculations
│   └── config.py              # API keys (from .env)
├── imc_template/
│   └── bot_template.py        # BaseBot class with CMI SSE stream
├── MAN/                       # Additional materials
├── submission_graphs/         # Strategy visualisations
├── jump.pdf                   # Full algorithm design document
└── requirements.txt
```

---

## Setup

```bash
# Requires Python >= 3.11
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API key for flight data
echo "PIHUB_API_KEY=your_key_here" > .env
