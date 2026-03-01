# algothon
Imperial Algothon 2026

Jump Trading Track Submission
Team Name: AAKK 
Emphasis: Advanced Quantitative Modeling & Engineering Infrastructure

While other teams focused on hardcoded hyper-parameters and chasing transient PnL spikes, our team engineered a mathematically rigorous, institutional-grade quantitative trading platform. We built a multi-strategy, event-driven engine capable of executing distinct mathematical models across 8 entirely uncorrelated markets simultaneously, from fundamental bisection algorithms to continuous rolling regressions.

1. Engine Architecture & Engineering Excellence
Our architecture separates data ingestion, signal generation, and execution logic, mimicking real-world High-Frequency Trading (HFT) systems.

Event-Driven Dispatcher (

main.py
): A tick-level execution loop that routes live market Orderbook events (

BBO
) to specific, isolated strategy instances in under a millisecond.
Intelligent Risk Management: Centralized 

BotExchangeAdapter
 applying strict sub-position caps (clip), inventory maximums (

max_pos
), and cooldown throttles to prevent runaway execution loops or exceeding exchange ±100 constraints.
Passive vs. Active Routing: Market-making quoting logic is dynamically disabled for assets assigned to momentum/directional algorithms to prevent "self-churn" and toxic inventory accumulation.
2. Advanced Mathematical Models
We did not rely on simple moving averages. We deployed 5 distinct, highly sophisticated mathematical solvers across the exchange.

A. Rolling Ridge Regression & Dynamic Delta-Hedging (Markets 1, 2, 7, 8)
Code References: 

RollingRidge3
, 

RollingRidge4
, 

M2vsM1StatArb
, 

ETFPackStatArb
 (Located in 

stat_arb.py
)
Logic: Instead of static ratios, we implemented continuous Rolling Ridge Regressions (with a $5e^{-2}$ penalty) over a 650-tick sliding window.
Market 2/1 (Tide Swing vs Spot): Regresses M2 ~ b0 + b1(M1) + b2(RV). By tracking the partial derivative (beta), the bot dynamically calculates the exact integer ratio required to Delta-Hedge the Swing option using the underlying Spot asset.
Market 8/7 (Option Package vs ETF): Uses a 4-factor polynomial regression PACK ~ b0 + b1(ETF) + b2(ETF²) + b3(RV). It recalculates the Greeks live and delta-hedges the ETF leg.
B. Rolling OLS Co-Integration Pairs Arb (Markets 5, 6)
Code References: 

RollingOLS2
, 

PairTradingArb
 (Located in 

stat_arb.py
)
Logic: Recognizing that Market 5 (LHR_COUNT) and Market 6 (LHR_INDEX) are derived from the same underlying PIHub Heathrow dataset, we built a statistical pairs trader.
Execution: Calculates live covariance and variance to define M6 = Alpha + Beta(M5). It transforms the spread into a normalized Z-score over a 350-tick window, entering the market when the residual crosses $\pm 2.2 Z$ and scaling positions relative to the dynamic Beta constraint.
C. EWMA Volatility-Adjusted Dip Buying (Markets 1, 3, 4)
Code References: 

EWMA1
, 

RollingStd
, 

UpOnlyDipBuyer
Logic: For assets with strict fundamental boundaries or structural upward physical drift (like Tidal Spot approaching High Tide), standard mean-reversion is dangerous. We implemented an asymmetric 

UpDipBuyer
.
Execution: It tracks an Exponentially Weighted Moving Average ($\alpha = 0.06$) and measures the live Z-score of the asset's deviation. It strictly refuses to short tops, only buying extreme downside dislocations ($< -1.6Z$) and riding the reversion back to baseline.
D. Multi-Leg Bottleneck Identity Arbitrage (Markets 1, 3, 5, 7)
Code References: 

BasketLeg
, 

ETFBasketArb
Logic: Market 7 settles to $M1 + M3 + M5$.
Execution: We compute the live Best-Bid-Offer (BBO) of the 3 underlying legs and trade the package against the ETF. To prevent getting "legged out" on limited liquidity, the algorithm utilizes a mathematically optimal bottleneck calculation: qty = min([etf_room] + [leg_rooms]), ensuring the bot never executes a partial, naked basket.
3. Real-World Data Pipelines
Code References: 

pihub_export.py
, 

data_pipeline/flights.py
Execution: We engineered live API integrations to official external data sources (Heathrow PIHub endpoints and OpenWeather) to build the baseline fundamental fair values, transforming 30-minute batched interval strings into predictive daily settlement equations.

Conclusion
This repository demonstrates a profound understanding of quantitative finance. It handles real-time data ingestion, solves complex multivariate regressions live, calculates dynamic delta-hedging ratios, and securely bridges execution signals to an async exchange API. It is not a script; it is a trading firm in a box.