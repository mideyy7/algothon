# IMCity Data Pipeline

Everything your trading bot needs to know about the London data layer.

---

## Quick-start (3 steps)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your RapidAPI key (only needed for flight data)
#    Edit data_pipeline/config.py → set RAPIDAPI_KEY = "..."

# 3. Use in your bot
python - <<'EOF'
from data_pipeline import DataPipeline, get_market_window
start, end = get_market_window()
pipeline   = DataPipeline(start, end)
snap       = pipeline.fetch_snapshot()
print(snap.estimates.as_dict())
EOF
```

---

## What does the pipeline do?

It fetches real London data from three external APIs in **parallel**, caches
the results to avoid hammering rate limits, and then computes your best
current estimate of where each of the 8 markets will settle.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DataPipeline.fetch_snapshot()                   │
│                                                                         │
│  Thread 1 → fetch_thames_readings()   →  M1 (TIDE_SPOT)               │
│                                          M2 (TIDE_SWING)               │
│  Thread 2 → fetch_weather_readings()  →  M3 (WX_SPOT)     ──►  M7    │
│                                          M4 (WX_SUM)        (LON_ETF) │
│  Thread 3 → fetch_flight_intervals()  →  M5 (LHR_COUNT)    ──►  M8   │
│                                          M6 (LHR_INDEX)   (LON_FLY)  │
└─────────────────────────────────────────────────────────────────────────┘
```

All three threads fire at the same time. The total latency is the slowest
single source (typically ~1 s), not the sum of all three.

---

## File-by-file explanation

### `data_pipeline/config.py`

All tuneable constants live here. **The only thing you need to change** is
`RAPIDAPI_KEY`. Everything else has sensible defaults.

| Constant | What it is |
|---|---|
| `RAPIDAPI_KEY` | Your AeroDataBox API key from RapidAPI.com |
| `THAMES_CACHE_TTL` | Seconds before Thames data is re-fetched (default 240 s) |
| `WEATHER_CACHE_TTL` | Seconds before weather data is re-fetched (default 600 s) |
| `FLIGHTS_CACHE_TTL` | Seconds before flight data is re-fetched (default 180 s) |
| `REQUEST_TIMEOUT` | HTTP timeout per request in seconds |
| `MAX_RETRIES` | How many times to retry a failed HTTP call |
| `MARKET_PRODUCTS` | Maps market numbers to CMI product names |

---

### `data_pipeline/thames.py`

**Data source:** UK Environment Agency Flood Monitoring API
**Cost:** Free, no key needed
**Update cadence:** Every 15 minutes

Fetches tidal height readings at London Bridge (station `0006`) measured in
MAOD (Metres Above Ordnance Datum). Values are typically negative at low
tide (e.g. −1.41 m) and positive at high tide.

#### `fetch_thames_readings(since, limit=200)`

Makes one GET request to the Environment Agency API and returns a list of
`ThamesReading(dt, value_maod)` sorted oldest-first. The `since` parameter
tells the API to only return readings after the start of our market window,
so we never get stale Saturday-before-last data.

**Error handling:** Retries up to `MAX_RETRIES` times with increasing
back-off. Raises `RuntimeError` if all attempts fail.

#### Market 1 — `compute_tide_spot(readings, settlement_dt)`

```
Settlement = ABS(reading closest to Sunday 12:00) × 1000
```

Finds the `ThamesReading` whose timestamp is nearest to `settlement_dt`
(Sunday noon). Returns `None` if no reading is within 20 minutes — this
prevents a stale reading from the previous day being used accidentally.

**Example (from PDF):** reading = −1.41 m → settlement = **1410**

#### Market 2 — `compute_tide_swing(readings, start_dt, end_dt)`

```
For each consecutive pair of 15-min readings:
    diff     = |reading[t] − reading[t−1]|
    strangle = max(0, 0.20 − diff) + max(0, diff − 0.25)
Settlement = sum(strangles) × 100
```

This is a **long strangle** on each 15-minute water-level move:
- If the move is **between 0.20 m and 0.25 m**, both options are out-of-the-money → payoff = 0
- If the move is **less than 0.20 m**, the put is in-the-money → payoff = (0.20 − diff)
- If the move is **greater than 0.25 m**, the call is in-the-money → payoff = (diff − 0.25)

Pairs separated by more than 20 minutes are skipped to prevent data outages
from inflating a single huge diff.

**PDF examples:**
| diff | payoff |
|------|--------|
| 0.09 | 0.11 (put ITM) |
| 0.33 | 0.08 (call ITM) |
| 0.21 | 0.00 (both OTM) |

#### `project_tide_swing(current_sum, readings_in_window)`

Linear projection: if we're 40% through the window and the running sum is X,
the projected final settlement is X ÷ 0.40.

---

### `data_pipeline/weather.py`

**Data source:** Open-Meteo API
**Cost:** Free, no key needed
**Update cadence:** ~15 minutes (model reruns)

**KEY COMPETITIVE ADVANTAGE:** Open-Meteo provides *forecast* data, not just
historical readings. By fetching `past_days=2` and `forecast_days=2` we get
the full 24-hour market window filled in with forecast values from the start
of trading. This means our WX_SPOT estimate on Saturday is already using
Open-Meteo's prediction of Sunday noon temperature — not just "whatever it
is right now."

#### `fetch_weather_readings(start_dt, end_dt)`

Calls Open-Meteo's `minutely_15` endpoint at London coordinates
(51.5074°N, −0.1278°E), requesting `temperature_2m` (in Fahrenheit) and
`relative_humidity_2m`. Returns a list of `WeatherReading(dt, temp_f, humidity)`
for the entire market window — **both past and forecast**.

#### Market 3 — `compute_wx_spot(readings, settlement_dt)`

```
Settlement = round(Temp_°F) × Humidity   at Sunday 12:00
```

Picks the reading closest in time to `settlement_dt`. Because the readings
include Open-Meteo forecast entries, this estimate is available immediately
at the start of the trading session.

**Example (from PDF):** T = 33°F, H = 29% → settlement = **957**

#### Market 4 — `compute_wx_sum(readings, start_dt, end_dt)`

```
Settlement = Σ (round(Temp_°F) × Humidity / 100)   for every 15-min interval
```

Sums over all 96 intervals in the window. Because we have forecast data for
the full window, this is effectively already a projected final value — not a
running partial sum.

---

### `data_pipeline/flights.py`

**Data source:** AeroDataBox via RapidAPI
**Cost:** Requires API key (free tier = 500 calls/month)
**Update cadence:** Near real-time

**USER ACTION:** Set `RAPIDAPI_KEY` in `config.py`.

AeroDataBox supports at most a 12-hour query window per API call. For our
24-hour market window we split into two calls (Saturday noon→midnight,
midnight→Sunday noon).

#### `fetch_flight_intervals(start_dt, end_dt, bucket_minutes=30)`

1. Splits the 24-hour window into two 12-hour chunks
2. Calls AeroDataBox for each chunk
3. For each flight, extracts the actual time (or scheduled if not yet departed)
4. Assigns each flight to a 30-minute bucket starting from `start_dt`
5. Returns a list of `FlightInterval(bucket_start, arrivals, departures)` sorted
   ascending, with **all** buckets pre-populated (zero-count buckets included)

**AeroDataBox response structure** (key paths used):
```
departures[i].departure.actualTime.local   (or scheduledTime.local)
arrivals[i].arrival.actualTime.local       (or scheduledTime.local)
```
If AeroDataBox changes their response format, these paths are clearly
labelled in `_fetch_chunk()`.

#### Market 5 — `compute_lhr_count(intervals)`

```
Settlement = total arrivals + total departures across all 30-min buckets
```

Simple count across the full day.

#### Market 6 — `compute_lhr_index(intervals)`

```
Per bucket: metric = 100 × (arrivals − departures) / max(arrivals + departures, 1)
Settlement = ABS(sum of all bucket metrics)
```

Measures how directionally imbalanced the airport is over the day. A perfectly
balanced day (same arrivals and departures) settles near zero. A day where
many more flights depart than arrive (or vice-versa) settles high.

**PDF example:** [18.18, −31.03, 15.38, …] → sum ≈ 2.53 → ABS = **2.53**

---

### `data_pipeline/settlement.py`

Pure math functions — no network calls.

#### `MarketEstimates` (dataclass)

Holds the current best estimate for all 8 markets plus `window_elapsed_fraction`
(0.0 = market just opened, 1.0 = at settlement). Call `.as_dict()` to get
a `{"TIDE_SPOT": 1410.0, ...}` dict keyed by CMI product name.

#### Market 7 — `compute_lon_etf(m1, m3, m5)`

```
Settlement = ABS(TIDE_SPOT + WX_SPOT + LHR_COUNT)
```

Simply sums the three settlement values. Returns `None` if any component is
`None` (e.g. flights not configured). ABS is applied because the specification
states the ETF cannot settle negative.

**PDF example:** water=1000, T×H=1500, flights=500 → ETF = **3000**

#### Market 8 — `compute_lon_fly(etf_value)`

```
Payoff = 2 × max(0, 6200 − S)     [long 2 puts at 6200]
       + 1 × max(0, S − 6200)     [long 1 call at 6200]
       − 2 × max(0, S − 6600)     [short 2 calls at 6600]
       + 3 × max(0, S − 7000)     [long 3 calls at 7000]
```

**Payoff shape:**

```
     payoff
       │  \                               /
       │   \                             /
       │    \        ___________        /
       │     \      /           \      /
  ─────┼──────\────/─────────────\────/───── ETF value
       │    6200  6300   6600  6700  7000
```

- ETF < 6200: payoff rises (2 puts dominate)
- 6200–6600: payoff dips slowly (1 call − 2 short calls)
- 6600–7000: payoff dips further (short calls bite)
- ETF > 7000: payoff rises steeply (3 long calls dominate)

**PDF example:** ETF = 6300 → settlement = **100**

---

### `data_pipeline/pipeline.py`

The main entry point for your bot.

#### `get_market_window() → (start, end)`

Returns timezone-aware London datetimes for the current or most recent
Saturday-noon → Sunday-noon window. Call this once at startup.

#### `DataPipeline(window_start, window_end, skip_flights=False)`

Create one instance per trading session. If `RAPIDAPI_KEY` is still the
placeholder string, `skip_flights` is automatically set to `True` and a
warning is logged — M5/M6/M7/M8 will be `None` until the key is added.

#### `DataPipeline.fetch_snapshot(use_cache=True) → PipelineSnapshot`

The main method your bot should call. Each call:
1. Checks the TTL cache for each of the three sources
2. Fires parallel HTTP fetches **only** for sources whose cache has expired
3. Assembles and returns a `PipelineSnapshot`

```python
snap = pipeline.fetch_snapshot()
snap.estimates.m1_tide_spot   # → 1410.0 (or None if not yet available)
snap.estimates.as_dict()      # → {"TIDE_SPOT": 1410.0, "WX_SPOT": 957.0, ...}
snap.errors                   # → {} on success, {"flights": "..."} on partial failure
snap.window_start             # → datetime
snap.thames_readings          # → list[ThamesReading] — raw data if you need it
```

#### TTL cache

Each source has its own time-to-live:

| Source | TTL | Why |
|--------|-----|-----|
| Thames | 240 s (4 min) | Data updates every 15 min; fetch early to catch each new reading |
| Weather | 600 s (10 min) | Forecast model updates ~hourly; no need to poll faster |
| Flights | 180 s (3 min) | Arrivals/departures update in near real-time; more frequent is fine |

Pass `use_cache=False` to force a full refresh (e.g. just before settlement).

---

## Integration with your trading bot

```python
from imc_template.bot_template import BaseBot, OrderBook, Trade, OrderRequest, Side
from data_pipeline import DataPipeline, get_market_window

class MyBot(BaseBot):

    def __init__(self, cmi_url, username, password):
        super().__init__(cmi_url, username, password)
        start, end = get_market_window()
        self.pipeline = DataPipeline(start, end)
        self._last_estimates = None

    def on_orderbook(self, orderbook: OrderBook) -> None:
        # Refresh estimates (cached, so only hits APIs when TTL expires)
        snap = self.pipeline.fetch_snapshot()
        estimates = snap.estimates.as_dict()

        fair_value = estimates.get(orderbook.product)
        if fair_value is None:
            return  # no estimate yet — don't trade blind

        # Simple directional trade: buy if mid-price is below fair value
        if orderbook.sell_orders:
            best_ask = orderbook.sell_orders[0].price
            if best_ask < fair_value * 0.995:   # at least 0.5% below estimate
                self.send_order(OrderRequest(
                    product=orderbook.product,
                    price=best_ask,
                    side=Side.BUY,
                    volume=1,
                ))

    def on_trades(self, trade: Trade) -> None:
        pass
```

---

## Running the tests

```bash
python -m pytest test_pipeline.py -v
```

All 47 tests should pass. No network calls are made — every external API
response is mocked. This means tests work offline and without an API key.

### What the tests cover

| Test class | What it verifies |
|---|---|
| `TestThamesSettlement` | Market 1 & 2 maths against PDF examples; edge cases |
| `TestWeatherSettlement` | Market 3 & 4 maths; temperature rounding |
| `TestFlightsSettlement` | Market 5 & 6 maths; metric formula; zero-flight edge cases |
| `TestDerivedMarkets` | Market 7 & 8 maths across the full payoff curve |
| `TestThamesFetcherMocked` | HTTP parsing; missing-field handling |
| `TestWeatherFetcherMocked` | HTTP parsing; window filtering |
| `TestFlightsFetcherMocked` | HTTP parsing; bucket counting; 401 error handling |
| `TestDataPipeline` | Full pipeline assembly; caching behaviour; skip_flights mode |
| `TestGetMarketWindow` | Window is 24 h; starts on Saturday noon |

---

## Things you still need to provide

| What | Where | Notes |
|---|---|---|
| **RapidAPI key** | `data_pipeline/config.py` → `RAPIDAPI_KEY` | Required for Markets 5, 6, 7, 8 |

Everything else (Thames, weather) is completely free and works out of the box.
