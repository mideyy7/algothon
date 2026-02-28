"""Pipeline configuration.

USER SETUP REQUIRED
-------------------
Only ONE thing needs your input: set RAPIDAPI_KEY below.

Thames (Market 1, 2) and Weather (Market 3, 4) APIs are completely free
with no authentication needed.

Flights (Market 5, 6) and the London ETF (Market 7, 8) require a
RapidAPI key for AeroDataBox. Steps:
  1. Go to https://rapidapi.com/aedbx-aedbx/api/aerodatabox
  2. Subscribe to a plan (Basic free tier gives 500 calls/month)
  3. Copy your key from "Header Parameters" → "X-RapidAPI-Key"
  4. Paste it below.
"""

# =============================================================================
# USER: Paste your RapidAPI key here (required for Heathrow flight data).
# Without this, Markets 5, 6, and the ETF/options legs will show None.
# =============================================================================
RAPIDAPI_KEY = "YOUR_RAPIDAPI_KEY_HERE"

RAPIDAPI_HOST = "aerodatabox.p.rapidapi.com"

# --- London coordinates (fixed, do not change) ---
LONDON_LAT = 51.5074
LONDON_LON = -0.1278
LONDON_TIMEZONE = "Europe/London"

# --- Airport ---
LHR_IATA = "LHR"

# --- HTTP behaviour ---
REQUEST_TIMEOUT = 12    # seconds before a single request is abandoned
MAX_RETRIES = 3         # how many times to retry on failure
RETRY_BACKOFF = 2.0     # seconds; multiplied by attempt number on each retry

# --- Cache time-to-live (seconds) ---
# Thames data updates every 15 min — cache for 4 min so we catch updates fast
THAMES_CACHE_TTL = 240
# Open-Meteo updates its model every ~15 min — cache for 10 min
WEATHER_CACHE_TTL = 600
# Flights update continuously — cache for 3 min to stay fresh without hammering
FLIGHTS_CACHE_TTL = 180

# --- CMI product names (must match what the exchange uses) ---
MARKET_PRODUCTS = {
    1: "TIDE_SPOT",
    2: "TIDE_SWING",
    3: "WX_SPOT",
    4: "WX_SUM",
    5: "LHR_COUNT",
    6: "LHR_INDEX",
    7: "LON_ETF",
    8: "LON_FLY",
}
