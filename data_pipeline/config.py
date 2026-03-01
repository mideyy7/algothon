import os
from pathlib import Path
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load variables from .env in the project root (one level up from this file)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Heathrow PIHub API (official Heathrow Airport data platform)
# ---------------------------------------------------------------------------
# Set PIHUB_API_KEY in your .env file if the PIHub endpoint requires auth.
# Leave blank / omit if the endpoint is publicly accessible.
PIHUB_API_KEY: str | None = os.environ.get("PIHUB_API_KEY")

# Base URL for PIHub REST endpoints (arrivals / departures appended by fetcher)
PIHUB_BASE_URL = "https://api-dp-prod.dp.heathrow.com/pihub/flights"

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

# Cache time-to-live (seconds) ---
# Thames data updates every 15 min — cache for 4 min so we catch updates fast
THAMES_CACHE_TTL = 240
# Open-Meteo updates its model every ~15 min — cache for 10 min
WEATHER_CACHE_TTL = 600
# Flights update continuously — cache for 10 min to avoid hammering PIHub
FLIGHTS_CACHE_TTL = 600

# --- Global HTTP Session with auto-retries ---
http_session = requests.Session()
_retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=RETRY_BACKOFF,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
_adapter = HTTPAdapter(max_retries=_retry_strategy)
http_session.mount("https://", _adapter)
http_session.mount("http://", _adapter)

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
