"""Exchange connectivity smoke test.

Verifies auth, product listing, positions, orderbooks, PnL,
and the live SSE stream — without placing any real orders.

Run with credentials from env vars (recommended):
    export CMI_USER=AAKK
    export CMI_PASS=aakk2026
    .venv/bin/python test_connection.py

Or pass them on the command line:
    .venv/bin/python test_connection.py AAKK aakk2026
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import requests
import sseclient
from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
CMI_URL = os.environ.get("CMI_URL", "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com")

if len(sys.argv) >= 3:
    USERNAME = sys.argv[1]
    PASSWORD = sys.argv[2]
else:
    USERNAME = os.environ.get("CMI_USER", "")
    PASSWORD = os.environ.get("CMI_PASS", "")

if not USERNAME or not PASSWORD:
    print("Usage:  .venv/bin/python test_connection.py <username> <password>")
    print("   or:  export CMI_USER=... CMI_PASS=... && .venv/bin/python test_connection.py")
    sys.exit(1)

HEADERS = {"Content-Type": "application/json; charset=utf-8"}
SSE_LISTEN_SECS = 15   # seconds to listen to the live stream


def hr(title=""):
    width = 60
    if title:
        print(f"\n{'─' * 4} {title} {'─' * max(0, width - len(title) - 6)}")
    else:
        print("─" * width)


def ok(msg):  print(f"  ✓  {msg}")
def fail(msg): print(f"  ✗  {msg}")
def info(msg): print(f"     {msg}")


# ── 1. Authentication ─────────────────────────────────────────────────────────
hr("1. Authentication")
try:
    resp = requests.post(
        f"{CMI_URL}/api/user/authenticate",
        headers=HEADERS,
        json={"username": USERNAME, "password": PASSWORD},
        timeout=10,
    )
    resp.raise_for_status()
    TOKEN = resp.headers["Authorization"]
    ok(f"Logged in as {USERNAME!r}  |  token: {TOKEN[:30]}…")
except requests.exceptions.ConnectionError as e:
    fail(f"Cannot reach exchange at {CMI_URL}")
    info(str(e))
    sys.exit(1)
except Exception as e:
    fail(f"Auth failed: {e}")
    if hasattr(e, "response") and e.response is not None:
        info(f"Response body: {e.response.text[:300]}")
    sys.exit(1)

AUTH_HEADERS = {**HEADERS, "Authorization": TOKEN}


# ── 2. Products ───────────────────────────────────────────────────────────────
hr("2. Products listed on exchange")
try:
    resp = requests.get(f"{CMI_URL}/api/product", headers=AUTH_HEADERS, timeout=10)
    resp.raise_for_status()
    products = resp.json()
    for p in products:
        info(f"  {p['symbol']:<18}  tick={p['tickSize']}  start={p['startingPrice']}")
    ok(f"{len(products)} products found")
    PRODUCT_SYMBOLS = [p["symbol"] for p in products]
except Exception as e:
    fail(f"Could not list products: {e}")
    PRODUCT_SYMBOLS = []


# ── 3. Positions ──────────────────────────────────────────────────────────────
hr("3. Current positions")
try:
    resp = requests.get(
        f"{CMI_URL}/api/position/current-user",
        headers=AUTH_HEADERS, timeout=10,
    )
    resp.raise_for_status()
    positions = resp.json()
    if positions:
        for p in positions:
            info(f"  {p['product']:<18}  net={p['netPosition']:+d}")
    else:
        info("  (all flat — no open positions)")
    ok("Positions fetched")
except Exception as e:
    fail(f"Could not fetch positions: {e}")


# ── 4. Open orders ────────────────────────────────────────────────────────────
hr("4. Current open orders")
try:
    resp = requests.get(
        f"{CMI_URL}/api/order/current-user",
        headers=AUTH_HEADERS, timeout=10,
    )
    resp.raise_for_status()
    orders = resp.json()
    if orders:
        for o in orders[:10]:
            info(f"  {o.get('product','?'):<18}  {o.get('side','?'):<5} "
                 f"price={o.get('price','?')}  vol={o.get('volume','?')}  "
                 f"id={str(o.get('id','?'))[:12]}…")
        if len(orders) > 10:
            info(f"  … and {len(orders)-10} more")
    else:
        info("  (no resting orders)")
    ok(f"{len(orders)} open orders")
except Exception as e:
    fail(f"Could not fetch orders: {e}")


# ── 5. PnL ────────────────────────────────────────────────────────────────────
hr("5. Profit & Loss")
try:
    resp = requests.get(
        f"{CMI_URL}/api/profit/current-user",
        headers=AUTH_HEADERS, timeout=10,
    )
    if resp.ok:
        pnl = resp.json()
        if isinstance(pnl, dict):
            for k, v in pnl.items():
                info(f"  {k}: {v}")
        else:
            info(str(pnl))
        ok("PnL fetched")
    else:
        info(f"  HTTP {resp.status_code}: {resp.text[:200]}")
        fail("PnL endpoint returned non-200")
except Exception as e:
    fail(f"Could not fetch PnL: {e}")


# ── 6. Orderbooks (first 3 products) ─────────────────────────────────────────
hr("6. Orderbook snapshots (first 3 products)")
for sym in PRODUCT_SYMBOLS[:3]:
    try:
        resp = requests.get(
            f"{CMI_URL}/api/product/{sym}/order-book/current-user",
            headers=AUTH_HEADERS, timeout=10,
        )
        if resp.ok:
            data = resp.json()
            buys  = data.get("buy",  [])
            sells = data.get("sell", [])
            top_bid = buys[0]["price"]  if buys  else "—"
            top_ask = sells[0]["price"] if sells else "—"
            info(f"  {sym:<18}  bid={top_bid}  ask={top_ask}  "
                 f"({len(buys)} bid levels, {len(sells)} ask levels)")
        else:
            info(f"  {sym:<18}  HTTP {resp.status_code}")
    except Exception as e:
        info(f"  {sym}: error — {e}")
ok("Orderbook snapshots done")


# ── 7. SSE live stream ────────────────────────────────────────────────────────
hr(f"7. SSE stream (listening {SSE_LISTEN_SECS}s)")
info(f"  Connecting to {CMI_URL}/api/market/stream …")

event_counts = {"order": 0, "trade": 0, "other": 0}
products_seen: set[str] = set()
lock = threading.Lock()
stop_flag = threading.Event()


def consume_sse():
    sse_headers = {
        "Authorization": TOKEN,
        "Accept": "text/event-stream; charset=utf-8",
    }
    try:
        stream = requests.get(
            f"{CMI_URL}/api/market/stream",
            headers=sse_headers, stream=True, timeout=30,
        )
        client = sseclient.SSEClient(stream)
        for event in client.events():
            if stop_flag.is_set():
                stream.close()
                break
            with lock:
                if event.event == "order":
                    event_counts["order"] += 1
                    try:
                        import json
                        data = json.loads(event.data)
                        products_seen.add(data.get("productsymbol", "?"))
                    except Exception:
                        pass
                elif event.event == "trade":
                    event_counts["trade"] += 1
                else:
                    event_counts["other"] += 1
    except Exception as e:
        with lock:
            event_counts["other"] += 1
        if not stop_flag.is_set():
            print(f"\n  SSE error: {e}")


t = threading.Thread(target=consume_sse, daemon=True)
t.start()

for i in range(SSE_LISTEN_SECS):
    time.sleep(1)
    with lock:
        print(f"\r  [{i+1:2d}s]  order_events={event_counts['order']}  "
              f"trade_events={event_counts['trade']}  "
              f"products_seen={len(products_seen)}", end="", flush=True)

stop_flag.set()
t.join(timeout=3)
print()  # newline after the carriage-return updates

total = sum(event_counts.values())
if total > 0:
    ok(f"{total} SSE events received  |  orderbook products: {sorted(products_seen)}")
else:
    fail("No SSE events received — stream may be inactive or market is closed")


# ── Summary ───────────────────────────────────────────────────────────────────
hr("Summary")
print(f"  Exchange : {CMI_URL}")
print(f"  User     : {USERNAME}")
print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Products : {PRODUCT_SYMBOLS}")
hr()
print("  If all checks passed, run the full bot with:")
print(f"    export CMI_USER={USERNAME}")
print(f"    export CMI_PASS=<your_password>")
print(f"    .venv/bin/python main.py")
hr()
