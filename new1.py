import time
import numpy as np
from collections import deque

# ============================================================
# Exchange API (plug in your platform)
# ============================================================
class ExchangeAPI:
    def get_mid(self, product: str) -> float:
        raise NotImplementedError

    def get_best_bid_ask(self, product: str) -> tuple[float, float]:
        raise NotImplementedError

    def place_order(self, product: str, side: str, price: float, qty: int):
        raise NotImplementedError

    def get_position(self, product: str) -> int:
        raise NotImplementedError


# ============================================================
# Common helpers
# ============================================================
def flatten_products(api: ExchangeAPI, products: list[str], clip=20):
    for p in products:
        pos = api.get_position(p)
        if pos == 0:
            continue
        bid, ask = api.get_best_bid_ask(p)
        q = int(min(abs(pos), clip))
        if pos > 0:
            api.place_order(p, "SELL", bid, q)
        else:
            api.place_order(p, "BUY", ask, q)


class RollingStd:
    def __init__(self, n=200):
        self.x = deque(maxlen=n)

    def add(self, v: float):
        self.x.append(float(v))

    def mean(self) -> float:
        return float(np.mean(self.x)) if len(self.x) else np.nan

    def std(self) -> float:
        if len(self.x) < 30:
            return np.nan
        return float(np.std(self.x) + 1e-9)


class EWMA1:
    def __init__(self, alpha=0.06):
        self.alpha = alpha
        self.v = None

    def update(self, x: float) -> float:
        x = float(x)
        self.v = x if self.v is None else (1 - self.alpha) * self.v + self.alpha * x
        return self.v


class RollingRV:
    def __init__(self, n=120):
        self.px = deque(maxlen=n + 1)

    def update(self, p: float) -> float:
        self.px.append(float(p))
        if len(self.px) < 25:
            return np.nan
        r = np.diff(np.log(np.array(self.px)))
        return float(np.sqrt(np.mean(r * r)))


# ============================================================
# Market 7/8: Pure statistical arb (no directional)
#   PACK ~ b0 + b1*ETF + b2*ETF^2 + b3*RV
# ============================================================
class RollingRidge4:
    def __init__(self, window=700, ridge=5e-2):
        self.X = deque(maxlen=window)
        self.y = deque(maxlen=window)
        self.ridge = ridge
        self.beta = np.zeros(4)

    def add(self, etf: float, rv: float, pack: float):
        if np.isnan(rv):
            return
        x = np.array([1.0, etf, etf * etf, rv], dtype=float)
        self.X.append(x)
        self.y.append(float(pack))

    def fit(self):
        if len(self.y) < 120:
            return self.beta
        X = np.vstack(self.X)
        y = np.array(self.y)
        XtX = X.T @ X
        Xty = X.T @ y
        lam = self.ridge * np.eye(4)
        self.beta = np.linalg.solve(XtX + lam, Xty)
        return self.beta

    def predict(self, etf: float, rv: float) -> float:
        x = np.array([1.0, etf, etf * etf, rv], dtype=float)
        return float(x @ self.beta)

    def dprice_detf(self, etf: float) -> float:
        return float(self.beta[1] + 2.0 * self.beta[2] * etf)


class RollingZ:
    def __init__(self, n=350):
        self.r = deque(maxlen=n)

    def add(self, v: float):
        self.r.append(float(v))

    def z(self, v: float) -> float:
        if len(self.r) < 80:
            return np.nan
        arr = np.array(self.r, dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std() + 1e-9)
        return (float(v) - mu) / sd


class ETFPackStatArb:
    def __init__(
        self,
        api: ExchangeAPI,
        etf: str,
        pack: str,
        max_pack_pos=18,
        max_etf_pos=35,
        clip_pack=3,
        clip_etf=6,
        z_enter=2.6,
        z_exit=0.7,
        cooldown=0.2,
    ):
        self.api = api
        self.etf = etf
        self.pack = pack

        self.max_pack_pos = max_pack_pos
        self.max_etf_pos = max_etf_pos
        self.clip_pack = clip_pack
        self.clip_etf = clip_etf
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.cooldown = cooldown
        self.last_trade = 0.0

        self.rv = RollingRV(n=120)
        self.reg = RollingRidge4(window=700, ridge=5e-2)
        self.resid = RollingZ(n=350)

    def flatten_pair(self, clip=12):
        flatten_products(self.api, [self.etf, self.pack], clip=clip)

    def step(self):
        etf_mid = self.api.get_mid(self.etf)
        pack_mid = self.api.get_mid(self.pack)
        bid_p, ask_p = self.api.get_best_bid_ask(self.pack)
        bid_e, ask_e = self.api.get_best_bid_ask(self.etf)

        rv = self.rv.update(etf_mid)

        self.reg.add(etf_mid, rv, pack_mid)
        self.reg.fit()

        fair = self.reg.predict(etf_mid, rv)
        r = pack_mid - fair
        self.resid.add(r)
        z = self.resid.z(r)

        pack_pos = self.api.get_position(self.pack)
        etf_pos = self.api.get_position(self.etf)

        now = time.time()
        if now - self.last_trade < self.cooldown:
            return

        if (not np.isnan(z)) and abs(z) < self.z_exit and (pack_pos != 0 or etf_pos != 0):
            self.flatten_pair(clip=10)
            self.last_trade = now
            return

        if np.isnan(z):
            return

        dP_dE = self.reg.dprice_detf(etf_mid)

        # pack rich => SELL pack + hedge ETF
        if z > self.z_enter:
            room_sell_pack = max(0, self.max_pack_pos + pack_pos)
            if room_sell_pack <= 0:
                return
            q_pack = int(min(self.clip_pack, room_sell_pack))
            self.api.place_order(self.pack, "SELL", bid_p, q_pack)

            hedge = int(np.clip(dP_dE * q_pack, -self.clip_etf, self.clip_etf))
            room_buy_etf = max(0, self.max_etf_pos - etf_pos)
            room_sell_etf = max(0, self.max_etf_pos + etf_pos)

            if hedge > 0 and room_buy_etf > 0:
                self.api.place_order(self.etf, "BUY", ask_e, int(min(hedge, room_buy_etf)))
            if hedge < 0 and room_sell_etf > 0:
                self.api.place_order(self.etf, "SELL", bid_e, int(min(-hedge, room_sell_etf)))

            self.last_trade = now
            return

        # pack cheap => BUY pack + hedge ETF
        if z < -self.z_enter:
            room_buy_pack = max(0, self.max_pack_pos - pack_pos)
            if room_buy_pack <= 0:
                return
            q_pack = int(min(self.clip_pack, room_buy_pack))
            self.api.place_order(self.pack, "BUY", ask_p, q_pack)

            hedge = int(np.clip(dP_dE * q_pack, -self.clip_etf, self.clip_etf))
            room_buy_etf = max(0, self.max_etf_pos - etf_pos)
            room_sell_etf = max(0, self.max_etf_pos + etf_pos)

            if hedge > 0 and room_sell_etf > 0:
                self.api.place_order(self.etf, "SELL", bid_e, int(min(hedge, room_sell_etf)))
            if hedge < 0 and room_buy_etf > 0:
                self.api.place_order(self.etf, "BUY", ask_e, int(min(-hedge, room_buy_etf)))

            self.last_trade = now
            return


# ============================================================
# Market 1/2: Relationship arb, with "M1 up" regime constraint:
#   - NEVER short M1 (no short hedge)
#   - only take the side that results in BUY M1
# ============================================================
class RollingRidge3:
    # y ~ b0 + b1*x + b2*rv
    def __init__(self, window=650, ridge=5e-2):
        self.X = deque(maxlen=window)
        self.y = deque(maxlen=window)
        self.ridge = ridge
        self.beta = np.zeros(3)

    def add(self, x: float, rv: float, y: float):
        if np.isnan(rv):
            return
        self.X.append(np.array([1.0, x, rv], dtype=float))
        self.y.append(float(y))

    def fit(self):
        if len(self.y) < 120:
            return self.beta
        X = np.vstack(self.X)
        y = np.array(self.y)
        XtX = X.T @ X
        Xty = X.T @ y
        lam = self.ridge * np.eye(3)
        self.beta = np.linalg.solve(XtX + lam, Xty)
        return self.beta

    def predict(self, x: float, rv: float) -> float:
        v = np.array([1.0, x, rv], dtype=float)
        return float(v @ self.beta)

    def dy_dx(self) -> float:
        return float(self.beta[1])


class M2vsM1StatArbUpM1:
    def __init__(
        self,
        api: ExchangeAPI,
        m1: str,
        m2: str,
        max_m2_pos=25,
        max_m1_pos=80,
        clip_m2=4,
        clip_m1=12,
        z_enter=2.4,
        z_exit=0.7,
        cooldown=0.2,
        min_m1_pos_floor=0,   # keep M1 >= floor
    ):
        self.api = api
        self.m1 = m1
        self.m2 = m2

        self.max_m2_pos = max_m2_pos
        self.max_m1_pos = max_m1_pos
        self.clip_m2 = clip_m2
        self.clip_m1 = clip_m1

        self.z_enter = z_enter
        self.z_exit = z_exit
        self.cooldown = cooldown
        self.last_trade = 0.0

        self.min_m1_pos_floor = min_m1_pos_floor

        self.rv1 = RollingRV(n=120)
        self.reg = RollingRidge3(window=650, ridge=5e-2)
        self.res = RollingStd(n=350)

    def flatten_spread(self, clip=12):
        # flatten M2 fully; reduce M1 only down to floor (never below)
        pos2 = self.api.get_position(self.m2)
        if pos2 != 0:
            bid2, ask2 = self.api.get_best_bid_ask(self.m2)
            q2 = int(min(abs(pos2), clip))
            if pos2 > 0:
                self.api.place_order(self.m2, "SELL", bid2, q2)
            else:
                self.api.place_order(self.m2, "BUY", ask2, q2)

        pos1 = self.api.get_position(self.m1)
        if pos1 > self.min_m1_pos_floor:
            bid1, _ = self.api.get_best_bid_ask(self.m1)
            q1 = int(min(pos1 - self.min_m1_pos_floor, clip))
            if q1 > 0:
                self.api.place_order(self.m1, "SELL", bid1, q1)

    def step(self):
        m1 = self.api.get_mid(self.m1)
        m2 = self.api.get_mid(self.m2)
        bid1, ask1 = self.api.get_best_bid_ask(self.m1)
        bid2, ask2 = self.api.get_best_bid_ask(self.m2)

        rv = self.rv1.update(m1)
        self.reg.add(m1, rv, m2)
        self.reg.fit()

        fair = self.reg.predict(m1, rv)
        resid = m2 - fair
        self.res.add(resid)

        mu = self.res.mean()
        sd = self.res.std()
        z = (resid - mu) / (sd if not np.isnan(sd) else np.nan)

        pos2 = self.api.get_position(self.m2)
        pos1 = self.api.get_position(self.m1)

        now = time.time()
        if now - self.last_trade < self.cooldown:
            return

        if (not np.isnan(z)) and abs(z) < self.z_exit and (pos2 != 0 or pos1 != self.min_m1_pos_floor):
            self.flatten_spread(clip=12)
            self.last_trade = now
            return

        if np.isnan(z):
            return

        beta = self.reg.dy_dx()

        room_buy1 = max(0, self.max_m1_pos - pos1)
        room_sell2 = max(0, self.max_m2_pos + pos2)

        # ONLY take the side that BUYS M1:
        # z high => M2 rich => SELL M2 + BUY M1
        if z > self.z_enter and room_sell2 > 0:
            q2 = int(min(self.clip_m2, room_sell2))
            self.api.place_order(self.m2, "SELL", bid2, q2)

            hedge = int(np.clip(beta * q2, 0, self.clip_m1))  # clamp to >=0 => BUY M1 only
            q1 = int(min(hedge, room_buy1))
            if q1 > 0:
                self.api.place_order(self.m1, "BUY", ask1, q1)

            self.last_trade = now
            return

        # z low => M2 cheap => would normally BUY M2 and SELL M1 -> SKIP (no short M1)
        return


# ============================================================
# Markets 1/3/4: Up-only dip buyers (never short)
# ============================================================
class UpOnlyDipBuyer:
    def __init__(self, api: ExchangeAPI, product: str, max_pos=50, clip=8, z_enter=1.6, z_take=0.3, cooldown=0.2):
        self.api = api
        self.product = product
        self.max_pos = max_pos
        self.clip = clip
        self.z_enter = z_enter
        self.z_take = z_take
        self.cooldown = cooldown
        self.last_trade = 0.0
        self.ema = EWMA1(alpha=0.06)
        self.dev = RollingStd(n=220)

    def step(self):
        mid = self.api.get_mid(self.product)
        bid, ask = self.api.get_best_bid_ask(self.product)
        pos = self.api.get_position(self.product)

        m = self.ema.update(mid)
        d = mid - m
        self.dev.add(d)
        sd = self.dev.std()
        z = d / (sd if not np.isnan(sd) else np.nan)

        now = time.time()
        if now - self.last_trade < self.cooldown:
            return

        if not np.isnan(z) and z < -self.z_enter:
            room_buy = max(0, self.max_pos - pos)
            if room_buy > 0:
                q = int(min(self.clip, room_buy))
                self.api.place_order(self.product, "BUY", ask, q)
                self.last_trade = now
                return

        if not np.isnan(z) and z > -self.z_take and pos > 0:
            q = int(min(self.clip, pos))
            self.api.place_order(self.product, "SELL", bid, q)
            self.last_trade = now
            return


# ============================================================
# Markets 5/6: Pair trading (stable relationship), no directional
#   Model: M6 ~ a + b*M5
# ============================================================
class RollingOLS2:
    def __init__(self, window=600):
        self.x = deque(maxlen=window)
        self.y = deque(maxlen=window)
        self.a = 0.0
        self.b = 1.0

    def add(self, x, y):
        self.x.append(float(x))
        self.y.append(float(y))

    def fit(self):
        if len(self.x) < 80:
            return self.a, self.b
        x = np.array(self.x, dtype=float)
        y = np.array(self.y, dtype=float)
        mx = float(x.mean())
        my = float(y.mean())
        vx = float(((x - mx) ** 2).mean()) + 1e-12
        cov = float(((x - mx) * (y - my)).mean())
        self.b = cov / vx
        self.a = my - self.b * mx
        return self.a, self.b


class PairTradingArb:
    def __init__(
        self,
        api: ExchangeAPI,
        pA: str,
        pB: str,
        max_pos_A=70,
        max_pos_B=70,
        clip_A=12,
        clip_B=12,
        z_enter=2.6,
        z_exit=0.7,
        cooldown=0.25,
        ols_window=700,
        resid_window=350,
        hedge_clip=12,
    ):
        self.api = api
        self.pA = pA
        self.pB = pB

        self.max_pos_A = max_pos_A
        self.max_pos_B = max_pos_B
        self.clip_A = clip_A
        self.clip_B = clip_B
        self.z_enter = z_enter
        self.z_exit = z_exit
        self.cooldown = cooldown
        self.last_trade = 0.0

        self.ols = RollingOLS2(window=ols_window)
        self.res = RollingZ(n=resid_window)
        self.hedge_clip = hedge_clip

    def flatten(self, clip=15):
        flatten_products(self.api, [self.pA, self.pB], clip=clip)

    def step(self):
        a_mid = self.api.get_mid(self.pA)
        b_mid = self.api.get_mid(self.pB)
        bidA, askA = self.api.get_best_bid_ask(self.pA)
        bidB, askB = self.api.get_best_bid_ask(self.pB)

        self.ols.add(a_mid, b_mid)
        a, beta = self.ols.fit()

        fairB = a + beta * a_mid
        resid = b_mid - fairB
        self.res.add(resid)
        z = self.res.z(resid)

        posA = self.api.get_position(self.pA)
        posB = self.api.get_position(self.pB)

        now = time.time()
        if now - self.last_trade < self.cooldown:
            return

        if (not np.isnan(z)) and abs(z) < self.z_exit and (posA != 0 or posB != 0):
            self.flatten(clip=12)
            self.last_trade = now
            return

        if np.isnan(z):
            return

        hedgeA = int(np.clip(beta, -self.hedge_clip, self.hedge_clip))
        hedgeA = hedgeA if hedgeA != 0 else 1

        room_buyA = max(0, self.max_pos_A - posA)
        room_sellA = max(0, self.max_pos_A + posA)
        room_buyB = max(0, self.max_pos_B - posB)
        room_sellB = max(0, self.max_pos_B + posB)

        # B rich => SELL B, BUY A
        if z > self.z_enter and room_sellB > 0:
            qB = int(min(self.clip_B, room_sellB))
            self.api.place_order(self.pB, "SELL", bidB, qB)

            qA = int(min(self.clip_A, room_buyA, abs(hedgeA) * qB))
            if qA > 0:
                self.api.place_order(self.pA, "BUY", askA, qA)

            self.last_trade = now
            return

        # B cheap => BUY B, SELL A
        if z < -self.z_enter and room_buyB > 0:
            qB = int(min(self.clip_B, room_buyB))
            self.api.place_order(self.pB, "BUY", askB, qB)

            qA = int(min(self.clip_A, room_sellA, abs(hedgeA) * qB))
            if qA > 0:
                self.api.place_order(self.pA, "SELL", bidA, qA)

            self.last_trade = now
            return


# ============================================================
# Unified runner (trades only the products you set)
# ============================================================
def run_all(api: ExchangeAPI):
    # ---- set tickers here ----
    # 7/8
    ETF7 = "LON_ETF"
    PACK8 = "ETF_PACK_8"

    # 1/2/3/4 (adjust to your platform)
    M1 = "TIDE_SPOT"     # Thames spot
    M2 = "TIDE_SWING"    # Thames derivative/strangle proxy
    M3 = "WX_SUM"        # temperature-related
    M4 = "WX_SPOT"       # humidity-related OR your true Market 4

    # 5/6 stable pair
    M5 = "LHR_COUNT"
    M6 = "LON_FLY"

    # ---- instantiate strategies ----
    s78 = ETFPackStatArb(api, etf=ETF7, pack=PACK8)

    s21 = M2vsM1StatArbUpM1(
        api,
        m1=M1,
        m2=M2,
        min_m1_pos_floor=0,   # set to 10 if you always want +10 long M1
    )

    b1 = UpOnlyDipBuyer(api, M1, max_pos=90, clip=12, z_enter=1.5, z_take=0.25)
    b3 = UpOnlyDipBuyer(api, M3, max_pos=50, clip=8,  z_enter=1.5, z_take=0.25)
    b4 = UpOnlyDipBuyer(api, M4, max_pos=50, clip=8,  z_enter=1.5, z_take=0.25)

    p56 = PairTradingArb(api, pA=M5, pB=M6)

    # ---- flatten touched products once at start ----
    for _ in range(6):
        s78.flatten_pair(clip=12)
        flatten_products(api, [M1, M2, M3, M4, M5, M6], clip=15)
        time.sleep(0.2)

    # ---- main loop ----
    while True:
        s78.step()
        s21.step()
        b1.step(); b3.step(); b4.step()
        p56.step()
        time.sleep(0.15)