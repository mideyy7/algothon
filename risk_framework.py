import time
import numpy as np
import pandas as pd
from scipy.stats import norm
import quantstats as qs

# ====== assume these already exist in your codebase ======
# get_live_ranking() -> dict like {"rank": 12, "score": 103.2, "leader_score": 115.7, "metric": "pnl"}
# get_latest_state() -> dict / snapshot your model needs (market/features/etc.)
# current_positions() -> dict mapping symbol to current net position (e.g. {"TIDE_SPOT": 20})
# model_predict(state) -> pd.DataFrame with columns ["symbol","side","price","qty","confidence"]
# submit_orders(df_orders) -> None
# get_recent_pnl_series(window=500) -> pd.Series indexed by time, returns cumulative or step pnl
# ========================================================


def score_to_expected_pnl(live: dict, prize_curve: pd.DataFrame) -> float:
    metric = live.get("metric", "pnl")
    score = float(live.get("score", 0.0))
    a = prize_curve[prize_curve["metric"] == metric].sort_values("score")
    expected = np.interp(score, a["score"].to_numpy(), a["expected_pnl"].to_numpy())
    return float(expected)


def pnl_risk_level(pnl: pd.Series) -> dict:
    rets = pnl.diff().fillna(0.0).to_numpy()
    vol = float(np.std(rets) + 1e-12)
    dd = float(qs.stats.max_drawdown(pnl))
    sharpe = float(qs.stats.sharpe(pd.Series(rets), rf=0.0))
    var_95 = float(-np.quantile(rets, 0.05))
    return {"vol": vol, "max_dd": dd, "sharpe": sharpe, "var_95": var_95}


def prize_pressure(live: dict) -> float:
    rank = float(live.get("rank", 9999))
    leader = float(live.get("leader_score", 0.0))
    score = float(live.get("score", 0.0))
    gap = max(0.0, leader - score)
    rank_pressure = 1.0 / (1.0 + np.log1p(rank))
    gap_pressure = 1.0 - np.exp(-gap / (abs(leader) + 1e-9 + 1.0))
    return float(np.clip(0.6 * rank_pressure + 0.4 * gap_pressure, 0.0, 1.0))


def risk_budget_from_prize(expected_pnl: float, risk: dict, pressure: float) -> dict:
    base = 1.0
    pnl_boost = np.tanh(expected_pnl / (abs(expected_pnl) + 100.0))
    risk_penalty = np.clip(0.7 * risk["var_95"] + 0.3 * abs(risk["max_dd"]), 0.0, 2.0)
    aggression = base + 0.8 * pressure + 0.4 * pnl_boost - 0.6 * risk_penalty
    aggression = float(np.clip(aggression, 0.2, 2.0))
    return {"aggression": aggression, "max_qty_mult": aggression, "min_conf": 0.45 + 0.15 / aggression}


def adjust_orders(orders: pd.DataFrame, budget: dict, current_positions: dict) -> pd.DataFrame:
    df = orders.copy()
    df["qty"] = np.floor(df["qty"].to_numpy() * budget["max_qty_mult"]).astype(int)
    df = df[df["confidence"] >= budget["min_conf"]]

    # CRITICAL IMC RULE: ±100 net position limit per product. Exceeding this = permanent ban.
    # We must cap the qty we try to buy or sell based on our current position.
    allowed_qtys = []
    for _, row in df.iterrows():
        sym = row["symbol"]
        side = row["side"].upper()
        qty = row["qty"]
        pos = current_positions.get(sym, 0)

        if side == "BUY":
            # Max we can buy is (100 - current position)
            space_left = max(0, 100 - pos)
        else:
            # Max we can sell is (100 + current position) -> going short down to -100
            space_left = max(0, 100 + pos)

        final_qty = min(qty, space_left)
        allowed_qtys.append(final_qty)

    df["qty"] = allowed_qtys
    df = df[df["qty"] > 0]
    return df.reset_index(drop=True)


def run_live_loop(prize_curve: pd.DataFrame, sleep_s: float = 0.3):
    while True:
        live = get_live_ranking()
        pnl = get_recent_pnl_series(window=800)
        expected_pnl = score_to_expected_pnl(live, prize_curve)
        risk = pnl_risk_level(pnl)
        pressure = prize_pressure(live)
        budget = risk_budget_from_prize(expected_pnl, risk, pressure)
        state = get_latest_state()
        pos = current_positions()
        raw_orders = model_predict(state)
        final_orders = adjust_orders(raw_orders, budget, pos)
        submit_orders(final_orders)
        time.sleep(sleep_s)


# example: make a simple "score -> expected_pnl" curve (you should fit this to your contest rules)
prize_curve = pd.DataFrame(
    {
        "metric": ["pnl"] * 6,
        "score":  [0, 20, 50, 80, 110, 150],
        "expected_pnl": [0, 10, 35, 70, 120, 200],
    }
)

# run_live_loop(prize_curve)
