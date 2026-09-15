"""シグナル発生ごとに取引をシミュレーションして期待値を出す。

ルール（戦略と同一）:
  エントリー: シグナル日の翌営業日の寄り
  利確 +5% / 損切り -2.5%（エントリー価格比）、両方同日なら損切り優先（保守的）
  寄りが損切り線を割っていれば寄り値で決済（ギャップダウンは -2.5% では止まらない）
  寄りが利確線を超えていれば寄り値で決済
  保有中に出た同一銘柄のシグナルは無視（重複計上しない）
  最大保有 MAX_HOLD 営業日、到達しなければ引け
  スリッページ+手数料 = 往復 COST

期待値(EV) = 勝率×平均利幅 + (1-勝率)×平均損幅  （1トレードあたり、ポジション対比%）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TP, SL, MAX_HOLD, COST = 0.05, -0.025, 10, 0.002


def simulate(g: pd.DataFrame, sig_col: str) -> pd.DataFrame:
    """1銘柄の指標付きデータからトレード一覧を返す"""
    g = g.reset_index(drop=True)
    o, h, l, c = g["open"].values, g["high"].values, g["low"].values, g["close"].values
    idx = np.where(g[sig_col].values)[0]
    rows = []
    n = len(g)
    busy_until = -1
    for i in idx:
        e = i + 1
        if e >= n or e <= busy_until:
            continue
        ep = o[e]
        if not np.isfinite(ep) or ep <= 0:
            continue
        ret, exit_i, why = None, None, None
        for k in range(e, min(e + MAX_HOLD, n)):
            if k > e and o[k] / ep - 1 <= SL:          # ギャップダウン: 寄りで決済
                ret, exit_i, why = o[k] / ep - 1, k, "SL_GAP"
                break
            if l[k] / ep - 1 <= SL:
                ret, exit_i, why = SL, k, "SL"
                break
            if k > e and o[k] / ep - 1 >= TP:
                ret, exit_i, why = o[k] / ep - 1, k, "TP"
                break
            if h[k] / ep - 1 >= TP:
                ret, exit_i, why = TP, k, "TP"
                break
        if ret is None:
            exit_i = min(e + MAX_HOLD, n) - 1
            ret, why = c[exit_i] / ep - 1, "TIME"
        busy_until = exit_i
        rows.append({
            "code": g["code"].iloc[0], "signal": sig_col.replace("sig_", ""),
            "signal_date": g["date"].iloc[i], "entry_date": g["date"].iloc[e],
            "exit_date": g["date"].iloc[exit_i], "hold_days": exit_i - e + 1,
            "ret": ret - COST, "exit": why,
            "r3": (c[min(e + 2, n - 1)] / ep - 1), "r5": (c[min(e + 4, n - 1)] / ep - 1),
            "ok_liquidity": bool(g["ok_liquidity"].iloc[i]), "ok_atr": bool(g["ok_atr"].iloc[i]),
            "ok_extended": bool(g["ok_extended"].iloc[i]), "ok_gap": bool(g["ok_gap"].iloc[i]),
        })
    return pd.DataFrame(rows)


def run_all(ind: pd.DataFrame, signals=("sig_breakout", "sig_pullback")) -> pd.DataFrame:
    parts = []
    for _, g in ind.groupby("code"):
        for s in signals:
            if g[s].any():
                parts.append(simulate(g, s))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def stats(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"n": 0}
    w = trades["ret"] > 0
    wins, losses = trades.loc[w, "ret"], trades.loc[~w, "ret"]
    wr = w.mean()
    aw, al = (wins.mean() if len(wins) else 0.0), (losses.mean() if len(losses) else 0.0)
    return {
        "n": int(len(trades)), "win_rate": float(wr),
        "avg_win": float(aw), "avg_loss": float(al),
        "ev": float(wr * aw + (1 - wr) * al),
        "tp_rate": float((trades["exit"] == "TP").mean()),
        "sl_rate": float(trades["exit"].isin(["SL", "SL_GAP"]).mean()),
        "gap_loss_rate": float((trades["exit"] == "SL_GAP").mean()),
        "avg_hold": float(trades["hold_days"].mean()),
        "r5_mean": float(trades["r5"].mean()),
    }


def stats_by(trades: pd.DataFrame, keys) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    return trades.groupby(list(keys)).apply(lambda x: pd.Series(stats(x))).reset_index()


def attach_regime(trades: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return trades
    return trades.merge(regime, left_on="signal_date", right_on="date", how="left").drop(columns=["date"])


def monthly_projection(ev: float, trades_per_month: int = 12, position_pct: float = 0.30) -> float:
    """月次期待リターン（資金対比）。サイズは自由なので position_pct は目安に過ぎない。"""
    return ev * trades_per_month * position_pct
