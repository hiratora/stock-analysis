"""指標とシグナル定義。

シグナルはすべて「終値確定後に判定し、翌営業日の寄りで買う」前提で書く。
その日の値を使ってその日のうちに買う設計にすると、バックテストが先読みになる。

breakout : close > 直近20日高値（当日を除く） かつ 出来高 >= 20日平均×1.5
pullback : 安値が25日線に接触（25日線×(1±tol)） かつ 終値 > 25日線 かつ 25日線が上向き かつ 25日線 > 75日線 かつ 出来高 >= 20日平均×1.0
出来高平均は当日を除く過去20日。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VOL_MULT_BREAKOUT = 1.5
VOL_MULT_PULLBACK = 1.0
PULLBACK_TOL = 0.01
MIN_TURNOVER = 5e8       # 20日平均売買代金 5億円
MAX_ATR_PCT = 0.025      # ATR14 / close <= 2.5%（損切り幅がノイズに埋まらない）
MAX_DEV25 = 0.07         # 25日線乖離 7% 以下（+5%/-2.5% の RR が成り立つ範囲）
MAX_RET1D = 0.06         # 当日騰落率 6% 以下（ギャップ直後の寄りは反落しやすい）


def add_indicators(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, h, l = g["close"], g["high"], g["low"]
    g["sma25"] = c.rolling(25).mean()
    g["sma75"] = c.rolling(75).mean()
    g["sma25_up"] = g["sma25"] > g["sma25"].shift(3)
    g["vol20"] = g["volume"].shift(1).rolling(20).mean()   # 当日を除く過去20日
    g["vol_ratio"] = g["volume"] / g["vol20"]
    g["turn20"] = g["turnover"].rolling(20).mean()
    g["hi20_prev"] = h.shift(1).rolling(20).max()
    g["hi52w_prev"] = h.shift(1).rolling(250, min_periods=120).max()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    g["atr14"] = tr.rolling(14).mean()
    g["atr_pct"] = g["atr14"] / c
    g["dev25"] = c / g["sma25"] - 1
    g["ret5d"] = c / c.shift(5) - 1
    g["ret1d"] = c / c.shift(1) - 1
    g["trend"] = g["sma25"] > g["sma75"]    # signals
    g["sig_breakout"] = (c > g["hi20_prev"]) & (g["vol_ratio"] >= VOL_MULT_BREAKOUT)
    touch = (l <= g["sma25"] * (1 + PULLBACK_TOL)) & (l >= g["sma25"] * (1 - PULLBACK_TOL * 2))
    g["sig_pullback"] = touch & (c > g["sma25"]) & g["sma25_up"] & g["trend"] & (g["vol_ratio"] >= VOL_MULT_PULLBACK)
    # liquidity / noise filters (applied at screening time, also recorded for backtest conditioning)
    g["ok_liquidity"] = g["turn20"] >= MIN_TURNOVER
    g["ok_atr"] = g["atr_pct"] <= MAX_ATR_PCT
    g["ok_extended"] = g["dev25"] <= MAX_DEV25      # 伸びすぎでない
    g["ok_gap"] = g["ret1d"] <= MAX_RET1D           # 当日の急騰後でない
    return g


def add_indicators_all(quotes: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([add_indicators(g) for _, g in quotes.groupby("code")], ignore_index=True)


def topix_regime(topix: pd.DataFrame) -> pd.DataFrame:
    """地合い: TOPIX終値が25日線の上=1 下=0"""
    t = topix.sort_values("date").copy()
    t["t_sma25"] = t["close"].rolling(25).mean()
    t["regime_up"] = (t["close"] > t["t_sma25"]).astype(int)
    return t[["date", "regime_up"]]
