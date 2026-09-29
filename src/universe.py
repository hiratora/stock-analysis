"""全銘柄スナップショットと基本スクリーニング。

目的: 候補を記事やシグナル発生銘柄からではなく、東証内国株の全銘柄から毎日選ぶ。

出力（outputs/latest/ にも同じものを置く）:
  universe.csv      全銘柄の指標（約3,800行）
  basic_screen.csv  基本スクリーニング通過銘柄をスコア順に（上位60）
  sector_rs.csv     33業種ごとの相対強度（資金がどこに向かっているか）

基本スクリーニング
  足切り（売買できるか）
    - 20日平均売買代金 3億円以上
    - 100株の購入代金 200万円以下
    - ATR14/終値 1.5%〜3.5%（-2.5%の損切りが日々のノイズで刺さらず、10営業日で+5%に届く値幅がある）
  足切り（上昇トレンドにあるか）
    - 終値 > 25日線 > 75日線、75日線が20日前より上
    - 終値が52週高値の85%以上
  足切り（伸び切っていないか）
    - 25日線乖離 +12%以下、5日騰落 +12%以下
  スコア（順位付け）
    - 相対強度: 20日・60日騰落率のTOPIX(1306.T)超過分の全銘柄内パーセンタイル（重み 0.35 / 0.25）
    - 52週高値への近さ（0.15）
    - 出来高の増加: 直近10日平均 / 60日平均（0.15）
    - 値幅の収縮: 直近10日の高安幅 / (ATR14×10)、小さいほど高得点（0.10）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

MIN_TURN = 3e8
MAX_LOT = 2_000_000
ATR_LO, ATR_HI = 0.015, 0.035
MIN_NEAR_HIGH = 0.85
MAX_DEV25, MAX_RET5 = 0.12, 0.12
TOP_N = 60


def snapshot(quotes: pd.DataFrame, topix: pd.DataFrame) -> pd.DataFrame:
    t = topix.sort_values("date").set_index("date")["close"]
    t20 = t.iloc[-1] / t.iloc[-21] - 1 if len(t) > 21 else np.nan
    t60 = t.iloc[-1] / t.iloc[-61] - 1 if len(t) > 61 else np.nan
    asof = quotes["date"].max()
    rows = []
    for code, g in quotes.groupby("code"):
        g = g.sort_values("date")
        if g["date"].iloc[-1] != asof or len(g) < 80:
            continue
        c, h, l, v = g["close"].values, g["high"].values, g["low"].values, g["volume"].values
        n = len(g)
        s25 = c[-25:].mean(); s75 = c[-75:].mean()
        s75_prev = c[-95:-20].mean() if n >= 95 else np.nan
        prev = np.r_[np.nan, c[:-1]]
        tr = np.nanmax(np.vstack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)
        atr = np.nanmean(tr[-14:])
        hi52 = h[-250:].max(); lo52 = l[-250:].min()
        turn20 = (c[-20:] * v[-20:]).mean()
        vol10 = v[-10:].mean(); vol60 = v[-60:].mean() if n >= 60 else np.nan
        rng10 = h[-10:].max() - l[-10:].min()
        rows.append({
            "code": code, "date": asof.date(), "close": c[-1],
            "ret1d": c[-1] / c[-2] - 1, "ret5d": c[-1] / c[-6] - 1,
            "ret20d": c[-1] / c[-21] - 1, "ret60d": c[-1] / c[-61] - 1 if n > 61 else np.nan,
            "rs20": (c[-1] / c[-21] - 1) - t20, "rs60": ((c[-1] / c[-61] - 1) - t60) if n > 61 else np.nan,
            "sma25": s25, "sma75": s75, "sma75_rising": bool(s75 > s75_prev) if s75_prev == s75_prev else False,
            "dev25": c[-1] / s25 - 1, "near_high": c[-1] / hi52, "hi52w": hi52, "lo52w": lo52,
            "atr_pct": atr / c[-1], "turn20_oku": turn20 / 1e8, "lot_yen": c[-1] * 100,
            "vol_trend": vol10 / vol60 if vol60 else np.nan,
            "contraction": rng10 / (atr * 10) if atr else np.nan,
        })
    return pd.DataFrame(rows)


def basic_screen(u: pd.DataFrame) -> pd.DataFrame:
    f = u[
        (u.turn20_oku * 1e8 >= MIN_TURN) & (u.lot_yen <= MAX_LOT)
        & u.atr_pct.between(ATR_LO, ATR_HI)
        & (u.close > u.sma25) & (u.sma25 > u.sma75) & u.sma75_rising
        & (u.near_high >= MIN_NEAR_HIGH)
        & (u.dev25 <= MAX_DEV25) & (u.ret5d <= MAX_RET5)
    ].copy()
    if f.empty:
        return f
    pct = lambda s: s.rank(pct=True)
    f["score"] = (0.35 * pct(f.rs20) + 0.25 * pct(f.rs60.fillna(f.rs60.median()))
                  + 0.15 * pct(f.near_high) + 0.15 * pct(f.vol_trend)
                  + 0.10 * pct(-f.contraction))
    return f.sort_values("score", ascending=False)


def sector_rs(u: pd.DataFrame) -> pd.DataFrame:
    g = u.groupby("sector33").agg(n=("code", "size"), rs20_med=("rs20", "median"), rs60_med=("rs60", "median"),
                                  above_trend=("sma75_rising", "mean"), near_high_med=("near_high", "median"))
    return g.sort_values("rs20_med", ascending=False).reset_index()


def run(offline: bool, out_root: Path = Path("outputs")) -> Path:
    if offline:
        quotes, topix, listed = pd.read_parquet(D.QUOTES), pd.read_parquet(D.TOPIX), D.load_listed()
    else:
        listed = D.fetch_listed()
        quotes = D.update_quotes(listed["code"].tolist())
        topix = D.update_topix()
    u = snapshot(quotes, topix)
    u = u.merge(listed[["code", "name", "market", "sector33"]], on="code", how="left")
    s = basic_screen(u)
    sec = sector_rs(u)
    asof = str(u["date"].iloc[0])
    for d in (out_root / asof, out_root / "latest"):
        d.mkdir(parents=True, exist_ok=True)
        u.round(4).to_csv(d / "universe.csv", index=False)
        s.head(TOP_N).round(4).to_csv(d / "basic_screen.csv", index=False)
        sec.round(4).to_csv(d / "sector_rs.csv", index=False)
    print(f"asof {asof}: universe {len(u)} / screen pass {len(s)} / top {min(TOP_N, len(s))}")
    return out_root / asof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    run(ap.parse_args().offline)
