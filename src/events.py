"""イベント候補: 直近で急落した銘柄の一覧（何が起きたかは判定しない。理由の確認は夜のセッションが Web 検索で行う）。

  python -m src.events --offline   # data/ のキャッシュから。Actions では screen の後に実行する

抽出（直近 LOOKBACK 営業日のどこかで）
  - 1日の下落率 ≤ DROP_1D、または2日累計 ≤ DROP_2D
  - その日の出来高 ≥ 直近20日平均 × VOL_MULT（当日を除く）
  - 20日平均売買代金 ≥ MIN_TURN
出力: outputs/<asof>/events.csv, events.md（latest にも）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

LOOKBACK = 7
DROP_1D, DROP_2D = -0.06, -0.09
VOL_MULT = 2.0
MIN_TURN = 2e8


def find_events(quotes: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    names = listed.set_index("code") if len(listed) else pd.DataFrame()
    asof = quotes["date"].max()
    rows = []
    for code, g in quotes.groupby("code"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < 80 or g["date"].iloc[-1] != asof:
            continue
        c, h, l, v = g["close"].values, g["high"].values, g["low"].values, g["volume"].values
        n = len(g)
        turn20 = (c[-20:] * v[-20:]).mean()
        if turn20 < MIN_TURN:
            continue
        r1 = c[1:] / c[:-1] - 1
        r1 = np.r_[np.nan, r1]
        r2 = np.r_[np.nan, np.nan, c[2:] / c[:-2] - 1]
        vol20 = pd.Series(v).shift(1).rolling(20).mean().values
        hit = None
        for i in range(n - LOOKBACK, n):
            if i < 25:
                continue
            if (r1[i] <= DROP_1D or r2[i] <= DROP_2D) and v[i] >= VOL_MULT * vol20[i]:
                hit = i
                break
        if hit is None:
            continue
        i = hit
        pre = c[i - 1] if r1[i] <= DROP_1D else c[i - 2]
        prev = np.r_[np.nan, c[:-1]]
        tr = np.nanmax(np.vstack([h - l, np.abs(h - prev), np.abs(l - prev)]), axis=0)
        atr = np.nanmean(tr[-14:])
        sma25, sma75 = c[-25:].mean(), c[-75:].mean()
        hi52 = h[-250:].max()
        meta = names.loc[code] if code in names.index else None
        rows.append({
            "code": code, "name": meta["name"] if meta is not None else "", "market": (meta["market"] if meta is not None else "").replace("（内国株式）", ""),
            "sector": meta["sector33"] if meta is not None else "",
            "event_date": g["date"].iloc[i].date(), "days_since": n - 1 - i,
            "d0_ret_pct": round(r1[i] * 100, 1), "d2_ret_pct": round(r2[i] * 100, 1) if np.isfinite(r2[i]) else np.nan,
            "vol_x": round(v[i] / vol20[i], 1) if vol20[i] else np.nan,
            "close": round(c[-1], 1), "vs_pre_pct": round((c[-1] / pre - 1) * 100, 1), "since_d0_pct": round((c[-1] / c[i] - 1) * 100, 1),
            "low_since_d0_pct": round((l[i:].min() / pre - 1) * 100, 1),
            "atr_pct": round(atr / c[-1] * 100, 2), "turn20_oku": round(turn20 / 1e8, 1), "lot_yen": round(c[-1] * 100),
            "vs_sma25_pct": round((c[-1] / sma25 - 1) * 100, 1), "vs_sma75_pct": round((c[-1] / sma75 - 1) * 100, 1),
            "near_52w_high": round(pre / hi52, 2), "ret60_pre_pct": round((pre / c[i - 61] - 1) * 100, 1) if i >= 61 else np.nan,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["days_since", "d0_ret_pct"]).reset_index(drop=True)
    return out


def _md(asof, ev: pd.DataFrame) -> str:
    L = [f"# イベント候補（急落銘柄） {asof.date()}", "",
         f"直近{LOOKBACK}営業日に 1日 {DROP_1D*100:.0f}% 以下または2日 {DROP_2D*100:.0f}% 以下の下落（出来高 {VOL_MULT:.0f}倍以上、代金{MIN_TURN/1e8:.0f}億以上）。"
         "何が起きたかはここでは判定しない。夜のセッションが Web 検索で分類し、PLAYBOOK のイベント枠の規則で判断する。", ""]
    if len(ev) == 0:
        L.append("該当なし")
    else:
        L.append("| code | name | 市場 | 業種 | 急落日 | 経過日 | 当日 | 2日 | 出来高比 | 終値 | 急落前比 | 急落日比 | 最安(前比) | ATR% | 代金(億) | 25日線 | 75日線 | 52w高比 | 60日前比(急落前) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in ev.iterrows():
            L.append(f"| {r['code']} | {r['name']} | {r['market']} | {r['sector']} | {r['event_date']} | {r['days_since']} | {r['d0_ret_pct']:+.1f}% | "
                     f"{r['d2_ret_pct']:+.1f}% | {r['vol_x']} | {r['close']:.0f} | {r['vs_pre_pct']:+.1f}% | {r['since_d0_pct']:+.1f}% | {r['low_since_d0_pct']:+.1f}% | "
                     f"{r['atr_pct']} | {r['turn20_oku']} | {r['vs_sma25_pct']:+.1f}% | {r['vs_sma75_pct']:+.1f}% | {r['near_52w_high']} | {r['ret60_pre_pct']:+.1f}% |")
    L += ["", "注: 急落前比=急落直前の終値に対する現在値。最安(前比)=急落日以降の安値の急落前比。52w高比と60日前比は急落前の終値基準（急落前に健全なトレンドにあったか）。"]
    return "\n".join(L)


def run(offline: bool = True, out_root: Path = Path("outputs")) -> Path:
    quotes = pd.read_parquet(D.QUOTES)
    quotes["date"] = pd.to_datetime(quotes["date"])
    listed = D.load_listed()
    ev = find_events(quotes, listed)
    asof = quotes["date"].max()
    for d in (out_root / asof.strftime("%Y-%m-%d"), out_root / "latest") if not D.latest_is_newer(asof, out_root) else (out_root / asof.strftime("%Y-%m-%d"),):
        d.mkdir(parents=True, exist_ok=True)
        ev.to_csv(d / "events.csv", index=False)
        (d / "events.md").write_text(_md(asof, ev))
    print(f"asof {asof.date()}: events {len(ev)}")
    return out_root / asof.strftime("%Y-%m-%d")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    run(ap.parse_args().offline)
