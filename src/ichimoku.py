"""一目均衡表: 雲上抜け間近の銘柄リストと、雲上抜けシグナルの期待値。

  python -m src.ichimoku --offline            # outputs/<asof>/ichimoku.csv, .md（latest にも）
  python -m src.ichimoku --offline --backtest # 400日での EV 比較（押し目シグナルと同じ決済規則）

定義（日足）
  転換線 = (9日高値+9日安値)/2、基準線 = (26日高値+26日安値)/2
  先行スパンA = (転換線+基準線)/2 を26日先へ、先行スパンB = (52日高値+52日安値)/2 を26日先へ
  今日の雲 = 26日前に計算されたスパンA/B。遅行スパン好転 = 終値 > 26日前の終値。
  先行雲 = 今日計算したスパンA/B（26日後の雲の色）。
抽出「上抜けしそう」
  終値が雲の上限の下（まだ抜けていない）かつ 上限まで 5% 以内 かつ 雲の下限×0.99 以上（雲の中か直下）
  転換線 > 基準線、終値 > 基準線、遅行スパン好転、先行雲が陽転（スパンA > スパンB）、5日騰落 > 0
  流動性: 20日平均売買代金 3億以上、100株 50万円以下
場中に実行しない（未確定バーが混ざる）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D

MIN_TURN = 3e8
MAX_LOT = 500_000
MAX_DIST = 0.05
TP, SL, H = 0.12, -0.06, 15
COST = 0.002


def ichimoku(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True).copy()
    h, l, c = g["high"], g["low"], g["close"]
    g["tenkan"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    g["kijun"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    g["span_a_raw"] = (g["tenkan"] + g["kijun"]) / 2
    g["span_b_raw"] = (h.rolling(52).max() + l.rolling(52).min()) / 2
    g["span_a"] = g["span_a_raw"].shift(26)
    g["span_b"] = g["span_b_raw"].shift(26)
    # 先行スパンA/B のどちらかが未定義（履歴 78 日未満）なら雲も未定義にする。片方だけで雲を作ると厚さ 0 の偽の雲になる
    g["cloud_top"] = g[["span_a", "span_b"]].max(axis=1, skipna=False)
    g["cloud_bot"] = g[["span_a", "span_b"]].min(axis=1, skipna=False)
    g["chikou_up"] = c > c.shift(26)
    g["future_bull"] = g["span_a_raw"] > g["span_b_raw"]
    valid = g["cloud_top"].notna()
    g["above_cloud"] = (c > g["cloud_top"]) & valid
    g["cross_up"] = g["above_cloud"] & ~g["above_cloud"].shift(1).fillna(False).astype(bool) & valid.shift(1).fillna(False).astype(bool)
    g["dist_top"] = g["cloud_top"] / c - 1
    g["cloud_thick"] = (g["cloud_top"] - g["cloud_bot"]) / c
    g["ret5"] = c / c.shift(5) - 1
    g["turn20"] = (c * g["volume"]).rolling(20).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    g["atr_pct"] = tr.rolling(14).mean() / c
    g["pre_break"] = (valid & (~g["above_cloud"]) & (g["dist_top"] <= MAX_DIST) & (c >= g["cloud_bot"] * 0.99)
                      & (g["tenkan"] > g["kijun"]) & (c > g["kijun"]) & g["chikou_up"] & g["future_bull"] & (g["ret5"] > 0))
    g["break_confirmed"] = g["cross_up"] & (g["tenkan"] > g["kijun"]) & g["chikou_up"] & g["future_bull"]
    return g


def screen(quotes: pd.DataFrame, listed: pd.DataFrame) -> pd.DataFrame:
    names = listed.set_index("code") if len(listed) else pd.DataFrame()
    asof = quotes["date"].max()
    rows = []
    for code, g in quotes.groupby("code"):
        if len(g) < 90 or g["date"].max() != asof:
            continue
        g = ichimoku(g)
        r = g.iloc[-1]
        if not np.isfinite(r["cloud_top"]) or r["turn20"] < MIN_TURN or r["close"] * 100 > MAX_LOT:
            continue
        recent_break = bool(g["cross_up"].iloc[-3:].any())
        if not (r["pre_break"] or recent_break):
            continue
        meta = names.loc[code] if code in names.index else None
        rows.append({
            "code": code, "name": meta["name"] if meta is not None else "", "market": (meta["market"] if meta is not None else "").replace("（内国株式）", ""),
            "sector": meta["sector33"] if meta is not None else "",
            "status": "上抜け済(3日内)" if recent_break else "上抜け間近",
            "close": round(r["close"], 1), "cloud_top": round(r["cloud_top"], 1), "cloud_bot": round(r["cloud_bot"], 1),
            "dist_top_pct": round(r["dist_top"] * 100, 2), "cloud_thick_pct": round(r["cloud_thick"] * 100, 2),
            "tenkan": round(r["tenkan"], 1), "kijun": round(r["kijun"], 1),
            "tenkan_gt_kijun": bool(r["tenkan"] > r["kijun"]), "chikou_up": bool(r["chikou_up"]), "future_bull": bool(r["future_bull"]),
            "ret5_pct": round(r["ret5"] * 100, 1), "atr_pct": round(r["atr_pct"] * 100, 2), "turn20_oku": round(r["turn20"] / 1e8, 1),
            "lot_yen": round(r["close"] * 100),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["_o"] = out["status"].map({"上抜け間近": 0, "上抜け済(3日内)": 1})
        out = out.sort_values(["_o", "dist_top_pct", "turn20_oku"], ascending=[True, True, False]).drop(columns="_o").reset_index(drop=True)
    return out


def _md(asof, df: pd.DataFrame) -> str:
    L = [f"# 一目均衡表 雲上抜け候補 {asof.date()}", "",
         f"上抜け間近 = 終値が雲の上限の下 {MAX_DIST*100:.0f}% 以内（雲の中か直下）、転換線>基準線、終値>基準線、遅行スパン好転、先行雲が陽転、5日騰落プラス。"
         f"代金{MIN_TURN/1e8:.0f}億以上、100株{MAX_LOT//10000}万円以下。上抜け済は直近3日に雲の上限を終値で上抜けたもの。", ""]
    if len(df) == 0:
        L.append("該当なし")
    else:
        L.append("| code | name | 市場 | 業種 | 状態 | 終値 | 雲上限 | 雲下限 | 上限まで | 雲の厚さ | 転換>基準 | 遅行好転 | 先行雲陽転 | 5日 | ATR% | 代金(億) | 100株 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in df.iterrows():
            L.append(f"| {r['code']} | {r['name']} | {r['market']} | {r['sector']} | {r['status']} | {r['close']:.0f} | {r['cloud_top']:.0f} | {r['cloud_bot']:.0f} | "
                     f"{r['dist_top_pct']:+.1f}% | {r['cloud_thick_pct']:.1f}% | {'○' if r['tenkan_gt_kijun'] else '×'} | {'○' if r['chikou_up'] else '×'} | {'○' if r['future_bull'] else '×'} | "
                     f"{r['ret5_pct']:+.1f}% | {r['atr_pct']} | {r['turn20_oku']} | {r['lot_yen']/10000:.0f}万 |")
    L += ["", "注: 雲は26日前の値で描かれるため、上限までの距離は「今の価格が雲の上限に届くまでの上昇率」。先行雲は26日後の雲の色。"]
    return "\n".join(L)


# ---------- backtest ----------
def _fwd(a, n):
    out = np.full((len(a), n), np.nan)
    for k in range(n):
        out[: len(a) - k, k] = a[k:]
    return out


def _outcomes(g: pd.DataFrame, tp: float, sl: float, hold: int):
    o, h, l, c = (g[x].values.astype(float) for x in ("open", "high", "low", "close"))
    n = len(o)
    O, Hh, L, C = _fwd(o, hold), _fwd(h, hold), _fwd(l, hold), _fwd(c, hold)
    ep = o[:, None]; k = np.arange(hold)[None, :]
    gap_sl = (k > 0) & (O / ep - 1 <= sl); sl_hit = L / ep - 1 <= sl
    gap_tp = (k > 0) & (O / ep - 1 >= tp); tp_hit = Hh / ep - 1 >= tp
    flag = gap_sl | sl_hit | gap_tp | tp_hit
    price = np.where(gap_sl, O, np.where(sl_hit, ep * (1 + sl), np.where(gap_tp, O, ep * (1 + tp))))
    first = np.where(flag.any(axis=1), flag.argmax(axis=1), hold - 1)
    rows = np.arange(n); hit = flag[rows, first]
    exit_price = np.where(hit, price[rows, first], C[:, hold - 1])
    ret = exit_price / o - 1 - COST
    full = (rows + hold - 1 < n) & np.isfinite(o) & (o > 0)
    ret[~full] = np.nan
    return ret, first + 1


def backtest(quotes: pd.DataFrame, topix: pd.DataFrame) -> pd.DataFrame:
    from . import signals as sg
    reg = sg.topix_regime(topix).set_index("date")["regime_up"]
    exits = [(TP, SL, H), (0.20, -0.10, 20)]
    parts = []
    for code, g in quotes.groupby("code"):
        if len(g) < 120:
            continue
        g = ichimoku(g)
        g = sg.add_indicators(g)
        g = g.sort_values("date").reset_index(drop=True)
        sig = {"雲上抜け(交差日)": g["cross_up"].values, "雲上抜け+三役(転換>基準・遅行・先行雲)": g["break_confirmed"].values,
               "上抜け間近(本リストの条件)": g["pre_break"].values, "押し目(比較)": g["sig_pullback"].values}
        liquid = g["turn20"].values >= MIN_TURN
        for tp, sl, hold in exits:
            ret, hd = _outcomes(g, tp, sl, hold)
            r = np.r_[ret[1:], np.nan]; d = np.r_[hd[1:], 0]     # シグナル日の翌日寄り建て
            regime = reg.reindex(g["date"]).values
            for name, m in sig.items():
                idx = np.where(m & liquid & np.isfinite(r))[0]
                if len(idx) == 0:
                    continue
                parts.append(pd.DataFrame({"setup": name, "exit": f"{tp*100:.0f}/{sl*100:.0f}/{hold}", "ret": r[idx], "hold": d[idx], "regime": regime[idx]}))
    big = pd.concat(parts, ignore_index=True)
    rows = []
    for (name, ex), x in big.groupby(["setup", "exit"]):
        w = x["ret"] > 0
        gw, gl = x.loc[w, "ret"].sum(), -x.loc[~w, "ret"].sum()
        rows.append({"setup": name, "exit": ex, "n": len(x), "win%": round(w.mean() * 100, 1), "EV%": round(x["ret"].mean() * 100, 2),
                     "med%": round(x["ret"].median() * 100, 2), "PF": round(gw / gl, 2) if gl else np.inf,
                     "hold": round(x["hold"].mean(), 1), "月換算%": round(x["ret"].mean() * 21 / x["hold"].mean() * 100, 2),
                     "EV_up": round(x.loc[x["regime"] == 1, "ret"].mean() * 100, 2), "EV_dn": round(x.loc[x["regime"] == 0, "ret"].mean() * 100, 2)})
    return pd.DataFrame(rows).sort_values(["exit", "月換算%"], ascending=[True, False])


def run(out_root: Path = Path("outputs")) -> Path:
    quotes = pd.read_parquet(D.QUOTES); quotes["date"] = pd.to_datetime(quotes["date"])
    listed = D.load_listed()
    df = screen(quotes, listed)
    asof = quotes["date"].max()
    for d in (out_root / asof.strftime("%Y-%m-%d"), out_root / "latest"):
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / "ichimoku.csv", index=False)
        (d / "ichimoku.md").write_text(_md(asof, df))
    print(f"asof {asof.date()}: 上抜け間近 {(df['status'] == '上抜け間近').sum() if len(df) else 0} / 上抜け済 {(df['status'] != '上抜け間近').sum() if len(df) else 0}")
    return out_root / asof.strftime("%Y-%m-%d")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    a = ap.parse_args()
    if a.backtest:
        q = pd.read_parquet(D.QUOTES); q["date"] = pd.to_datetime(q["date"])
        pd.set_option("display.width", 220)
        print(backtest(q, pd.read_parquet(D.TOPIX)).to_string(index=False))
    else:
        run()
