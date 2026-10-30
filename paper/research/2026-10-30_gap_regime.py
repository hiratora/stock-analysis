"""週次レビュー（2026-10-30）: 押し目 4 フラグ OK（3 年）を翌日寄り建て 12/-6/15 で、
  (1) 翌日寄りのギャップ（建値 ÷ シグナル日終値 - 1）別の EV
  (2) 地合いの深さ（1306.T の 25 日線乖離）と 25 日線の向き別の EV
を見る。問い: ロート（+1.9% で寄って 3 日で SL）のような上ギャップ建ては不利か。地合い下の建玉は止めるべきか。
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/stock-analysis")
from src import data as D, signals as sg

OUT = "/home/user/stock-analysis/paper/research/2026-10-30_gap_regime"
TP, SL, MAX_HOLD, COST = 0.12, -0.06, 15, 0.002
q = pd.read_parquet(D.QUOTES); q["date"] = pd.to_datetime(q["date"])
topix = pd.read_parquet(D.TOPIX).sort_values("date"); topix["date"] = pd.to_datetime(topix["date"])
topix["sma25"] = topix["close"].rolling(25).mean()
topix["dev"] = topix["close"] / topix["sma25"] - 1
topix["sma_up"] = topix["sma25"] > topix["sma25"].shift(5)
topix["ret5"] = topix["close"] / topix["close"].shift(5) - 1
tinfo = topix.set_index("date")[["dev", "sma_up", "ret5"]]

def sim(g):
    o, h, l, c = g["open"].values, g["high"].values, g["low"].values, g["close"].values
    idx = np.where(g["sig_pullback"].values & g["ok_liquidity"].values & g["ok_atr"].values & g["ok_extended"].values & g["ok_gap"].values)[0]
    n = len(g); busy = -1; rows = []
    for i in idx:
        e = i + 1
        if e >= n or e <= busy: continue
        ep = o[e]
        if not np.isfinite(ep) or ep <= 0: continue
        ret, exit_i, why = None, None, None
        for k in range(e, min(e + MAX_HOLD, n)):
            if k > e and o[k] / ep - 1 <= SL: ret, exit_i, why = o[k] / ep - 1, k, "SL_GAP"; break
            if l[k] / ep - 1 <= SL: ret, exit_i, why = SL, k, "SL"; break
            if k > e and o[k] / ep - 1 >= TP: ret, exit_i, why = o[k] / ep - 1, k, "TP"; break
            if h[k] / ep - 1 >= TP: ret, exit_i, why = TP, k, "TP"; break
        if ret is None:
            exit_i = min(e + MAX_HOLD, n) - 1; ret, why = c[exit_i] / ep - 1, "TIME"
        busy = exit_i
        d = g["date"].iloc[i]
        ti = tinfo.loc[d] if d in tinfo.index else None
        rows.append({"code": g["code"].iloc[0], "signal_date": d, "gap": ep / c[i] - 1, "ret": ret - COST, "why": why, "hold": exit_i - e + 1,
                     "t_dev": None if ti is None else ti["dev"], "t_sma_up": None if ti is None else bool(ti["sma_up"]), "t_ret5": None if ti is None else ti["ret5"]})
    return rows

rows = []
for code, g in q.groupby("code"):
    if len(g) < 120: continue
    rows += sim(sg.add_indicators(g))
t = pd.DataFrame(rows).dropna(subset=["t_dev"])
t["regime"] = np.where(t.t_dev > 0, "上", "下")
pd.set_option("display.width", 250)
lines = [f"# 寄りギャップ別・地合いの深さ別の押し目 EV（2026-10-30）\n\n押し目 4 フラグ OK、n={len(t)}（{t.signal_date.min().date()}〜{t.signal_date.max().date()}）、翌日寄り建て 12/-6/15、往復 0.2%。\n"]
def summ(d, key, label):
    out = []
    for k, r in d.groupby(key, observed=True):
        x = r["ret"]
        out.append({key: k, "n": len(x), "勝率%": round((x > 0).mean() * 100, 1), "EV%": round(x.mean() * 100, 2), "中央値%": round(x.median() * 100, 2),
                    "PF": round(x[x > 0].sum() / max(1e-9, -x[x < 0].sum()), 2), "SL率%": round(r["why"].str.startswith("SL").mean() * 100, 1), "TP率%": round((r["why"] == "TP").mean() * 100, 1)})
    df = pd.DataFrame(out)
    lines.append(f"\n### {label}\n"); lines.append(df.to_markdown(index=False)); print("\n==", label); print(df.to_string(index=False))
t["gap_b"] = pd.cut(t.gap * 100, [-99, -2, -1, 0, 1, 2, 99], labels=["<-2%", "-2〜-1%", "-1〜0%", "0〜+1%", "+1〜+2%", ">+2%"])
summ(t, "gap_b", "寄りギャップ別（全期間）")
summ(t[t.regime == "上"], "gap_b", "寄りギャップ別（地合い上）")
summ(t[t.regime == "下"], "gap_b", "寄りギャップ別（地合い下）")
t["dev_b"] = pd.cut(t.t_dev * 100, [-99, -3, -2, -1, 0, 1, 2, 3, 99], labels=["<-3%", "-3〜-2%", "-2〜-1%", "-1〜0%", "0〜+1%", "+1〜+2%", "+2〜+3%", ">+3%"])
summ(t, "dev_b", "1306.T の 25 日線乖離別")
t["dn_type"] = np.where(t.regime == "上", "上", np.where(t.t_sma_up, "下・25日線は上向き", "下・25日線は下向き"))
summ(t, "dn_type", "地合い下を 1306.T 25 日線の向き（5 日前比）で分割")
t["t5_b"] = pd.cut(t.t_ret5 * 100, [-99, -2, 0, 2, 99], labels=["<-2%", "-2〜0%", "0〜+2%", ">+2%"])
summ(t[t.regime == "下"], "t5_b", "地合い下を 1306.T 直近 5 日騰落で分割")
t.to_csv(OUT + ".csv", index=False)
open(OUT + "_tables.md", "w", encoding="utf-8").write("\n".join(lines))
print("done")
