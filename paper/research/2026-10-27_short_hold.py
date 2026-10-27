"""決算シーズン対応（2026-10-27）: 押し目 4 フラグ OK を 12/-6 で、最大保有を 5/7/9/12/15 日に短縮したときの EV。
「決算前日まで N 営業日以上あれば max_hold をそこまで短縮して建玉可」の N を決めるため。
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/stock-analysis")
from src import data as D, signals as sg

OUT = "/home/user/stock-analysis/paper/research/2026-10-27_short_hold"
TP, SL, COST = 0.12, -0.06, 0.002
HOLDS = [3, 5, 7, 9, 12, 15]
q = pd.read_parquet(D.QUOTES); q["date"] = pd.to_datetime(q["date"])
topix = pd.read_parquet(D.TOPIX).sort_values("date"); topix["date"] = pd.to_datetime(topix["date"])
reg = sg.topix_regime(topix).set_index("date")["regime_up"]

def sim(g):
    o, h, l, c = g["open"].values, g["high"].values, g["low"].values, g["close"].values
    idx = np.where(g["sig_pullback"].values & g["ok_liquidity"].values & g["ok_atr"].values & g["ok_extended"].values & g["ok_gap"].values)[0]
    n = len(g); busy = -1; rows = []
    for i in idx:
        e = i + 1
        if e >= n or e <= busy: continue
        ep = o[e]
        if not np.isfinite(ep) or ep <= 0: continue
        if e + 15 >= n: continue
        res = {}
        for H in HOLDS:
            ret, exit_i = None, None
            for k in range(e, min(e + H, n)):
                if k > e and o[k] / ep - 1 <= SL: ret, exit_i = o[k] / ep - 1, k; break
                if l[k] / ep - 1 <= SL: ret, exit_i = SL, k; break
                if k > e and o[k] / ep - 1 >= TP: ret, exit_i = o[k] / ep - 1, k; break
                if h[k] / ep - 1 >= TP: ret, exit_i = TP, k; break
            if ret is None:
                exit_i = min(e + H, n) - 1; ret = c[exit_i] / ep - 1
            res[H] = ret - COST
        busy = e + 15 - 1
        rows.append({"code": g["code"].iloc[0], "signal_date": g["date"].iloc[i], "regime": reg.get(g["date"].iloc[i], np.nan), **{f"ret_{H}": res[H] for H in HOLDS}})
    return rows

rows = []
for code, g in q.groupby("code"):
    if len(g) < 120: continue
    rows += sim(sg.add_indicators(g))
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
lines = [f"# 保有日数を短縮したときの押し目 EV（2026-10-27）\n\n押し目 4 フラグ OK、n={len(t)}、12/-6、往復 0.2%。\n"]
def summ(d, label):
    out = []
    for H in HOLDS:
        r = d[f"ret_{H}"]
        out.append({"最大保有": H, "n": len(r), "勝率%": round((r > 0).mean() * 100, 1), "EV%": round(r.mean() * 100, 2), "中央値%": round(r.median() * 100, 2),
                    "PF": round(r[r > 0].sum() / max(1e-9, -r[r < 0].sum()), 2), "EV/日%": round(r.mean() * 100 / H, 3)})
    df = pd.DataFrame(out)
    lines.append(f"\n### {label}\n"); lines.append(df.to_markdown(index=False)); print("\n==", label); print(df.to_string(index=False))
summ(t, "全期間")
summ(t[t.regime == 1], "地合い上で建玉")
summ(t[t.regime == 0], "地合い下で建玉")
t.to_csv(OUT + ".csv", index=False)
open(OUT + "_tables.md", "w", encoding="utf-8").write("\n".join(lines))
print("done")
