"""週次レビュー（2026-10-23）: 「保有中に業種 rs60 が下位 1/3 に落ちたら手仕舞う」は、素の 12/-6/15 より EV を上げるか。

押し目シグナル（4 フラグ OK、3 年）を翌日寄り建てで 12/-6/15 を基準に、
  A: 基準（TP/SL/TIME のみ）
  B: 保有中に業種 rs60 三分位が bot になった日の引けで手仕舞い（TP/SL が先なら従う）
  C: B のうち、その時点で含み損のときだけ手仕舞い（含み益なら持ち続ける）
  D: 保有中に個別が 25 日線を終値で割った日の引けで手仕舞い
を比較する。
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/stock-analysis")
from src import data as D, signals as sg

OUT = "/home/user/stock-analysis/paper/research/2026-10-23_sector_exit"
TP, SL, MAX_HOLD, COST = 0.12, -0.06, 15, 0.002
q = pd.read_parquet(D.QUOTES); q["date"] = pd.to_datetime(q["date"])
listed = pd.read_parquet(D.LISTED)[["code", "sector33"]]
topix = pd.read_parquet(D.TOPIX).sort_values("date"); topix["date"] = pd.to_datetime(topix["date"])
topix["t60"] = topix["close"] / topix["close"].shift(60) - 1
reg = sg.topix_regime(topix).set_index("date")["regime_up"]
tmap = topix.set_index("date")["t60"]

# 1) 指標と rs60
frames = {}
rs_rows = []
for code, g in q.groupby("code"):
    if len(g) < 120:
        continue
    g = sg.add_indicators(g)
    g["rs60"] = (g["close"] / g["close"].shift(60) - 1).values - tmap.reindex(g["date"]).values
    frames[code] = g
    rs_rows.append(g[["code", "date", "rs60"]])
rs = pd.concat(rs_rows, ignore_index=True).merge(listed, on="code", how="left")
sec = rs.groupby(["date", "sector33"]).agg(rs60_med=("rs60", "median"), n=("code", "size")).reset_index()
sec = sec[sec["n"] >= 5]
def tercile(s):
    r = s.rank(ascending=False, method="first"); k = len(s)
    return np.where(r <= k / 3, "top", np.where(r > 2 * k / 3, "bot", "mid"))
sec["ter"] = sec.groupby("date")["rs60_med"].transform(tercile)
bot = sec[sec.ter == "bot"].set_index(["date", "sector33"]).index
sector_of = dict(zip(listed.code, listed.sector33))

def sim(g, bot_flag, below25):
    o, h, l, c = g["open"].values, g["high"].values, g["low"].values, g["close"].values
    idx = np.where(g["sig_pullback"].values & g["ok_liquidity"].values & g["ok_atr"].values & g["ok_extended"].values & g["ok_gap"].values)[0]
    n = len(g); busy = -1; rows = []
    for i in idx:
        e = i + 1
        if e >= n or e <= busy: continue
        ep = o[e]
        if not np.isfinite(ep) or ep <= 0: continue
        res = {}
        for variant in ("A", "B", "C", "D"):
            ret, exit_i, why = None, None, None
            for k in range(e, min(e + MAX_HOLD, n)):
                if k > e and o[k] / ep - 1 <= SL: ret, exit_i, why = o[k] / ep - 1, k, "SL_GAP"; break
                if l[k] / ep - 1 <= SL: ret, exit_i, why = SL, k, "SL"; break
                if k > e and o[k] / ep - 1 >= TP: ret, exit_i, why = o[k] / ep - 1, k, "TP"; break
                if h[k] / ep - 1 >= TP: ret, exit_i, why = TP, k, "TP"; break
                if k > e:
                    cur = c[k] / ep - 1
                    if variant == "B" and bot_flag[k]: ret, exit_i, why = cur, k, "SEC"; break
                    if variant == "C" and bot_flag[k] and cur < 0: ret, exit_i, why = cur, k, "SEC"; break
                    if variant == "D" and below25[k]: ret, exit_i, why = cur, k, "MA25"; break
            if ret is None:
                exit_i = min(e + MAX_HOLD, n) - 1; ret, why = c[exit_i] / ep - 1, "TIME"
            res[variant] = (ret - COST, why, exit_i - e + 1)
        busy = res["A"][2] + e - 1
        rows.append({"code": g["code"].iloc[0], "signal_date": g["date"].iloc[i], "regime": reg.get(g["date"].iloc[i], np.nan),
                     **{f"ret_{v}": res[v][0] for v in res}, **{f"why_{v}": res[v][1] for v in res}, **{f"hold_{v}": res[v][2] for v in res}})
    return rows

rows = []
for code, g in frames.items():
    s = sector_of.get(code)
    if s is None: continue
    keys = pd.MultiIndex.from_arrays([g["date"].values, [s] * len(g)])
    bot_flag = keys.isin(bot)
    below25 = (g["close"] < g["sma25"]).values
    rows += sim(g, bot_flag, below25)
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
lines = [f"# 業種 RS 崩れ / 25 日線割れでの手仕舞いは EV を上げるか（2026-10-23）\n\n押し目 4 フラグ OK、n={len(t)}（{t.signal_date.min().date()}〜{t.signal_date.max().date()}）、12/-6/15 基準。\n"]
def summ(d, label):
    out = []
    for v in "ABCD":
        r = d[f"ret_{v}"]
        out.append({"案": v, "n": len(r), "勝率%": round((r > 0).mean() * 100, 1), "EV%": round(r.mean() * 100, 2), "中央値%": round(r.median() * 100, 2),
                    "PF": round(r[r > 0].sum() / max(1e-9, -r[r < 0].sum()), 2), "平均保有": round(d[f"hold_{v}"].mean(), 1),
                    "発動率%": round((d[f"why_{v}"].isin(["SEC", "MA25"])).mean() * 100, 1)})
    df = pd.DataFrame(out)
    lines.append(f"\n### {label}\n"); lines.append(df.to_markdown(index=False)); print("\n==", label); print(df.to_string(index=False))
summ(t, "全期間")
summ(t[t.regime == 1], "地合い上で建玉")
summ(t[t.regime == 0], "地合い下で建玉")
fired = t[t.why_B == "SEC"]
summ(fired, "B が発動したトレードだけ（A との差が規則の効果）")
t.to_csv(OUT + ".csv", index=False)
open(OUT + "_tables.md", "w", encoding="utf-8").write("\n".join(lines))
print("done")
