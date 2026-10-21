"""候補の供給を増やすために、どのフィルタを緩めると EV がいくら落ちるかを測る（2026-10-21）。

押し目シグナル全件（4フラグを問わず）を 12/-6/15 でシミュレーションし、
ATR・25日乖離・代金・業種RS三分位・地合い ごとに EV を出す。
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/stock-analysis")
from src import data as D, signals as sg, backtest as bt

OUT = "/home/user/stock-analysis/paper/research/2026-10-21_supply"
q = pd.read_parquet(D.QUOTES); q["date"] = pd.to_datetime(q["date"])
listed = pd.read_parquet(D.LISTED)[["code", "sector33"]]
topix = pd.read_parquet(D.TOPIX).sort_values("date"); topix["date"] = pd.to_datetime(topix["date"])
topix["t60"] = topix["close"] / topix["close"].shift(60) - 1
reg = sg.topix_regime(topix).set_index("date")["regime_up"]
tmap = topix.set_index("date")[["t60"]]

trades, rs_rows = [], []
for code, g in q.groupby("code"):
    if len(g) < 120:
        continue
    g = sg.add_indicators(g)
    g["rs60"] = (g["close"] / g["close"].shift(60) - 1).values - tmap.reindex(g["date"])["t60"].values
    rs_rows.append(g[["code", "date", "rs60"]])
    t = bt.simulate(g, "sig_pullback")
    if t.empty:
        continue
    t = t.merge(g[["date", "rs60", "atr_pct", "dev25", "turn20", "ret1d"]].rename(columns={"date": "signal_date"}), on="signal_date", how="left")
    trades.append(t)
tr = pd.concat(trades, ignore_index=True).merge(listed, on="code", how="left")
tr["regime"] = reg.reindex(tr["signal_date"]).values
rs = pd.concat(rs_rows, ignore_index=True).merge(listed, on="code", how="left")
sec = rs.groupby(["date", "sector33"]).agg(rs60_med=("rs60", "median"), n=("code", "size")).reset_index()
sec = sec[sec["n"] >= 5]
def tercile(s):
    r = s.rank(ascending=False, method="first"); k = len(s)
    return np.where(r <= k / 3, "top", np.where(r > 2 * k / 3, "bot", "mid"))
sec["sec60"] = sec.groupby("date")["rs60_med"].transform(tercile)
tr = tr.merge(sec[["date", "sector33", "sec60"]].rename(columns={"date": "signal_date"}), on=["signal_date", "sector33"], how="left")
tr["turn_oku"] = tr["turn20"] / 1e8

def table(d, key):
    g = d.groupby(key, observed=True)["ret"]
    out = pd.DataFrame({"n": g.size(), "勝率%": (g.apply(lambda s: (s > 0).mean()) * 100).round(1), "EV%": (g.mean() * 100).round(2)})
    out["PF"] = d.groupby(key, observed=True)["ret"].apply(lambda s: s[s > 0].sum() / max(1e-9, -s[s < 0].sum())).round(2)
    return out

pd.set_option("display.width", 250)
lines = [f"# フィルタ別 EV（押し目全件、12/-6/15）2026-10-21\n\n全期間 n={len(tr)}（{tr.signal_date.min().date()}〜{tr.signal_date.max().date()}）\n"]
def emit(title, df):
    lines.append(f"\n### {title}\n"); lines.append(df.to_markdown()); print("\n==", title); print(df.to_string())

base = tr[tr.ok_liquidity & tr.ok_atr & tr.ok_extended & tr.ok_gap]
emit("基準: 4フラグ全OK × 地合い", table(base, "regime"))
emit("基準: 4フラグ全OK × 業種rs60三分位 × 地合い", table(base, ["regime", "sec60"]))
liq = tr[tr.ok_atr & tr.ok_extended & tr.ok_gap].copy()
liq["liq"] = pd.cut(liq["turn_oku"], [0, 2, 3, 5, 20, 1e9], labels=["<2億", "2-3億", "3-5億", "5-20億", ">20億"])
emit("代金の帯（他3フラグOK）", table(liq, "liq"))
atr = tr[tr.ok_liquidity & tr.ok_extended & tr.ok_gap].copy()
atr["atr"] = pd.cut(atr["atr_pct"], [0, 0.015, 0.025, 0.03, 0.035, 1], labels=["<1.5%", "1.5-2.5%", "2.5-3.0%", "3.0-3.5%", ">3.5%"])
emit("ATR の帯（他3フラグOK）", table(atr, "atr"))
emit("ATR の帯 × 地合い", table(atr, ["regime", "atr"]))
dev = tr[tr.ok_liquidity & tr.ok_atr & tr.ok_gap].copy()
dev["dev"] = pd.cut(dev["dev25"], [-1, 0.02, 0.04, 0.07, 0.10, 1], labels=["<2%", "2-4%", "4-7%", "7-10%", ">10%"])
emit("25日乖離の帯（他3フラグOK）", table(dev, "dev"))
gap = tr[tr.ok_liquidity & tr.ok_atr & tr.ok_extended].copy()
gap["gap"] = pd.cut(gap["ret1d"], [-1, 0, 0.03, 0.06, 0.09, 1], labels=["<=0", "0-3%", "3-6%", "6-9%", ">9%"])
emit("当日騰落の帯（他3フラグOK）", table(gap, "gap"))
b = base.copy(); b["rs60b"] = pd.cut(b["rs60"], [-9, -0.10, 0, 0.10, 0.30, 9], labels=["<-10%", "-10..0", "0..10%", "10..30%", ">30%"])
emit("4フラグOK × 地合い下 × 個別rs60", table(b[b.regime == 0], "rs60b"))
emit("4フラグOK × 地合い下 × 業種三分位 × 個別rs60>0", table(b[b.regime == 0].assign(pos=np.where(b[b.regime == 0].rs60 > 0, ">0", "<=0")), ["sec60", "pos"]))
# 1日あたりの候補数（4フラグOK、業種 bot 除外）
daily = base[base.sec60 != "bot"].groupby("signal_date").size()
lines.append(f"\n### 1日あたりの候補数（4フラグOK・業種下位1/3除外）\n\n平均 {daily.mean():.2f} 件/日、中央値 {daily.median():.0f}、ゼロの日の割合 {(daily.reindex(base.signal_date.unique()).fillna(0)==0).mean()*100:.0f}%\n")
print(lines[-1])
tr.to_csv(OUT + ".csv", index=False)
open(OUT + "_tables.md", "w", encoding="utf-8").write("\n".join(lines))
print("done")
