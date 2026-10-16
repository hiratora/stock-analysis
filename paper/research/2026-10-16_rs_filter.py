"""週次レビュー（2026-10-16）: 押し目（4フラグOK）の EV を「個別 RS」と「業種 RS（rs20 / rs60 のどちらで測るか）」で分ける。

問い1: 個別の rs60 が負の銘柄（JAL のような『出遅れの戻り』）は押し目として EV が低いか。
問い2: 業種 RS は rs20_med と rs60_med のどちらで上位1/3を選ぶと EV が高いか。
決済は 12/-6/15（backtest.simulate と同一）。
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/stock-analysis")
from src import data as D, signals as sg, backtest as bt

OUT = "/home/user/stock-analysis/paper/research/2026-10-16_rs_filter"
q = pd.read_parquet(D.QUOTES); q["date"] = pd.to_datetime(q["date"])
listed = pd.read_parquet(D.LISTED)[["code", "sector33"]]
topix = pd.read_parquet(D.TOPIX).sort_values("date"); topix["date"] = pd.to_datetime(topix["date"])
topix["t20"] = topix["close"] / topix["close"].shift(20) - 1
topix["t60"] = topix["close"] / topix["close"].shift(60) - 1
reg = sg.topix_regime(topix).set_index("date")["regime_up"]
tmap = topix.set_index("date")[["t20", "t60"]]

trades, rs_rows = [], []
for code, g in q.groupby("code"):
    if len(g) < 120:
        continue
    g = sg.add_indicators(g)
    c = g["close"]
    g["rs20"] = (c / c.shift(20) - 1).values - tmap.reindex(g["date"])["t20"].values
    g["rs60"] = (c / c.shift(60) - 1).values - tmap.reindex(g["date"])["t60"].values
    rs_rows.append(g[["code", "date", "rs20", "rs60"]])
    t = bt.simulate(g, "sig_pullback")
    if t.empty:
        continue
    t = t.merge(g[["date", "rs20", "rs60", "turn20"]].rename(columns={"date": "signal_date"}), on="signal_date", how="left")
    trades.append(t)
trades = pd.concat(trades, ignore_index=True)
rs = pd.concat(rs_rows, ignore_index=True).merge(listed, on="code", how="left")
trades = trades.merge(listed, on="code", how="left")
trades["regime"] = reg.reindex(trades["signal_date"]).values

# 業種 RS（日付×業種の中央値）と、その日の順位（上位1/3=top, 中=mid, 下位1/3=bot）
sec = rs.groupby(["date", "sector33"]).agg(rs20_med=("rs20", "median"), rs60_med=("rs60", "median"), n=("code", "size")).reset_index()
sec = sec[sec["n"] >= 5]
def tercile(s):
    r = s.rank(ascending=False, method="first"); k = len(s)
    return np.where(r <= k / 3, "top", np.where(r > 2 * k / 3, "bot", "mid"))
sec["sec20"] = sec.groupby("date")["rs20_med"].transform(tercile)
sec["sec60"] = sec.groupby("date")["rs60_med"].transform(tercile)
trades = trades.merge(sec[["date", "sector33", "sec20", "sec60"]].rename(columns={"date": "signal_date"}), on=["signal_date", "sector33"], how="left")

ok = trades[trades.ok_liquidity & trades.ok_atr & trades.ok_extended & trades.ok_gap].copy()
ok["stock_rs60"] = pd.cut(ok["rs60"], [-9, -0.10, 0, 0.10, 0.30, 9], labels=["<-10%", "-10..0", "0..10%", "10..30%", ">30%"])
ok["stock_rs20"] = pd.cut(ok["rs20"], [-9, -0.05, 0, 0.05, 0.15, 9], labels=["<-5%", "-5..0", "0..5%", "5..15%", ">15%"])
recent = ok[ok["signal_date"] >= ok["signal_date"].max() - pd.Timedelta(days=400)]

def table(d, key):
    g = d.groupby(key, observed=True)["ret"]
    out = pd.DataFrame({"n": g.size(), "勝率%": (g.apply(lambda s: (s > 0).mean()) * 100).round(1), "EV%": (g.mean() * 100).round(2),
                        "中央値%": (g.median() * 100).round(2)})
    pf = d.groupby(key, observed=True)["ret"].apply(lambda s: s[s > 0].sum() / max(1e-9, -s[s < 0].sum()))
    out["PF"] = pf.round(2)
    return out

pd.set_option("display.width", 250)
lines = []
def emit(title, df):
    lines.append(f"\n### {title}\n"); lines.append(df.to_markdown()); print(title); print(df.to_string())
lines.append(f"# 押し目（4フラグOK）の EV を RS で分ける（2026-10-16）\n\n決済 12/-6/15、往復 0.2%。全期間 n={len(ok)}（{ok.signal_date.min().date()}〜{ok.signal_date.max().date()}）、直近400日 n={len(recent)}。\n")
emit("問い1: 個別 rs60（対TOPIX、60日）別 — 全期間", table(ok, "stock_rs60"))
emit("問い1: 個別 rs60 別 — 直近400日", table(recent, "stock_rs60"))
emit("個別 rs20 別 — 全期間", table(ok, "stock_rs20"))
emit("問い2: 業種RS を rs20_med で三分 — 全期間", table(ok, "sec20"))
emit("問い2: 業種RS を rs60_med で三分 — 全期間", table(ok, "sec60"))
emit("問い2: 業種RS rs20 三分 — 直近400日", table(recent, "sec20"))
emit("問い2: 業種RS rs60 三分 — 直近400日", table(recent, "sec60"))
emit("業種 rs20 × rs60 の組み合わせ — 全期間", table(ok, ["sec20", "sec60"]))
emit("個別 rs60 正負 × 業種(rs20) 三分 — 全期間", table(ok.assign(pos=np.where(ok.rs60 > 0, "rs60>0", "rs60<=0")), ["pos", "sec20"]))
emit("地合い × 個別 rs60 正負 — 全期間", table(ok.assign(pos=np.where(ok.rs60 > 0, "rs60>0", "rs60<=0")), ["regime", "pos"]))
emit("決済理由の分布（個別 rs60 正負）", ok.assign(pos=np.where(ok.rs60 > 0, "rs60>0", "rs60<=0")).groupby(["pos", "exit"])["ret"].agg(n="size", EV=lambda s: round(s.mean() * 100, 2)).unstack("exit"))
ok.to_csv(OUT + ".csv", index=False)
open(OUT + "_tables.md", "w", encoding="utf-8").write("\n".join(lines))
print("done")
