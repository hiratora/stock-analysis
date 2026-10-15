"""雲上抜けを雲の厚さ（率・円・ATR倍）で分けて検証"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/stock-analysis")
from src import ichimoku as I, data as D, signals as sg
q = pd.read_parquet(D.QUOTES); q["date"] = pd.to_datetime(q["date"])
topix = pd.read_parquet(D.TOPIX); reg = sg.topix_regime(topix).set_index("date")["regime_up"]
rows = []
for code, g in q.groupby("code"):
    if len(g) < 120: continue
    g = I.ichimoku(g)
    c = g["close"].values; h = g["high"].values; l = g["low"].values; o = g["open"].values
    n = len(g)
    r12, d12 = I._outcomes(g, 0.12, -0.06, 15); r20, d20 = I._outcomes(g, 0.20, -0.10, 20)
    idx = np.where(g["cross_up"].values & (g["turn20"].values >= I.MIN_TURN))[0]
    for i in idx:
        e = i + 1
        if e + 20 >= n: continue
        ep = o[e]
        atr = g["atr_pct"].values[i] * c[i]
        rows.append({
            "code": code, "date": g["date"].iloc[i], "close": c[i],
            "thick_pct": g["cloud_thick"].values[i] * 100, "thick_yen": g["cloud_top"].values[i] - g["cloud_bot"].values[i],
            "thick_atr": (g["cloud_top"].values[i] - g["cloud_bot"].values[i]) / atr if atr > 0 else np.nan,
            "sanyaku": bool(g["break_confirmed"].values[i]), "regime": reg.get(g["date"].iloc[i], np.nan),
            "ret_12_6_15": r12[e], "ret_20_10_20": r20[e],
            "fwd10": c[e + 10] / ep - 1, "fwd20": c[e + 20] / ep - 1,
            "mfe20": h[e:e + 21].max() / ep - 1, "mae20": l[e:e + 21].min() / ep - 1,
            "mfe20_yen": h[e:e + 21].max() - ep, "fwd20_yen": c[e + 20] - ep,
            "back_in_cloud_10d": bool((c[e:e + 11] < g["cloud_top"].values[e:e + 11]).any()),
        })
d = pd.DataFrame(rows).dropna(subset=["thick_atr"])
print(f"雲上抜け（代金3億+）n={len(d)}  期間 {d.date.min().date()}〜{d.date.max().date()}")
pd.set_option("display.width", 250)
def summarize(d, col, bins, labels):
    d = d.copy(); d["bucket"] = pd.cut(d[col], bins=bins, labels=labels)
    out = []
    for b, x in d.groupby("bucket", observed=True):
        w = x["ret_12_6_15"] > 0; w2 = x["ret_20_10_20"] > 0
        out.append({"厚さ": b, "n": len(x), "中央値株価": round(x["close"].median()), "厚さ中央値(円)": round(x["thick_yen"].median(), 1), "厚さ中央値(%)": round(x["thick_pct"].median(), 2),
                    "勝率12/-6": round(w.mean()*100, 1), "EV12/-6%": round(x["ret_12_6_15"].mean()*100, 2),
                    "勝率20/-10": round(w2.mean()*100, 1), "EV20/-10%": round(x["ret_20_10_20"].mean()*100, 2),
                    "+10d%": round(x["fwd10"].mean()*100, 2), "+20d%": round(x["fwd20"].mean()*100, 2), "+20d中央%": round(x["fwd20"].median()*100, 2),
                    "最大上昇20d%": round(x["mfe20"].mean()*100, 1), "最大下落20d%": round(x["mae20"].mean()*100, 1),
                    "+20d円(中央)": round(x["fwd20_yen"].median(), 1), "雲に戻る率10d": round(x["back_in_cloud_10d"].mean()*100)})
    return pd.DataFrame(out)
print("\n=== 率（雲の厚さ ÷ 株価）===")
print(summarize(d, "thick_pct", [0, 1, 2, 4, 8, 100], ["<1%", "1-2%", "2-4%", "4-8%", ">8%"]).to_string(index=False))
print("\n=== ATR倍（雲の厚さ ÷ ATR14 = 1日の値動き何日分か）===")
print(summarize(d, "thick_atr", [0, 0.5, 1, 2, 4, 100], ["<0.5ATR", "0.5-1", "1-2", "2-4", ">4ATR"]).to_string(index=False))
print("\n=== 値幅（円）===")
qs = d["thick_yen"].quantile([0.2, 0.4, 0.6, 0.8]).round(1).tolist()
print("五分位の境界(円):", qs)
print(summarize(d, "thick_yen", [-1] + qs + [1e9], [f"〜{qs[0]}", f"{qs[0]}-{qs[1]}", f"{qs[1]}-{qs[2]}", f"{qs[2]}-{qs[3]}", f"{qs[3]}〜"]).to_string(index=False))
print("\n=== 率 × 三役の有無（EV 20/-10/20）===")
d["bucket"] = pd.cut(d["thick_pct"], [0, 1, 2, 4, 8, 100], labels=["<1%", "1-2%", "2-4%", "4-8%", ">8%"])
print(d.groupby(["bucket", "sanyaku"], observed=True).agg(n=("ret_20_10_20", "size"), EV20=("ret_20_10_20", lambda s: round(s.mean()*100, 2)), EV12=("ret_12_6_15", lambda s: round(s.mean()*100, 2)), fwd20=("fwd20", lambda s: round(s.mean()*100, 2))).to_string())
print("\n=== 率 × 地合い（EV 20/-10/20）===")
print(d.groupby(["bucket", "regime"], observed=True).agg(n=("ret_20_10_20", "size"), EV20=("ret_20_10_20", lambda s: round(s.mean()*100, 2))).to_string())
d.to_csv("/tmp/claude-0/-home-user-stock-analysis/f30d4514-2a12-5632-93d4-123cc0bdbc34/scratchpad/kumo_thickness.csv", index=False)
