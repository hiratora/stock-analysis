"""国内の情報漏洩・サイバー・不祥事イベント後の株価経路。day0 = 公表日付近で最も下げた日（終値基準）。"""
import yfinance as yf, pandas as pd, numpy as np, sys, warnings
warnings.filterwarnings("ignore")
# (code, name, 公表/報道日, category, memo)  category: A=サイバー/漏洩  B=品質・検査不正  C=製品事故/健康被害  D=会計/ガバナンス  X=未確認(2026, 日付は自動検出)
EVENTS = [
    ("9468", "KADOKAWA", "2024-06-10", "A", "ランサムウェア、ニコニコ停止"),
    ("2502", "アサヒGHD", "2025-09-29", "A", "ランサムウェア、受注出荷停止"),
    ("2678", "アスクル", "2025-10-20", "A", "ランサムウェア、出荷停止"),
    ("7453", "良品計画", "2025-10-20", "A", "アスクル経由でネットストア停止"),
    ("4689", "LINEヤフー", "2023-11-27", "A", "44万件漏えい"),
    ("9783", "ベネッセHD", "2014-07-09", "A", "2,070万件漏えい（本業に直撃）"),
    ("8056", "BIPROGY", "2022-06-23", "A", "尼崎市USB紛失"),
    ("6952", "カシオ", "2024-10-11", "A", "ランサムウェア"),
    ("9531", "東京ガス", "2024-07-17", "A", "子会社416万件漏えい"),
    ("9201", "日本航空", "2024-12-26", "A", "DDoS、遅延"),
    ("4967", "小林製薬", "2024-03-22", "C", "紅麹 健康被害"),
    ("5406", "神戸製鋼所", "2017-10-10", "B", "データ改ざん"),
    ("7205", "日野自動車", "2022-03-04", "B", "エンジン認証不正"),
    ("6201", "豊田自動織機", "2024-01-29", "B", "エンジン認証不正"),
    ("3097", "物語コーポ", "2026-10-05", "A", "1,078万件漏えい（直近）"),
    ("6082", "ライドオンEX", "2026-10-13", "A", "650万件漏えいの可能性、注文サイト停止"),
    ("9433", "KDDI", "2026-06-23", "A", "メール1,223万件漏えい"),
    ("8601", "大和証券G", "2026-10-05", "A", "委託先不正アクセス、22万件"),
    ("7458", "第一興商", "2026-10-08", "A", "委託先マルウェア、872万件"),
    ("2871", "ニチレイ", "2026-07-13", "A", "ランサムウェア、出荷停止（7/24復旧）"),
    ("9008", "京王電鉄", "2026-09-26", "A", "ランサムウェア、グループ営業システム障害"),
]
H = [1, 3, 5, 10, 20, 40, 60]
rows = []
for code, name, date, cat, memo in EVENTS:
    try:
        df = yf.download(f"{code}.T", start="2013-06-01", auto_adjust=True, progress=False, threads=False)
    except Exception as e:
        print("fail", code, e); continue
    if df is None or df.empty: print("nodata", code); continue
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"]); c = df["Close"]; o = df["Open"]; v = df["Volume"]
    r = c.pct_change()
    if date == "auto":
        y = r[r.index >= "2026-01-01"]
        if y.empty: print("no2026", code); continue
        d0 = y.idxmin()
    else:
        d = pd.Timestamp(date); win = r[(r.index >= d - pd.Timedelta(days=2)) & (r.index <= d + pd.Timedelta(days=6))]
        if win.empty: print("nowin", code); continue
        d0 = win.idxmin()
    i0 = c.index.get_loc(d0); pre = c.iloc[i0 - 1]; c0 = c.iloc[i0]
    vol_ratio = v.iloc[i0] / v.iloc[i0-21:i0-1].mean() if i0 > 25 else np.nan
    row = {"code": code, "name": name, "cat": cat, "day0": d0.date(), "d0_ret%": round((c0/pre-1)*100, 1),
           "d-5→d0%": round((c0/c.iloc[i0-5]-1)*100, 1) if i0 >= 5 else np.nan, "vol_x": round(vol_ratio, 1)}
    fut = c.iloc[i0: i0+121]
    for h in H:
        row[f"+{h}d%"] = round((c.iloc[i0+h]/c0-1)*100, 1) if i0+h < len(c) else np.nan
    f40 = c.iloc[i0: i0+41]
    row["min40d%"] = round((f40.min()/c0-1)*100, 1); row["min_day"] = int(f40.values.argmin())
    row["max60d%"] = round((c.iloc[i0: i0+61].max()/c0-1)*100, 1)
    # 回復（公表前日の終値を上回った最初の日）
    rec = fut[fut >= pre]
    row["recover_day"] = int(c.index.get_loc(rec.index[0]) - i0) if len(rec) else ">120"
    # 翌日寄りで買った場合
    if i0+1 < len(o):
        ep = o.iloc[i0+1]
        for h in (10, 20, 40):
            row[f"buy@+1open→+{h}d%"] = round((c.iloc[i0+h]/ep-1)*100, 1) if i0+h < len(c) else np.nan
        # 翌寄り買いの最大ドローダウン（40日）
        row["dd_from_entry40%"] = round((c.iloc[i0+1:i0+41].min()/ep-1)*100, 1)
    row["memo"] = memo
    rows.append(row)
out = pd.DataFrame(rows)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 40)
print(out.drop(columns=["memo"]).to_string(index=False))
print("\n--- memo ---"); print(out[["code","name","day0","memo"]].to_string(index=False))
out.to_csv("/tmp/claude-0/-home-user-stock-analysis/f30d4514-2a12-5632-93d4-123cc0bdbc34/scratchpad/events/event_study.csv", index=False)
