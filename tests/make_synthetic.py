"""J-Quants が使えない環境で pipeline を検証するための合成データ。
ランダムウォーク + 時々の出来高スパイクを 40 銘柄 × 3 年分生成し、data/ に置く。
期待値の数字に意味はない。動作確認専用。
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(0)
DATA = Path("data"); DATA.mkdir(exist_ok=True)
dates = pd.bdate_range("2023-09-01", "2026-09-14")
rows, listed = [], []
for i in range(40):
    code = f"{1000 + i * 137:04d}"
    listed.append({"code": code, "name": f"SYN{i}", "market": "プライム", "sector33": "テスト"})
    p = 1000 * np.exp(np.cumsum(rng.normal(0.0003, 0.015 + 0.01 * (i % 3), len(dates))))
    vol = rng.lognormal(13.5, 0.4, len(dates))
    spikes = rng.random(len(dates)) < 0.04
    vol[spikes] *= 2.5
    p[spikes] *= 1 + rng.normal(0.02, 0.01, spikes.sum())
    o = p * (1 + rng.normal(0, 0.004, len(dates)))
    h = np.maximum(o, p) * (1 + abs(rng.normal(0, 0.006, len(dates))))
    l = np.minimum(o, p) * (1 - abs(rng.normal(0, 0.006, len(dates))))
    for d, oo, hh, ll, cc, vv in zip(dates, o, h, l, p, vol):
        rows.append({"code": code, "date": d, "open": oo, "high": hh, "low": ll, "close": cc,
                     "volume": vv, "turnover": vv * cc})
pd.DataFrame(rows).to_parquet(DATA / "quotes.parquet")
pd.DataFrame(listed).to_parquet(DATA / "listed.parquet")
t = 2500 * np.exp(np.cumsum(rng.normal(0.0002, 0.009, len(dates))))
pd.DataFrame({"date": dates, "open": t, "high": t, "low": t, "close": t}).to_parquet(DATA / "topix.parquet")
print("synthetic data written")
