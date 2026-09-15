"""yfinance の MultiIndex 出力を模した DataFrame で normalize_yf / update_quotes を検証（ネットワーク不要）"""
import numpy as np, pandas as pd, os, tempfile
os.environ["SS_DATA_DIR"] = tempfile.mkdtemp()
from src import data as D

def fake_download(tickers, start, **kw):
    idx = pd.bdate_range(start, periods=30, tz="Asia/Tokyo")
    cols = pd.MultiIndex.from_product([tickers, ["Open", "High", "Low", "Close", "Volume"]])
    df = pd.DataFrame(np.random.rand(len(idx), len(cols)) * 1000 + 100, index=idx, columns=cols)
    return df

q = D.update_quotes(["7203", "6758"], downloader=fake_download)
assert set(q["code"]) == {"7203", "6758"} and len(q) == 60 and q["date"].dt.tz is None
q2 = D.update_quotes(["7203", "6758"], downloader=fake_download)
assert len(q2) >= 60 and not q2.duplicated(["code", "date"]).any()
t = D.update_topix(downloader=fake_download)
assert len(t) == 30
print("normalize/update ok", len(q2))
