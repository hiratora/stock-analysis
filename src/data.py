"""無料データ層（yfinance）。

- 銘柄一覧: JPX公開の東証上場銘柄一覧（data_j.xlsx）。内国株式のプライム/スタンダード/グロースに絞る
- 日足: yfinance（<code>.T）。分割調整済み(auto_adjust=True)。data/quotes.parquet に増分キャッシュ
- 地合い: TOPIX連動ETF 1306.T の終値を TOPIX の代用として data/topix.parquet に保存
- 決算発表予定日: 無料経路では取れない。候補に残った銘柄だけ手動で確認する（README参照）

このモジュール内でネットワークに触るのは fetch_listed / update_quotes / update_topix の3つだけ。
"""
from __future__ import annotations

import datetime as dt
import io
import os
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(os.environ.get("SS_DATA_DIR", "data"))
QUOTES = DATA_DIR / "quotes.parquet"
TOPIX = DATA_DIR / "topix.parquet"
LISTED = DATA_DIR / "listed.parquet"

JPX_LIST_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
TOPIX_PROXY = "1306.T"
MARKETS = ("プライム（内国株式）", "スタンダード（内国株式）", "グロース（内国株式）")
CHUNK = 150          # yfinance に一度に渡す銘柄数
LOOKBACK_DAYS = 3 * 365


# ---------- listed ----------
def fetch_listed() -> pd.DataFrame:
    r = requests.get(JPX_LIST_URL, timeout=60)
    r.raise_for_status()
    raw = pd.read_excel(io.BytesIO(r.content))
    df = pd.DataFrame({
        "code": raw["コード"].astype(str).str.zfill(4),
        "name": raw["銘柄名"], "market": raw["市場・商品区分"], "sector33": raw["33業種区分"],
        "scale": raw.get("規模区分", ""),
    })
    df = df[df["market"].isin(MARKETS)].reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LISTED)
    return df


def load_listed() -> pd.DataFrame:
    if LISTED.exists():
        return pd.read_parquet(LISTED)
    return pd.DataFrame(columns=["code", "name", "market", "sector33", "scale"])


# ---------- quotes ----------
def _download(tickers: list[str], start: dt.date, downloader=None) -> pd.DataFrame:
    """yfinance の MultiIndex 出力を long 形式に正規化。downloader は検証用の差し替え口。"""
    import yfinance as yf
    dl = downloader or yf.download
    frames = []
    for i in range(0, len(tickers), CHUNK):
        part = tickers[i:i + CHUNK]
        for attempt in range(3):
            try:
                raw = dl(part, start=start.isoformat(), auto_adjust=True, group_by="ticker",
                         threads=True, progress=False)
                break
            except Exception as e:  # noqa
                if attempt == 2:
                    raise
                time.sleep(10 * (attempt + 1))
        frames.append(normalize_yf(raw, part))
        time.sleep(1.0)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_yf(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["code", "date", "open", "high", "low", "close", "volume", "turnover"])
    single = not isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        g = raw if single else (raw[t] if t in raw.columns.get_level_values(0) else None)
        if g is None or g.empty:
            continue
        g = g.dropna(subset=["Close"])
        if g.empty:
            continue
        out = pd.DataFrame({
            "code": t.replace(".T", ""), "date": pd.to_datetime(g.index).tz_localize(None),
            "open": g["Open"].values, "high": g["High"].values, "low": g["Low"].values,
            "close": g["Close"].values, "volume": g["Volume"].values,
        })
        out["turnover"] = out["close"] * out["volume"]
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["code", "date", "open", "high", "low", "close", "volume", "turnover"])


def update_quotes(codes: list[str], downloader=None) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cached = pd.read_parquet(QUOTES) if QUOTES.exists() else pd.DataFrame()
    if cached.empty:
        start = dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)
    else:
        # 直近5営業日を取り直して、Yahoo側の遅延・修正を吸収
        start = cached["date"].max().date() - dt.timedelta(days=7)
    new = _download([f"{c}.T" for c in codes], start, downloader)
    if not new.empty:
        cached = pd.concat([cached, new], ignore_index=True)
        cached = cached.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"])
        # 3年より古い行は落として肥大化を防ぐ
        cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS))
        cached = cached[cached["date"] >= cutoff]
        cached.to_parquet(QUOTES)
    return cached


def quotes_for(code: str, downloader=None) -> pd.DataFrame:
    """単一銘柄。キャッシュがあればそれを、なければ取得。"""
    if QUOTES.exists():
        q = pd.read_parquet(QUOTES)
        q = q[q["code"] == code]
        if len(q) > 100:
            return q
    start = dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)
    return _download([f"{code}.T"], start, downloader)


# ---------- TOPIX proxy ----------
def update_topix(downloader=None) -> pd.DataFrame:
    start = dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)
    df = _download([TOPIX_PROXY], start, downloader)
    if df.empty:
        raise RuntimeError("TOPIX代用(1306.T)が取得できない")
    df = df[["date", "open", "high", "low", "close"]].sort_values("date")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TOPIX)
    return df


def load_topix() -> pd.DataFrame:
    return pd.read_parquet(TOPIX)
