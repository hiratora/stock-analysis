"""watchlist.txt にある銘柄をまとめて分析する（1行1コード、# はコメント）。
GitHub のスマホアプリから watchlist.txt に追記するだけで、次回実行時にレポートが出る。"""
import argparse
from pathlib import Path
from . import analyze

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--offline", action="store_true"); a = ap.parse_args()
    wl = Path("watchlist.txt")
    codes = [l.split("#")[0].strip() for l in wl.read_text().splitlines()] if wl.exists() else []
    codes = [c for c in codes if c]
    for c in codes:
        try:
            p = analyze.run(c, a.offline)
            latest = Path("outputs/analysis/latest"); latest.mkdir(parents=True, exist_ok=True)
            (latest / f"{analyze.norm_code(c)}.md").write_text(p.read_text())
            print("ok", c)
        except Exception as e:
            print("fail", c, e)
