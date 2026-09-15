"""個別銘柄の期待値分析。

  python -m src.analyze 7203            # 4桁でも5桁でも可
  python -m src.analyze 7203 --offline  # キャッシュのみ

出力: outputs/analysis/<code>_<asof>.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import backtest as bt
from . import signals as sg
from . import data as D


def norm_code(c: str) -> str:
    c = str(c).strip().upper().replace(".T", "")
    return c[:4] if len(c) == 5 and c.endswith("0") else c


def run(code: str, offline: bool, out_root: Path = Path("outputs/analysis")) -> Path:
    code = norm_code(code)
    if offline:
        q = pd.read_parquet(D.QUOTES)
        q = q[q["code"] == code]
        topix = pd.read_parquet(D.TOPIX)
    else:
        q = D.quotes_for(code)
        topix = D.load_topix() if D.TOPIX.exists() else D.update_topix()
    listed = D.load_listed()
    if q.empty:
        raise SystemExit(f"{code}: データなし")

    g = sg.add_indicators(q)
    regime = sg.topix_regime(topix)
    asof = g["date"].max()
    last = g.iloc[-1]
    name = listed.loc[listed["code"] == code, "name"].iloc[0] if len(listed) and (listed["code"] == code).any() else ""

    trades = pd.concat([bt.simulate(g, "sig_breakout"), bt.simulate(g, "sig_pullback")], ignore_index=True)
    trades = bt.attach_regime(trades, regime)
    L = [f"# {code} {name} 期待値分析 (as of {asof.date()})", ""]
    L += ["## 現状", f"- 終値 {last['close']:.0f} / 25日線乖離 {last['dev25']*100:+.1f}% / 75日線 {'上' if last['close'] > last['sma75'] else '下'}",
          f"- 出来高比(20日) {last['vol_ratio']:.2f} / ATR% {last['atr_pct']*100:.2f}% ({'OK' if last['ok_atr'] else '損切り幅がノイズ以下'})",
          f"- 20日平均売買代金 {last['turn20']/1e8:.1f}億円 ({'OK' if last['ok_liquidity'] else '流動性不足'})",
          f"- 当日騰落 {last['ret1d']*100:+.1f}% ({'OK' if last['ok_gap'] else 'ギャップ直後'}) / 25日線乖離 ({'OK' if last['ok_extended'] else '伸びすぎ'})",
          f"- 本日シグナル: breakout={bool(last['sig_breakout'])} pullback={bool(last['sig_pullback'])}",
          f"- 利確目標 {last['close']*1.05:.0f} / 損切り {last['close']*0.975:.0f}", ""]

    L.append("## この銘柄でのシグナル実績（過去データ全期間）")
    if len(trades) == 0:
        L.append("シグナル発生なし → 銘柄固有の期待値は算出不能。")
    else:
        L.append("| signal | n | 勝率 | 平均利 | 平均損 | EV/trade | 平均保有 |")
        L.append("|---|---|---|---|---|---|---|")
        for s in ("breakout", "pullback"):
            st = bt.stats(trades[trades["signal"] == s])
            if st["n"] == 0:
                L.append(f"| {s} | 0 | - | - | - | - | - |")
            else:
                L.append(f"| {s} | {st['n']} | {st['win_rate']*100:.1f}% | {st['avg_win']*100:+.2f}% | {st['avg_loss']*100:+.2f}% | {st['ev']*100:+.2f}% | {st['avg_hold']:.1f}日 |")
        L.append("")
        L.append("直近10トレード:")
        L.append("| signal日 | signal | 決済 | 損益 | 保有 | 地合い |")
        L.append("|---|---|---|---|---|---|")
        for _, t in trades.sort_values("signal_date").tail(10).iterrows():
            L.append(f"| {t['signal_date'].date()} | {t['signal']} | {t['exit']} | {t['ret']*100:+.2f}% | {t['hold_days']}日 | {t.get('regime_up', '-')} |")
        n = len(trades)
        L.append("")
        L.append(f"注: n={n}。{'20件未満なので統計としては弱い。全体EVと併読すること。' if n < 20 else ''}")
    L.append("")

    L.append("## ファンダメンタルズ")
    L.append("- 無料データ経路では決算日・信用残・進捗率は取れない。買う前に kabutan 等で次回決算日を確認すること（跨ぎ禁止）。")
    L.append("")
    L.append("## 判定材料の整理")
    flags = []
    if not last["ok_liquidity"]:
        flags.append("流動性不足 → フルベット不可")
    if not last["ok_atr"]:
        flags.append("ボラ過大 → -2.5%損切りが機能しにくい")
    if not last["ok_extended"]:
        flags.append("25日線乖離7%超 → +5%の余地が薄い")
    if not last["ok_gap"]:
        flags.append("当日6%超の急騰 → 翌日寄りは反落しやすい")
    if len(trades) and bt.stats(trades)["ev"] < 0:
        flags.append("銘柄固有EVがマイナス")
    L.append("- " + ("; ".join(flags) if flags else "機械的な除外条件には該当なし"))

    out_root.mkdir(parents=True, exist_ok=True)
    p = out_root / f"{code}_{asof.date()}.md"
    p.write_text("\n".join(L))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    p = run(a.code, a.offline)
    print(p.read_text())
