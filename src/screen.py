"""日次スクリーニング。

  python -m src.screen            # J-Quants から更新して実行
  python -m src.screen --offline  # data/ のキャッシュだけで実行（検証用）

出力: outputs/YYYY-MM-DD/screen.md, screen.csv, stats.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from . import backtest as bt
from . import signals as sg
from . import data as D



def run(offline: bool, out_root: Path = Path("outputs")) -> Path:
    if offline:
        quotes = pd.read_parquet(D.QUOTES)
        topix = pd.read_parquet(D.TOPIX)
        listed = D.load_listed()
    else:
        listed = D.fetch_listed()
        quotes = D.update_quotes(listed["code"].tolist())
        topix = D.update_topix()

    ind = sg.add_indicators_all(quotes)
    regime = sg.topix_regime(topix)
    asof = ind["date"].max()
    reg_today = int(regime.loc[regime["date"] == asof, "regime_up"].iloc[0]) if (regime["date"] == asof).any() else -1

    # --- 期待値の母集団: 全銘柄・過去全期間のシグナル（フィルタ通過分のみ） ---
    trades = bt.run_all(ind)
    trades = bt.attach_regime(trades, regime)
    tr_f = trades[trades["ok_liquidity"] & trades["ok_atr"] & trades["ok_extended"] & trades["ok_gap"]] if len(trades) else trades
    glob = {s: bt.stats(tr_f[tr_f["signal"] == s]) for s in ("breakout", "pullback")}
    by_regime = bt.stats_by(tr_f, ["signal", "regime_up"])
    per_code = bt.stats_by(tr_f, ["code", "signal"])

    def cond_ev(sig):
        """今日の地合いでのシグナルEV。地合い別が薄い(n<30)ときは全体値。"""
        if len(by_regime):
            r = by_regime[(by_regime["signal"] == sig) & (by_regime["regime_up"] == reg_today)]
            if len(r) and r["n"].iloc[0] >= 30:
                return float(r["ev"].iloc[0]), float(r["win_rate"].iloc[0]), int(r["n"].iloc[0]), "地合い別"
        g = glob[sig]
        return g.get("ev", float("nan")), g.get("win_rate", float("nan")), g.get("n", 0), "全体"

    # --- 当日のシグナル ---
    today = ind[ind["date"] == asof]
    cand = today[(today["sig_breakout"] | today["sig_pullback"])].copy()
    cand["signal"] = cand.apply(lambda r: "breakout" if r["sig_breakout"] else "pullback", axis=1)

    rows = []
    for _, r in cand.iterrows():
        code = r["code"]
        meta = listed[listed["code"] == code]
        pc = per_code[(per_code["code"] == code) & (per_code["signal"] == r["signal"])]
        ev, wr, n, src = cond_ev(r["signal"])
        pc_ev = float(pc["ev"].iloc[0]) if len(pc) else float("nan")
        pc_n = int(pc["n"].iloc[0]) if len(pc) else 0
        rows.append({
            "code": code, "name": meta["name"].iloc[0] if len(meta) else "",
            "market": _short_market(meta["market"].iloc[0]) if len(meta) else "",
            "sector": meta["sector33"].iloc[0] if len(meta) and "sector33" in meta else "",
            "signal": r["signal"], "close": r["close"],
            "ret5d_pct": round(r["ret5d"] * 100, 2), "vol_ratio": round(r["vol_ratio"], 2),
            "atr_pct": round(r["atr_pct"] * 100, 2), "turn20_oku": round(r["turn20"] / 1e8, 1),
            "dev25_pct": round(r["dev25"] * 100, 2), "above75": bool(r["close"] > r["sma75"]) if pd.notna(r["sma75"]) else None,
            "hi52w": bool(r["close"] >= r["hi52w_prev"]) if pd.notna(r["hi52w_prev"]) else None,
            "ret1d_pct": round(r["ret1d"] * 100, 2),
            "flag_liquidity": "OK" if r["ok_liquidity"] else "NG",
            "flag_atr": "OK" if r["ok_atr"] else "NG",
            "flag_extended": "OK" if r["ok_extended"] else "NG",
            "flag_gap": "OK" if r["ok_gap"] else "NG",
            "ev_pct": round(ev * 100, 2), "win_rate": round(wr * 100, 1), "n_hist": n, "ev_source": src,
            "ev_stock_pct": round(pc_ev * 100, 2) if pc_n else None, "n_stock": pc_n,
            "tp_ref": round(r["close"] * 1.05, 1), "sl_ref": round(r["close"] * 0.975, 1),
        })
    res = pd.DataFrame(rows)
    if len(res):
        res["_ok"] = ((res["flag_liquidity"] == "OK") & (res["flag_atr"] == "OK")
                      & (res["flag_extended"] == "OK") & (res["flag_gap"] == "OK"))
        res = res.sort_values(["_ok", "ev_pct", "turn20_oku"], ascending=[False, False, False]).drop(columns="_ok")

    out = out_root / asof.strftime("%Y-%m-%d")
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "screen.csv", index=False)
    (out / "stats.json").write_text(json.dumps({
        "asof": str(asof.date()), "regime_up": reg_today, "global": glob,
        "by_regime": by_regime.to_dict(orient="records") if len(by_regime) else [],
        "n_trades_total": int(len(tr_f)),
    }, ensure_ascii=False, indent=2, default=float))
    (out / "screen.md").write_text(_md(asof, reg_today, glob, by_regime, res))
    latest = out_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for f in ("screen.md", "screen.csv", "stats.json"):
        (latest / f).write_text((out / f).read_text())
    return out


def _short_market(m: str) -> str:
    return m.replace("（内国株式）", "") if isinstance(m, str) else ""


def _md(asof, reg, glob, by_regime, res: pd.DataFrame) -> str:
    L = [f"# スクリーニング {asof.date()}", ""]
    L.append(f"地合い(TOPIX代用1306.T>25日線): {'上' if reg == 1 else '下' if reg == 0 else '不明'}")
    L.append("")
    L.append("## シグナル別 期待値（全銘柄・フィルタ通過・履歴全期間）")
    L.append("| signal | n | 勝率 | 平均利 | 平均損 | EV/trade | TP到達 | SL到達 | うちギャップ損 | 平均保有 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s, g in glob.items():
        if g.get("n", 0) == 0:
            L.append(f"| {s} | 0 | - | - | - | - | - | - | - | - |")
            continue
        L.append(f"| {s} | {g['n']} | {g['win_rate']*100:.1f}% | {g['avg_win']*100:+.2f}% | {g['avg_loss']*100:+.2f}% | "
                 f"{g['ev']*100:+.2f}% | {g['tp_rate']*100:.0f}% | {g['sl_rate']*100:.0f}% | {g['gap_loss_rate']*100:.0f}% | {g['avg_hold']:.1f}日 |")
    if len(by_regime):
        L.append("")
        L.append("## 地合い別（regime_up=1: TOPIXが25日線の上）")
        L.append("| signal | regime | n | 勝率 | EV/trade |")
        L.append("|---|---|---|---|---|")
        for _, r in by_regime.iterrows():
            L.append(f"| {r['signal']} | {int(r['regime_up']) if pd.notna(r['regime_up']) else '-'} | {int(r['n'])} | {r['win_rate']*100:.1f}% | {r['ev']*100:+.2f}% |")
    L.append("")
    n_ok = int(((res["flag_liquidity"] == "OK") & (res["flag_atr"] == "OK") & (res["flag_extended"] == "OK") & (res["flag_gap"] == "OK")).sum()) if len(res) else 0
    L.append(f"## 本日のシグナル {len(res)} 件（4フラグ全OK {n_ok} 件を先頭に、地合い別EV降順）")
    if len(res) == 0:
        L.append("該当なし")
    else:
        L.append("| code | name | 市場 | 業種 | signal | 終値 | 当日 | 5日 | 出来高比 | ATR% | 代金(億) | 25日乖離 | 75日 | 52w高 | 流動性 | ATR | 乖離 | ギャップ | EV | 勝率 | n | 根拠 | 銘柄EV(n) | 利確目安 | 損切目安 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in res.iterrows():
            stock_ev = f"{r['ev_stock_pct']:+.2f}%({r['n_stock']})" if r["n_stock"] else "-"
            L.append(f"| {r['code']} | {r['name']} | {r['market']} | {r['sector']} | {r['signal']} | {r['close']:.0f} | {r['ret1d_pct']:+.1f}% | {r['ret5d_pct']:+.1f}% | {r['vol_ratio']} | {r['atr_pct']} | {r['turn20_oku']} | "
                     f"{r['dev25_pct']:+.1f}% | {'上' if r['above75'] else '下' if r['above75'] is not None else '-'} | {'○' if r['hi52w'] else ''} | {r['flag_liquidity']} | {r['flag_atr']} | {r['flag_extended']} | {r['flag_gap']} | "
                     f"{r['ev_pct']:+.2f}% | {r['win_rate']}% | {r['n_hist']} | {r['ev_source']} | {stock_ev} | {r['tp_ref']} | {r['sl_ref']} |")
    L.append("")
    L.append("注: EV=勝率×平均利幅+(1-勝率)×平均損幅、1トレード・ポジション対比、往復コスト0.2%込み、ギャップダウン損失込み。"
             "主EVは「シグナル種別×今日の地合い」の全銘柄実績（n<30なら全体）。銘柄EVは参考値で、n が小さいほど信用しない。"
             "フラグ: 流動性=20日平均売買代金5億以上 / ATR=ATR14÷終値2.5%以下 / 乖離=25日線乖離7%以下 / ギャップ=当日騰落6%以下。"
             "利確・損切目安は終値基準。実際の建値は翌日寄りなので、寄り後に建値×1.05／×0.975で引き直す。次回決算日は無料データに無いので買う前に確認する。")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    p = run(a.offline)
    print(f"written: {p}")
    print((p / "screen.md").read_text())
