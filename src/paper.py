"""仮想取引（ペーパートレード）の約定・決済・記録。

  python -m src.paper settle   # data/quotes.parquet の最新日まで注文を約定し、保有の決済を判定し、記録を更新
  python -m src.paper report   # 現状の markdown を表示

ファイル（paper/）
  orders.csv     Claude が書く注文。fill_after の翌営業日の寄りで約定する（終値確定後に出した注文、の意）
  positions.csv  保有中
  trades.csv     決済済み
  equity.csv     日次の資産推移
  state.json     現金・開始資金

決済規則（注文ごとに指定、空欄は既定値 DEFAULTS）
  sl_pct / sl_atr      初期ストップ（建値比 % または ATR14 の倍数）
  trail_pct / trail_atr トレーリングストップ（エントリー後の最高終値からの下落率 / ATR 倍数）。引け後に更新し翌日から有効
  be_pct               建値撤退: 最高終値が建値×(1+be_pct) を超えたらストップを建値以上に引き上げ、トレールもそこから開始
  tp_pct               利確（空欄なら利確せず引っ張る）
  ma_exit              10 or 25: 終値がその移動平均を割ったら引けで撤退（エントリー3日目以降）
  max_hold             最大保有営業日。到達したら引け
判定順序は毎日: 寄りがストップ以下→寄り値 / 安値がストップ以下→ストップ値 / 寄りが利確以上→寄り値 / 高値が利確以上→利確値 / 引け後にストップ更新。
コストは往復 0.2% を片道 0.1% ずつ約定値に乗せる。
データは終値確定後のものだけを使う前提。場中に回すと未確定バーで判定してしまう。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest as bt
from . import data as D

PAPER_DIR = Path(os.environ.get("SS_PAPER_DIR", "paper"))
ORDERS, POSITIONS, TRADES, EQUITY, STATE = (PAPER_DIR / f for f in
                                            ("orders.csv", "positions.csv", "trades.csv", "equity.csv", "state.json"))
START_CAPITAL = 2_000_000
MONTHLY_GOAL = 100_000
HALF_COST = bt.COST / 2
# 既定の決済規則（paper/research の探索結果に基づく。変更は PLAYBOOK の変更履歴に残す）
DEFAULTS = {"sl_pct": -0.06, "sl_atr": None, "tp_pct": 0.12, "trail_pct": None, "trail_atr": None,
            "be_pct": None, "ma_exit": None, "max_hold": 15}
EXIT_PARAMS = ["sl_pct", "sl_atr", "tp_pct", "trail_pct", "trail_atr", "be_pct", "ma_exit", "max_hold"]

ORDER_COLS = ["id", "fill_after", "code", "side", "qty", "tp_pct", "sl_pct", "max_hold", "trail_pct", "be_pct",
              "sl_atr", "trail_atr", "ma_exit", "reason", "status", "fill_date", "fill_price", "note"]
POS_COLS = ["code", "name", "entry_date", "entry_price", "qty", "tp_price", "sl_price", "stop_price", "high_close",
            "atr0", "trail_pct", "trail_atr", "be_pct", "ma_exit", "max_hold", "hold_days",
            "last_date", "last_close", "unrealized_pct", "unrealized_yen", "order_id", "reason"]
TRADE_COLS = ["code", "name", "entry_date", "entry_price", "exit_date", "exit_price", "qty", "ret_pct", "pnl_yen",
              "exit_reason", "hold_days", "order_id", "reason"]
EQ_COLS = ["date", "cash", "position_value", "equity", "realized_cum", "n_positions"]


# ---------- io ----------
def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, dtype=str)      # 空セルは NaN。数値は使う側で float()/int() に変換する
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df.astype(object)
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.strip().str.zfill(4)
    return df[cols]


def _write(df: pd.DataFrame, path: Path) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"cash": START_CAPITAL, "start_capital": START_CAPITAL, "start_date": None}


def save_state(s: dict) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=2))


# ---------- core ----------
def _bars(quotes: pd.DataFrame, code: str) -> pd.DataFrame:
    g = quotes[quotes["code"] == code].sort_values("date").reset_index(drop=True)
    return g


def _f(v, default=None):
    """csv の文字列/NaN を float か None に"""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None"):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _atr0(g: pd.DataFrame, e: int) -> float:
    """エントリー前日（シグナル日）時点の ATR14"""
    h, l, c = g["high"], g["low"], g["close"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    a = tr.rolling(14).mean()
    v = a.iloc[e - 1] if e >= 1 else np.nan
    return float(v) if np.isfinite(v) else float(a.dropna().iloc[-1]) if a.notna().any() else 0.0


def _exit_walk(g: pd.DataFrame, e: int, ep: float, prm: dict, st: dict, start_k: int, sell_k: int | None = None):
    """経路依存の決済判定。prm = 決済規則、st = {"stop", "hi", "sl0"}（更新される）。
    start_k から走査し (exit_index, exit_price, reason) か None。sell_k は手動売りが約定するバー。"""
    o, h, l, c = g["open"].values, g["high"].values, g["low"].values, g["close"].values
    n = len(g)
    max_hold = int(prm["max_hold"])
    tp = ep * (1 + prm["tp_pct"]) if prm.get("tp_pct") is not None else np.inf
    ma = None
    if prm.get("ma_exit"):
        ma = g["close"].rolling(int(prm["ma_exit"])).mean().values
    end = min(e + max_hold, n)
    if sell_k is not None:
        sell_k = max(sell_k, start_k)
        end = min(end, sell_k)
    for k in range(start_k, end):
        stop, sl0 = st["stop"], st["sl0"]
        trailing = stop > sl0 + 1e-9
        if k > e and o[k] <= stop:
            return k, o[k], "TRAIL_GAP" if trailing else "SL_GAP"
        if l[k] <= stop:
            return k, stop, "TRAIL" if trailing else "SL"
        if k > e and o[k] >= tp:
            return k, o[k], "TP"
        if h[k] >= tp:
            return k, tp, "TP"
        # 引け後の更新（翌日から有効）
        st["hi"] = max(st["hi"], c[k])
        if ma is not None and k > e + 2 and np.isfinite(ma[k]) and c[k] < ma[k]:
            return k, c[k], "MA"
        hi = st["hi"]
        be = prm.get("be_pct")
        armed = be is None or hi >= ep * (1 + be)
        if prm.get("trail_pct") is not None and armed:
            st["stop"] = max(st["stop"], hi * (1 - prm["trail_pct"]), ep if be is not None else -np.inf)
        if prm.get("trail_atr") is not None and armed and st.get("atr0"):
            st["stop"] = max(st["stop"], hi - prm["trail_atr"] * st["atr0"], ep if be is not None else -np.inf)
        if k == e + max_hold - 1:
            return k, c[k], "TIME"
    if sell_k is not None and sell_k < e + max_hold and sell_k <= n - 1:
        return sell_k, o[sell_k], "MANUAL"
    return None


def _params_from_order(od) -> dict:
    prm = {}
    for k in EXIT_PARAMS:
        v = _f(od.get(k)) if hasattr(od, "get") else None
        prm[k] = v if v is not None else DEFAULTS[k]
    if prm["sl_pct"] is None and prm["sl_atr"] is None:
        prm["sl_pct"] = -0.04
    prm["max_hold"] = int(prm["max_hold"])
    return prm


def _params_from_position(p) -> dict:
    return {"sl_pct": None, "sl_atr": None, "tp_pct": (_f(p["tp_price"]) / float(p["entry_price"]) - 1) if _f(p["tp_price"]) else None,
            "trail_pct": _f(p["trail_pct"]), "trail_atr": _f(p["trail_atr"]), "be_pct": _f(p["be_pct"]),
            "ma_exit": _f(p["ma_exit"]), "max_hold": int(float(p["max_hold"]))}


def settle(quotes: pd.DataFrame | None = None, listed: pd.DataFrame | None = None) -> str:
    quotes = pd.read_parquet(D.QUOTES) if quotes is None else quotes
    listed = D.load_listed() if listed is None else listed
    quotes = quotes.copy()
    quotes["date"] = pd.to_datetime(quotes["date"])
    names = dict(zip(listed["code"], listed["name"])) if len(listed) else {}
    asof = quotes["date"].max()

    orders = _read(ORDERS, ORDER_COLS)
    pos = _read(POSITIONS, POS_COLS)
    trades = _read(TRADES, TRADE_COLS)
    equity = _read(EQUITY, EQ_COLS)
    state = load_state()
    if not state.get("start_date"):
        state["start_date"] = str(asof.date())
    cash = float(state["cash"])
    log = []

    def close_position(p: pd.Series, k_date, price: float, why: str, hold_days: int):
        nonlocal cash, trades
        px = price * (1 - HALF_COST)
        qty = int(p["qty"])
        pnl = (px - float(p["entry_price"])) * qty
        cash += px * qty
        trades = pd.concat([trades, pd.DataFrame([{
            "code": p["code"], "name": p["name"], "entry_date": p["entry_date"], "entry_price": round(float(p["entry_price"]), 2),
            "exit_date": str(pd.Timestamp(k_date).date()), "exit_price": round(px, 2), "qty": qty,
            "ret_pct": round((px / float(p["entry_price"]) - 1) * 100, 2), "pnl_yen": round(pnl),
            "exit_reason": why, "hold_days": hold_days, "order_id": p["order_id"], "reason": p["reason"],
        }])], ignore_index=True)
        log.append(f"決済 {p['code']} {p['name']} {why} {px:.1f} ({(px / float(p['entry_price']) - 1) * 100:+.2f}%, {pnl:+,.0f}円)")

    def pending_sell(code: str, g: pd.DataFrame):
        """code に pending の売りがあれば (order_index, 約定バー index or None)"""
        m = orders[(orders["status"] == "pending") & (orders["side"] == "sell") & (orders["code"] == code)]
        if m.empty:
            return None, None
        i = m.index[0]
        nxt = g[g["date"] > pd.Timestamp(orders.loc[i, "fill_after"])]
        return i, (int(nxt.index[0]) if not nxt.empty else None)

    def mark_sell_filled(i, k_date, price):
        orders.loc[i, ["status", "fill_date", "fill_price"]] = ["filled", str(pd.Timestamp(k_date).date()), round(price * (1 - HALF_COST), 2)]

    # 2) 保有中の決済判定（last_date の翌バーから）
    keep = []
    for _, p in pos.iterrows():
        g = _bars(quotes, p["code"])
        if g.empty or not (g["date"] == pd.Timestamp(p["entry_date"])).any():
            keep.append(p); continue
        e = int(g.index[g["date"] == pd.Timestamp(p["entry_date"])][0])
        last_idx = int(g.index[g["date"] == pd.Timestamp(p["last_date"])][0]) if (g["date"] == pd.Timestamp(p["last_date"])).any() else e - 1
        start_k = last_idx + 1
        if start_k > len(g) - 1:
            keep.append(p); continue
        ep = float(p["entry_price"])
        prm = _params_from_position(p)
        st = {"stop": _f(p["stop_price"], _f(p["sl_price"])), "sl0": _f(p["sl_price"]), "hi": _f(p["high_close"], ep), "atr0": _f(p["atr0"], 0.0)}
        si, sk = pending_sell(p["code"], g)
        r = _exit_walk(g, e, ep, prm, st, start_k, sk)
        if r is not None:
            k, price, why = r
            close_position(p, g["date"].iloc[k], price, why, k - e + 1)
            if why == "MANUAL":
                mark_sell_filled(si, g["date"].iloc[k], price)
        else:
            k = len(g) - 1
            p = p.copy()
            p["stop_price"] = round(st["stop"], 2); p["high_close"] = round(st["hi"], 2)
            p["hold_days"] = k - e + 1
            p["last_date"] = str(g["date"].iloc[k].date())
            p["last_close"] = round(float(g["close"].iloc[k]), 2)
            p["unrealized_pct"] = round((float(g["close"].iloc[k]) / ep - 1) * 100, 2)
            p["unrealized_yen"] = round((float(g["close"].iloc[k]) - ep) * int(p["qty"]))
            keep.append(p)
    pos = pd.DataFrame(keep, columns=POS_COLS) if keep else pd.DataFrame(columns=POS_COLS)

    # 3) 買い注文 → 翌営業日の寄りで約定し、その日から決済判定
    for i, od in orders.iterrows():
        if od["status"] != "pending" or od["side"] != "buy":
            continue
        g = _bars(quotes, od["code"])
        after = pd.Timestamp(od["fill_after"])
        nxt = g[g["date"] > after]
        if nxt.empty:
            continue
        if (pos["code"] == od["code"]).any():
            orders.loc[i, ["status", "note"]] = ["rejected", "既に保有中"]
            continue
        e = int(nxt.index[0])
        ep = float(g["open"].iloc[e]) * (1 + HALF_COST)
        qty = int(od["qty"])
        if not np.isfinite(ep) or ep <= 0 or qty <= 0:
            orders.loc[i, ["status", "note"]] = ["rejected", "寄り値なし"]
            continue
        if ep * qty > cash:
            orders.loc[i, ["status", "note"]] = ["rejected", f"現金不足 {cash:,.0f} < {ep * qty:,.0f}"]
            continue
        prm = _params_from_order(od)
        atr0 = _atr0(g, e)
        sl0 = ep * (1 + prm["sl_pct"]) if prm["sl_pct"] is not None else ep - prm["sl_atr"] * atr0
        cash -= ep * qty
        orders.loc[i, ["status", "fill_date", "fill_price"]] = ["filled", str(g["date"].iloc[e].date()), round(ep, 2)]
        p = pd.Series({
            "code": od["code"], "name": names.get(od["code"], ""), "entry_date": str(g["date"].iloc[e].date()),
            "entry_price": round(ep, 2), "qty": qty,
            "tp_price": round(ep * (1 + prm["tp_pct"]), 2) if prm["tp_pct"] is not None else "",
            "sl_price": round(sl0, 2), "stop_price": round(sl0, 2), "high_close": round(ep, 2), "atr0": round(atr0, 2),
            "trail_pct": prm["trail_pct"] if prm["trail_pct"] is not None else "", "trail_atr": prm["trail_atr"] if prm["trail_atr"] is not None else "",
            "be_pct": prm["be_pct"] if prm["be_pct"] is not None else "", "ma_exit": int(prm["ma_exit"]) if prm["ma_exit"] else "",
            "max_hold": prm["max_hold"], "hold_days": 1, "last_date": str(g["date"].iloc[e].date()),
            "last_close": round(float(g["close"].iloc[e]), 2), "unrealized_pct": 0.0, "unrealized_yen": 0,
            "order_id": od["id"], "reason": od["reason"],
        })
        log.append(f"約定 {od['code']} {p['name']} {qty}株 @{ep:.1f} ({str(g['date'].iloc[e].date())}) 初期SL {sl0:.1f}")
        st = {"stop": sl0, "sl0": sl0, "hi": ep, "atr0": atr0}
        si, sk = pending_sell(od["code"], g)
        r = _exit_walk(g, e, ep, prm, st, e, sk)
        if r is not None:
            k, price, why = r
            close_position(p, g["date"].iloc[k], price, why, k - e + 1)
            if why == "MANUAL":
                mark_sell_filled(si, g["date"].iloc[k], price)
        else:
            k = len(g) - 1
            p["stop_price"] = round(st["stop"], 2); p["high_close"] = round(st["hi"], 2)
            p["hold_days"] = k - e + 1
            p["last_date"] = str(g["date"].iloc[k].date())
            p["last_close"] = round(float(g["close"].iloc[k]), 2)
            p["unrealized_pct"] = round((float(g["close"].iloc[k]) / ep - 1) * 100, 2)
            p["unrealized_yen"] = round((float(g["close"].iloc[k]) - ep) * qty)
            pos = pd.concat([pos, p.to_frame().T], ignore_index=True)

    # 4) 保有が無く、約定待ちの買いも無い売り注文は却下
    for i, od in orders.iterrows():
        if od["status"] == "pending" and od["side"] == "sell":
            held = (pos["code"] == od["code"]).any()
            buying = ((orders["status"] == "pending") & (orders["side"] == "buy") & (orders["code"] == od["code"])).any()
            if not held and not buying:
                orders.loc[i, ["status", "note"]] = ["rejected", "保有なし"]

    # 5) 記録
    pos_value = float((pos["last_close"].astype(float) * pos["qty"].astype(int)).sum()) if len(pos) else 0.0
    realized = float(trades["pnl_yen"].astype(float).sum()) if len(trades) else 0.0
    eq_row = {"date": str(asof.date()), "cash": round(cash), "position_value": round(pos_value),
              "equity": round(cash + pos_value), "realized_cum": round(realized), "n_positions": int(len(pos))}
    equity = equity[equity["date"] != eq_row["date"]]
    equity = pd.concat([equity, pd.DataFrame([eq_row])], ignore_index=True).sort_values("date")
    state["cash"] = round(cash, 2)
    _write(orders, ORDERS); _write(pos, POSITIONS); _write(trades, TRADES); _write(equity, EQUITY); save_state(state)
    md = report(asof, orders, pos, trades, equity, state, log)
    return md


# ---------- report ----------
def describe_defaults() -> str:
    d = DEFAULTS; parts = []
    parts.append(f"初期SL {d['sl_pct']*100:+.0f}%" if d["sl_pct"] is not None else f"初期SL ATR×{d['sl_atr']}")
    parts.append(f"利確 {d['tp_pct']*100:+.0f}%" if d["tp_pct"] is not None else "利確なし")
    if d["trail_pct"] is not None: parts.append(f"トレール {d['trail_pct']*100:.0f}%")
    if d["trail_atr"] is not None: parts.append(f"トレール ATR×{d['trail_atr']}")
    if d["be_pct"] is not None: parts.append(f"建値撤退 +{d['be_pct']*100:.0f}%で発動")
    if d["ma_exit"]: parts.append(f"{int(d['ma_exit'])}日線割れで撤退")
    parts.append(f"最大 {d['max_hold']} 営業日")
    return " / ".join(parts)


def report(asof=None, orders=None, pos=None, trades=None, equity=None, state=None, log=None) -> str:
    orders = _read(ORDERS, ORDER_COLS) if orders is None else orders
    pos = _read(POSITIONS, POS_COLS) if pos is None else pos
    trades = _read(TRADES, TRADE_COLS) if trades is None else trades
    equity = _read(EQUITY, EQ_COLS) if equity is None else equity
    state = load_state() if state is None else state
    asof = (equity["date"].max() if len(equity) else "-") if asof is None else str(pd.Timestamp(asof).date())
    start = float(state["start_capital"])
    eq = float(equity["equity"].iloc[-1]) if len(equity) else start
    realized = float(trades["pnl_yen"].astype(float).sum()) if len(trades) else 0.0
    unreal = float(pos["unrealized_yen"].astype(float).sum()) if len(pos) else 0.0
    days = (pd.Timestamp(asof) - pd.Timestamp(state["start_date"])).days if state.get("start_date") and asof != "-" else 0
    L = [f"# 仮想取引 {asof}", ""]
    L.append(f"資産 **{eq:,.0f}円** (開始 {start:,.0f} / {eq - start:+,.0f}円 {((eq / start) - 1) * 100:+.2f}%) "
             f"現金 {float(state['cash']):,.0f} / 確定損益 {realized:+,.0f} / 含み {unreal:+,.0f} / 経過 {days}日")
    pace = realized / max(days, 1) * 30 if days else 0.0
    L.append(f"目標 月{MONTHLY_GOAL:,}円 に対し確定損益ベースの月換算ペース {pace:+,.0f}円")
    if log:
        L += ["", "## 本日の約定・決済"] + [f"- {x}" for x in log]
    L += ["", f"## 保有 {len(pos)} 件"]
    if len(pos):
        L.append("| code | name | 建日 | 建値 | 株数 | 現在ストップ | 初期SL | 最高終値 | 利確 | 保有日 | 終値 | 含み% | 含み円 | 根拠 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, p in pos.iterrows():
            tpv = _f(p["tp_price"]); stp = _f(p["stop_price"], _f(p["sl_price"])); hc = _f(p["high_close"], float(p["entry_price"]))
            L.append(f"| {p['code']} | {p['name']} | {p['entry_date']} | {float(p['entry_price']):.1f} | {int(p['qty'])} | "
                     f"{stp:.1f} ({(stp / float(p['entry_price']) - 1) * 100:+.1f}%) | {float(p['sl_price']):.1f} | {hc:.1f} | {f'{tpv:.1f}' if tpv else 'なし'} | "
                     f"{int(p['hold_days'])}/{int(float(p['max_hold']))} | "
                     f"{float(p['last_close']):.1f} | {float(p['unrealized_pct']):+.2f}% | {float(p['unrealized_yen']):+,.0f} | {p['reason']} |")
    else:
        L.append("なし")
    pend = orders[orders["status"] == "pending"]
    L += ["", f"## 未約定の注文 {len(pend)} 件"]
    if len(pend):
        L.append("| id | 約定予定 | code | side | 株数 | 根拠 |")
        L.append("|---|---|---|---|---|---|")
        for _, o in pend.iterrows():
            L.append(f"| {o['id']} | {o['fill_after']} の翌営業日寄り | {o['code']} | {o['side']} | {o['qty']} | {o['reason']} |")
    L += ["", f"## 決済済み {len(trades)} 件"]
    if len(trades):
        w = trades["ret_pct"].astype(float) > 0
        wr = w.mean()
        aw = trades.loc[w, "ret_pct"].astype(float).mean() if w.any() else 0.0
        al = trades.loc[~w, "ret_pct"].astype(float).mean() if (~w).any() else 0.0
        L.append(f"勝率 {wr * 100:.0f}% / 平均利 {aw:+.2f}% / 平均損 {al:+.2f}% / 実現EV {wr * aw + (1 - wr) * al:+.2f}%/trade / "
                 f"平均保有 {trades['hold_days'].astype(float).mean():.1f}日")
        by = trades["exit_reason"].value_counts().to_dict()
        L.append("決済理由: " + ", ".join(f"{k} {v}" for k, v in by.items()))
        L.append("")
        L.append("| code | name | 建日 | 建値 | 決済日 | 決済値 | 株数 | 損益% | 損益円 | 理由 | 保有日 | 根拠 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, t in trades.tail(30).iloc[::-1].iterrows():
            L.append(f"| {t['code']} | {t['name']} | {t['entry_date']} | {float(t['entry_price']):.1f} | {t['exit_date']} | "
                     f"{float(t['exit_price']):.1f} | {int(t['qty'])} | {float(t['ret_pct']):+.2f}% | {float(t['pnl_yen']):+,.0f} | "
                     f"{t['exit_reason']} | {int(t['hold_days'])} | {t['reason']} |")
    if len(equity) > 1:
        L += ["", "## 資産推移（直近）", "| date | 資産 | 現金 | 保有評価 | 確定損益累計 | 保有数 |", "|---|---|---|---|---|---|"]
        for _, r in equity.tail(15).iterrows():
            L.append(f"| {r['date']} | {float(r['equity']):,.0f} | {float(r['cash']):,.0f} | {float(r['position_value']):,.0f} | "
                     f"{float(r['realized_cum']):+,.0f} | {int(r['n_positions'])} |")
    L += ["", "注: 約定は注文日の翌営業日の寄り値、片道0.1%のコスト込み。既定の決済: " + describe_defaults() + "。注文ごとに上書き可。"]
    return "\n".join(L)


def write_outputs(md: str, asof: str, out_root: Path = Path("outputs")) -> None:
    for d in (out_root / asof, out_root / "latest"):
        d.mkdir(parents=True, exist_ok=True)
        (d / "paper.md").write_text(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["settle", "report"])
    a = ap.parse_args()
    if a.cmd == "settle":
        md = settle()
        eq = _read(EQUITY, EQ_COLS)
        write_outputs(md, str(eq["date"].max()))
        print(md)
    else:
        print(report())
