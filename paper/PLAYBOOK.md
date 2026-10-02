# 仮想取引 PLAYBOOK

夜と朝の定期セッション（Claude）が従う手順。状態はすべてこのリポジトリにある。会話の記憶に頼らない。

## 前提
- 軍資金 200万円（仮想）。目標は月 +10万円（+5%）。数日〜数週間のスイング。
- 約定・決済は `src/paper.py settle` が GitHub Actions 内で機械的に行う（翌営業日寄り建て、利確 +5% / 損切り -2.5% / 最大保有10営業日、片道0.1%コスト）。Claude の仕事は **何を・いつ・いくら買うか、何を手仕舞うか** の判断と、その記録。
- データは Yahoo の日足のみ。決算日・信用残・業績進捗は無い。**決算日は買う前に必ず Web 検索で確認する（kabutan 等）。決算またぎ禁止。**
- 場中に Actions を回さない（未確定バーで判定してしまう）。`workflow_dispatch` はこの環境から叩けない。
- 定時実行が遅れて outputs が古いときは、引け後（15:30 JST 以降）であればセッション内で `python -m src.screen && python -m src.universe --offline && python -m src.paper settle` を直接回してよい（約10〜20分）。その結果の `outputs/` と `paper/` はコミットしてよい（後で bot が同じ日付で上書きする）。

## ファイル
| ファイル | 誰が書く | 内容 |
|---|---|---|
| `paper/orders.csv` | Claude | 注文。`fill_after` の翌営業日の寄りで約定。`status` は pending → filled / cancelled / rejected |
| `paper/positions.csv` `trades.csv` `equity.csv` `state.json` | Actions | 保有・決済済み・資産推移・現金。**手で編集しない** |
| `paper/journal.md` | Claude | 日誌。判断理由・振り返り・学び。新しい日付を上に |
| `outputs/latest/paper.md` | Actions | 現状サマリ |
| `outputs/latest/screen.md` `screen.csv` | Actions | シグナル銘柄（breakout / pullback）と EV |
| `outputs/latest/basic_screen.csv` `universe.csv` `sector_rs.csv` | Actions | 全銘柄の指標、トレンド足切り通過の上位60、業種RS |

### orders.csv の書き方
```
id,fill_after,code,side,qty,tp_pct,sl_pct,max_hold,reason,status,fill_date,fill_price,note
20261005-01,2026-10-02,4502,buy,100,,,,pullback/4flagOK/業種RS上位,pending,,,
20261005-02,2026-10-02,8601,sell,,,,,"含み益が伸びず、業種RS低下",pending,,,
```
- `id` = 約定予定日-連番。`fill_after` = 判断に使った日足の日付（outputs の as-of）。
- `qty` は100株単位。`tp_pct` `sl_pct` `max_hold` は空なら既定（0.05 / -0.025 / 10）。変えるなら理由を reason に書く。
- `sell` は保有の全株を翌寄りで手仕舞い。
- 取り消しは `status` を `cancelled` にして `note` に理由。

## 資金管理（破ってはいけない）
- 1銘柄の建玉 ≤ 資産の 25%（約50万円）。100株の代金が50万を超える銘柄は買わない。
- 同時保有 ≤ 5 銘柄。同一33業種 ≤ 2 銘柄。1日の新規 ≤ 3 銘柄。
- 現金が足りない注文は rejected になる。注文前に `state.json` の cash を見る。
- 地合い（1306.T が25日線の下）の日は新規を最大1銘柄に絞り、breakout は見送る（地合い下の breakout EV はマイナス）。

## 銘柄選定の順序
1. `outputs/latest/screen.md` の「地合い」と、シグナル別・地合い別 EV を読む。EV が負のシグナル種別はその日は使わない。
2. `screen.csv` から 4フラグ全OK かつ `ev_pct > 0` の銘柄を取る。
3. それを `basic_screen.csv`（トレンド足切り通過・スコア順）と突き合わせる。両方に載る銘柄を最優先。片方だけなら basic_screen の `score` と `rs20` `near_high` `contraction` で順位付け。
4. `sector_rs.csv` の `rs20_med` 上位3分の1の業種を優先、下位3分の1は見送る。
5. 残った候補について Web 検索で決算日と直近の材料（上方修正、増配、大口受注、不祥事）を確認。10営業日以内に決算があるものは外す。
6. 建値の目安は前日終値。TP/SL は約定後に paper.py が建値基準で引き直す。
7. 1〜3 銘柄に絞る。選ばなかった理由も journal に書く（後で検証できるように）。

## 夜の手順（22:20 JST 起動。19:17 と予備 21:43 の定時実行が終わった後）
1. `git pull origin main`。`outputs/latest/screen.md` の as-of が今日（JST）か確認。
   - 今日でなければ `git log --oneline -5` で bot の「outputs <日付>」コミットの有無を見る。無ければ定時実行が遅延か失敗（coverage 不足）。
     その場合は **新規注文を出さず**、journal に「データ未更新のため見送り」と書いて終了する。古いデータで注文しない。
   - 定期セッションでは GitHub API や send_later 等のコネクタは使えない前提で動く（git と Web 検索だけで完結させる）。
   - `paper/journal.md` に既に今日の「夜」の節があれば、別セッションが済ませている。その場合は注文を追加せず、内容を確認して矛盾があれば journal に追記するだけで終える。
2. `outputs/latest/paper.md` を読む。本日の約定・決済があれば **1件ずつ振り返り** を journal に書く: 選定理由は妥当だったか、決済理由（TP/SL/SL_GAP/TIME）は想定内か、同じ判断を次もするか。
3. 保有銘柄の手仕舞い判断。規則上は TP/SL/TIME で自動決済されるので、手動 sell は「前提が崩れた」とき（業種RSが崩れた、地合いが下に転換、悪材料）に限る。
4. 上の「銘柄選定の順序」で新規候補を決め、`orders.csv` に追記。
5. `journal.md` の先頭に今日の節を追加: 地合い / 約定・決済の振り返り / 新規注文と理由 / 見送った候補と理由 / 仮説。
6. `git add paper && git commit && git push origin main`。
7. 最後に短く報告: 資産、本日の損益、新規注文、見送り理由、気づき。

## 朝の手順（07:15 JST 起動）
1. `git pull origin main`。`paper/orders.csv` の pending 注文と `positions.csv` の保有を確認。
2. Web 検索で確認: (a) pending 注文・保有銘柄の決算発表日と前夜以降の開示、(b) 米国市場の前夜の動き（S&P500、半導体）、日経平均先物の夜間終値。急落（先物 -2% 超）なら新規 pending を全部 cancelled にする。
3. 問題がある注文は `status=cancelled`、`note` に理由。保有で前提が崩れたものは `sell` を追加（今日の寄りで決済される）。
4. `journal.md` の今日の節に「朝メモ」を追記。変更があれば commit & push。
5. 短く報告。何も変えなければ「変更なし」と一行。

## 週次レビュー（金曜の夜の手順の最後に）
- `trades.csv` を集計: 勝率、実現EV、決済理由の分布、`reason` 別（pullback/breakout、業種、地合い）の成績。
- 月換算ペースが目標 +10万円に届かない場合、何を変えるか仮説を1つ立てる（例: 業種RS条件を強める、地合い下は全休、TP を 4% に下げる）。
- 仮説はこの PLAYBOOK の「選定の順序」や「資金管理」を書き換えて反映し、変更履歴を下の表に残す。ルール変更は週1回まで（過学習を避ける）。

## 変更履歴
| 日付 | 変更 | 理由 |
|---|---|---|
| 2026-10-02 | 初版 | — |
| 2026-10-02 | 夜の起動を 22:20 JST に。遅延時はセッション内で取得 | 定時実行が数時間遅れるため |
