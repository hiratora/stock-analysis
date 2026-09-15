# swing-screener

日本株スイング戦略（利確+5% / 損切り-2.5% / 最大保有10営業日）の日次スクリーニングと期待値試算。
**費用ゼロ**: 株価は yfinance（Yahoo Finance）、銘柄一覧は JPX 公開シート、実行は GitHub Actions 無料枠。

## できること / できないこと
- できる: 東証内国株（約3,800銘柄）の日足から breakout / pullback が出た全銘柄を、指標・フラグ・過去実績のEV付きで毎日出す。どれを買うかの判断はしない（材料出しまで）。watchlist の銘柄は個別レポート。
- できない: 次回決算発表日、信用残、業績進捗率。無料データに無い。**候補に残った銘柄は買う前に kabutan 等で決算日を確認する（跨ぎ禁止ルール）。**
- 地合いは TOPIX そのものではなく TOPIX連動ETF 1306.T の終値で代用。
- Yahoo のデータは無償・無保証。分割調整は yfinance の auto_adjust に依存。欠損・遅延はあり得る。

## セットアップ（5分）
1. このリポジトリを GitHub に作る（public にすると Claude のチャットから結果を直接読める。private でも動く）
2. Actions を有効化。Secrets は不要
3. Actions → daily-screen → Run workflow で初回実行（初回は約3年分のダウンロードで 15〜30 分）
4. 以降、毎営業日 18:30 JST に `outputs/latest/screen.md` が更新される

## 使い方
- 今日の候補: `outputs/latest/screen.md`
- 銘柄を調べたい: `watchlist.txt` に 4桁コードを1行追記して commit（スマホの GitHub アプリで可）→ 数分で `outputs/analysis/latest/<code>.md`
- 調べ終わったら watchlist から消す（残しておけば毎日更新される）

ローカル実行:
```
pip install -r requirements.txt
python -m src.screen
python -m src.analyze 7203
```

動作確認（ネットワーク不要）:
```
python tests/make_synthetic.py && python -m src.screen --offline
PYTHONPATH=. python tests/test_yf_normalize.py
```

## 出力の読み方
- `EV/trade`: 勝率×平均利幅 + (1-勝率)×平均損幅。1トレード・ポジション対比。往復コスト0.2%込み。
- 主EVは「シグナル種別×今日の地合い」の全銘柄実績（地合い別 n<30 なら全体値）。銘柄EVは参考列で、n が小さいほど信用しない（銘柄固有EVで並べると多重比較でノイズが上位に来るため、並び順には使わない）。
- `地合い別`: 1306.T が25日線の上/下で分けたシグナル成績。
- フラグ（落とさず印を付ける）: 流動性=20日平均売買代金5億以上 / ATR=ATR14÷終値2.5%以下 / 乖離=25日線乖離7%以下 / ギャップ=当日騰落6%以下。EVの母集団は4フラグ全OKのシグナルのみ。閾値は `src/signals.py` 先頭。

## 設計上の注意
- シグナルは終値確定後に判定、翌営業日寄りで買う。同日買いはバックテストが先読みになる。
- 利確・損切りが同日に両方ヒットした場合は損切り扱い。寄りが損切り線を割っていれば寄り値で決済（ギャップダウン損失を計上）。保有中の同一銘柄シグナルは無視（重複計上しない）。
- 合成データ（ランダムウォーク）で EV ≈ 0 を確認済み。実データで EV が出るなら、それはランダムでは説明できない分の優位性。
