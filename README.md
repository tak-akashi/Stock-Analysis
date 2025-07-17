# Stock-Analysis (株式データ分析プロジェクト)

## 概要

このプロジェクトは、日本の株式市場に関するデータを収集、保存、分析するためのツール群です。J-Quants APIおよびyfinanceライブラリを利用して、日々の株価、企業情報、銘柄マスターなどを自動で取得・更新し、SQLiteデータベースに格納します。

収集したデータは、様々な分析手法（ミネルヴィニ戦略、高値・安値比率、相対力、チャートパターン分類など）を用いて評価され、統合的な視点から銘柄選定や投資戦略の構築に活用できます。

## 主な機能

*   **データ収集と管理:**
    *   **日次株価取得 (J-Quants):** 平日の夜間にJ-Quants APIから最新の株価四本値（始値, 高値, 安値, 終値）および出来高を取得し、`data/jquants.db` に保存します。
    *   **週次企業情報取得 (yfinance):** 週末にyfinanceライブラリを通じて、各企業の属性情報（セクター、業種など）を取得し、`data/yfinance.db` に保存します。
    *   **月次マスターデータ更新:** 毎月1回、最新の銘柄一覧（マスターデータ）を更新し、`data/master.db` に保存します。

*   **株式分析:**
    *   **ミネルヴィニ戦略:** マーク・ミネルヴィニの投資基準に基づき、銘柄のトレンドと強さを評価します。
    *   **高値・安値比率 (HL Ratio):** 過去一定期間の高値と安値に対する現在の株価の位置を評価し、買われすぎ・売られすぎを判断します。
    *   **相対力 (Relative Strength):** 市場全体や他の銘柄と比較した株価の相対的な強さを評価します。
    *   **チャートパターン分類:** 機械学習を用いて、株価チャートの形状を自動的に分類し、特定のパターン（上昇、下落、もみ合いなど）を識別します。データの可用性に基づいて動的にウィンドウサイズを選択（1200日または960日の長期パターン分析を含む）し、高性能なバッチ処理により効率的に実行されます。
    *   **統合分析:** 上記の各分析結果を組み合わせ、複合的なスコアや条件フィルタリングにより、多角的な視点から銘柄を評価します。

## 分析機能の詳細

`backend/analysis` ディレクトリには、以下の分析プログラムが含まれており、それぞれが特定の分析手法を実装しています。

*   **`minervini.py`**: マーク・ミネルヴィニの株式スクリーニング戦略を実装しています。株価と移動平均線の関係、52週高値・安値からの乖離率などを基に、銘柄のトレンドと強さを評価します。計算結果は `data/analysis_results.db` の `minervini` テーブルに保存されます。
*   **`high_low_ratio.py`**: 銘柄の高値・安値比率を計算するロジックです。過去52週間の高値と安値の範囲内で、現在の株価がどの位置にあるかを示します。結果は `data/analysis_results.db` の `hl_ratio` テーブルに保存されます。
*   **`relative_strength.py`**: 相対力（Relative Strength Percentage: RSP）および相対力指数（Relative Strength Index: RSI）を計算するロジックです。銘柄の市場に対する相対的なパフォーマンスを評価します。結果は `data/analysis_results.db` の `relative_strength` テーブルに保存されます。
*   **`chart_classification.py`**: 株価チャートのパターンを分類するロジックです。過去の株価データから特定の期間（20日、60日、120日、240日、および動的選択される960日または1200日）のチャート形状を抽出し、「上昇」「下落」「もみ合い」などのパターンに分類します。データの可用性に基づいて自動的に長期ウィンドウ（1200日 ≥ 960日）を選択し、バッチ処理による高性能な分析を提供します。結果は `data/analysis_results.db` の `classification_results` テーブルに保存されます。
*   **`integrated_analysis.py`**: 上記の各分析プログラムによって生成された結果（`hl_ratio`, `minervini`, `relative_strength`, `classification_results`）を `data/analysis_results.db` から読み込み、それらを統合して複合的な評価を行うためのユーティリティ関数を提供します。これにより、複数の指標を横断的に評価し、より精度の高い銘柄選定を支援します。このスクリプト自体はデータを生成せず、既存のデータをクエリ・集計します。
*   **`integrated_analysis2.py`**: `integrated_analysis.py` の機能を利用し、各分析結果を統合して、最終的な分析結果をExcelファイルとして `output` フォルダに出力します。

## ディレクトリ構成

```
.
├── backend/         # データ処理のコアロジック
│   ├── analysis/    # 各種分析ロジック
│   ├── jquants/     # J-Quants API関連の処理
│   ├── master/      # 銘柄マスター関連の処理
│   └── yfinance/    # yfinance関連の処理
├── data/            # データベースファイル（.sqlite, .db）を格納
├── logs/            # cronジョブの実行ログ
├── output/          # 分析結果のExcelファイルやエラーログなどを格納
├── notebook/        # (現在未使用)
├── scripts/         # 定期実行用のスクリプト群
├── test/            # テストコード
├── pyproject.toml   # プロジェクトの依存関係定義
└── README.md        # このファイル
```

## データベース構造

このプロジェクトでは、主に以下のSQLiteデータベースファイルを使用します。

*   **`data/jquants.db`**:
    *   J-Quants APIから取得した日次株価データ（`daily_quotes`テーブル）や企業情報が格納されます。
    *   `prices`テーブルも含まれる可能性があります。
*   **`data/yfinance.db`**:
    *   yfinanceから取得した企業の属性情報が格納されます。
*   **`data/master.db`**:
    *   銘柄マスターデータ（`stocks_master`テーブルなど）が格納されます。
*   **`data/analysis_results.db`**:
    *   各種分析プログラム（`minervini.py`, `high_low_ratio.py`, `relative_strength.py`, `chart_classification.py`）によって計算された結果が格納されます。
    *   主なテーブル:
        *   `minervini`: ミネルヴィニ戦略の評価結果
        *   `hl_ratio`: 高値・安値比率の計算結果
        *   `relative_strength`: 相対力（RSP, RSI）の計算結果
        *   `classification_results`: チャートパターン分類の結果

## セットアップ方法

1.  **リポジトリのクローン:**
    ```bash
    git clone <repository_url>
    cd Stock-Analysis
    ```

2.  **Python環境:**
    このプロジェクトは Python 3.10 以上を要求します。

3.  **依存ライブラリのインストール:**
    `uv` または `pip` を使用して、必要なライブラリをインストールします。
    ```bash
    # uvを使用する場合
    uv pip install -r requirements.txt

    # pipを使用する場合
    pip install -r requirements.txt
    ```
    ※ `requirements.txt` がない場合は、`pyproject.toml` から生成するか、直接インストールしてください。

4.  **環境変数の設定:**
    J-Quants APIを利用するために、APIキーなどの設定が必要です。プロジェクトルートに `.env` ファイルを作成し、以下の内容を記述してください。
    ```
    JQUANTS_API_KEY="YOUR_API_KEY"
    ```

## 使用方法

### 手動での実行

各スクリプトを直接実行することで、任意のタイミングでデータ取得や分析を実行できます。

*   **J-Quants日次データ取得:**
    J-Quants APIから日次株価データを取得します。
    ```bash
    python scripts/run_jquants_daily.py
    ```

*   **週次タスク (yfinanceデータ取得 & チャート分類 & 統合分析):**
    yfinanceから企業属性情報を取得し、チャートパターン分類、統合分析を実行します。
    ```bash
    python scripts/run_weekly_tasks.py
    ```

*   **月次マスターデータ更新:**
    銘柄マスターデータを更新します。
    ```bash
    python scripts/run_master_monthly.py
    ```

*   **日次分析フロー:**
    高値・安値比率、ミネルヴィニ戦略、相対力（RSP/RSI）の計算とデータベースへの保存、およびその日の分析サマリーのログ出力を実行します。
    ```bash
    python scripts/run_daily_analysis.py
    ```

*   **統合分析:**
    `integrated_analysis2.py` を直接実行することで、統合分析を行い、結果をExcelファイルに出力します。
    ```bash
    python backend/analysis/integrated_analysis2.py
    ```

*   **チャートパターン分類:**
    チャートパターンの分類分析を実行します。複数の実行モードが利用可能です。
    ```bash
    # サンプル実行（基本的なパターン分析）
    python backend/analysis/chart_classification.py --mode sample
    
    # アダプティブウィンドウのサンプル実行（1200/960日の動的選択をテスト）
    python backend/analysis/chart_classification.py --mode sample-adaptive
    
    # 全銘柄での高性能分析（アダプティブウィンドウ付き）
    python backend/analysis/chart_classification.py --mode full-optimized
    
    # バッチサイズの調整（メモリ使用量の制御）
    python backend/analysis/chart_classification.py --mode full --batch-size 50
    ```

### cronによる自動実行

`cron_schedule.txt` に記載されている設定を参考に、cronにジョブを登録することで、データ取得・更新・分析を自動化できます。

**設定例:**
```crontab
# 平日22時にJ-Quantsで株価データを取得
0 22 * * 1-5 /path/to/python /Users/tak/Markets/Stocks/Stock-Analysis/scripts/run_jquants_daily.py >> /Users/tak/Markets/Stocks/Stock-Analysis/logs/jquants_daily.log 2>&1

# 日曜20時に週次タスク（yfinanceデータ取得、チャート分類、統合分析）を実行
0 20 * * 0 /path/to/python /Users/tak/Markets/Stocks/Stock-Analysis/scripts/run_weekly_tasks.py >> /Users/tak/Markets/Stocks/Stock-Analysis/logs/weekly_tasks.log 2>&1

# 毎月1日18時にマスターデータを更新
0 18 1 * * /path/to/python /Users/tak/Markets/Stocks/Stock-Analysis/scripts/run_master_monthly.py >> /Users/tak/Markets/Stocks/Stock-Analysis/logs/master_monthly.log 2>&1

# 平日23時に日次分析フローを実行
0 23 * * 1-5 /path/to/python /Users/tak/Markets/Stocks/Stock-Analysis/scripts/run_daily_analysis.py >> /Users/tak/Markets/Stocks/Stock-Analysis/logs/daily_analysis.log 2>&1
```
**注意:** `/path/to/python` の部分は、使用しているPython実行環境の絶対パスに置き換えてください。ログファイルのパスも適宜調整してください。

## 依存ライブラリ

このプロジェクトでは、以下の主要なライブラリを使用しています。

*   pandas: データ操作と分析
*   numpy: 数値計算
*   sqlite3: SQLiteデータベース操作
*   yfinance: Yahoo Financeからのデータ取得
*   requests: HTTPリクエスト
*   backtrader: バックテストフレームワーク（現在未使用の可能性あり）
*   matplotlib: グラフ描画（チャート分類で使用）
*   japanize_matplotlib: matplotlibの日本語表示対応（チャート分類で使用）
*   scipy: 科学技術計算（チャート分類で使用）
*   scikit-learn: 機械学習（チャート分類で使用）
*   talib: テクニカル分析ライブラリ（Minerviniで使用、オプション）
*   pytest: テストフレームワーク
*   openpyxl: Excelファイルの読み書き