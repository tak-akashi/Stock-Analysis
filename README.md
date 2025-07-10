# Stock-Analysis (株式データ分析プロジェクト)

## 概要

このプロジェクトは、日本の株式市場に関するデータを収集、保存、分析するためのツール群です。J-Quants APIおよびyfinanceライブラリを利用して、日々の株価、企業情報、銘柄マスターなどを自動で取得・更新し、SQLiteデータベースに格納します。

収集したデータは、さらなる分析やバックテスト戦略の構築などに活用できます。

## 主な機能

*   **日次株価取得 (J-Quants):**
    *   平日の夜間にJ-Quants APIから最新の株価四本値（始値, 高値, 安値, 終値）および出来高を取得します。
    *   取得データは `data/jquants.db` に保存されます。
*   **週次企業情報取得 (yfinance):**
    *   週末にyfinanceライブラリを通じて、各企業の属性情報（セクター、業種など）を取得します。
*   **月次マスターデータ更新:**
    *   毎月1回、最新の銘柄一覧（マスターデータ）を更新します。

## 分析機能

`backend/analysis` ディレクトリには、以下の分析プログラムが含まれています。

*   **`chart_classification.py`**: チャートパターンを分類するロジック。
*   **`high_low_ratio.py`**: 高値・安値比率を計算するロジック。
*   **`minervini.py`**: マーク・ミネルヴィニの株式スクリーニング戦略を実装。
*   **`relative_strength.py`**: 相対力（Relative Strength）を計算するロジック。
*   **`integrated_analysis.py`**: 上記の分析結果を統合し、複合的な評価を行うロジック。
*   **`demo_integrated_analysis.py`**: `integrated_analysis.py` の利用例を示すデモスクリプト。

これらのプログラムは、収集した株価データや企業情報を用いて、特定の投資戦略やテクニカル分析に基づいた銘柄の評価を行います。

### 分析機能の利用方法

`integrated_analysis.py` は、`high_low_ratio.py`、`minervini.py`、`relative_strength.py` などで生成された分析結果を統合し、複合的なスコア（composite score）を算出します。これにより、複数の指標を横断的に評価し、より精度の高い銘柄選定を支援します。

`demo_integrated_analysis.py` を実行することで、`integrated_analysis.py` の各種機能（特定日付の総合分析、上位銘柄ランキング、条件フィルタリング、複数日付の時系列分析、サマリー統計など）のデモンストレーションを確認できます。

```bash
python backend/analysis/demo_integrated_analysis.py
```

このデモスクリプトを実行する前に、各分析スクリプト（`high_low_ratio.py`、`minervini.py`、`relative_strength.py`）を実行して、`data/analysis_results.db` に分析結果が格納されていることを確認してください。

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
├── notebook/        # (現在未使用)
├── scripts/         # 定期実行用のスクリプト群
├── test/            # テストコード
├── pyproject.toml   # プロジェクトの依存関係定義
└── README.md        # このファイル
```

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

### 手動でのデータ取得

各スクリプトを直接実行することで、任意のタイミングでデータを取得・更新できます。

*   **J-Quants日次データ取得:**
    ```bash
    python scripts/run_jquants_daily.py
    ```
*   **yfinance週次データ取得:**
    ```bash
    python scripts/run_yfinance_weekly.py
    ```
*   **マスターデータ月次更新:**
    ```bash
    python scripts/run_master_monthly.py
    ```

### cronによる自動実行

`cron_schedule.txt` に記載されている設定を参考に、cronにジョブを登録することで、データ取得・更新を自動化できます。

**設定例:**
```crontab
# 平日22時にJ-Quantsで株価データを取得
0 22 * * 1-5 /path/to/python /Users/tak/Markets/Stocks/Stock-Analysis/scripts/run_jquants_daily.py >> /Users/tak/Markets/Stocks/Stock-Analysis/logs/jquants_daily.log 2>&1

# 日曜20時にyfinanceで属性情報を取得
0 20 * * 0 /path/to/python /Users/tak/Markets/Stocks/Stock-Analysis/scripts/run_yfinance_weekly.py >> /Users/tak/Markets/Stocks/Stock-Analysis/logs/yfinance_weekly.log 2>&1

# 毎月1日18時にマスターデータを更新
0 18 1 * * /path/to/python /Users/tak/Markets/Stocks/Stock-Analysis/scripts/run_master_monthly.py >> /Users/tak/Markets/Stocks/Stock-Analysis/logs/master_monthly.log 2>&1
```
**注意:** `/path/to/python` の部分は、使用しているPython実行環境の絶対パスに置き換えてください。

## 依存ライブラリ

このプロジェクトでは、以下の主要なライブラリを使用しています。

*   pandas
*   yfinance
*   requests
*   backtrader
*   matplotlib（未使用）
*   plotly（未使用）
*   seaborn（未使用）
*   pytest

詳細なリストは `pyproject.toml` を参照してください。
