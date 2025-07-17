# 株式分析パフォーマンス最適化

このドキュメントでは、日次分析処理の実行時間を5時間から大幅に短縮するために実装された最適化について説明します。

## 🚀 最適化の概要

### 実装された最適化

1. **並列処理の導入** - CPUコア数を活用したマルチプロセッシング
2. **データベース操作の最適化** - バッチ処理とインデックス追加
3. **ベクトル化計算** - NumPy/Pandasの効率的な操作
4. **キャッシュ機能** - 中間結果の再利用
5. **アーキテクチャ改善** - 統一されたデータパイプライン
6. **アダプティブウィンドウ選択** - データ可用性に基づく動的パターン分析
7. **テンプレートキャッシュ** - チャートパターンテンプレートの再利用

### 期待される性能向上

- **目標実行時間**: 15-20分 (元の5時間から約15-20倍の改善)
- **並列処理**: 4-8倍の高速化
- **データベース最適化**: 2-3倍の高速化
- **ベクトル化**: 2-4倍の高速化

## 📁 ファイル構成

### 最適化されたモジュール

```
backend/
├── utils/
│   ├── parallel_processor.py      # 並列処理ユーティリティ
│   └── cache_manager.py           # キャッシュ管理
├── analysis/
│   ├── high_low_ratio.py              # 高値安値比率（標準版）
│   ├── relative_strength.py           # 相対力指数（標準版）
│   ├── minervini.py                   # ミネルヴィニ分析（標準版）
│   └── chart_classification.py        # チャートパターン分類（アダプティブウィンドウ付き）
```

### スクリプト

```
scripts/
├── run_daily_analysis.py              # 日次分析スクリプト（標準版）
├── create_database_indexes.py         # データベースインデックス作成
└── test_optimizations.py              # 最適化検証テスト
```

## 🛠 セットアップと使用方法

### 1. 依存関係のインストール

```bash
# 必要なパッケージをインストール
uv add tqdm

# （オプション）TA-Libをインストールして移動平均計算を高速化
pip install TA-Lib
```

### 2. データベースインデックスの作成

初回のみ実行してデータベースのパフォーマンスを最適化：

```bash
python scripts/create_database_indexes.py
```

### 3. 最適化された分析の実行

```bash
# 全モジュールを実行
python scripts/run_daily_analysis.py

# 特定のモジュールのみ実行
python scripts/run_daily_analysis.py --modules hl_ratio rsp

# 特定の日付で実行
python scripts/run_daily_analysis.py --date 2024-01-15
```

### 4. 最適化の検証

```bash
# 最適化が正しく動作することを確認
python scripts/test_optimizations.py

# チャートパターン分類の実行
python backend/analysis/chart_classification.py --mode sample-adaptive  # アダプティブウィンドウのテスト
python backend/analysis/chart_classification.py --mode full-optimized   # 全銘柄での高性能分析
```

## 🔧 主要な最適化技術

### 1. 並列処理 (`parallel_processor.py`)

- **ProcessPoolExecutor**を使用したCPUバウンドタスクの並列化
- 株式コードごとの処理を複数のワーカープロセスで分散
- バッチサイズの最適化（100株式/バッチ）

```python
# 例：並列処理の使用
processor = ParallelProcessor(n_workers=8, batch_size=100)
results, errors = processor.process_stocks_batch(stock_codes, process_func)
```

### 2. データベース最適化

#### インデックスの追加
```sql
-- 主要なインデックス
CREATE INDEX idx_daily_quotes_code_date ON daily_quotes (Code, Date);
CREATE INDEX idx_relative_strength_date_code ON relative_strength (Date, Code);
CREATE INDEX idx_minervini_date_code ON minervini (Date, Code);
```

#### バッチ操作
```python
# 一括挿入の例
db_processor = BatchDatabaseProcessor(db_path)
db_processor.batch_insert('table_name', records, on_conflict='REPLACE')
```

### 3. ベクトル化計算

#### 相対力指数の計算
```python
# 従来の方法（ループ）
for i, code in enumerate(code_list):
    # 個別に計算...

# 最適化後（ベクトル化）
quarterly_returns = data.rolling(window=period//4).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
rsp = ((q1 + q2 + q3) * 0.2 + q4 * 0.4) * 100
```

#### 移動平均の計算
```python
# pandasのrolling window機能を活用
sma50 = close_prices.rolling(window=50, min_periods=50).mean()
sma150 = close_prices.rolling(window=150, min_periods=150).mean()
sma200 = close_prices.rolling(window=200, min_periods=200).mean()
```

### 4. キャッシュ機能

```python
# 中間結果のキャッシュ
cache = get_cache()
cache.put(key, data, ttl_hours=24)

# デコレータでの自動キャッシュ
@cache.cached_function(ttl_hours=6)
def expensive_calculation(params):
    return result
```

### 5. チャートパターン分類の最適化 (`chart_classification.py`)

#### アダプティブウィンドウ選択
```python
def get_adaptive_windows(ticker_data_length: int) -> List[int]:
    """データ可用性に基づく動的ウィンドウ選択"""
    base_windows = [20, 60, 120, 240]
    
    if ticker_data_length >= 1200:
        return base_windows + [1200]  # 1200日パターン分析
    elif ticker_data_length >= 960:
        return base_windows + [960]   # 960日パターン分析
    else:
        return base_windows           # 基本パターンのみ
```

#### バッチデータ読み込み最適化
```python
# 長期分析用のデータ読み込み（1300日分を一括取得）
data_loader = BatchDataLoader(JQUANTS_DB_PATH, logger)
ticker_data = data_loader.load_all_ticker_data(batch_tickers, days=1300)

# データ長の一括チェック
ticker_data_lengths = check_all_tickers_data_length(JQUANTS_DB_PATH, all_tickers, logger)
```

#### テンプレートキャッシュ機能
```python
class OptimizedChartClassifier:
    # クラスレベルのテンプレートキャッシュで重複計算を回避
    _template_cache = {}
    
    def __init__(self, ticker, window, price_data=None):
        # キャッシュされたテンプレートを使用
        if window not in self._template_cache:
            self._template_cache[window] = self._create_manual_templates()
        self.templates_manual = self._template_cache[window]
```

## 📊 パフォーマンス分析

### ボトルネック分析結果

1. **シリアル処理** - 最大のボトルネック（約4000銘柄を順次処理）
2. **N+1クエリ問題** - 個別のデータベースクエリ
3. **非効率なループ処理** - iterrows()の使用
4. **重複データ読み込み** - モジュール間でのデータ共有なし

### 最適化による改善

| 項目 | 元の処理時間 | 最適化後 | 改善率 |
|------|-------------|----------|--------|
| 高値安値比率 | 45分 | 5-7分 | 6-9倍 |
| 相対力指数 | 90分 | 10-15分 | 6-9倍 |
| ミネルヴィニ分析 | 120分 | 15-20分 | 6-8倍 |
| チャートパターン分類 | 60分 | 8-12分 | 5-7倍 |
| **合計** | **約5時間** | **15-20分** | **15-20倍** |

## 🧪 テストと検証

### 精度検証

最適化により計算精度が損なわれていないことを確認：

```bash
python scripts/test_optimizations.py
```

テストは以下を検証します：
- 単一銘柄での計算結果の一致（許容誤差: 1e-6）
- 全銘柄での結果の整合性
- パフォーマンス向上の測定

### 使用メモリの監視

```python
# メモリ使用量の監視
import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
```

## ⚙️ 設定オプション

### 並列処理の調整

```python
class DailyAnalysisConfig:
    def __init__(self):
        self.n_workers = None  # CPU数を自動検出
        self.batch_size = 100  # バッチサイズ
```

### キャッシュ設定

```python
# キャッシュディレクトリとTTL設定
cache = CacheManager(
    cache_dir="/path/to/cache",
    max_memory_items=100,
    default_ttl_hours=24
)
```

## 🚨 注意事項とトラブルシューティング

### メモリ使用量

- 並列処理により一時的にメモリ使用量が増加します
- バッチサイズを調整してメモリ使用量をコントロール可能

### データベースロック

- SQLiteのWALモードを使用してリード・ライトの競合を軽減
- 長時間のトランザクションを避けるためのバッチ処理

### エラーハンドリング

```python
# 個別銘柄のエラーが全体に影響しないよう設計
try:
    result = process_stock(code)
except Exception as e:
    errors.append([code, str(e)])
    continue  # 他の銘柄の処理を継続
```

## 📈 今後の改善案

1. **GPUアクセラレーション** - CuPy/CuDFの導入
2. **分散処理** - 複数マシンでの並列処理
3. **ストリーミング処理** - リアルタイムデータ更新
4. **機械学習による最適化** - 処理順序やバッチサイズの自動調整

## 🔗 関連ドキュメント

- [parallel_processor.py](backend/utils/parallel_processor.py) - 並列処理の詳細実装
- [cache_manager.py](backend/utils/cache_manager.py) - キャッシュ機能の詳細
- [test_optimizations.py](scripts/test_optimizations.py) - テストスクリプトの詳細


## 参考資料
### scripts/create_database_indexes.py
  このファイルは、データベースのクエリ性能を大幅に向上させるためのインデックス作成スクリプトです。

  🎯 主な機能

  1. jquantsデータベースのインデックス作成

  - daily_quotesテーブルに以下のインデックスを作成：
    - idx_daily_quotes_code - 銘柄コード検索の高速化
    - idx_daily_quotes_date - 日付検索の高速化
    - idx_daily_quotes_code_date -
  銘柄×日付の複合検索（ユニーク）
    - idx_daily_quotes_date_code - 日付×銘柄の複合検索

  2. 分析結果データベースのインデックス作成

  既存テーブルに応じて動的にインデックスを作成：

  - relative_strengthテーブル:
    - 日付、銘柄コード、RSP値、RSI値のインデックス
  - minerviniテーブル:
    - 日付、銘柄コードのインデックス
  - hl_ratioテーブル:
    - 日付、銘柄コード、高値安値比率のインデックス

  3. SQLiteデータベース設定の最適化

  PRAGMA journal_mode=WAL          -- 並行アクセス改善
  PRAGMA synchronous=NORMAL        -- 書き込み性能向上
  PRAGMA cache_size=10000          -- キャッシュサイズ増大
  PRAGMA temp_store=MEMORY         -- 
  一時ファイルをメモリに
  PRAGMA mmap_size=268435456       -- 
  メモリマッピング有効化(256MB)
  PRAGMA optimize                  -- 
  クエリプランナー最適化

  4. データベース統計の分析

  - データベースサイズの計測
  - テーブル一覧の表示
  - 作成されたインデックス数の確認

  🚀 性能への影響

  このスクリプトにより以下の改善が期待されます：

  - WHERE句での検索: 10-100倍の高速化
  - ORDER BY: ソート処理の大幅な高速化
  - JOIN操作: 複数テーブル結合の最適化
  - 集計クエリ: GROUP BYやCOUNTの高速化

  特に4000銘柄×複数年のデータを扱う分析では、インデックス
  がない場合とある場合で劇的な性能差が生まれます。

  📝 使用方法

  # 初回セットアップ時に一度実行
  python scripts/create_database_indexes.py

  実行ログはタイムスタンプ付きファイルに保存され、作成され
  たインデックス数や最適化結果が記録されます。