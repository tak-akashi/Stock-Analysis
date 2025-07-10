"""
Chart Classification (Batch Processing)
=========================================

This script performs chart pattern classification on stock price data.
It can be run in two modes:

1.  **Sample Mode (`--mode sample`)**:
    -   Analyzes a predefined list of stock tickers.
    -   Saves the resulting classification plots as PNG images in the output directory.

2.  **Full Mode (`--mode full`)**:
    -   Fetches all tickers from the master database.
    -   Runs classification for all tickers across all specified time windows.
    -   Saves the classification results (label, score) into a SQLite database (`analysis_results.db`).

Usage:
------
-   For a sample run: `python chart_classification.py --mode sample`
-   For a full run:   `python chart_classification.py --mode full`
"""

import argparse
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# --- Constants ---
JQUANTS_DB_PATH = "/Users/tak/Markets/Stocks/Stock-Analysis/data/jquants.db"
MASTER_DB_PATH = "/Users/tak/Markets/Stocks/Stock-Analysis/data/master.db" # Assumes master.db is in the data directory
OUTPUT_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/output"
DATA_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/data"
LOGS_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/logs"
RESULTS_DB_PATH = os.path.join(DATA_DIR, "analysis_results.db")

def setup_logging():
    """Setup logging configuration"""
    log_filename = os.path.join(LOGS_DIR, f"chart_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

class ChartClassifier:
    """
    A class to classify chart patterns for a given stock ticker and window.
    """

    def __init__(self, ticker: str, window: int, db_path: str = JQUANTS_DB_PATH):
        self.ticker = ticker
        self.window = window
        self.db_path = db_path
        self.price_data = self._get_stock_data()
        self.templates_manual = self._create_manual_templates()

    def _get_stock_data(self, days: int = 500) -> pd.Series:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                SELECT Date, AdjustmentClose 
                FROM daily_quotes 
                WHERE Code = ? AND Date BETWEEN ? AND ?
                ORDER BY Date
                """
                df = pd.read_sql_query(
                    query,
                    conn,
                    params=[self.ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')],
                    parse_dates=['Date']
                )
        except sqlite3.Error as e:
            raise ConnectionError(f"Database connection or query failed: {e}")

        if len(df) < self.window:
            raise ValueError(f"Not enough data for ticker {self.ticker} with window {self.window} (found {len(df)} days)")

        return df.set_index('Date')['AdjustmentClose'].dropna()

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler()
        return scaler.fit_transform(arr.reshape(-1, 1)).flatten()

    def _create_manual_templates(self) -> Dict[str, np.ndarray]:
        half1 = self.window // 2
        half2 = self.window - half1
        templates = {
            "上昇ストップ": np.concatenate([np.linspace(0, 1, half1), np.full(half2, 1)]),
            "上昇": np.linspace(0, 1, self.window),
            "急上昇": np.concatenate([np.full(half1, 0), np.linspace(0, 1, half2)]),
            "調整": np.concatenate([np.linspace(0, 1, half1), np.linspace(1, 0, half2)]),
            "もみ合い": np.sin(np.linspace(0, 4 * np.pi, self.window)),
            "リバウンド": np.concatenate([np.linspace(1, 0, half1), np.linspace(0, 1, half2)]),
            "急落": np.concatenate([np.full(half1, 1), np.linspace(1, 0, half2)]),
            "下落": np.linspace(1, 0, self.window),
            "下げとまった": np.concatenate([np.linspace(1, 0, half1), np.full(half2, 0)]),
        }
        return {name: self._normalize(template) for name, template in templates.items()}

    def _find_best_match(self, series: np.ndarray, templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
        normalized_series = self._normalize(series)
        best_label, best_score = None, -np.inf
        for label, tpl in templates.items():
            score, _ = pearsonr(normalized_series, tpl)
            if np.isnan(score):
                score = 0
            if score > best_score:
                best_label, best_score = label, score
        return best_label, best_score

    def classify_latest(self) -> Tuple[str, float]:
        latest_data = self.price_data.iloc[-self.window:].values
        return self._find_best_match(latest_data, self.templates_manual)

    def save_classification_plot(self, label: str, score: float, output_dir: str):
        latest_data = self.price_data.iloc[-self.window:].values
        normalized_latest = self._normalize(latest_data)
        template = self.templates_manual[label]

        fig = plt.figure(figsize=(10, 5))
        plt.plot(normalized_latest, label="最新の株価", linewidth=2)
        plt.plot(template, "--", label=f"テンプレート: {label}")
        plt.title(f"銘柄: {self.ticker} (直近{self.window}日) vs. パターン: {label} (r={score:.3f})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{self.ticker}_window{self.window}_{label}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close(fig)
        print(f"Plot saved to {filepath}")

# --- Database Utility Functions ---

def get_all_tickers(db_path: str) -> List[str]:
    """Fetches all unique ticker codes from the master database."""
    print(f"Reading all tickers from {db_path}...")
    try:
        with sqlite3.connect(db_path) as conn:
            # Assuming the table is named 'master' or 'stocks'. Adjust if necessary.
            df = pd.read_sql_query("SELECT * FROM stocks_master", conn)
        tickers = df['jquants_code'].astype(str).tolist()
        print(f"Found {len(tickers)} unique tickers.")
        return tickers
    except Exception as e:
        print(f"Error reading from master database: {e}")
        return []

def init_results_db(db_path: str):
    """Initializes the results database and creates the table if it doesn't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS classification_results (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            window INTEGER NOT NULL,
            pattern_label TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY (date, ticker, window)
        )
        """
        )
        conn.commit()

def save_result_to_db(db_path: str, date: str, ticker: str, window: int, label: str, score: float):
    """Saves a single classification result to the database."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO classification_results (date, ticker, window, pattern_label, score)
        VALUES (?, ?, ?, ?, ?)
        """
        , (date, ticker, window, label, score))
        conn.commit()

# --- Main Execution Functions ---

def main_sample():
    """Runs classification for a sample of tickers and saves plots."""
    TICKERS = ["74530", "99840", "67580"]  # Example: Fast Retailing, Softbank, Sony
    WINDOWS = [20, 60, 120, 240]

    print("==== サンプル実行開始 ====")
    for ticker in TICKERS:
        for window in WINDOWS:
            try:
                classifier = ChartClassifier(ticker=ticker, window=window)
                label, score = classifier.classify_latest()
                print(f"[銘柄: {ticker}, 期間: {window}日] -> 分類: {label} (r={score:.3f})")
                classifier.save_classification_plot(label, score, OUTPUT_DIR)
            except (ValueError, ConnectionError) as e:
                print(f"エラー (銘柄: {ticker}, 期間: {window}日): {e}")
            print("---")
    print("==== サンプル実行完了 ====")

def main_full_run():
    """Runs classification for all tickers and saves results to the database."""
    WINDOWS = [20, 60, 120, 240]
    all_tickers = get_all_tickers(MASTER_DB_PATH)
    today_str = datetime.today().strftime('%Y-%m-%d')

    if not all_tickers:
        print("銘柄リストが空のため、処理を終了します。")
        return

    print(f"==== 全銘柄分類処理開始 ({len(all_tickers)}銘柄) ====")
    init_results_db(RESULTS_DB_PATH)

    for i, ticker in enumerate(all_tickers):
        print(f"\n--- 処理中: {ticker} ({i+1}/{len(all_tickers)}) ---")
        for window in WINDOWS:
            try:
                classifier = ChartClassifier(ticker=ticker, window=window)
                label, score = classifier.classify_latest()
                save_result_to_db(RESULTS_DB_PATH, today_str, ticker, window, label, score)
                print(f"  [期間: {window}日] -> {label} (r={score:.3f}) ... DB保存済み")
            except (ValueError, ConnectionError) as e:
                print(f"  [期間: {window}日] -> エラー: {e}")

    print("\n==== 全ての処理が完了しました ====")
    print(f"結果は {RESULTS_DB_PATH} に保存されています。")

def main():
    """Main function that handles argument parsing and dispatches to appropriate execution mode."""
    parser = argparse.ArgumentParser(description="Chart Pattern Classification for Stocks.")
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['sample', 'full'],
        help="Execution mode: 'sample' to run on a few examples and save plots, 'full' to run on all tickers and save to DB."
    )
    args = parser.parse_args()

    if args.mode == 'sample':
        main_sample()
    elif args.mode == 'full':
        main_full_run()

if __name__ == "__main__":
    main()