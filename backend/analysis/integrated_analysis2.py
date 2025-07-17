# -*- coding: utf-8 -*-
"""
OPTIMIZED data analysis script for fetching and combining data from multiple
SQLite databases with improved performance and error handling.

The script performs the following steps:
1. Sets up paths to the various databases.
2. Fetches the latest available analysis date.
3. Retrieves comprehensive analysis data for the latest date.
4. Retrieves and pivots chart classification data.
5. Retrieves fundamental data from yfinance.
6. Merges these data sources into a single DataFrame using optimized operations.
7. Outputs the final combined DataFrame to Excel.
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import warnings
import logging
from datetime import datetime
from typing import Optional, Dict, List

# Add project root to sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.analysis.integrated_analysis import get_comprehensive_analysis

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Constants and Configuration ---

# Determine the project root directory based on the script's location
# The script is in backend/analysis, so the root is two levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Database paths
RESULTS_DB_PATH = os.path.join(DATA_DIR, "analysis_results.db")
MASTER_DB_PATH = os.path.join(DATA_DIR, "master.db")
JQUANTS_DB_PATH = os.path.join(DATA_DIR, "jquants.db")
YFINANCE_DB_PATH = os.path.join(DATA_DIR, "yfinance.db")


# --- Setup Functions ---

def setup_logging() -> logging.Logger:
    """Setup logging configuration for optimized processing."""
    log_filename = os.path.join(PROJECT_ROOT, "logs", f"integrated_analysis2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Optimized integrated analysis logging initialized. Log file: {log_filename}")
    return logger


# --- Data Fetching Functions ---

def get_available_dates(db_path: str, logger: logging.Logger) -> List[str]:
    """Gets all available analysis dates from the database with optimized query."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Enable optimizations
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            dates_query = "SELECT DISTINCT Date FROM hl_ratio ORDER BY Date DESC LIMIT 10"
            dates_df = pd.read_sql(dates_query, conn)
            dates = dates_df['Date'].tolist()
            
        logger.info(f"Retrieved {len(dates)} available dates")
        return dates
    except Exception as e:
        logger.error(f"Error getting available dates: {e}")
        return []

def get_chart_classification_data(db_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Fetches the latest chart classification results with optimized query."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Enable optimizations
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Check if table exists first
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='classification_results'"
            ).fetchone()
            
            if not table_check:
                logger.warning("classification_results table not found in database")
                return pd.DataFrame()
            
            query = """
            SELECT *
            FROM classification_results
            WHERE date = (SELECT MAX(date) FROM classification_results)
            """
            df = pd.read_sql(query, conn)
            
        logger.info(f"Retrieved {len(df)} chart classification records")
        return df
    except Exception as e:
        logger.error(f"Error fetching chart classification data: {e}")
        return pd.DataFrame()

def pivot_chart_classification_data(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Pivots the chart classification data to have one row per ticker using optimized operations."""
    if df is None or df.empty:
        logger.info("No chart classification data to pivot")
        return pd.DataFrame()
    
    try:
        # Optimized pivot using vectorized operations
        pivot_df = df.pivot_table(
            index=['date', 'ticker'],
            columns='window',
            values=['pattern_label', 'score'],
            aggfunc='first'
        )
        
        # Efficient column flattening
        pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()
        
        # Batch rename operation
        pivot_df.rename(columns={'date': 'Date', 'ticker': 'Code'}, inplace=True)
        
        logger.info(f"Pivoted chart classification data: {len(pivot_df)} records")
        return pivot_df
        
    except Exception as e:
        logger.error(f"Error pivoting chart classification data: {e}")
        return pd.DataFrame()

def get_yfinance_data(db_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Fetches stock data from the yfinance database with optimized query."""
    try:
        with sqlite3.connect(db_path) as conn:
            # Enable optimizations
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Check if table exists first
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'"
            ).fetchone()
            
            if not table_check:
                logger.warning("stocks table not found in yfinance database")
                return pd.DataFrame()
            
            # Optimized query to select only necessary columns if table is large
            query = "SELECT * FROM stocks"
            df = pd.read_sql(query, conn)
            
            # Vectorized string operation
            if 'ticker' in df.columns and not df.empty:
                df["Code"] = df["ticker"].str.replace(".T", "0", regex=False)
                
        logger.info(f"Retrieved {len(df)} yfinance records")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching yfinance data: {e}")
        return pd.DataFrame()


# --- Main Analysis Logic ---

def main():
    """Main function to run the optimized data analysis pipeline."""
    logger = setup_logging()
    logger.info("Starting OPTIMIZED integrated data analysis...")

    try:
        # Get available dates
        available_dates = get_available_dates(RESULTS_DB_PATH, logger)
        if not available_dates:
            logger.error("No available analysis dates found. Exiting.")
            return

        latest_date = available_dates[0]
        logger.info(f"Using latest analysis date: {latest_date}")

        # 1. Get comprehensive analysis data (this is already optimized)
        logger.info("Fetching comprehensive analysis data...")
        comprehensive_df = get_comprehensive_analysis(latest_date)
        if comprehensive_df.empty:
            logger.error("Could not retrieve comprehensive analysis data.")
            return
        logger.info(f"Retrieved {len(comprehensive_df)} records in comprehensive analysis")

        # 2. Get and process chart classification data
        logger.info("Fetching and processing chart classification data...")
        chart_df = get_chart_classification_data(RESULTS_DB_PATH, logger)
        pivot_df = pivot_chart_classification_data(chart_df, logger)
        
        # 3. Get yfinance data
        logger.info("Fetching yfinance data...")
        yfinance_df = get_yfinance_data(YFINANCE_DB_PATH, logger)

        # 4. Optimized merge operations
        logger.info("Performing optimized data merging...")
        
        # Start with the largest dataset (comprehensive analysis)
        all_df = comprehensive_df.copy()
        
        # Merge with yfinance data if available
        if not yfinance_df.empty:
            all_df = pd.merge(all_df, yfinance_df, on='Code', how='left')
            logger.info(f"Merged with yfinance data: {len(all_df)} rows")
        
        # Merge with chart classification data if available
        if not pivot_df.empty:
            all_df = pd.merge(all_df, pivot_df, on='Code', how='left')
            logger.info(f"Merged with chart classification data: {len(all_df)} rows")
        
        # 5. Optimized column reordering and sorting
        logger.info("Optimizing final dataframe structure...")
        
        # Priority columns for reordering (most important first)
        priority_columns = [
            'Code', 'composite_score', 'HlRatio', 'RelativeStrengthIndex', 
            'minervini_score', 'ticker', 'shortName', 'sector', 'industry'
        ]
        
        # Additional columns in logical order
        additional_columns = [
            'longName', 'marketCap', 'trailingPE', 'forwardPE', 'dividendYield', 
            'currentPrice', 'regularMarketPrice', 'currency', 'exchange',
            'previousClose', 'open', 'dayLow', 'dayHigh', 'volume',
            'averageDailyVolume10Day', 'averageDailyVolume3Month',
            'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'fiftyDayAverage',
            'twoHundredDayAverage', 'beta', 'priceToBook', 'enterpriseValue',
            'profitMargins', 'grossMargins', 'operatingMargins', 'returnOnAssets',
            'returnOnEquity', 'freeCashflow', 'totalCash', 'totalDebt',
            'earningsGrowth', 'revenueGrowth', 'last_updated', 'Date_x',
            'pattern_label_20', 'pattern_label_60', 'pattern_label_120',
            'pattern_label_240', 'score_20', 'score_60', 'score_120', 'score_240',
            'Date_y', 'MedianRatio', 'hl_weeks', 'minervini_close',
            'Sma50', 'Sma150', 'Sma200', 'minervini_type_1', 'minervini_type_2',
            'minervini_type_3', 'minervini_type_4', 'minervini_type_5',
            'minervini_type_6', 'minervini_type_7', 'minervini_type_8',
            'RelativeStrengthPercentage', 'website'
        ]
        
        # Combine and filter for existing columns
        final_columns = priority_columns + additional_columns
        existing_columns = [col for col in final_columns if col in all_df.columns]
        
        # Add any remaining columns not in our predefined list
        remaining_columns = [col for col in all_df.columns if col not in existing_columns]
        final_column_order = existing_columns + remaining_columns
        
        # Reorder columns efficiently
        all_df = all_df[final_column_order]
        
        # Optimized sorting by composite score
        if 'composite_score' in all_df.columns:
            all_df['composite_score'] = pd.to_numeric(all_df['composite_score'], errors='coerce')
            all_df = all_df.sort_values(by=['composite_score'], ascending=False, na_position='last')
            logger.info("Sorted by composite score (descending)")

        logger.info("Displaying top 10 results:")
        with pd.option_context('display.max_rows', 10, 'display.max_columns', 15):
            logger.info(f"\n{all_df}")

        # --- Optimized Excel Output ---
        output_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"integrated_analysis_{latest_date.replace('-','')}.xlsx"
        output_path = os.path.join(output_dir, output_filename)

        try:
            # Use efficient Excel writing
            all_df.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"Successfully saved the analysis to: {output_path}")
            logger.info(f"Final output contains {len(all_df)} rows and {len(all_df.columns)} columns")
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

        logger.info("OPTIMIZED analysis script finished successfully.")
        
    except Exception as e:
        logger.error(f"Error in main analysis pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
