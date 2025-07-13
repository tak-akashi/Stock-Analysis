# -*- coding: utf-8 -*-
"""
This script performs data analysis by fetching and combining data from multiple
SQLite databases. It is a script version of the analysis originally performed in
the data_analysis_from_sqlite.ipynb notebook.

The script performs the following steps:
1. Sets up paths to the various databases.
2. Fetches the latest available analysis date.
3. Retrieves comprehensive analysis data for the latest date.
4. Retrieves and pivots chart classification data.
5. Retrieves fundamental data from yfinance.
6. Merges these data sources into a single DataFrame.
7. Prints the final combined DataFrame.
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import warnings

# Since this script is in the same directory as integrated_analysis,
# the sys.path manipulation from the notebook is no longer needed.
from integrated_analysis import get_comprehensive_analysis

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


# --- Data Fetching Functions ---

def get_available_dates(db_path):
    """Gets all available analysis dates from the database."""
    try:
        with sqlite3.connect(db_path) as conn:
            dates_query = "SELECT DISTINCT Date FROM hl_ratio ORDER BY Date DESC"
            dates_df = pd.read_sql(dates_query, conn)
            return dates_df['Date'].tolist()
    except Exception as e:
        print(f"Error getting available dates: {e}")
        return []

def get_chart_classification_data(db_path):
    """Fetches the latest chart classification results."""
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT *
            FROM classification_results
            WHERE date = (SELECT MAX(date) FROM classification_results)
            """
            return pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error fetching chart classification data: {e}")
        return pd.DataFrame()

def pivot_chart_classification_data(df):
    """Pivots the chart classification data to have one row per ticker."""
    if df is None or df.empty:
        return pd.DataFrame()
    pivot_df = df.pivot_table(
        index=['date', 'ticker'],
        columns='window',
        values=['pattern_label', 'score'],
        aggfunc='first'
    )
    pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    pivot_df.rename(columns={'date': 'Date', 'ticker': 'Code'}, inplace=True)
    return pivot_df

def get_yfinance_data(db_path):
    """Fetches stock data from the yfinance database."""
    try:
        with sqlite3.connect(db_path) as conn:
            query = "SELECT * FROM stocks"
            df = pd.read_sql(query, conn)
            # Ensure 'ticker' column exists before trying to modify it
            if 'ticker' in df.columns:
                df["Code"] = df["ticker"].str.replace(".T", "0", regex=False)
            return df
    except Exception as e:
        print(f"Error fetching yfinance data: {e}")
        return pd.DataFrame()


# --- Main Analysis Logic ---

def main():
    """Main function to run the data analysis pipeline."""
    print("Starting data analysis...")

    # Get available dates
    available_dates = get_available_dates(RESULTS_DB_PATH)
    if not available_dates:
        print("No available analysis dates found. Exiting.")
        return

    latest_date = available_dates[0]
    print(f"\nLatest analysis date: {latest_date}")

    # 1. Get comprehensive analysis data
    print("\nFetching comprehensive analysis data...")
    comprehensive_df = get_comprehensive_analysis(latest_date)
    if comprehensive_df.empty:
        print("Could not retrieve comprehensive analysis data.")
        return
    print(f"Found {len(comprehensive_df)} records in comprehensive analysis.")

    # 2. Get and process chart classification data
    print("\nFetching and processing chart classification data...")
    chart_df = get_chart_classification_data(RESULTS_DB_PATH)
    pivot_df = pivot_chart_classification_data(chart_df)
    if pivot_df.empty:
        print("Could not retrieve chart classification data.")
        # Continue without this data if not critical
    else:
        print(f"Found {len(pivot_df)} records in chart classification.")


    # 3. Get yfinance data
    print("\nFetching yfinance data...")
    yfinance_df = get_yfinance_data(YFINANCE_DB_PATH)
    if yfinance_df.empty:
        print("Could not retrieve yfinance data.")
        # Continue without this data if not critical
    else:
        print(f"Found {len(yfinance_df)} records in yfinance data.")

    # 4. Merge all dataframes
    print("\nMerging data sources...")
    # Merge yfinance and chart patterns first
    merged_df = pd.merge(yfinance_df, pivot_df, on='Code', how='outer')
    # Merge with comprehensive analysis data
    all_df = pd.merge(merged_df, comprehensive_df, on="Code", how="outer")
    print(f"Final merged dataframe has {len(all_df)} rows.")

    # Reorder columns to match the notebook's final structure
    final_columns = [
        'Code', 'ticker', 'longName', 'sector', 'industry', 'marketCap', 'trailingPE',
        'forwardPE', 'dividendYield', 'website', 'currentPrice',
        'regularMarketPrice', 'currency', 'exchange', 'shortName',
        'previousClose', 'open', 'dayLow', 'dayHigh', 'volume',
        'averageDailyVolume10Day', 'averageDailyVolume3Month',
        'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'fiftyDayAverage',
        'twoHundredDayAverage', 'beta', 'priceToBook', 'enterpriseValue',
        'profitMargins', 'grossMargins', 'operatingMargins', 'returnOnAssets',
        'returnOnEquity', 'freeCashflow', 'totalCash', 'totalDebt',
        'earningsGrowth', 'revenueGrowth', 'last_updated', 'Date_x',
        'pattern_label_20', 'pattern_label_60', 'pattern_label_120',
        'pattern_label_240', 'score_20', 'score_60', 'score_120', 'score_240',
        'Date_y', 'HlRatio', 'MedianRatio', 'hl_weeks', 'minervini_close',
        'Sma50', 'Sma150', 'Sma200', 'minervini_type_1', 'minervini_type_2',
        'minervini_type_3', 'minervini_type_4', 'minervini_type_5',
        'minervini_type_6', 'minervini_type_7', 'minervini_type_8',
        'RelativeStrengthPercentage', 'RelativeStrengthIndex',
        'minervini_score', 'composite_score'
    ]
    
    # Filter for columns that actually exist in the merged dataframe
    existing_columns = [col for col in final_columns if col in all_df.columns]
    all_df = all_df[existing_columns]
    all_df = all_df.sort_values(by='composite_score', ascending=False)

    print("\n--- Final Merged Data ---")
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
        print(all_df)

    # --- Output to Excel ---
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
    output_filename = f"comprehensive_analysis_{latest_date.replace('-','')}.xlsx"
    output_path = os.path.join(output_dir, output_filename)

    try:
        all_df.to_excel(output_path, index=False)
        print(f"\nSuccessfully saved the analysis to: {output_path}")
    except Exception as e:
        print(f"\nError saving to Excel: {e}")

    print("\nAnalysis script finished.")


if __name__ == "__main__":
    main()
