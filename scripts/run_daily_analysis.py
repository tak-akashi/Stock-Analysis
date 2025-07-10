import sqlite3
import logging
import os
from datetime import datetime, timedelta

# Import functions from other analysis modules
from backend.analysis.high_low_ratio import calc_hl_ratio_for_all
from backend.analysis.minervini import update_minervini_db, update_type8_db, MinerviniConfig, setup_logging
from backend.analysis.relative_strength import update_rsp_db, update_rsi_db
from backend.analysis.integrated_analysis import create_analysis_summary

# Constants
DATA_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/data"
JQUANTS_DB_PATH = os.path.join(DATA_DIR, "jquants.db")
RESULTS_DB_PATH = os.path.join(DATA_DIR, "analysis_results.db")

def run_daily_analysis():
    config = MinerviniConfig() # Reusing MinerviniConfig for paths and logging setup
    logger = setup_logging(config)
    logger.info("Starting daily analysis workflow.")

    conn_jquants = None
    conn_results = None

    try:
        conn_jquants = sqlite3.connect(JQUANTS_DB_PATH)
        conn_results = sqlite3.connect(RESULTS_DB_PATH)

        # Get all stock codes from jquants.db (assuming 'prices' table exists)
        cursor = conn_jquants.cursor()
        cursor.execute("SELECT DISTINCT code FROM prices")
        code_list = [row[0] for row in cursor.fetchall()]
        logger.info(f"Found {len(code_list)} stock codes for analysis.")

        # Define date range for updates
        end_date = datetime.now()
        # For daily updates, we typically update for the latest day or a few recent days.
        # Let's use a 5-day window for updates to catch any late data or re-runs.
        calc_end_date_str = end_date.strftime('%Y-%m-%d')
        calc_start_date_str = (end_date - timedelta(days=5)).strftime('%Y-%m-%d')

        # --- Analysis Steps ---

        # 1. Relative Strength Percentage (RSP) Update
        logger.info("Running Relative Strength Percentage (RSP) update...")
        update_rsp_db(db_path=JQUANTS_DB_PATH, result_db_path=RESULTS_DB_PATH,
                      calc_start_date=calc_start_date_str, calc_end_date=calc_end_date_str, period=-5)
        logger.info("RSP update completed.")

        # 2. Relative Strength Index (RSI) Update (depends on RSP)
        logger.info("Running Relative Strength Index (RSI) update...")
        # update_rsi_db can fetch recent dates from DB, or we can pass a list of dates.
        # Let's pass the last 5 days for RSI calculation to ensure consistency.
        date_list_for_rsi = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)]
        update_rsi_db(result_db_path=RESULTS_DB_PATH, date_list=date_list_for_rsi, period=-5)
        logger.info("RSI update completed.")

        # 3. Minervini Analysis Update
        logger.info("Running Minervini analysis update...")
        # update_minervini_db expects a jquants connection, code list, and date range.
        update_minervini_db(conn_jquants, code_list, calc_start_date_str, calc_end_date_str, period=5)
        logger.info("Minervini analysis update completed.")

        # 4. Minervini Type 8 Update (depends on RSI, so it must run after RSI update)
        logger.info("Running Minervini Type 8 update...")
        # update_type8_db expects a results connection and a list of dates.
        update_type8_db(conn_results, date_list_for_rsi, period=-5)
        logger.info("Minervini Type 8 update completed.")

        # 5. High-Low Ratio Calculation
        logger.info("Running High-Low Ratio calculation...")
        # calc_hl_ratio_for_all takes jquants db path and end date.
        calc_hl_ratio_for_all(db_path=JQUANTS_DB_PATH, end_date=end_date.strftime('%Y-%m-%d'), weeks=52)
        logger.info("High-Low Ratio calculation completed.")

        # 6. Create Analysis Summary
        logger.info("Creating daily analysis summary...")
        summary_date = end_date.strftime('%Y-%m-%d')
        summary = create_analysis_summary(date=summary_date, db_path=RESULTS_DB_PATH)
        if summary:
            logger.info(f"Daily Analysis Summary for {summary_date}:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.warning(f"No summary data generated for {summary_date}.")
        logger.info("Analysis summary creation completed.")

        logger.info("Daily analysis workflow finished successfully.")

    except Exception as e:
        logger.error(f"An error occurred during daily analysis workflow: {e}", exc_info=True)
    finally:
        if conn_jquants:
            conn_jquants.close()
        if conn_results:
            conn_results.close()

if __name__ == "__main__":
    run_daily_analysis()
