import sqlite3
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import functions from other analysis modules
from backend.analysis.high_low_ratio import calc_hl_ratio_for_all, init_hl_ratio_db
from backend.analysis.minervini import update_minervini_db, update_type8_db, MinerviniConfig, setup_logging, init_minervini_db
from backend.analysis.relative_strength import update_rsp_db, update_rsi_db, init_rsp_db, init_results_db
from backend.analysis.integrated_analysis import create_analysis_summary


class DatabaseManager:
    """Database connection manager for analysis workflow."""
    
    def __init__(self, jquants_db_path: str, results_db_path: str):
        self.jquants_db_path = jquants_db_path
        self.results_db_path = results_db_path
        self.jquants_conn = None
        self.results_conn = None
    
    def __enter__(self):
        """Enter context manager and open connections."""
        self.jquants_conn = sqlite3.connect(self.jquants_db_path)
        self.results_conn = sqlite3.connect(self.results_db_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close connections."""
        if self.jquants_conn:
            self.jquants_conn.close()
        if self.results_conn:
            self.results_conn.close()


class DailyAnalysisConfig:
    """Configuration for daily analysis workflow."""
    
    def __init__(self, base_config: Optional[MinerviniConfig] = None):
        self.base_config = base_config or MinerviniConfig()
        
        # Analysis periods
        self.rsp_period_days = 360  # Days to look back for RSP calculation
        self.update_window_days = 5  # Days to update in recent period
        self.hl_ratio_weeks = 52    # Weeks for high-low ratio calculation
        
        # Database paths from MinerviniConfig
        self.jquants_db_path = str(self.base_config.jquants_db_path)
        self.results_db_path = str(self.base_config.results_db_path)
    
    def setup_logger(self) -> logging.Logger:
        """Setup and return a logger instance."""
        return setup_logging(self.base_config)
    
    def get_database_manager(self) -> DatabaseManager:
        """Get a database manager instance."""
        return DatabaseManager(self.jquants_db_path, self.results_db_path)

def run_daily_analysis() -> bool:
    """
    Run the daily analysis workflow.
    
    Returns:
        bool: True if all analysis steps completed successfully, False otherwise
    """
    analysis_config = DailyAnalysisConfig()
    logger = analysis_config.setup_logger()
    logger.info("Starting daily analysis workflow.")

    success = True

    try:
        with analysis_config.get_database_manager() as db_manager:
            # Get all stock codes from jquants.db
            cursor = db_manager.jquants_conn.cursor()
            cursor.execute("SELECT DISTINCT Code FROM daily_quotes")
            code_list = [row[0] for row in cursor.fetchall()]
            logger.info(f"Found {len(code_list)} stock codes for analysis.")

            # Define date range for updates
            end_date = datetime.now()
            calc_end_date_str = end_date.strftime('%Y-%m-%d')
            calc_start_date_str = (end_date - timedelta(days=analysis_config.rsp_period_days)).strftime('%Y-%m-%d')

            # --- Analysis Steps ---

            # 1. Relative Strength Percentage (RSP) Update
            logger.info("Running Relative Strength Percentage (RSP) update...")
            try:
                cursor = db_manager.results_conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='relative_strength'")
                table_exists = cursor.fetchone() is not None
                
                if not table_exists:
                    logger.info("relative_strength table not found. Initializing with full data...")
                    processed, errors = init_rsp_db(result_db_path=analysis_config.results_db_path)
                    logger.info("relative_strength table initialization completed.")
                else:
                    logger.info("relative_strength table found. Updating recent data...")
                    processed, errors = update_rsp_db(
                        db_path=analysis_config.jquants_db_path, 
                        result_db_path=analysis_config.results_db_path,
                        calc_start_date=calc_start_date_str, 
                        calc_end_date=calc_end_date_str, 
                        period=-analysis_config.update_window_days
                    )
                
                if errors > 0:
                    logger.warning(f"RSP update completed with {errors} errors. Processed {processed} stocks.")
                    success = False
                else:
                    logger.info(f"RSP update completed successfully. Processed {processed} stocks.")
                    
            except Exception as e:
                logger.error(f"Error in RSP update: {e}", exc_info=True)
                success = False

            # 2. Relative Strength Index (RSI) Update (depends on RSP)
            logger.info("Running Relative Strength Index (RSI) update...")
            try:
                date_list_for_rsi = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') 
                                   for i in range(analysis_config.update_window_days)]
                errors = update_rsi_db(
                    result_db_path=analysis_config.results_db_path, 
                    date_list=date_list_for_rsi, 
                    period=-analysis_config.update_window_days
                )
                
                if errors > 0:
                    logger.warning(f"RSI update completed with {errors} errors.")
                    success = False
                else:
                    logger.info("RSI update completed successfully.")
                    
            except Exception as e:
                logger.error(f"Error in RSI update: {e}", exc_info=True)
                success = False

            # 3. Minervini Analysis Update
            logger.info("Running Minervini analysis update...")
            try:
                cursor = db_manager.results_conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='minervini'")
                table_exists = cursor.fetchone() is not None

                if not table_exists:
                    logger.info("'minervini' table not found. Initializing with full data...")
                    init_minervini_db(db_manager.jquants_conn, db_manager.results_conn, code_list)
                    logger.info("'minervini' table initialization completed.")
                else:
                    logger.info("'minervini' table found. Updating recent data...")
                    update_minervini_db(
                        db_manager.jquants_conn, # source_conn
                        db_manager.results_conn, # dest_conn
                        code_list, 
                        calc_start_date_str, 
                        calc_end_date_str, 
                        period=analysis_config.update_window_days
                    )
                logger.info("Minervini analysis update completed successfully.")
            except Exception as e:
                logger.error(f"Error in Minervini update: {e}", exc_info=True)
                success = False

            # 4. Minervini Type 8 Update (depends on RSI, so it must run after RSI update)
            logger.info("Running Minervini Type 8 update...")
            try:
                update_type8_db(
                    db_manager.results_conn, 
                    date_list_for_rsi, 
                    period=-analysis_config.update_window_days
                )
                logger.info("Minervini Type 8 update completed successfully.")
            except Exception as e:
                logger.error(f"Error in Minervini Type 8 update: {e}", exc_info=True)
                success = False

            # 5. High-Low Ratio Calculation
            logger.info("Running High-Low Ratio calculation...")
            try:
                # Ensure hl_ratio table exists
                cursor = db_manager.results_conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hl_ratio'")
                table_exists = cursor.fetchone() is not None
                if not table_exists:
                    logger.info("'hl_ratio' table not found. Initializing...")
                    init_hl_ratio_db(db_path=analysis_config.results_db_path)
                    logger.info("'hl_ratio' table initialized.")

                # Calculate HL Ratio
                result_df = calc_hl_ratio_for_all(
                    db_path=analysis_config.jquants_db_path,
                    end_date=end_date.strftime('%Y-%m-%d'),
                    weeks=analysis_config.hl_ratio_weeks
                )

                if result_df is not None and not result_df.empty:
                    # Save results to the database, replacing old data
                    result_df.to_sql(
                        'hl_ratio',
                        db_manager.results_conn,
                        if_exists='replace',
                        index=False,
                        dtype={'Date': 'TEXT', 'Code': 'TEXT', 'HlRatio': 'REAL', 'Weeks': 'INTEGER'}
                    )
                    logger.info(f"High-Low Ratio calculation completed and saved for {len(result_df)} stocks.")
                else:
                    logger.warning("High-Low Ratio calculation returned no results.")
                    success = False
            except Exception as e:
                logger.error(f"Error in High-Low Ratio calculation: {e}", exc_info=True)
                success = False

            # 6. Create Analysis Summary
            logger.info("Creating daily analysis summary...")
            try:
                summary_date = end_date.strftime('%Y-%m-%d')
                summary = create_analysis_summary(date=summary_date, db_path=analysis_config.results_db_path)
                if summary:
                    logger.info(f"Daily Analysis Summary for {summary_date}:")
                    for key, value in summary.items():
                        logger.info(f"  {key}: {value}")
                else:
                    logger.warning(f"No summary data generated for {summary_date}.")
                    success = False
                logger.info("Analysis summary creation completed.")
            except Exception as e:
                logger.error(f"Error creating analysis summary: {e}", exc_info=True)
                success = False

        status_msg = "Daily analysis workflow finished successfully." if success else "Daily analysis workflow completed with errors."
        logger.info(status_msg)

    except Exception as e:
        logger.error(f"An error occurred during daily analysis workflow: {e}", exc_info=True)
        success = False
    
    return success

if __name__ == "__main__":
    run_daily_analysis()
