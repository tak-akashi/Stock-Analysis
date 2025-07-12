import numpy as np
import pandas as pd
import sqlite3
import datetime
import logging
import os
from datetime import timedelta

# --- Constants ---
JQUANTS_DB_PATH = "/Users/tak/Markets/Stocks/Stock-Analysis/data/jquants.db"
DATA_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/data"
LOGS_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/logs"
OUTPUT_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/output"
RESULTS_DB_PATH = os.path.join(DATA_DIR, "analysis_results.db")

def setup_logging():
    """Setup logging configuration"""
    log_filename = os.path.join(LOGS_DIR, f"relative_strength_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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

def fill_non_vals(close_arr, rsp, period=200):
    """Fill non-values with NaN for the beginning of the array"""
    result = np.full(len(close_arr), np.nan)
    result[period:] = rsp
    return result


def relative_strength_percentage(close_arr, period=200):
    """
    Relative Strengthを算出する。
    [計算方法] http://www.great-trades.com/Help/rs.htm
    @param close_arr: numpy.array
    @param period: int
    @return: numpy.array
    """
    logger = logging.getLogger(__name__)
    
    if len(close_arr) < period:
        logger.warning(f"Insufficient data for RSP calculation: {len(close_arr)} < {period}")
        return np.full(len(close_arr), np.nan)
    
    try:
        # Helper function to safely calculate percentage change
        def safe_pct_change(current, previous):
            if previous is None or previous == 0 or np.isnan(previous) or current is None or np.isnan(current):
                return np.nan
            return (current - previous) / previous
        
        q1 = np.array([safe_pct_change(close_arr[idx - int(period * 3 / 4) + 1], close_arr[idx - period + 1]) 
                       for idx in range(period, len(close_arr))])
        q2 = np.array([safe_pct_change(close_arr[idx - int(period * 2 / 4) + 1], close_arr[idx - int(period * 3 / 4) + 1]) 
                       for idx in range(period, len(close_arr))])
        q3 = np.array([safe_pct_change(close_arr[idx - int(period * 1 / 4) + 1], close_arr[idx - int(period * 2 / 4) + 1]) 
                       for idx in range(period, len(close_arr))])
        q4 = np.array([safe_pct_change(close_arr[idx], close_arr[idx - int(period * 1 / 4) + 1]) 
                       for idx in range(period, len(close_arr))])
        
        # Use numpy nanmean to handle NaN values
        rsp = ((np.nan_to_num(q1, 0) + np.nan_to_num(q2, 0) + np.nan_to_num(q3, 0)) * 0.2 + np.nan_to_num(q4, 0) * 0.4) * 100
        rsp = fill_non_vals(close_arr, rsp, period)
        logger.debug(f"RSP calculation completed for {len(close_arr)} data points")
        return rsp
    except Exception as e:
        logger.error(f"Error in RSP calculation: {e}")
        return np.full(len(close_arr), np.nan)


def init_rsp_db(db_path=JQUANTS_DB_PATH, result_db_path=RESULTS_DB_PATH):
    """Initialize relative strength database with all stock data"""
    logger = setup_logging()
    
    # Get all tickers from jquants database
    try:
        with sqlite3.connect(db_path) as conn:
            code_df = pd.read_sql(
                "SELECT DISTINCT Code FROM daily_quotes ORDER BY Code",
                conn
            )
        code_list = code_df['Code'].tolist()
    except sqlite3.Error as e:
        logger.error(f"Error getting code list: {e}")
        raise
    
    logger.info(f"Initializing relative strength analysis for {len(code_list)} stocks")
    
    # Initialize results database
    init_results_db(result_db_path)
    
    errors = []
    processed = 0
    
    for i, c in enumerate(code_list):
        try:
            with sqlite3.connect(db_path) as conn:
                each_df = pd.read_sql(
                    'SELECT Code, Date, AdjustmentClose '
                    'FROM daily_quotes '
                    'WHERE Code = ? '
                    'ORDER BY Date',
                    conn,
                    params=(c,),
                    parse_dates=['Date']
                )
            
            if len(each_df) < 200:
                logger.warning(f"Insufficient data for {c}: {len(each_df)} days")
                continue
                
            each_df = each_df.set_index('Date')
            
            # Safely process close prices
            close_series = each_df['AdjustmentClose'].replace('', np.nan).replace('None', np.nan)
            close_series = pd.to_numeric(close_series, errors='coerce').ffill()
            close = close_series.values
            
            # Skip if all values are NaN
            if np.all(np.isnan(close)):
                logger.warning(f"All close prices are NaN for {c}")
                continue
                
            rsp = relative_strength_percentage(close)
            each_df['RelativeStrengthPercentage'] = rsp
            each_df = each_df[['Code', 'RelativeStrengthPercentage']]
            each_df.index = each_df.index.date
            
            try:
                with sqlite3.connect(result_db_path) as result_conn:
                    each_df.to_sql('relative_strength', result_conn, schema=None, if_exists='append',
                                   index_label='Date', method='multi')
                processed += 1
                logger.debug(f"Processed {c} successfully")
            except Exception as e:
                logger.error(f"Error saving {c}: {e}")
                errors.append([c, str(e)])
            
            if (i + 1) % 100 == 0:
                logger.info(f"{i + 1}/{len(code_list)} stocks processed - {processed} successful")
                
        except Exception as e:
            logger.error(f"Error processing {c}: {e}")
            errors.append([c, str(e)])
    
    if errors:
        error_file = os.path.join(OUTPUT_DIR, f"errors_relative_strength_init_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
        error_df = pd.DataFrame(errors, columns=['code', 'error'])
        error_df.to_csv(error_file, index=False)
        logger.warning(f"Errors saved to {error_file}")
    
    logger.info(f"Relative strength initialization completed. Processed {processed} stocks successfully")
    return processed, len(errors)

def init_results_db(db_path):
    """Initialize results database for relative strength analysis"""
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing relative strength results database at {db_path}")
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS relative_strength (
            Date TEXT NOT NULL,
            Code TEXT NOT NULL,
            RelativeStrengthPercentage REAL,
            RelativeStrengthIndex REAL,
            PRIMARY KEY (Date, Code)
        )
        """)
        conn.commit()
    
    logger.info("Relative strength results database initialized successfully")


def update_rsp_db(db_path=JQUANTS_DB_PATH, result_db_path=RESULTS_DB_PATH, calc_start_date=None, calc_end_date=None, period=-5):
    """Update relative strength database with recent data"""
    logger = setup_logging()
    
    if calc_end_date is None:
        calc_end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    if calc_start_date is None:
        calc_start_date = (datetime.datetime.strptime(calc_end_date, '%Y-%m-%d') - timedelta(days=250)).strftime('%Y-%m-%d')
    
    # Get all tickers
    try:
        with sqlite3.connect(db_path) as conn:
            code_df = pd.read_sql(
                "SELECT DISTINCT Code FROM daily_quotes ORDER BY Code",
                conn
            )
        code_list = code_df['Code'].tolist()
    except sqlite3.Error as e:
        logger.error(f"Error getting code list: {e}")
        raise
    
    logger.info(f"Updating relative strength for {len(code_list)} stocks from {calc_start_date} to {calc_end_date}")
    
    errors = []
    processed = 0
    
    for i, c in enumerate(code_list):
        try:
            with sqlite3.connect(db_path) as conn:
                each_df = pd.read_sql(
                    'SELECT Code, Date, AdjustmentClose '
                    'FROM daily_quotes '
                    'WHERE Code = ? '
                    'AND Date BETWEEN ? AND ? '
                    'ORDER BY Date',
                    conn,
                    params=(c, calc_start_date, calc_end_date),
                    parse_dates=['Date']
                )
            
            if len(each_df) < 200:
                logger.warning(f"Insufficient data for {c}: {len(each_df)} days")
                continue
                
            each_df = each_df.set_index('Date')
            
            # Safely process close prices
            close_series = each_df['AdjustmentClose'].replace('', np.nan).replace('None', np.nan)
            close_series = pd.to_numeric(close_series, errors='coerce').ffill()
            close = close_series.values
            
            # Skip if all values are NaN
            if np.all(np.isnan(close)):
                logger.warning(f"All close prices are NaN for {c}")
                continue
                
            rsp = relative_strength_percentage(close)
            each_df['RelativeStrengthPercentage'] = rsp
            each_df = each_df[['Code', 'RelativeStrengthPercentage']]
            each_df = each_df.reset_index().rename(columns={'index': 'Date'})
            each_df['Date'] = each_df['Date'].dt.date
            
            with sqlite3.connect(result_db_path) as result_conn:
                for _, row in each_df[period:].iterrows():
                    try:
                        sql = """
                        INSERT OR REPLACE INTO relative_strength(Date, Code, RelativeStrengthPercentage)
                        VALUES(?,?,?)
                        """
                        result_conn.execute(sql, (str(row['Date']), row['Code'], row['RelativeStrengthPercentage']))
                    except Exception as e:
                        logger.error(f"Error inserting data for {c}: {e}")
                        errors.append([row['Date'], c, str(e)])
            processed += 1
            logger.debug(f"Updated {c} successfully")
            
            if (i + 1) % 100 == 0:
                logger.info(f"{i + 1}/{len(code_list)} stocks processed - {processed} successful")
                
        except Exception as e:
            logger.error(f"Error updating {c}: {e}")
            errors.append([calc_end_date, c, str(e)])
    
    if errors:
        error_file = os.path.join(OUTPUT_DIR, f"errors_relative_strength_update_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
        error_df = pd.DataFrame(errors, columns=['Date', 'Code', 'error'])
        error_df.to_csv(error_file, index=False)
        logger.warning(f"Errors saved to {error_file}")
    
    logger.info(f"Relative strength update completed. Processed {processed} stocks successfully")
    return processed, len(errors)


def update_rsi_db(result_db_path=RESULTS_DB_PATH, date_list=None, period=-5):
    """Update relative strength index for multiple dates"""
    logger = setup_logging()
    
    if date_list is None:
        # Get recent dates from the database that have RSP but no RSI
        try:
            with sqlite3.connect(result_db_path) as conn:
                date_df = pd.read_sql(
                    """SELECT DISTINCT Date FROM relative_strength 
                       WHERE RelativeStrengthPercentage IS NOT NULL 
                       AND RelativeStrengthIndex IS NULL 
                       ORDER BY Date DESC LIMIT 20""",
                    conn
                )
                if date_df.empty:
                    # Fallback to all recent dates
                    date_df = pd.read_sql(
                        "SELECT DISTINCT Date FROM relative_strength ORDER BY Date DESC LIMIT 10",
                        conn
                    )
            date_list = date_df['Date'].tolist()
        except sqlite3.Error as e:
            logger.error(f"Error getting date list: {e}")
            return
    
    logger.info(f"Updating relative strength index for {len(date_list[period:])} dates")
    
    errors = []
    
    for i, date in enumerate(date_list[period:]):
        try:
            with sqlite3.connect(result_db_path) as conn:
                each_df = pd.read_sql(
                    'SELECT Code, Date, RelativeStrengthPercentage '
                    'FROM relative_strength '
                    'WHERE Date = ?'
                    'ORDER BY Code',
                    conn,
                    params=(date,),
                    parse_dates=['Date']
                )
            
            if each_df.empty:
                logger.warning(f"No data found for {date}")
                continue
            
            each_df['RelativeStrengthPercentage'] = pd.to_numeric(each_df['RelativeStrengthPercentage'], errors='coerce')
            each_df = each_df.sort_values(by='RelativeStrengthPercentage', ascending=False, na_position='last').reset_index(drop=True)
            
            # Calculate RSI ranking - highest RSP gets RSI close to 99, lowest gets close to 1
            valid_count = each_df['RelativeStrengthPercentage'].notna().sum()
            if valid_count == 0:
                logger.warning(f"No valid RSP data for {date}")
                continue
                
            each_df['RelativeStrengthIndex'] = np.nan
            valid_mask = each_df['RelativeStrengthPercentage'].notna()
            
            # Assign ranking: top performer gets 99, bottom gets 1
            valid_indices = each_df[valid_mask].index
            for rank, idx in enumerate(valid_indices):
                each_df.loc[idx, 'RelativeStrengthIndex'] = 99 - (rank / (valid_count - 1)) * 98 if valid_count > 1 else 50
            
            with sqlite3.connect(result_db_path) as conn:
                cursor = conn.cursor()
                sql = """
                 UPDATE relative_strength SET RelativeStrengthIndex = ?
                 WHERE Code = ? AND Date = ?
                 """
                
                for _, row in each_df.iterrows():
                    try:
                        if pd.notna(row['RelativeStrengthIndex']):
                            cursor.execute(sql, (row['RelativeStrengthIndex'], row['Code'], str(row['Date'].date())))
                    except Exception as e:
                        logger.error(f"Error updating RSI for {row['Code']} on {date}: {e}")
                        errors.append([date, row['Code'], str(e)])
                
                conn.commit()  # Ensure changes are committed
            
            logger.info(f"Updated RSI for {len(each_df)} stocks on {date}")
            
            if (i + 1) % 10 == 0:
                logger.info(f"{i + 1} dates processed")
                
        except Exception as e:
            logger.error(f"Error processing date {date}: {e}")
            errors.append([date, 'ALL', str(e)])
    
    if errors:
        error_file = os.path.join(OUTPUT_DIR, f"errors_rsi_update_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
        error_df = pd.DataFrame(errors, columns=['Date', 'Code', 'error'])
        error_df.to_csv(error_file, index=False)
        logger.warning(f"Errors saved to {error_file}")
    
    logger.info(f"RSI update completed. Total errors: {len(errors)}")
    return len(errors)



