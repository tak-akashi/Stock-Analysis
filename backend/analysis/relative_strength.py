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
        q1 = np.array([((close_arr[idx - int(period * 3 / 4) + 1] - close_arr[idx - period + 1]) /
                       close_arr[idx - period + 1]) for idx in range(period, len(close_arr))])
        q2 = np.array([((close_arr[idx - int(period * 2 / 4) + 1] - close_arr[idx - int(period * 3 / 4) + 1]) /
                       close_arr[idx - int(period * 3 / 4) + 1]) for idx in range(period, len(close_arr))])
        q3 = np.array([((close_arr[idx - int(period * 1 / 4) + 1] - close_arr[idx - int(period * 2 / 4) + 1]) /
                       close_arr[idx - int(period * 2 / 4) + 1]) for idx in range(period, len(close_arr))])
        q4 = np.array([((close_arr[idx] - close_arr[idx - int(period * 1 / 4) + 1]) /
                       close_arr[idx - int(period * 1 / 4) + 1]) for idx in range(period, len(close_arr))])
        rsp = ((q1 + q2 + q3) * 0.2 + q4 * 0.4) * 100
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
            close = each_df['AdjustmentClose'].replace('', np.nan).ffill().values
            rsp = relative_strength_percentage(close)
            each_df['relative_strength_percentage'] = rsp
            each_df = each_df[['Code', 'relative_strength_percentage']]
            each_df.index = each_df.index.date
            
            try:
                with sqlite3.connect(result_db_path) as result_conn:
                    each_df.to_sql('relative_strength', result_conn, schema=None, if_exists='append',
                                   index_label='date', method='multi')
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
            date TEXT NOT NULL,
            code TEXT NOT NULL,
            relative_strength_percentage REAL,
            relative_strength_index REAL,
            PRIMARY KEY (date, code)
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
            close = each_df['AdjustmentClose'].replace('', np.nan).ffill().values
            rsp = relative_strength_percentage(close)
            each_df['relative_strength_percentage'] = rsp
            each_df = each_df[['Code', 'relative_strength_percentage']]
            each_df = each_df.reset_index().rename(columns={'index': 'date'})
            each_df['date'] = each_df['date'].dt.date
            
            with sqlite3.connect(result_db_path) as result_conn:
                for _, row in each_df[period:].iterrows():
                    try:
                        sql = """
                        INSERT OR REPLACE INTO relative_strength(date, code, relative_strength_percentage)
                        VALUES(?,?,?)
                        """
                        result_conn.execute(sql, (str(row['date']), row['Code'], row['relative_strength_percentage']))
                    except Exception as e:
                        logger.error(f"Error inserting data for {c}: {e}")
                        errors.append([row['date'], c, str(e)])
            processed += 1
            logger.debug(f"Updated {c} successfully")
            
            if (i + 1) % 100 == 0:
                logger.info(f"{i + 1}/{len(code_list)} stocks processed - {processed} successful")
                
        except Exception as e:
            logger.error(f"Error updating {c}: {e}")
            errors.append([calc_end_date, c, str(e)])
    
    if errors:
        error_file = os.path.join(OUTPUT_DIR, f"errors_relative_strength_update_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
        error_df = pd.DataFrame(errors, columns=['date', 'code', 'error'])
        error_df.to_csv(error_file, index=False)
        logger.warning(f"Errors saved to {error_file}")
    
    logger.info(f"Relative strength update completed. Processed {processed} stocks successfully")
    return processed, len(errors)


def update_rsi_db(result_db_path=RESULTS_DB_PATH, date_list=None, period=-5):
    """Update relative strength index for multiple dates"""
    logger = setup_logging()
    
    if date_list is None:
        # Get recent dates from the database
        try:
            with sqlite3.connect(result_db_path) as conn:
                date_df = pd.read_sql(
                    "SELECT DISTINCT date FROM relative_strength ORDER BY date DESC LIMIT 10",
                    conn
                )
            date_list = date_df['date'].tolist()
        except sqlite3.Error as e:
            logger.error(f"Error getting date list: {e}")
            return
    
    logger.info(f"Updating relative strength index for {len(date_list[period:])} dates")
    
    errors = []
    
    for i, date in enumerate(date_list[period:]):
        try:
            with sqlite3.connect(result_db_path) as conn:
                each_df = pd.read_sql(
                    'SELECT code, date, relative_strength_percentage '
                    'FROM relative_strength '
                    'WHERE date = ?'
                    'ORDER BY code',
                    conn,
                    params=(date,),
                    parse_dates=['date']
                )
            
            if each_df.empty:
                logger.warning(f"No data found for {date}")
                continue
            
            each_df['relative_strength_percentage'] = pd.to_numeric(each_df['relative_strength_percentage'], errors='coerce')
            each_df = each_df.sort_values(by='relative_strength_percentage', ascending=False).reset_index(drop=True)
            _s = each_df['relative_strength_percentage']
            each_df['relative_strength_index'] = np.array([x if np.isnan(x) else i for i, x in enumerate(_s)])
            # 最高が99, 最低が1になるように調整
            each_df['relative_strength_index'] = each_df['relative_strength_index'].apply(
                lambda x: 99 - x / len(_s.dropna()) * 99 if not (np.isnan(x)) else x)
            
            with sqlite3.connect(result_db_path) as conn:
                for _, row in each_df.iterrows():
                    try:
                        sql = """
                         UPDATE relative_strength SET relative_strength_index = ?
                         WHERE code = ? AND date = ?
                         """
                        conn.execute(sql, (row['relative_strength_index'], row['code'], str(row['date'].date())))
                    except Exception as e:
                        logger.error(f"Error updating RSI for {row['code']} on {date}: {e}")
                        errors.append([date, row['code'], str(e)])
            
            logger.info(f"Updated RSI for {len(each_df)} stocks on {date}")
            
            if (i + 1) % 10 == 0:
                logger.info(f"{i + 1} dates processed")
                
        except Exception as e:
            logger.error(f"Error processing date {date}: {e}")
            errors.append([date, 'ALL', str(e)])
    
    if errors:
        error_file = os.path.join(OUTPUT_DIR, f"errors_rsi_update_{datetime.datetime.now().strftime('%Y%m%d')}.csv")
        error_df = pd.DataFrame(errors, columns=['date', 'code', 'error'])
        error_df.to_csv(error_file, index=False)
        logger.warning(f"Errors saved to {error_file}")
    
    logger.info(f"RSI update completed. Total errors: {len(errors)}")
    return len(errors)



