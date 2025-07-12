import sqlite3
import datetime
import logging
import os
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

# --- Constants ---
JQUANTS_DB_PATH = "/Users/tak/Markets/Stocks/Stock-Analysis/data/jquants.db"
DATA_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/data"
LOGS_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/logs"
OUTPUT_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/output"
RESULTS_DB_PATH = os.path.join(DATA_DIR, "analysis_results.db")

def setup_logging():
    """Setup logging configuration"""
    log_filename = os.path.join(LOGS_DIR, f"high_low_ratio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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


def init_hl_ratio_db(db_path=RESULTS_DB_PATH):
    """Initialize hl_ratio table in results database"""
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing hl_ratio results database at {db_path}")
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS hl_ratio (
            Date TEXT NOT NULL,
            Code TEXT NOT NULL,
            HlRatio REAL NOT NULL,
            MedianRatio REAL NOT NULL,
            Weeks INTEGER NOT NULL,
            PRIMARY KEY (Date, Code)
        )
        """)
        conn.commit()
    
    logger.info("HL ratio results database initialized successfully")


def calc_median_ratio(price_df, weeks=52):
    """Calculate Median-Low ratio for given price data"""
    logger = logging.getLogger(__name__)
    days = weeks * 5
    
    # Adapt column names to match jquants database schema
    high_col = 'High'
    low_col = 'Low'
    close_col = 'AdjustmentClose' if 'AdjustmentClose' in price_df.columns else 'Close'
    
    period_data = price_df[-days:]
    
    # Convert to numeric and handle invalid data
    high_series = pd.to_numeric(period_data[high_col], errors='coerce')
    low_series = pd.to_numeric(period_data[low_col], errors='coerce')
    close_series = pd.to_numeric(period_data[close_col], errors='coerce')
    
    # Check if we have valid data
    if high_series.isna().all() or low_series.isna().all() or close_series.isna().all():
        logger.warning(f"No valid data available for median ratio calculation")
        return np.nan
    
    highest_price = high_series.max()
    lowest_price = low_series.min()
    
    # Check for valid median calculation
    valid_close_data = close_series.dropna()
    if len(valid_close_data) == 0:
        logger.warning(f"No valid close price data for median calculation")
        return np.nan
    
    median_price = valid_close_data.median()
    
    if pd.isna(highest_price) or pd.isna(lowest_price) or pd.isna(median_price):
        logger.warning(f"Invalid price data (NaN values)")
        return np.nan
    
    if highest_price == lowest_price:
        logger.warning(f"Highest and lowest prices are equal, returning 50.0")
        return 50.0
    
    ratio = (median_price - lowest_price) / (highest_price - lowest_price) * 100
    logger.debug(f"Median ratio calculated: {ratio:.2f}%")
    return ratio


def calc_hl_ratio(price_df, weeks=52):
    """Calculate High-Low ratio for given price data"""
    logger = logging.getLogger(__name__)
    days = weeks * 5
    
    # Adapt column names to match jquants database schema
    high_col = 'High'
    low_col = 'Low'
    close_col = 'AdjustmentClose' if 'AdjustmentClose' in price_df.columns else 'Close'
    
    # Convert to numeric and handle invalid data
    high_series = pd.to_numeric(price_df[high_col][-days:], errors='coerce')
    low_series = pd.to_numeric(price_df[low_col][-days:], errors='coerce')
    close_series = pd.to_numeric(price_df[close_col], errors='coerce')
    
    # Check if we have valid data
    if high_series.isna().all() or low_series.isna().all() or close_series.isna().all():
        logger.warning(f"No valid data available for HL ratio calculation")
        return np.nan
    
    highest_price = high_series.max()
    lowest_price = low_series.min()
    
    # Get current price (last valid close price)
    valid_close_data = close_series.dropna()
    if len(valid_close_data) == 0:
        logger.warning(f"No valid close price data for current price")
        return np.nan
    
    current_price = valid_close_data.iloc[-1]
    
    if pd.isna(highest_price) or pd.isna(lowest_price) or pd.isna(current_price):
        logger.warning(f"Invalid price data (NaN values)")
        return np.nan
    
    if highest_price == lowest_price:
        logger.warning(f"Highest and lowest prices are equal, returning 50.0")
        return 50.0
    
    ratio = (current_price - lowest_price) / (highest_price - lowest_price) * 100
    logger.debug(f"HL ratio calculated: {ratio:.2f}%")
    return ratio


def calc_hl_ratio_for_all(db_path=JQUANTS_DB_PATH, end_date=None, weeks=52):
    """
    Calculate HL ratio for all stocks in the database and return a DataFrame.
    This function focuses on calculation only and does not handle DB writing.
    """
    logger = logging.getLogger(__name__)

    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    buffers = 30
    start_date = end_date - relativedelta(days=weeks * 7 + buffers)

    logger.info(f"Calculating HL ratio for all stocks from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    try:
        with sqlite3.connect(db_path) as conn:
            price_df = pd.read_sql(
                """
                SELECT Date, Code, High, Low, AdjustmentClose
                FROM daily_quotes
                WHERE Date BETWEEN ? AND ?
                ORDER BY Date
                """,
                conn,
                params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')),
                parse_dates=['Date']
            )
    except sqlite3.Error as e:
        logger.error(f"Database error while reading daily_quotes: {e}")
        raise

    price_df = price_df.replace('', np.nan)
    code_list = price_df['Code'].sort_values().unique()
    ratio_dict = dict()
    errors = []

    logger.info(f"Processing {len(code_list)} stocks for HL Ratio.")

    for i, code in enumerate(code_list):
        try:
            df = price_df[price_df['Code'] == code].set_index('Date')
            if len(df) < weeks * 5:
                logger.warning(f"Insufficient data for {code}: {len(df)} days")
                continue
            df = df.ffill()
            hl_ratio = calc_hl_ratio(df, weeks)
            median_ratio = calc_median_ratio(df, weeks)
            
            # Only include results if both ratios are valid (not NaN)
            if not (pd.isna(hl_ratio) or pd.isna(median_ratio)):
                ratio_dict[code] = {'HlRatio': hl_ratio, 'MedianRatio': median_ratio}
            else:
                logger.debug(f"Skipping {code} due to invalid ratio calculations")

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(code_list)} stocks for HL Ratio")

        except Exception as e:
            logger.error(f"Error processing HL Ratio for {code}: {e}")
            errors.append([code, str(e)])

    if not ratio_dict:
        logger.warning("No HL ratios were calculated.")
        return pd.DataFrame()

    ratio_data = []
    for code, ratios in ratio_dict.items():
        ratio_data.append({'Code': code, 'HlRatio': ratios['HlRatio'], 'MedianRatio': ratios['MedianRatio']})
    
    ratio_df = pd.DataFrame(ratio_data)
    ratio_df = ratio_df.sort_values('HlRatio', ascending=False).reset_index(drop=True)
    ratio_df['Date'] = end_date.strftime('%Y-%m-%d')
    ratio_df['Weeks'] = weeks

    if errors:
        logger.warning(f"Encountered {len(errors)} errors during HL ratio calculation.")

    logger.info(f"HL ratio calculation completed. Calculated for {len(ratio_dict)} stocks.")
    return ratio_df


def calc_hl_ratio_by_code(code, db_path=JQUANTS_DB_PATH, end_date=None, weeks=52, save_to_db=True):
    """Calculate HL ratio for a specific stock code"""
    logger = logging.getLogger(__name__)
    
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    buffers = 30
    start_date = end_date - relativedelta(days=weeks * 7 + buffers)
    
    logger.info(f"Calculating HL ratio for {code} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            price_df = pd.read_sql(
                """
                SELECT Date, Code, High, Low, AdjustmentClose
                FROM daily_quotes
                WHERE Date BETWEEN ? AND ? 
                AND Code = ?
                ORDER BY Date
                """,
                conn,
                params=(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), code),
                parse_dates=['Date']
            )
    except sqlite3.Error as e:
        logger.error(f"Database error for {code}: {e}")
        raise
    
    if len(price_df) < weeks * 5:
        logger.warning(f"Insufficient data for {code}: {len(price_df)} days")
        return None, price_df
    
    price_df = price_df.set_index('Date')
    price_df = price_df.ffill()
    
    try:
        hl_ratio = calc_hl_ratio(price_df, weeks)
        median_ratio = calc_median_ratio(price_df, weeks)
        
        # Check if results are valid
        if pd.isna(hl_ratio) or pd.isna(median_ratio):
            logger.warning(f"Invalid ratio calculations for {code}: HL={hl_ratio}, Median={median_ratio}")
            return None, price_df
        
        logger.info(f"HL ratio for {code}: {hl_ratio:.2f}%, Median ratio: {median_ratio:.2f}%")
        
        # Save to database if requested
        if save_to_db and hl_ratio is not None and median_ratio is not None:
            try:
                init_hl_ratio_db()
                with sqlite3.connect(RESULTS_DB_PATH) as result_conn:
                    sql = """
                    INSERT OR REPLACE INTO hl_ratio(Date, Code, HlRatio, MedianRatio, Weeks)
                    VALUES(?,?,?,?,?)
                    """
                    result_conn.execute(sql, (end_date.strftime('%Y-%m-%d'), code, hl_ratio, median_ratio, weeks))
                logger.debug(f"HL ratio and Median ratio for {code} saved to database")
            except Exception as e:
                logger.error(f"Error saving ratios for {code} to database: {e}")
        
        return {'HlRatio': hl_ratio, 'MedianRatio': median_ratio}, price_df
    except Exception as e:
        logger.error(f"Error calculating HL ratio for {code}: {e}")
        return None, price_df


if __name__ == "__main__":
    calc_hl_ratio_for_all()