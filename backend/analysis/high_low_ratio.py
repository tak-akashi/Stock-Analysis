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
            date TEXT NOT NULL,
            code TEXT NOT NULL,
            hl_ratio REAL NOT NULL,
            weeks INTEGER NOT NULL,
            PRIMARY KEY (date, code, weeks)
        )
        """)
        conn.commit()
    
    logger.info("HL ratio results database initialized successfully")


def calc_hl_ratio(price_df, weeks=52):
    """Calculate High-Low ratio for given price data"""
    logger = logging.getLogger(__name__)
    days = weeks * 5
    
    # Adapt column names to match jquants database schema
    high_col = 'High' if 'High' in price_df.columns else 'high'
    low_col = 'Low' if 'Low' in price_df.columns else 'low'
    close_col = 'AdjustmentClose' if 'AdjustmentClose' in price_df.columns else 'close'
    
    highest_price = price_df[high_col][-days:].astype('float').max()
    lowest_price = price_df[low_col][-days:].astype('float').min()
    current_price = float(price_df[close_col].iloc[-1])
    
    if highest_price == lowest_price:
        logger.warning(f"Highest and lowest prices are equal, returning 50.0")
        return 50.0
    
    ratio = (current_price - lowest_price) / (highest_price - lowest_price) * 100
    logger.debug(f"HL ratio calculated: {ratio:.2f}%")
    return ratio


def calc_hl_ratio_for_all(db_path=JQUANTS_DB_PATH, end_date=None, weeks=52):
    """Calculate HL ratio for all stocks in the database"""
    logger = setup_logging()
    
    if end_date is None:
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    buffers = 30
    start_date = end_date - relativedelta(days=weeks * 7 + buffers)
    
    logger.info(f"Calculating HL ratio for all stocks from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Adapt to jquants database schema
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
        logger.error(f"Database error: {e}")
        raise
    
    price_df = price_df.replace('', np.nan)
    code_list = price_df['Code'].sort_values().unique()
    ratio_dict = dict()
    errors = []
    
    logger.info(f"Processing {len(code_list)} stocks")
    
    for i, code in enumerate(code_list):
        try:
            df = price_df[price_df['Code'] == code].set_index('Date')
            if len(df) < weeks * 5:
                logger.warning(f"Insufficient data for {code}: {len(df)} days")
                continue
            df = df.ffill()
            ratio = calc_hl_ratio(df, weeks)
            ratio_dict[code] = ratio
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(code_list)} stocks")
                
        except Exception as e:
            logger.error(f"Error processing {code}: {e}")
            errors.append([code, str(e)])
    
    ratio_df = pd.DataFrame(list(ratio_dict.items()), columns=['code', 'hl_ratio'])
    ratio_df = ratio_df.sort_values('hl_ratio', ascending=False).reset_index(drop=True)
    ratio_df['date'] = end_date.strftime('%Y-%m-%d')
    ratio_df['weeks'] = weeks
    
    # Initialize database if it doesn't exist
    init_hl_ratio_db()
    
    # Save results to database
    try:
        with sqlite3.connect(RESULTS_DB_PATH) as result_conn:
            for _, row in ratio_df.iterrows():
                try:
                    sql = """
                    INSERT OR REPLACE INTO hl_ratio(date, code, hl_ratio, weeks)
                    VALUES(?,?,?,?)
                    """
                    result_conn.execute(sql, (row['date'], row['code'], row['hl_ratio'], row['weeks']))
                except Exception as e:
                    logger.error(f"Error saving {row['code']}: {e}")
                    errors.append([row['code'], str(e)])
        logger.info(f"Results saved to database: {len(ratio_dict)} records")
    except Exception as e:
        logger.error(f"Database save error: {e}")
        # Fallback to CSV if database fails
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"hl_ratio_{end_date.strftime('%Y%m%d')}.csv")
        ratio_df.to_csv(output_file, index=False)
        logger.warning(f"Fallback: Results saved to {output_file}")
    
    if errors:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        error_file = os.path.join(OUTPUT_DIR, f"errors_hl_ratio_{end_date.strftime('%Y%m%d')}.csv")
        errors_df = pd.DataFrame(errors, columns=['code', 'error'])
        errors_df.to_csv(error_file, index=False)
        logger.warning(f"Errors logged to {error_file}")
    
    logger.info(f"HL ratio calculation completed. Processed {len(ratio_dict)} stocks successfully")
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
        ratio = calc_hl_ratio(price_df, weeks)
        logger.info(f"HL ratio for {code}: {ratio:.2f}%")
        
        # Save to database if requested
        if save_to_db and ratio is not None:
            try:
                init_hl_ratio_db()
                with sqlite3.connect(RESULTS_DB_PATH) as result_conn:
                    sql = """
                    INSERT OR REPLACE INTO hl_ratio(date, code, hl_ratio, weeks)
                    VALUES(?,?,?,?)
                    """
                    result_conn.execute(sql, (end_date.strftime('%Y-%m-%d'), code, ratio, weeks))
                logger.debug(f"HL ratio for {code} saved to database")
            except Exception as e:
                logger.error(f"Error saving HL ratio for {code} to database: {e}")
        
        return ratio, price_df
    except Exception as e:
        logger.error(f"Error calculating HL ratio for {code}: {e}")
        return None, price_df


if __name__ == "__main__":
    calc_hl_ratio_for_all()