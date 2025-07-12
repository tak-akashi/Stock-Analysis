from __future__ import annotations

import os
import logging
import sqlite3
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    warnings.warn("talib not available, using simple moving average implementation")


class MinerviniConfig:
    """Configuration settings for Minervini analysis."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path("/Users/tak/Markets/Stocks/Stock-Analysis")
        
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.output_dir = self.base_dir / "output"
        
        self.jquants_db_path = self.data_dir / "jquants.db"
        self.results_db_path = self.data_dir / "analysis_results.db"
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.logs_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def error_output_dir(self) -> Path:
        """Get error output directory, creating if necessary."""
        error_dir = self.output_dir / "errors"
        error_dir.mkdir(parents=True, exist_ok=True)
        return error_dir


def setup_logging(config: MinerviniConfig) -> logging.Logger:
    """Setup logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = config.logs_dir / f"minervini_{timestamp}.log"
    
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


def simple_sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average implementation as fallback."""
    if len(data) < period:
        return np.full(len(data), np.nan)
    
    result = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        result[i] = np.mean(data[i - period + 1:i + 1])
    return result


class MinerviniAnalyzer:
    """Minervini strategy analyzer."""
    
    def __init__(self, config: MinerviniConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_strategy(self, close_arr: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        ミネルヴィニの投資基準をチェックし、ndarray形式で返す
        
        Args:
            close_arr: 終値のnumpy配列
            
        Returns:
            Tuple containing:
            - sma50, sma150, sma200: 移動平均線
            - type_1-8: 各投資基準の結果
            
        投資基準:
            type_1: 現在の株価が150日(30週)と200日(40週)の移動平均線を上回っているか
            type_2: 150日移動平均線は200日移動平均線を上回っているか
            type_3: 200日移動平均線は少なくとも1ヶ月上昇トレンドにあるか
            type_4: 50日(10週)移動平均線は150日移動平均線と200日移動平均線を上回っているか
            type_5: 現在の株価は50日移動平均線を上回っているか
            type_6: 現在の株価は52週安値(260日）よりも、少なくとも30％高いか
            type_7: 現在の株価は52週高値から少なくとも25％以内にあるか
            type_8: レラティブストレングスのランキングは70％以上か（別途算出）
        """
        # Ensure close_arr is a numpy array to avoid pandas Series indexing warnings
        close_arr = np.array(close_arr)
        
        sma50, sma150, sma200 = self._calculate_moving_averages(close_arr)
        
        # Calculate each type of criteria
        type_1 = (close_arr > sma150) & (close_arr > sma200)
        type_2 = sma150 > sma200
        type_3 = self._calculate_type3(sma200)
        type_4 = (sma50 > sma150) & (sma50 > sma200)
        type_5 = close_arr > sma50
        type_6, type_7 = self._calculate_52week_criteria(close_arr)
        type_8 = np.full(len(close_arr), np.nan)  # Calculated separately

        return (sma50, sma150, sma200, type_1, type_2, type_3, type_4, type_5,
                type_6, type_7, type_8)
    
    def _calculate_moving_averages(self, close_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate moving averages using talib or simple implementation."""
        if HAS_TALIB:
            sma50 = np.array(talib.SMA(close_arr, timeperiod=50))
            sma150 = np.array(talib.SMA(close_arr, timeperiod=150))
            sma200 = np.array(talib.SMA(close_arr, timeperiod=200))
        else:
            self.logger.debug("Using simple moving average implementation")
            sma50 = simple_sma(close_arr, 50)
            sma150 = simple_sma(close_arr, 150)
            sma200 = simple_sma(close_arr, 200)
        
        return sma50, sma150, sma200
    
    def _calculate_type3(self, sma200: np.ndarray) -> np.ndarray:
        """Calculate type 3: 200日移動平均線は少なくとも1ヶ月上昇トレンドにあるか"""
        type_3 = np.full(len(sma200), np.nan)
        for idx in range(20, len(sma200)):
            type_3[idx] = sma200[idx] > sma200[idx - 20]
        return type_3
    
    def _calculate_52week_criteria(self, close_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate 52-week high/low criteria (type 6 and 7)."""
        if len(close_arr) <= 260:
            return (np.full(len(close_arr), np.nan), 
                   np.full(len(close_arr), np.nan))
        
        type_6 = np.full(len(close_arr), np.nan)
        type_7 = np.full(len(close_arr), np.nan)
        
        for idx in range(260, len(close_arr)):
            week_52_data = close_arr[idx - 260:idx]
            week_52_low = np.min(week_52_data)
            week_52_high = np.max(week_52_data)
            
            # Type 6: At least 30% above 52-week low
            type_6[idx] = close_arr[idx] >= week_52_low * 1.3
            
            # Type 7: Within 25% of 52-week high  
            type_7[idx] = close_arr[idx] >= week_52_high * 0.75
        
        return type_6, type_7

    def make_dataframe(self, code_array: np.ndarray, date_index: pd.DatetimeIndex, 
                      close_array: np.ndarray) -> pd.DataFrame:
        """
        ミネルヴィニの投資基準をチェックし、DataFrame形式で返す
        
        Args:
            code_array: 証券コード配列
            date_index: 日付インデックス
            close_array: 終値配列
            
        Returns:
            分析結果のDataFrame
        """
        results = self.calculate_strategy(close_array)
        sma50, sma150, sma200, type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8 = results

        data = {
            'Code': code_array,
            'Close': close_array,
            'Sma50': sma50,
            'Sma150': sma150,
            'Sma200': sma200,
            'Type_1': type_1.astype(float),
            'Type_2': type_2.astype(float),
            'Type_3': type_3,
            'Type_4': type_4.astype(float),
            'Type_5': type_5.astype(float),
            'Type_6': type_6,
            'Type_7': type_7,
            'Type_8': type_8
        }
        
        return pd.DataFrame(data, index=date_index)


class MinerviniDatabase:
    """Database operations for Minervini analysis."""
    
    def __init__(self, config: MinerviniConfig, analyzer: MinerviniAnalyzer):
        self.config = config
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
    
    def init_database(self, source_conn: sqlite3.Connection, dest_conn: sqlite3.Connection, code_list: List[str]) -> None:
        """Initialize Minervini database with historical data."""
        errors = []
        
        for i, code in enumerate(code_list):
            try:
                stock_data = self._fetch_stock_data(source_conn, code)
                if stock_data is None:
                    continue
                    
                date_index, code_array, close_array = stock_data
                
                if len(close_array) >= 260:
                    df = self.analyzer.make_dataframe(code_array, date_index, close_array)
                    df.to_sql('minervini', dest_conn, schema=None, if_exists='append',
                              index_label='Date', method='multi')
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f'{i + 1} stocks - code {code} finished.')
                    
            except Exception as e:
                self.logger.error(f"Error processing code {code}: {e}")
                errors.append([code, str(e)])
        
        self._save_errors(errors, 'errors_minervini_init.csv')
    
    def _fetch_stock_data(self, conn: sqlite3.Connection, code: str) -> Optional[Tuple]:
        """Fetch stock data for a given code."""
        try:
            data = pd.read_sql(
                'SELECT Code, Date, AdjustmentClose '
                'FROM daily_quotes '
                'WHERE Code = ? '
                'ORDER BY Date',
                conn,
                params=(code,),
                parse_dates=('Date',),
                index_col='Date'
            )
            
            if data.empty:
                return None
            
            date_index = data.index
            code_array = data['Code'].values
            close_array = data['AdjustmentClose'].apply(
                lambda x: np.nan if x == '' else x
            ).ffill().astype(np.float64)
            
            return date_index, code_array, close_array
            
        except Exception as e:
            self.logger.error(f"Error fetching data for code {code}: {e}")
            return None
    
    def _save_errors(self, errors: List, filename: str) -> None:
        """Save errors to CSV file."""
        if errors:
            error_df = pd.DataFrame(errors, columns=['Code', 'Error'])
            error_path = self.config.error_output_dir / filename
            error_df.to_csv(error_path, index=False)
            self.logger.info(f"Saved {len(errors)} errors to {error_path}")

    def update_database(self, source_conn: sqlite3.Connection, dest_conn: sqlite3.Connection, code_list: List[str],
                       calc_start_date: str, calc_end_date: str, period: int = 5) -> None:
        """Update Minervini database with new data."""
        errors = []
        
        for i, code in enumerate(code_list):
            try:
                stock_data = self._fetch_stock_data_range(
                    source_conn, code, calc_start_date, calc_end_date
                )
                if stock_data is None:
                    continue
                    
                date_index, code_array, close_array = stock_data
                
                if len(close_array) >= 260:
                    df = self.analyzer.make_dataframe(code_array, date_index, close_array)
                    df = df.reset_index().rename(columns={'index': 'Date'})
                    
                    self._insert_recent_data(dest_conn, df, period, errors)

                # 100件ごとに進捗をログ出力
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Updated {i + 1}/{len(code_list)} stocks. Last processed code: {code}")
                    
            except Exception as e:
                self.logger.error(f"Error updating code {code}: {e}")
                errors.append([code, str(e)])
        
        self._save_errors(errors, 'errors_minervini_update.csv')
    
    def _fetch_stock_data_range(self, conn: sqlite3.Connection, code: str, 
                               start_date: str, end_date: str) -> Optional[Tuple]:
        """Fetch stock data for a given code and date range."""
        try:
            data = pd.read_sql(
                'SELECT Code, Date, AdjustmentClose '
                'FROM daily_quotes '
                'WHERE Code = ? '
                'AND Date >= ? AND Date <= ? '
                'ORDER BY Date',
                conn,
                params=(code, start_date, end_date),
                parse_dates=('Date',),
                index_col='Date'
            )
            
            if data.empty:
                return None
            
            date_index = data.index
            code_array = data['Code'].values
            close_array = data['AdjustmentClose'].apply(
                lambda x: np.nan if x == '' else x
            ).ffill().astype(np.float64)
            
            return date_index, code_array, close_array
            
        except Exception as e:
            self.logger.error(f"Error fetching data range for code {code}: {e}")
            return None
    
    def _insert_recent_data(self, conn: sqlite3.Connection, df: pd.DataFrame, 
                           period: int, errors: List) -> None:
        """Insert recent data into database."""
        sql = """
        INSERT OR REPLACE INTO minervini(Date, Code, Close, Sma50, Sma150, Sma200, Type_1, Type_2,
        Type_3, Type_4, Type_5, Type_6, Type_7, Type_8)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        
        with conn:
            for row in df.tail(period).itertuples():
                try:
                    params = (
                        str(row.Date), row.Code, row.Close, row.Sma50, row.Sma150, row.Sma200,
                        float(row.Type_1), float(row.Type_2), row.Type_3, float(row.Type_4),
                        float(row.Type_5), row.Type_6, row.Type_7, row.Type_8
                    )
                    conn.execute(sql, params)
                except Exception as e:
                    self.logger.error(f"Error inserting row: {e}")
                    errors.append([row.Date, row.Code, str(e)])

    def update_type8_by_date(self, conn: sqlite3.Connection, date: str) -> List:
        """Update type 8 (relative strength) for a specific date."""
        try:
            # Handle date format normalization - extract just the date part
            date_part = date.split()[0] if ' ' in date else date
            
            # Query with flexible date matching
            data = pd.read_sql(
                """
                SELECT Code, Date, RelativeStrengthIndex 
                FROM relative_strength 
                WHERE Date = ? OR Date = ? OR Date LIKE ?
                ORDER BY Code
                """,
                conn,
                params=(date, date_part, date_part + '%'),
                parse_dates=('Date',),
                index_col='Date'
            )
            
            if data.empty:
                self.logger.warning(f"No relative strength data found for date {date}")
                return []
            
            self.logger.info(f"Processing Type_8 update for {len(data)} stocks on {date}")
            
            data['RelativeStrengthIndex'] = data['RelativeStrengthIndex'].astype('float')
            data['Type_8'] = data['RelativeStrengthIndex'].apply(
                lambda x: 1.0 if x >= 70 else 0.0 if not np.isnan(x) else np.nan
            )
            data = data.reset_index()
            data['Date'] = data['Date'].dt.date

            errors = []
            updated_count = 0
            sql = """
             UPDATE minervini SET Type_8 = ?
             WHERE Code = ? AND (
                 Date = ? OR 
                 Date = ? OR 
                 Date LIKE ? OR 
                 substr(Date, 1, 10) = ?
             )
             """
            
            with conn:
                conn.execute('BEGIN TRANSACTION')
                for i, (_, row) in enumerate(data.iterrows()):
                    try:
                        # Normalize date formats for better matching
                        date_str = str(row['Date'])
                        date_only = date_str.split()[0] if ' ' in date_str else date_str
                        date_pattern = date_only + '%'
                        
                        # Try multiple date format variations for better matching
                        result = conn.execute(sql, (
                            row['Type_8'], row['Code'], 
                            date_str, date_only, date_pattern, date_only
                        ))
                        
                        if result.rowcount == 0:
                            self.logger.warning(f"No minervini records updated for {row['Code']} on {date_only}")
                        else:
                            updated_count += result.rowcount
                            self.logger.debug(f"Updated {result.rowcount} records for {row['Code']} on {date_only}")
                        
                        # Progress report every 100 stocks
                        if (i + 1) % 100 == 0:
                            self.logger.info(f"Type_8 update progress: {i + 1}/{len(data)} stocks processed")
                            
                    except Exception as e:
                        self.logger.error(f"Error updating type_8 for {row['Code']}: {e}")
                        errors.append([row['Date'], row['Code'], str(e)])
                conn.commit()
            
            self.logger.info(f"Type_8 update completed: {updated_count} records updated, {len(errors)} errors for date {date}")
            
            return errors
            
        except Exception as e:
            self.logger.error(f"Error updating type_8 for date {date}: {e}")
            return [[date, 'ALL', str(e)]]

    def update_type8_bulk(self, conn: sqlite3.Connection, date_list: List[str], 
                         period: int = -5) -> None:
        """Update type 8 for multiple dates."""
        all_errors = []
        
        dates_to_process = date_list[period:] if period < 0 else date_list[-period:]
        
        for i, date in enumerate(dates_to_process):
            errors = self.update_type8_by_date(conn, date)
            all_errors.extend(errors)
            
            if (i + 1) % 100 == 0:
                self.logger.info(f"{i + 1} dates processed - {date} finished.")
        
        if all_errors:
            error_df = pd.DataFrame(all_errors, columns=['Date', 'Code', 'Error'])
            error_path = self.config.error_output_dir / 'errors_update_type8.csv'
            error_df.to_csv(error_path, index=False)
            self.logger.info(f"Saved {len(all_errors)} errors to {error_path}")


# Compatibility functions for backward compatibility
def minervini_strategy(close_arr: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Backward compatibility function."""
    config = MinerviniConfig()
    analyzer = MinerviniAnalyzer(config)
    return analyzer.calculate_strategy(close_arr)


def make_minervini_df(code_array: np.ndarray, date_index: pd.DatetimeIndex, 
                     close_array: np.ndarray) -> pd.DataFrame:
    """Backward compatibility function."""
    config = MinerviniConfig()
    analyzer = MinerviniAnalyzer(config)
    return analyzer.make_dataframe(code_array, date_index, close_array)


def init_minervini_db(source_conn: sqlite3.Connection, dest_conn: sqlite3.Connection, code_list: List[str]) -> None:
    """Backward compatibility function."""
    config = MinerviniConfig()
    analyzer = MinerviniAnalyzer(config)
    database = MinerviniDatabase(config, analyzer)
    database.init_database(source_conn, dest_conn, code_list)


def update_minervini_db(source_conn: sqlite3.Connection, dest_conn: sqlite3.Connection, code_list: List[str], 
                       calc_start_date: str, calc_end_date: str, period: int = 5) -> None:
    """Backward compatibility function."""
    config = MinerviniConfig()
    analyzer = MinerviniAnalyzer(config)
    database = MinerviniDatabase(config, analyzer)
    database.update_database(source_conn, dest_conn, code_list, calc_start_date, calc_end_date, period)


def update_type8_db_by_date(conn: sqlite3.Connection, date: str) -> List:
    """Backward compatibility function."""
    config = MinerviniConfig()
    analyzer = MinerviniAnalyzer(config)
    database = MinerviniDatabase(config, analyzer)
    return database.update_type8_by_date(conn, date)


def update_type8_db(conn: sqlite3.Connection, date_list: List[str], period: int = -5) -> None:
    """Backward compatibility function."""
    config = MinerviniConfig()
    analyzer = MinerviniAnalyzer(config)
    database = MinerviniDatabase(config, analyzer)
    database.update_type8_bulk(conn, date_list, period)