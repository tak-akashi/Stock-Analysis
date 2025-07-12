import sqlite3
import datetime
import logging
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Union

# --- Constants ---
DATA_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/data"
LOGS_DIR = "/Users/tak/Markets/Stocks/Stock-Analysis/logs"
RESULTS_DB_PATH = os.path.join(DATA_DIR, "analysis_results.db")

def setup_logging():
    """Setup logging configuration"""
    log_filename = os.path.join(LOGS_DIR, f"integrated_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
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


def get_comprehensive_analysis(date: str, code: Optional[str] = None, 
                             db_path: str = RESULTS_DB_PATH) -> pd.DataFrame:
    """
    Get comprehensive analysis results for a specific date and optionally a specific code.
    Combines HL ratio, Minervini criteria, and Relative Strength data.
    
    Args:
        date: Analysis date in YYYY-MM-DD format
        code: Optional stock code filter
        db_path: Path to the analysis results database
        
    Returns:
        DataFrame with combined analysis results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Getting comprehensive analysis for date: {date}" + 
                (f", code: {code}" if code else " (all codes)"))
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Base query for all analysis data
            base_query = """
            SELECT 
                h.Date,
                h.Code,
                h.HlRatio,
                h.MedianRatio,
                h.Weeks as hl_weeks,
                MAX(m.Close) as minervini_close,
                MAX(m.Sma50) as Sma50,
                MAX(m.Sma150) as Sma150, 
                MAX(m.Sma200) as Sma200,
                MAX(m.Type_1) as minervini_type_1,
                MAX(m.Type_2) as minervini_type_2,
                MAX(m.Type_3) as minervini_type_3,
                MAX(m.Type_4) as minervini_type_4,
                MAX(m.Type_5) as minervini_type_5,
                MAX(m.Type_6) as minervini_type_6,
                MAX(m.Type_7) as minervini_type_7,
                MAX(m.Type_8) as minervini_type_8,
                MAX(r.RelativeStrengthPercentage) as RelativeStrengthPercentage,
                MAX(r.RelativeStrengthIndex) as RelativeStrengthIndex
            FROM hl_ratio h
            LEFT JOIN minervini m ON substr(m.Date, 1, 10) = h.Date AND h.Code = m.Code
            LEFT JOIN relative_strength r ON substr(r.Date, 1, 10) = h.Date AND h.Code = r.Code
            WHERE h.Date = ?
            GROUP BY h.Date, h.Code, h.HlRatio, h.MedianRatio, h.Weeks
            """
            
            params = [date]
            if code:
                base_query += " AND h.Code = ?"
                params.append(code)
            
            base_query += " ORDER BY h.HlRatio DESC"
            
            df = pd.read_sql(base_query, conn, params=params)
            
        if df.empty:
            logger.warning(f"No comprehensive analysis data found for date: {date}")
            return pd.DataFrame()
        
        # Calculate composite scores
        df = _calculate_composite_scores(df)
        
        logger.info(f"Retrieved comprehensive analysis for {len(df)} stocks")
        return df
        
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving comprehensive analysis: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise


def get_multi_date_analysis(start_date: str, end_date: str, code: str,
                          db_path: str = RESULTS_DB_PATH) -> pd.DataFrame:
    """
    Get analysis results for a specific stock across multiple dates.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        code: Stock code
        db_path: Path to the analysis results database
        
    Returns:
        DataFrame with time series analysis results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Getting multi-date analysis for {code} from {start_date} to {end_date}")
    
    try:
        with sqlite3.connect(db_path) as conn:
            query = """
            SELECT 
                h.Date,
                h.Code,
                h.HlRatio,
                MAX(m.Close) as minervini_close,
                MAX(m.Sma50) as Sma50,
                MAX(m.Sma150) as Sma150,
                MAX(m.Sma200) as Sma200,
                MAX(m.Type_1) + MAX(m.Type_2) + MAX(m.Type_3) + MAX(m.Type_4) + MAX(m.Type_5) + 
                MAX(m.Type_6) + MAX(m.Type_7) + MAX(m.Type_8) as minervini_score,
                MAX(r.RelativeStrengthPercentage) as RelativeStrengthPercentage,
                MAX(r.RelativeStrengthIndex) as RelativeStrengthIndex
            FROM hl_ratio h
            LEFT JOIN minervini m ON substr(m.Date, 1, 10) = h.Date AND h.Code = m.Code
            LEFT JOIN relative_strength r ON substr(r.Date, 1, 10) = h.Date AND h.Code = r.Code
            WHERE h.Code = ? AND h.Date BETWEEN ? AND ?
            GROUP BY h.Date, h.Code, h.HlRatio
            ORDER BY h.Date ASC
            """
            
            df = pd.read_sql(query, conn, params=[code, start_date, end_date])
            
        if df.empty:
            logger.warning(f"No multi-date analysis data found for {code}")
            return pd.DataFrame()
            
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Retrieved {len(df)} records for {code}")
        return df
        
    except sqlite3.Error as e:
        logger.error(f"Database error retrieving multi-date analysis: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in multi-date analysis: {e}")
        raise


def get_top_stocks_by_criteria(date: str, criteria: str = 'composite', 
                             limit: int = 50, include_median_ratio: bool = False, 
                             db_path: str = RESULTS_DB_PATH) -> pd.DataFrame:
    """
    Get top stocks based on various criteria.
    
    Args:
        date: Analysis date in YYYY-MM-DD format
        criteria: Ranking criteria ('hl_ratio', 'rsi', 'minervini', 'composite')
        limit: Number of top stocks to return
        include_median_ratio: Whether to include MedianRatio in hl_ratio sorting
        db_path: Path to the analysis results database
        
    Returns:
        DataFrame with top stocks ranked by criteria
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Getting top {limit} stocks by {criteria} for date: {date}")
    
    df = get_comprehensive_analysis(date, db_path=db_path)
    
    if df.empty:
        logger.warning(f"No data available for ranking on {date}")
        return pd.DataFrame()
    
    # Apply ranking based on criteria
    if criteria == 'hl_ratio':
        if include_median_ratio and 'MedianRatio' in df.columns:
            df_ranked = df.sort_values(['HlRatio', 'MedianRatio'], ascending=[False, True])
        else:
            df_ranked = df.sort_values('HlRatio', ascending=False)
    elif criteria == 'rsi':
        df_ranked = df.sort_values('RelativeStrengthIndex', ascending=False, na_position='last')
    elif criteria == 'minervini':
        df_ranked = df.sort_values('minervini_score', ascending=False, na_position='last')
    elif criteria == 'composite':
        df_ranked = df.sort_values('composite_score', ascending=False, na_position='last')
    else:
        logger.error(f"Unknown criteria: {criteria}")
        raise ValueError(f"Unknown criteria: {criteria}")
    
    result = df_ranked.head(limit)
    logger.info(f"Ranked {len(result)} stocks by {criteria}")
    return result


def get_stocks_meeting_criteria(date: str, hl_ratio_min: float = 80.0,
                              rsi_min: float = 70.0, minervini_min: int = 5,
                              db_path: str = RESULTS_DB_PATH) -> pd.DataFrame:
    """
    Get stocks that meet minimum criteria across all indicators.
    
    Args:
        date: Analysis date in YYYY-MM-DD format
        hl_ratio_min: Minimum HL ratio threshold
        rsi_min: Minimum RSI threshold
        minervini_min: Minimum number of Minervini criteria met
        db_path: Path to the analysis results database
        
    Returns:
        DataFrame with stocks meeting all criteria
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Finding stocks meeting criteria on {date}: HL>{hl_ratio_min}, RSI>{rsi_min}, Minervini>{minervini_min}")
    
    df = get_comprehensive_analysis(date, db_path=db_path)
    
    if df.empty:
        logger.warning(f"No data available for filtering on {date}")
        return pd.DataFrame()
    
    # Apply filters
    filtered_df = df[
        (df['HlRatio'] >= hl_ratio_min) &
        (df['RelativeStrengthIndex'] >= rsi_min) &
        (df['minervini_score'] >= minervini_min)
    ].copy()
    
    # Sort by composite score
    filtered_df = filtered_df.sort_values('composite_score', ascending=False, na_position='last')
    
    logger.info(f"Found {len(filtered_df)} stocks meeting all criteria")
    return filtered_df


def _calculate_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate composite scores combining all analysis indicators.
    
    Args:
        df: DataFrame with individual analysis results
        
    Returns:
        DataFrame with added composite score columns
    """
    df = df.copy()
    
    # Calculate Minervini score (sum of type_1 through type_8)
    minervini_cols = [f'minervini_type_{i}' for i in range(1, 9)]
    df['minervini_score'] = df[minervini_cols].fillna(0).sum(axis=1)
    
    # Normalize scores to 0-100 scale for composite calculation
    df['hl_ratio_norm'] = df['HlRatio'].fillna(0)  # Already 0-100
    df['rsi_norm'] = df['RelativeStrengthIndex'].fillna(0)  # Already 0-99, treat as 0-100
    df['minervini_norm'] = (df['minervini_score'] / 8.0) * 100  # Convert 0-8 to 0-100
    
    # Calculate composite score (weighted average)
    # HL ratio: 40%, RSI: 40%, Minervini: 20%
    df['composite_score'] = (
        df['hl_ratio_norm'] * 0.4 +
        df['rsi_norm'] * 0.4 +
        df['minervini_norm'] * 0.2
    )
    
    # Clean up temporary columns
    df = df.drop(['hl_ratio_norm', 'rsi_norm', 'minervini_norm'], axis=1)
    
    return df


def create_analysis_summary(date: str, db_path: str = RESULTS_DB_PATH) -> Dict[str, Union[int, float]]:
    """
    Create a summary of analysis results for a given date.
    
    Args:
        date: Analysis date in YYYY-MM-DD format
        db_path: Path to the analysis results database
        
    Returns:
        Dictionary with summary statistics
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating analysis summary for date: {date}")
    
    df = get_comprehensive_analysis(date, db_path=db_path)
    
    if df.empty:
        logger.warning(f"No data available for summary on {date}")
        return {}
    
    summary = {
        'total_stocks': len(df),
        'avg_hl_ratio': df['HlRatio'].mean(),
        'avg_rsi': df['RelativeStrengthIndex'].mean(),
        'avg_minervini_score': df['minervini_score'].mean(),
        'avg_composite_score': df['composite_score'].mean(),
        'high_hl_ratio_count': len(df[df['HlRatio'] >= 80]),
        'high_rsi_count': len(df[df['RelativeStrengthIndex'] >= 70]),
        'strong_minervini_count': len(df[df['minervini_score'] >= 5]),
        'strong_composite_count': len(df[df['composite_score'] >= 70])
    }
    
    logger.info(f"Analysis summary completed for {summary['total_stocks']} stocks")
    return summary


def check_database_coverage(db_path: str = RESULTS_DB_PATH) -> Dict[str, int]:
    """
    Check the coverage of data across different analysis tables.
    
    Args:
        db_path: Path to the analysis results database
        
    Returns:
        Dictionary with coverage information
    """
    logger = logging.getLogger(__name__)
    logger.info("Checking database coverage across analysis tables")
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Get counts for each table
            hl_count = conn.execute("SELECT COUNT(*) FROM hl_ratio").fetchone()[0]
            minervini_count = conn.execute("SELECT COUNT(*) FROM minervini").fetchone()[0]
            rs_count = conn.execute("SELECT COUNT(*) FROM relative_strength").fetchone()[0]
            
            # Get unique dates
            hl_dates = conn.execute("SELECT COUNT(DISTINCT Date) FROM hl_ratio").fetchone()[0]
            minervini_dates = conn.execute("SELECT COUNT(DISTINCT Date) FROM minervini").fetchone()[0]
            rs_dates = conn.execute("SELECT COUNT(DISTINCT Date) FROM relative_strength").fetchone()[0]
            
            # Get unique codes
            hl_codes = conn.execute("SELECT COUNT(DISTINCT Code) FROM hl_ratio").fetchone()[0]
            minervini_codes = conn.execute("SELECT COUNT(DISTINCT Code) FROM minervini").fetchone()[0]
            rs_codes = conn.execute("SELECT COUNT(DISTINCT Code) FROM relative_strength").fetchone()[0]
            
        coverage = {
            'hl_ratio_records': hl_count,
            'minervini_records': minervini_count,
            'relative_strength_records': rs_count,
            'hl_ratio_dates': hl_dates,
            'minervini_dates': minervini_dates,
            'relative_strength_dates': rs_dates,
            'hl_ratio_codes': hl_codes,
            'minervini_codes': minervini_codes,
            'relative_strength_codes': rs_codes
        }
        
        logger.info(f"Database coverage check completed")
        return coverage
        
    except sqlite3.Error as e:
        logger.error(f"Database error checking coverage: {e}")
        raise
    except Exception as e:
        logger.error(f"Error checking database coverage: {e}")
        raise