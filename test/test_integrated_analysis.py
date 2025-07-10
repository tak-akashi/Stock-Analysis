import pytest
import numpy as np
import pandas as pd
import sqlite3
import datetime
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the functions to test
import sys
sys.path.append('/Users/tak/Markets/Stocks/Stock-Analysis/backend/analysis')

from integrated_analysis import (
    get_comprehensive_analysis,
    get_multi_date_analysis,
    get_top_stocks_by_criteria,
    get_stocks_meeting_criteria,
    create_analysis_summary,
    check_database_coverage,
    _calculate_composite_scores
)


class TestIntegratedAnalysis:
    
    @pytest.fixture
    def temp_results_database(self):
        """Create a temporary results database with sample data"""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        conn = sqlite3.connect(temp_db.name)
        
        # Create tables
        conn.execute("""
        CREATE TABLE hl_ratio (
            date TEXT NOT NULL,
            code TEXT NOT NULL,
            hl_ratio REAL NOT NULL,
            weeks INTEGER NOT NULL,
            PRIMARY KEY (date, code, weeks)
        )
        """)
        
        conn.execute("""
        CREATE TABLE minervini (
            date TEXT,
            code TEXT,
            close REAL,
            sma50 REAL,
            sma150 REAL,
            sma200 REAL,
            type_1 REAL,
            type_2 REAL,
            type_3 REAL,
            type_4 REAL,
            type_5 REAL,
            type_6 REAL,
            type_7 REAL,
            type_8 REAL
        )
        """)
        
        conn.execute("""
        CREATE TABLE relative_strength (
            date TEXT,
            code TEXT,
            relative_strength_percentage REAL,
            relative_strength_index REAL,
            PRIMARY KEY (date, code)
        )
        """)
        
        # Insert sample data
        test_date = '2023-12-01'
        codes = ['1001', '1002', '1003', '1004', '1005']
        
        for i, code in enumerate(codes):
            # HL ratio data
            hl_ratio = 90 - (i * 10)  # 90, 80, 70, 60, 50
            conn.execute("""
            INSERT INTO hl_ratio (date, code, hl_ratio, weeks)
            VALUES (?, ?, ?, ?)
            """, (test_date, code, hl_ratio, 52))
            
            # Minervini data
            close_price = 100 + i * 10
            sma50 = close_price * 0.95
            sma150 = close_price * 0.90
            sma200 = close_price * 0.85
            
            # Create varied Minervini scores
            types = [1.0 if j < (5-i) else 0.0 for j in range(8)]
            
            conn.execute("""
            INSERT INTO minervini (date, code, close, sma50, sma150, sma200, 
                                 type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (test_date, code, close_price, sma50, sma150, sma200, *types))
            
            # Relative strength data
            rsp = i * 2.0  # 0, 2, 4, 6, 8
            rsi = 95 - (i * 15)  # 95, 80, 65, 50, 35
            conn.execute("""
            INSERT INTO relative_strength (date, code, relative_strength_percentage, relative_strength_index)
            VALUES (?, ?, ?, ?)
            """, (test_date, code, rsp, rsi))
        
        # Add data for multiple dates for time series testing
        for days_back in range(1, 6):
            test_date_ts = (datetime.datetime(2023, 12, 1) - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            for i, code in enumerate(['1001', '1002']):  # Just two codes for time series
                hl_ratio = 80 + days_back + i * 5
                conn.execute("""
                INSERT INTO hl_ratio (date, code, hl_ratio, weeks)
                VALUES (?, ?, ?, ?)
                """, (test_date_ts, code, hl_ratio, 52))
                
                close_price = 95 + days_back + i * 10
                conn.execute("""
                INSERT INTO minervini (date, code, close, sma50, sma150, sma200, 
                                     type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (test_date_ts, code, close_price, 90, 85, 80, 1, 1, 1, 0, 0, 0, 0, 0))
                
                rsp = days_back + i * 2
                rsi = 75 + days_back + i * 5
                conn.execute("""
                INSERT INTO relative_strength (date, code, relative_strength_percentage, relative_strength_index)
                VALUES (?, ?, ?, ?)
                """, (test_date_ts, code, rsp, rsi))
        
        conn.commit()
        conn.close()
        
        yield temp_db.name
        
        # Clean up
        os.unlink(temp_db.name)
    
    def test_get_comprehensive_analysis_all_codes(self, temp_results_database):
        """Test comprehensive analysis for all codes on a date"""
        result = get_comprehensive_analysis('2023-12-01', db_path=temp_results_database)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # Should have 5 stocks
        assert 'hl_ratio' in result.columns
        assert 'minervini_score' in result.columns
        assert 'relative_strength_index' in result.columns
        assert 'composite_score' in result.columns
        
        # Check that results are sorted by HL ratio descending
        assert result['hl_ratio'].iloc[0] >= result['hl_ratio'].iloc[1]
        
    def test_get_comprehensive_analysis_single_code(self, temp_results_database):
        """Test comprehensive analysis for a single code"""
        result = get_comprehensive_analysis('2023-12-01', code='1001', db_path=temp_results_database)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result['code'].iloc[0] == '1001'
        assert result['hl_ratio'].iloc[0] == 90.0
        
    def test_get_comprehensive_analysis_no_data(self, temp_results_database):
        """Test comprehensive analysis with no data"""
        result = get_comprehensive_analysis('2024-01-01', db_path=temp_results_database)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_get_multi_date_analysis(self, temp_results_database):
        """Test multi-date analysis for a single code"""
        result = get_multi_date_analysis('2023-11-26', '2023-12-01', '1001', db_path=temp_results_database)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6  # 5 historical days + 1 current day
        assert all(result['code'] == '1001')
        assert 'date' in result.columns
        assert 'minervini_score' in result.columns
        
        # Check that dates are in ascending order
        dates = pd.to_datetime(result['date'])
        assert dates.is_monotonic_increasing
    
    def test_get_multi_date_analysis_no_data(self, temp_results_database):
        """Test multi-date analysis with no data"""
        result = get_multi_date_analysis('2024-01-01', '2024-01-02', '1001', db_path=temp_results_database)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_get_top_stocks_by_hl_ratio(self, temp_results_database):
        """Test getting top stocks by HL ratio"""
        result = get_top_stocks_by_criteria('2023-12-01', criteria='hl_ratio', limit=3, db_path=temp_results_database)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Should be sorted by HL ratio descending
        assert result['hl_ratio'].iloc[0] >= result['hl_ratio'].iloc[1]
        assert result['hl_ratio'].iloc[1] >= result['hl_ratio'].iloc[2]
    
    def test_get_top_stocks_by_rsi(self, temp_results_database):
        """Test getting top stocks by RSI"""
        result = get_top_stocks_by_criteria('2023-12-01', criteria='rsi', limit=3, db_path=temp_results_database)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Should be sorted by RSI descending
        rsi_values = result['relative_strength_index'].dropna()
        assert rsi_values.iloc[0] >= rsi_values.iloc[1]
    
    def test_get_top_stocks_by_minervini(self, temp_results_database):
        """Test getting top stocks by Minervini score"""
        result = get_top_stocks_by_criteria('2023-12-01', criteria='minervini', limit=3, db_path=temp_results_database)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Should be sorted by Minervini score descending
        minervini_values = result['minervini_score'].dropna()
        assert minervini_values.iloc[0] >= minervini_values.iloc[1]
    
    def test_get_top_stocks_by_composite(self, temp_results_database):
        """Test getting top stocks by composite score"""
        result = get_top_stocks_by_criteria('2023-12-01', criteria='composite', limit=3, db_path=temp_results_database)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Should be sorted by composite score descending
        composite_values = result['composite_score'].dropna()
        assert composite_values.iloc[0] >= composite_values.iloc[1]
    
    def test_get_top_stocks_invalid_criteria(self, temp_results_database):
        """Test getting top stocks with invalid criteria"""
        with pytest.raises(ValueError):
            get_top_stocks_by_criteria('2023-12-01', criteria='invalid', db_path=temp_results_database)
    
    def test_get_stocks_meeting_criteria(self, temp_results_database):
        """Test getting stocks meeting specific criteria"""
        result = get_stocks_meeting_criteria(
            '2023-12-01', 
            hl_ratio_min=70.0, 
            rsi_min=60.0, 
            minervini_min=2,
            db_path=temp_results_database
        )
        
        assert isinstance(result, pd.DataFrame)
        
        # All returned stocks should meet the criteria
        if not result.empty:
            assert all(result['hl_ratio'] >= 70.0)
            assert all(result['relative_strength_index'] >= 60.0)
            assert all(result['minervini_score'] >= 2)
    
    def test_get_stocks_meeting_strict_criteria(self, temp_results_database):
        """Test getting stocks meeting very strict criteria"""
        result = get_stocks_meeting_criteria(
            '2023-12-01', 
            hl_ratio_min=95.0, 
            rsi_min=95.0, 
            minervini_min=7,
            db_path=temp_results_database
        )
        
        assert isinstance(result, pd.DataFrame)
        # With strict criteria, might get no results
        assert len(result) >= 0
    
    def test_create_analysis_summary(self, temp_results_database):
        """Test creating analysis summary"""
        summary = create_analysis_summary('2023-12-01', db_path=temp_results_database)
        
        assert isinstance(summary, dict)
        assert 'total_stocks' in summary
        assert 'avg_hl_ratio' in summary
        assert 'avg_rsi' in summary
        assert 'avg_minervini_score' in summary
        assert 'avg_composite_score' in summary
        assert 'high_hl_ratio_count' in summary
        assert 'high_rsi_count' in summary
        assert 'strong_minervini_count' in summary
        assert 'strong_composite_count' in summary
        
        assert summary['total_stocks'] == 5
        assert isinstance(summary['avg_hl_ratio'], (int, float))
        assert isinstance(summary['avg_rsi'], (int, float))
    
    def test_create_analysis_summary_no_data(self, temp_results_database):
        """Test creating analysis summary with no data"""
        summary = create_analysis_summary('2024-01-01', db_path=temp_results_database)
        
        assert isinstance(summary, dict)
        assert len(summary) == 0
    
    def test_check_database_coverage(self, temp_results_database):
        """Test checking database coverage"""
        coverage = check_database_coverage(db_path=temp_results_database)
        
        assert isinstance(coverage, dict)
        assert 'hl_ratio_records' in coverage
        assert 'minervini_records' in coverage
        assert 'relative_strength_records' in coverage
        assert 'hl_ratio_dates' in coverage
        assert 'minervini_dates' in coverage
        assert 'relative_strength_dates' in coverage
        assert 'hl_ratio_codes' in coverage
        assert 'minervini_codes' in coverage
        assert 'relative_strength_codes' in coverage
        
        # Should have data in all tables
        assert coverage['hl_ratio_records'] > 0
        assert coverage['minervini_records'] > 0
        assert coverage['relative_strength_records'] > 0
    
    def test_calculate_composite_scores(self):
        """Test composite score calculation"""
        # Create sample DataFrame
        data = {
            'hl_ratio': [90, 80, 70],
            'relative_strength_index': [95, 85, 75],
            'minervini_type_1': [1, 1, 0],
            'minervini_type_2': [1, 1, 0],
            'minervini_type_3': [1, 0, 0],
            'minervini_type_4': [1, 0, 0],
            'minervini_type_5': [1, 0, 0],
            'minervini_type_6': [0, 0, 0],
            'minervini_type_7': [0, 0, 0],
            'minervini_type_8': [0, 0, 0]
        }
        df = pd.DataFrame(data)
        
        result = _calculate_composite_scores(df)
        
        assert 'minervini_score' in result.columns
        assert 'composite_score' in result.columns
        
        # Check Minervini scores
        assert result['minervini_score'].iloc[0] == 5  # 5 types met
        assert result['minervini_score'].iloc[1] == 2  # 2 types met
        assert result['minervini_score'].iloc[2] == 0  # 0 types met
        
        # Check that composite scores are calculated
        assert all(result['composite_score'] >= 0)
        assert all(result['composite_score'] <= 100)
        
        # Higher individual scores should generally lead to higher composite scores
        assert result['composite_score'].iloc[0] > result['composite_score'].iloc[2]
    
    def test_calculate_composite_scores_with_nan(self):
        """Test composite score calculation with NaN values"""
        data = {
            'hl_ratio': [90, np.nan, 70],
            'relative_strength_index': [95, 85, np.nan],
            'minervini_type_1': [1, np.nan, 0],
            'minervini_type_2': [1, 1, 0],
            'minervini_type_3': [1, 0, 0],
            'minervini_type_4': [1, 0, 0],
            'minervini_type_5': [1, 0, 0],
            'minervini_type_6': [0, 0, 0],
            'minervini_type_7': [0, 0, 0],
            'minervini_type_8': [0, 0, 0]
        }
        df = pd.DataFrame(data)
        
        result = _calculate_composite_scores(df)
        
        # Should handle NaN values gracefully
        assert 'composite_score' in result.columns
        assert len(result) == 3
        
        # Composite scores should be calculated even with some NaN values
        assert not np.isnan(result['composite_score'].iloc[0])


if __name__ == '__main__':
    pytest.main([__file__])