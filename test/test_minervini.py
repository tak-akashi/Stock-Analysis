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

from minervini import (
    simple_sma, 
    minervini_strategy, 
    make_minervini_df,
    init_minervini_db,
    update_minervini_db,
    update_type8_db_by_date
)


class TestMinervini:
    
    @pytest.fixture
    def sample_close_data(self):
        """Create sample close price data for testing"""
        np.random.seed(42)
        
        # Generate 300 days of price data with upward trend
        days = 300
        base_price = 100.0
        prices = []
        current_price = base_price
        
        for i in range(days):
            # Add trend and volatility
            trend = 0.001  # Slight upward trend
            volatility = np.random.normal(0, 0.02)
            current_price *= (1 + trend + volatility)
            prices.append(current_price)
        
        return np.array(prices)
    
    @pytest.fixture
    def temp_database(self):
        """Create a temporary database for testing"""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        conn = sqlite3.connect(temp_db.name)
        
        # Create tables
        conn.execute("""
        CREATE TABLE daily_quotes (
            Date TEXT,
            Code TEXT,
            AdjustmentClose REAL
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
            type_8 REAL,
            PRIMARY KEY (date, code)
        )
        """)
        
        conn.execute("""
        CREATE TABLE relative_strength (
            date TEXT,
            code TEXT,
            relative_strength_index REAL,
            PRIMARY KEY (date, code)
        )
        """)
        
        # Insert sample data
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        codes = ['1001', '1002']
        
        for code in codes:
            base_price = np.random.uniform(50, 200)
            current_price = base_price
            
            for date in dates:
                change = np.random.normal(0.001, 0.02)  # Slight upward trend with volatility
                current_price *= (1 + change)
                
                conn.execute("""
                INSERT INTO daily_quotes (Date, Code, AdjustmentClose)
                VALUES (?, ?, ?)
                """, (date.strftime('%Y-%m-%d'), code, current_price))
            
            # Insert some relative strength data for type_8 testing
            for i, date in enumerate(dates[-10:]):
                rsi = np.random.uniform(50, 95)  # Random RSI values
                conn.execute("""
                INSERT INTO relative_strength (date, code, relative_strength_index)
                VALUES (?, ?, ?)
                """, (date.strftime('%Y-%m-%d'), code, rsi))
        
        conn.commit()
        conn.close()
        
        yield temp_db.name
        
        # Clean up
        os.unlink(temp_db.name)
    
    def test_simple_sma_basic(self):
        """Test simple moving average calculation"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = simple_sma(data, 3)
        
        # First 2 values should be NaN, then moving averages
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0  # (1+2+3)/3
        assert result[3] == 3.0  # (2+3+4)/3
        assert result[-1] == 9.0  # (8+9+10)/3
    
    def test_simple_sma_insufficient_data(self):
        """Test simple moving average with insufficient data"""
        data = np.array([1, 2])
        result = simple_sma(data, 5)
        
        # All values should be NaN
        assert all(np.isnan(result))
    
    def test_simple_sma_edge_case(self):
        """Test simple moving average edge cases"""
        # Test with period equal to data length
        data = np.array([1, 2, 3, 4, 5])
        result = simple_sma(data, 5)
        
        assert all(np.isnan(result[:-1]))
        assert result[-1] == 3.0  # (1+2+3+4+5)/5
    
    @patch('minervini.HAS_TALIB', False)
    def test_minervini_strategy_without_talib(self, sample_close_data):
        """Test Minervini strategy calculation without talib"""
        with patch('minervini.logging.getLogger') as mock_logger:
            mock_logger.return_value.debug = MagicMock()
            
            result = minervini_strategy(sample_close_data)
            
            # Should return tuple of 11 elements
            assert len(result) == 11
            
            sma50, sma150, sma200, type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8 = result
            
            # Check that moving averages are calculated
            assert len(sma50) == len(sample_close_data)
            assert len(sma150) == len(sample_close_data)
            assert len(sma200) == len(sample_close_data)
            
            # Check that types are boolean arrays
            assert len(type_1) == len(sample_close_data)
            assert len(type_2) == len(sample_close_data)
            
            # Check that type_8 is all NaN (as specified in function)
            assert all(np.isnan(type_8))
    
    @patch('minervini.HAS_TALIB', True)
    def test_minervini_strategy_with_talib(self, sample_close_data):
        """Test Minervini strategy calculation with talib"""
        with patch('minervini.talib') as mock_talib:
            # Mock talib SMA function
            mock_talib.SMA.side_effect = lambda data, timeperiod: simple_sma(data, timeperiod)
            
            result = minervini_strategy(sample_close_data)
            
            # Should return tuple of 11 elements
            assert len(result) == 11
            
            # Verify talib was called
            assert mock_talib.SMA.call_count == 3
    
    def test_make_minervini_df(self, sample_close_data):
        """Test creating Minervini DataFrame"""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=len(sample_close_data), freq='D')
        codes = np.array(['1001'] * len(sample_close_data))
        
        with patch('minervini.HAS_TALIB', False):
            df = make_minervini_df(codes, dates.date, sample_close_data)
            
            # Check DataFrame structure
            expected_columns = ['code', 'close', 'sma50', 'sma150', 'sma200', 
                              'type_1', 'type_2', 'type_3', 'type_4', 'type_5', 
                              'type_6', 'type_7', 'type_8']
            assert list(df.columns) == expected_columns
            assert len(df) == len(sample_close_data)
            assert all(df['code'] == '1001')
    
    def test_init_minervini_db_legacy_function(self, temp_database):
        """Test the legacy init_minervini_db function"""
        conn = sqlite3.connect(temp_database)
        
        # Note: This tests the legacy function that hasn't been fully updated
        code_list = ['1001', '1002']
        
        with patch('minervini.pd.read_sql') as mock_read_sql:
            # Mock the SQL read to return empty data to avoid processing
            mock_read_sql.return_value = pd.DataFrame({
                'code': [],
                'close': []
            }, index=pd.DatetimeIndex([]))
            
            with patch('builtins.print'):  # Suppress print statements
                init_minervini_db(conn, code_list)
        
        conn.close()
    
    def test_update_type8_db_by_date(self, temp_database):
        """Test updating type_8 values by date"""
        conn = sqlite3.connect(temp_database)
        
        # Insert some test data into minervini table first
        test_date = '2023-12-01'
        conn.execute("""
        INSERT INTO minervini (date, code, close, sma50, sma150, sma200, type_1, type_2,
                              type_3, type_4, type_5, type_6, type_7, type_8)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (test_date, '1001', 100, 95, 90, 85, 1, 1, 1, 1, 1, 1, 1, None))
        
        conn.execute("""
        INSERT INTO minervini (date, code, close, sma50, sma150, sma200, type_1, type_2,
                              type_3, type_4, type_5, type_6, type_7, type_8)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (test_date, '1002', 100, 95, 90, 85, 1, 1, 1, 1, 1, 1, 1, None))
        
        # Insert relative strength data
        conn.execute("""
        INSERT OR REPLACE INTO relative_strength (date, code, relative_strength_index)
        VALUES (?, ?, ?)
        """, (test_date, '1001', 75))  # Above 70 threshold
        
        conn.execute("""
        INSERT OR REPLACE INTO relative_strength (date, code, relative_strength_index)
        VALUES (?, ?, ?)
        """, (test_date, '1002', 65))  # Below 70 threshold
        
        conn.commit()
        
        # Test the function
        errors = update_type8_db_by_date(conn, test_date)
        
        # Check that no errors occurred
        assert len(errors) == 0
        
        # Verify that type_8 values were updated correctly
        result = conn.execute("""
        SELECT code, type_8 FROM minervini WHERE date = ?
        ORDER BY code
        """, (test_date,)).fetchall()
        
        assert len(result) == 2
        assert result[0][1] == 1.0  # Code 1001 should have type_8 = 1.0 (RSI >= 70)
        assert result[1][1] == 0.0  # Code 1002 should have type_8 = 0.0 (RSI < 70)
        
        conn.close()
    
    def test_update_type8_db_by_date_no_data(self, temp_database):
        """Test updating type_8 when no relative strength data exists"""
        conn = sqlite3.connect(temp_database)
        
        errors = update_type8_db_by_date(conn, '2023-01-01')
        
        # Should return empty list (no errors, but no data to process)
        assert isinstance(errors, list)
        
        conn.close()
    
    def test_minervini_strategy_edge_cases(self):
        """Test Minervini strategy with edge cases"""
        # Test with minimum required data (260 days)
        close_data = np.random.uniform(50, 150, 260)
        
        with patch('minervini.HAS_TALIB', False):
            result = minervini_strategy(close_data)
            
            assert len(result) == 11
            assert all(len(arr) == 260 for arr in result)
    
    def test_minervini_strategy_type_calculations(self):
        """Test specific Minervini type calculations"""
        # Create controlled data for testing specific conditions
        close_data = np.linspace(50, 150, 300)  # Steadily increasing prices
        
        with patch('minervini.HAS_TALIB', False):
            sma50, sma150, sma200, type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8 = minervini_strategy(close_data)
            
            # With steadily increasing prices, later values should meet criteria
            # type_1: current price > sma150 and sma200
            assert any(type_1[-50:])  # Should be True for recent data
            
            # type_2: sma150 > sma200 (should be True with uptrend)
            assert any(type_2[-50:])
            
            # type_3: sma200 trending up (compare with 20 days ago)
            # Should be True for most recent data in uptrend
            assert any(type_3[-50:])
    
    def test_make_minervini_df_data_types(self):
        """Test that DataFrame contains correct data types"""
        close_data = np.random.uniform(50, 150, 300)
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        codes = np.array(['TEST'] * 300)
        
        with patch('minervini.HAS_TALIB', False):
            df = make_minervini_df(codes, dates.date, close_data)
            
            # Check that numeric columns can be converted to float
            numeric_columns = ['close', 'sma50', 'sma150', 'sma200']
            for col in numeric_columns:
                assert df[col].dtype == object  # Initially stored as object in hstack
                # Should be convertible to numeric
                pd.to_numeric(df[col], errors='coerce')
    
    def test_update_minervini_db_legacy(self, temp_database):
        """Test the legacy update_minervini_db function"""
        conn = sqlite3.connect(temp_database)
        
        code_list = ['1001']
        calc_start_date = '2023-11-01'
        calc_end_date = '2023-12-31'
        
        with patch('minervini.pd.read_sql') as mock_read_sql:
            # Mock SQL read to return insufficient data
            mock_read_sql.return_value = pd.DataFrame({
                'code': ['1001'] * 50,  # Insufficient data (< 260)
                'close': np.random.uniform(50, 150, 50)
            }, index=pd.date_range('2023-11-01', periods=50, freq='D'))
            
            # Should not raise an error even with insufficient data
            update_minervini_db(conn, code_list, calc_start_date, calc_end_date)
        
        conn.close()


if __name__ == '__main__':
    pytest.main([__file__])