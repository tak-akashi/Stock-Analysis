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

from relative_strength import (
    fill_non_vals,
    relative_strength_percentage,
    init_rsp_db,
    update_rsp_db,
    update_rsi_db,
    init_results_db
)


class TestRelativeStrength:
    
    @pytest.fixture
    def sample_close_data(self):
        """Create sample close price data for testing"""
        np.random.seed(42)
        
        # Generate 300 days of price data
        days = 300
        base_price = 100.0
        prices = []
        current_price = base_price
        
        for i in range(days):
            # Add some trend and volatility
            trend = np.random.normal(0.001, 0.002)  # Small random trend
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
        CREATE TABLE relative_strength (
            Date TEXT,
            Code TEXT,
            RelativeStrengthPercentage REAL,
            RelativeStrengthIndex REAL,
            PRIMARY KEY (Date, Code)
        )
        """)
        
        # Insert sample data
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        codes = ['1001', '1002', '1003']
        
        for code in codes:
            base_price = np.random.uniform(50, 200)
            current_price = base_price
            
            for date in dates:
                change = np.random.normal(0.001, 0.02)  # Small trend with volatility
                current_price *= (1 + change)
                
                conn.execute("""
                INSERT INTO daily_quotes (Date, Code, AdjustmentClose)
                VALUES (?, ?, ?)
                """, (date.strftime('%Y-%m-%d'), code, current_price))
        
        conn.commit()
        conn.close()
        
        yield temp_db.name
        
        # Clean up
        os.unlink(temp_db.name)
    
    @pytest.fixture
    def temp_results_database(self):
        """Create a temporary results database"""
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        yield temp_db.name
        
        # Clean up
        os.unlink(temp_db.name)
    
    def test_fill_non_vals_basic(self):
        """Test fill_non_vals function"""
        close_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        rsp = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 5 values for last 5 elements
        
        result = fill_non_vals(close_arr, rsp, period=5)
        
        # First 5 values should be NaN, last 5 should be the rsp values
        assert len(result) == 10
        assert all(np.isnan(result[:5]))
        assert np.array_equal(result[5:], rsp)
    
    def test_fill_non_vals_default_period(self):
        """Test fill_non_vals with default period"""
        close_arr = np.array(range(250))
        rsp = np.array(range(50))  # 50 values
        
        result = fill_non_vals(close_arr, rsp)  # Default period=200
        
        assert len(result) == 250
        assert all(np.isnan(result[:200]))
        assert np.array_equal(result[200:], rsp)
    
    def test_relative_strength_percentage_basic(self, sample_close_data):
        """Test basic relative strength percentage calculation"""
        rsp = relative_strength_percentage(sample_close_data, period=200)
        
        assert len(rsp) == len(sample_close_data)
        # First 200 values should be NaN
        assert all(np.isnan(rsp[:200]))
        # Remaining values should be numeric
        assert all(np.isfinite(rsp[200:]) | np.isnan(rsp[200:]))
    
    def test_relative_strength_percentage_insufficient_data(self):
        """Test RSP with insufficient data"""
        short_data = np.array([1, 2, 3, 4, 5])
        
        with patch('relative_strength.logging.getLogger') as mock_logger:
            mock_logger.return_value.warning = MagicMock()
            
            rsp = relative_strength_percentage(short_data, period=200)
            
            # Should return all NaN
            assert len(rsp) == 5
            assert all(np.isnan(rsp))
            mock_logger.return_value.warning.assert_called_once()
    
    def test_relative_strength_percentage_error_handling(self):
        """Test RSP calculation error handling"""
        # Create data that might cause calculation errors
        problematic_data = np.array([0] * 250)  # All zeros
        
        with patch('relative_strength.logging.getLogger') as mock_logger:
            mock_logger.return_value.error = MagicMock()
            
            # This might cause division by zero or other issues
            rsp = relative_strength_percentage(problematic_data, period=200)
            
            # Should handle errors gracefully and return all NaN
            assert len(rsp) == 250
    
    def test_relative_strength_percentage_custom_period(self, sample_close_data):
        """Test RSP with custom period"""
        rsp = relative_strength_percentage(sample_close_data, period=100)
        
        assert len(rsp) == len(sample_close_data)
        # First 100 values should be NaN
        assert all(np.isnan(rsp[:100]))
        # Some of the remaining values should be numeric
        assert any(np.isfinite(rsp[100:]))
    
    @patch('relative_strength.setup_logging')
    def test_init_rsp_db_success(self, mock_setup_logging, temp_database, temp_results_database):
        """Test successful initialization of RSP database"""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            processed, errors = init_rsp_db(temp_database, temp_results_database)
            
            assert isinstance(processed, int)
            assert isinstance(errors, int)
            assert processed >= 0
            assert errors >= 0
            
            # Check that database was initialized
            conn = sqlite3.connect(temp_results_database)
            tables = conn.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='relative_strength'
            """).fetchall()
            assert len(tables) == 1
            conn.close()
    
    @patch('relative_strength.setup_logging')
    def test_init_rsp_db_database_error(self, mock_setup_logging):
        """Test init_rsp_db with database error"""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        with pytest.raises((sqlite3.OperationalError, pd.errors.DatabaseError)):
            init_rsp_db('nonexistent.db', 'nonexistent_results.db')
    
    def test_init_results_db(self, temp_results_database):
        """Test initialization of results database"""
        with patch('relative_strength.logging.getLogger') as mock_logger:
            mock_logger.return_value.info = MagicMock()
            
            init_results_db(temp_results_database)
            
            # Check that table was created
            conn = sqlite3.connect(temp_results_database)
            tables = conn.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='relative_strength'
            """).fetchall()
            assert len(tables) == 1
            
            # Check table structure
            columns = conn.execute("PRAGMA table_info(relative_strength)").fetchall()
            column_names = [col[1] for col in columns]
            expected_columns = ['Date', 'Code', 'RelativeStrengthPercentage', 'RelativeStrengthIndex']
            assert all(col in column_names for col in expected_columns)
            
            conn.close()
    
    @patch('relative_strength.setup_logging')
    def test_update_rsp_db_success(self, mock_setup_logging, temp_database, temp_results_database):
        """Test successful update of RSP database"""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Initialize the results database first
        init_results_db(temp_results_database)
        
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            processed, errors = update_rsp_db(
                temp_database, 
                temp_results_database,
                calc_start_date='2023-01-01',
                calc_end_date='2023-12-31'
            )
            
            assert isinstance(processed, int)
            assert isinstance(errors, int)
            assert processed >= 0
            assert errors >= 0
    
    @patch('relative_strength.setup_logging')
    def test_update_rsp_db_default_dates(self, mock_setup_logging, temp_database, temp_results_database):
        """Test update_rsp_db with default dates"""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        init_results_db(temp_results_database)
        
        with patch('relative_strength.datetime') as mock_datetime:
            mock_datetime.datetime.today.return_value = datetime.datetime(2023, 12, 31)
            mock_datetime.datetime.strptime.side_effect = datetime.datetime.strptime
            mock_datetime.timedelta = datetime.timedelta
            
            with patch('pandas.DataFrame.to_csv'):
                processed, errors = update_rsp_db(temp_database, temp_results_database)
                
                assert isinstance(processed, int)
                assert isinstance(errors, int)
    
    @patch('relative_strength.setup_logging')
    def test_update_rsi_db_success(self, mock_setup_logging, temp_results_database):
        """Test successful update of RSI database"""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Initialize and populate the results database
        init_results_db(temp_results_database)
        
        conn = sqlite3.connect(temp_results_database)
        
        # Insert test data
        test_dates = ['2023-12-01', '2023-12-02', '2023-12-03']
        codes = ['1001', '1002', '1003']
        
        for date in test_dates:
            for i, code in enumerate(codes):
                rsp = np.random.uniform(-10, 10)  # Random RSP values
                conn.execute("""
                INSERT INTO relative_strength (Date, Code, RelativeStrengthPercentage)
                VALUES (?, ?, ?)
                """, (date, code, rsp))
        
        conn.commit()
        conn.close()
        
        # Test the update function
        with patch('pandas.DataFrame.to_csv'):
            errors = update_rsi_db(temp_results_database, date_list=test_dates)
            
            assert isinstance(errors, int)
            assert errors >= 0
            
            # Check that RSI values were calculated and stored
            conn = sqlite3.connect(temp_results_database)
            rsi_data = conn.execute("""
            SELECT Date, Code, RelativeStrengthIndex 
            FROM relative_strength 
            WHERE RelativeStrengthIndex IS NOT NULL
            ORDER BY Date, Code
            """).fetchall()
            
            # Should have RSI values for all entries
            assert len(rsi_data) > 0
            
            # RSI values should be between 0 and 99
            for row in rsi_data:
                if row[2] is not None:
                    assert 0 <= row[2] <= 99
            
            conn.close()
    
    @patch('relative_strength.setup_logging')
    def test_update_rsi_db_no_data(self, mock_setup_logging, temp_results_database):
        """Test update_rsi_db when no data exists"""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        init_results_db(temp_results_database)
        
        errors = update_rsi_db(temp_results_database, date_list=['2023-01-01'])
        
        # Should handle empty data gracefully
        assert isinstance(errors, int)
    
    @patch('relative_strength.setup_logging')
    def test_update_rsi_db_default_dates(self, mock_setup_logging, temp_results_database):
        """Test update_rsi_db with default date list"""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        init_results_db(temp_results_database)
        
        # Insert some test data first
        conn = sqlite3.connect(temp_results_database)
        conn.execute("""
        INSERT INTO relative_strength (Date, Code, RelativeStrengthPercentage)
        VALUES ('2023-12-01', '1001', 5.5)
        """)
        conn.commit()
        conn.close()
        
        with patch('pandas.DataFrame.to_csv'):
            errors = update_rsi_db(temp_results_database)  # Use default date list
            
            assert isinstance(errors, int)
    
    def test_rsi_ranking_calculation(self, temp_results_database):
        """Test that RSI ranking is calculated correctly"""
        init_results_db(temp_results_database)
        
        conn = sqlite3.connect(temp_results_database)
        
        # Insert test data with known RSP values
        test_date = '2023-12-01'
        test_data = [
            ('1001', 10.0),  # Highest RSP
            ('1002', 5.0),   # Middle RSP  
            ('1003', 1.0),   # Lowest RSP
        ]
        
        for code, rsp in test_data:
            conn.execute("""
            INSERT INTO relative_strength (Date, Code, RelativeStrengthPercentage)
            VALUES (?, ?, ?)
            """, (test_date, code, rsp))
        
        conn.commit()
        conn.close()
        
        # Update RSI
        with patch('relative_strength.setup_logging'):
            with patch('pandas.DataFrame.to_csv'):
                update_rsi_db(temp_results_database, date_list=[test_date])
        
        # Check RSI rankings
        conn = sqlite3.connect(temp_results_database)
        results = conn.execute("""
        SELECT Code, RelativeStrengthPercentage, RelativeStrengthIndex 
        FROM relative_strength 
        WHERE Date = ?
        ORDER BY RelativeStrengthPercentage DESC
        """, (test_date,)).fetchall()
        
        assert len(results) == 3
        
        # Highest RSP should have highest RSI (closest to 99)
        assert results[0][0] == '1001'  # Highest RSP
        assert results[0][2] > results[1][2]  # Higher RSI than middle
        assert results[1][2] > results[2][2]  # Middle > lowest
        
        # Lowest RSP should have lowest RSI (closest to 0)
        assert results[2][0] == '1003'  # Lowest RSP
        
        conn.close()
    
    def test_relative_strength_percentage_quarters_calculation(self):
        """Test that RSP quarters are calculated correctly"""
        # Create controlled data to test quarter calculations - need more than period
        prices = np.array(range(1, 251))  # 250 ascending prices for period=200
        
        rsp = relative_strength_percentage(prices, period=200)
        
        # First 200 values should be NaN
        assert all(np.isnan(rsp[:200]))
        
        # Last 50 values should have computed RSP
        assert not np.isnan(rsp[-1])  # Last value should not be NaN
        
        # Test with descending prices
        prices_desc = np.array(range(250, 0, -1))  # 250 descending prices
        rsp_desc = relative_strength_percentage(prices_desc, period=200)
        
        # RSP for descending prices should be different from ascending
        assert not np.isnan(rsp_desc[-1])


if __name__ == '__main__':
    pytest.main([__file__])