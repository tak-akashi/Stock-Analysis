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

from high_low_ratio import calc_hl_ratio, calc_hl_ratio_by_code, calc_hl_ratio_for_all, init_hl_ratio_db


class TestHighLowRatio:
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data with trend
        base_price = 100
        prices = []
        current_price = base_price
        
        for i in range(300):
            # Add some volatility
            change = np.random.normal(0, 0.02) * current_price
            current_price += change
            prices.append(current_price)
        
        prices = np.array(prices)
        
        # Create high/low based on close with some spread
        high_prices = prices * (1 + np.random.uniform(0, 0.03, 300))
        low_prices = prices * (1 - np.random.uniform(0, 0.03, 300))
        
        df = pd.DataFrame({
            'Date': dates,
            'High': high_prices,
            'Low': low_prices,
            'AdjustmentClose': prices
        })
        
        return df.set_index('Date')
    
    @pytest.fixture
    def sample_price_data_legacy(self):
        """Create sample price data with legacy column names"""
        dates = pd.date_range('2023-01-01', periods=300, freq='D')
        np.random.seed(42)
        
        base_price = 100
        prices = []
        current_price = base_price
        
        for i in range(300):
            change = np.random.normal(0, 0.02) * current_price
            current_price += change
            prices.append(current_price)
        
        prices = np.array(prices)
        high_prices = prices * (1 + np.random.uniform(0, 0.03, 300))
        low_prices = prices * (1 - np.random.uniform(0, 0.03, 300))
        
        df = pd.DataFrame({
            'date': dates,
            'high': high_prices,
            'low': low_prices,
            'close': prices
        })
        
        return df.set_index('date')
    
    @pytest.fixture
    def temp_database(self):
        """Create a temporary database for testing"""
        # Create temporary database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db.close()
        
        conn = sqlite3.connect(temp_db.name)
        
        # Create table structure
        conn.execute("""
        CREATE TABLE daily_quotes (
            Date TEXT,
            Code TEXT,
            High REAL,
            Low REAL,
            AdjustmentClose REAL
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
                change = np.random.normal(0, 0.02) * current_price
                current_price += change
                
                high = current_price * (1 + np.random.uniform(0, 0.02))
                low = current_price * (1 - np.random.uniform(0, 0.02))
                
                conn.execute("""
                INSERT INTO daily_quotes (Date, Code, High, Low, AdjustmentClose)
                VALUES (?, ?, ?, ?, ?)
                """, (date.strftime('%Y-%m-%d'), code, high, low, current_price))
        
        conn.commit()
        conn.close()
        
        yield temp_db.name
        
        # Clean up
        os.unlink(temp_db.name)
    
    def test_calc_hl_ratio_basic(self, sample_price_data):
        """Test basic HL ratio calculation"""
        ratio = calc_hl_ratio(sample_price_data, weeks=52)
        
        assert isinstance(ratio, (int, float))
        assert 0 <= ratio <= 100
    
    def test_calc_hl_ratio_legacy_columns(self, sample_price_data_legacy):
        """Test HL ratio calculation with legacy column names"""
        ratio = calc_hl_ratio(sample_price_data_legacy, weeks=52)
        
        assert isinstance(ratio, (int, float))
        assert 0 <= ratio <= 100
    
    def test_calc_hl_ratio_equal_high_low(self):
        """Test HL ratio when highest and lowest prices are equal"""
        # Create data where all prices are the same
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'High': [100.0] * 100,
            'Low': [100.0] * 100,
            'AdjustmentClose': [100.0] * 100
        }, index=dates)
        
        with patch('high_low_ratio.logging.getLogger') as mock_logger:
            mock_logger.return_value.warning = MagicMock()
            ratio = calc_hl_ratio(df, weeks=10)
            
            assert ratio == 50.0
            mock_logger.return_value.warning.assert_called_once()
    
    def test_calc_hl_ratio_short_period(self, sample_price_data):
        """Test HL ratio calculation with shorter period"""
        ratio = calc_hl_ratio(sample_price_data, weeks=4)
        
        assert isinstance(ratio, (int, float))
        assert 0 <= ratio <= 100
    
    def test_calc_hl_ratio_by_code_success(self, temp_database):
        """Test HL ratio calculation for specific code with database save"""
        temp_results_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_results_db.close()
        
        try:
            with patch('high_low_ratio.logging.getLogger') as mock_logger:
                mock_logger.return_value.info = MagicMock()
                mock_logger.return_value.debug = MagicMock()
                
                # Initialize the database first
                init_hl_ratio_db(temp_results_db.name)
                
                with patch('high_low_ratio.RESULTS_DB_PATH', temp_results_db.name):
                    ratio, data = calc_hl_ratio_by_code(
                        '1001', 
                        db_path=temp_database, 
                        end_date='2023-12-31', 
                        weeks=20,
                        save_to_db=True
                    )
                    
                    assert isinstance(ratio, (int, float))
                    assert 0 <= ratio <= 100
                    assert isinstance(data, pd.DataFrame)
                    assert not data.empty
                    
                    # Verify data was saved to database
                    conn = sqlite3.connect(temp_results_db.name)
                    db_results = conn.execute("SELECT * FROM hl_ratio WHERE code = '1001'").fetchall()
                    assert len(db_results) == 1
                    assert db_results[0][2] == ratio  # hl_ratio column
                    conn.close()
        finally:
            os.unlink(temp_results_db.name)
    
    def test_calc_hl_ratio_by_code_insufficient_data(self, temp_database):
        """Test HL ratio calculation with insufficient data"""
        with patch('high_low_ratio.logging.getLogger') as mock_logger:
            mock_logger.return_value.warning = MagicMock()
            
            ratio, data = calc_hl_ratio_by_code(
                '1001', 
                db_path=temp_database, 
                end_date='2023-02-15',  # Early date with insufficient data
                weeks=52
            )
            
            assert ratio is None
            mock_logger.return_value.warning.assert_called()
    
    def test_calc_hl_ratio_by_code_nonexistent_code(self, temp_database):
        """Test HL ratio calculation for nonexistent code"""
        with patch('high_low_ratio.logging.getLogger') as mock_logger:
            mock_logger.return_value.warning = MagicMock()
            
            ratio, data = calc_hl_ratio_by_code(
                'NONEXISTENT', 
                db_path=temp_database, 
                end_date='2023-12-31', 
                weeks=20
            )
            
            assert ratio is None
            assert data.empty
    
    @patch('high_low_ratio.setup_logging')
    @patch('os.makedirs')
    def test_calc_hl_ratio_for_all_success(self, mock_makedirs, mock_setup_logging, temp_database):
        """Test HL ratio calculation for all stocks with database storage"""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Create temporary results database
        temp_results_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_results_db.close()
        
        try:
            # Initialize the database first
            init_hl_ratio_db(temp_results_db.name)
            
            with patch('high_low_ratio.RESULTS_DB_PATH', temp_results_db.name):
                result = calc_hl_ratio_for_all(
                    db_path=temp_database,
                    end_date='2023-12-31',
                    weeks=20
                )
                
                assert isinstance(result, pd.DataFrame)
                assert 'code' in result.columns
                assert 'hl_ratio' in result.columns
                assert 'date' in result.columns
                assert 'weeks' in result.columns
                assert len(result) > 0
                
                # Check if results are sorted by hl_ratio descending
                assert all(result['hl_ratio'].iloc[i] >= result['hl_ratio'].iloc[i+1] 
                          for i in range(len(result)-1))
                
                # Verify data was saved to database
                conn = sqlite3.connect(temp_results_db.name)
                db_results = conn.execute("SELECT * FROM hl_ratio").fetchall()
                assert len(db_results) > 0
                conn.close()
        finally:
            os.unlink(temp_results_db.name)
    
    @patch('high_low_ratio.setup_logging')
    def test_calc_hl_ratio_for_all_database_error(self, mock_setup_logging):
        """Test HL ratio calculation with database error"""
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        with pytest.raises((sqlite3.OperationalError, pd.errors.DatabaseError)):
            calc_hl_ratio_for_all(db_path='nonexistent.db')
    
    def test_calc_hl_ratio_date_string_input(self, temp_database):
        """Test that string dates are properly converted"""
        with patch('high_low_ratio.logging.getLogger'):
            ratio, data = calc_hl_ratio_by_code(
                '1001', 
                db_path=temp_database, 
                end_date='2023-12-31',  # String date
                weeks=20
            )
            
            assert isinstance(ratio, (int, float)) or ratio is None
    
    def test_calc_hl_ratio_default_date(self, temp_database):
        """Test default date handling (today)"""
        with patch('high_low_ratio.logging.getLogger'):
            with patch('high_low_ratio.datetime') as mock_datetime:
                mock_datetime.datetime.today.return_value = datetime.datetime(2023, 12, 31)
                mock_datetime.datetime.strptime.side_effect = datetime.datetime.strptime
                
                ratio, data = calc_hl_ratio_by_code('1001', db_path=temp_database)
                
                # Should not raise an error
                assert isinstance(ratio, (int, float)) or ratio is None
    
    def test_calc_hl_ratio_with_nan_values(self):
        """Test HL ratio calculation with NaN values in data"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'High': [100.0] * 100,
            'Low': [90.0] * 100,
            'AdjustmentClose': [95.0] * 100
        }, index=dates)
        
        # Introduce some NaN values
        df.loc[df.index[10:15], 'High'] = np.nan
        df.loc[df.index[20:25], 'Low'] = np.nan
        
        # Should handle NaN values through fillna
        ratio = calc_hl_ratio(df, weeks=10)
        assert isinstance(ratio, (int, float))
        assert 0 <= ratio <= 100
    
    def test_init_hl_ratio_db(self):
        """Test initialization of HL ratio database"""
        temp_results_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_results_db.close()
        
        try:
            with patch('high_low_ratio.logging.getLogger') as mock_logger:
                mock_logger.return_value.info = MagicMock()
                
                init_hl_ratio_db(temp_results_db.name)
                
                # Check that table was created
                conn = sqlite3.connect(temp_results_db.name)
                tables = conn.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='hl_ratio'
                """).fetchall()
                assert len(tables) == 1
                
                # Check table structure
                columns = conn.execute("PRAGMA table_info(hl_ratio)").fetchall()
                column_names = [col[1] for col in columns]
                expected_columns = ['date', 'code', 'hl_ratio', 'weeks']
                assert all(col in column_names for col in expected_columns)
                
                conn.close()
        finally:
            os.unlink(temp_results_db.name)
    
    def test_calc_hl_ratio_by_code_no_save(self, temp_database):
        """Test HL ratio calculation without saving to database"""
        with patch('high_low_ratio.logging.getLogger') as mock_logger:
            mock_logger.return_value.info = MagicMock()
            
            ratio, data = calc_hl_ratio_by_code(
                '1001', 
                db_path=temp_database, 
                end_date='2023-12-31', 
                weeks=20,
                save_to_db=False
            )
            
            assert isinstance(ratio, (int, float))
            assert 0 <= ratio <= 100
            assert isinstance(data, pd.DataFrame)
            assert not data.empty


if __name__ == '__main__':
    pytest.main([__file__])