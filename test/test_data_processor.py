"""
Test module for TSE Stock Data Processor
"""

import pytest
import sqlite3
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime

from backend.agent.data_processor import (
    init_db,
    save_stock_info_to_db,
    download_tse_listed_stocks,
    fetch_and_store_tse_data,
    TSEDataProcessor,
    DATA_DIR,
    DB_PATH
)


class TestDatabaseFunctions:
    """Test database initialization and data saving functions."""
    
    def setup_method(self):
        """Setup test database in temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_stocks.db")
    
    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_init_db_creates_table(self):
        """Test that init_db creates the stocks table correctly."""
        init_db(self.test_db_path)
        
        # Check if database file was created
        assert os.path.exists(self.test_db_path)
        
        # Check if table exists and has correct schema
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'")
        table_exists = cursor.fetchone()
        assert table_exists is not None
        
        # Check table schema
        cursor.execute("PRAGMA table_info(stocks)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        expected_columns = [
            'ticker', 'longName', 'sector', 'industry', 'marketCap',
            'trailingPE', 'forwardPE', 'dividendYield', 'website',
            'currentPrice', 'regularMarketPrice', 'currency', 'exchange',
            'shortName', 'previousClose', 'open', 'dayLow', 'dayHigh',
            'volume', 'averageDailyVolume10Day', 'averageDailyVolume3Month',
            'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'fiftyDayAverage',
            'twoHundredDayAverage', 'beta', 'priceToBook', 'enterpriseValue',
            'profitMargins', 'grossMargins', 'operatingMargins',
            'returnOnAssets', 'returnOnEquity', 'freeCashflow',
            'totalCash', 'totalDebt', 'earningsGrowth', 'revenueGrowth',
            'last_updated'
        ]
        
        for col in expected_columns:
            assert col in column_names
        
        conn.close()
    
    def test_save_stock_info_to_db(self):
        """Test saving stock info to database."""
        init_db(self.test_db_path)
        
        # Mock stock info data
        mock_info = {
            'symbol': '1234.T',
            'longName': 'Test Company',
            'sector': 'Technology',
            'industry': 'Software',
            'marketCap': 1000000000,
            'trailingPE': 15.5,
            'forwardPE': 12.3,
            'dividendYield': 0.02,
            'website': 'https://test.com',
            'currentPrice': 100.0,
            'regularMarketPrice': 100.0,
            'currency': 'JPY',
            'exchange': 'TSE',
            'shortName': 'TestCo',
            'previousClose': 99.5,
            'open': 100.5,
            'dayLow': 99.0,
            'dayHigh': 101.0,
            'volume': 1000000,
            'averageDailyVolume10Day': 950000,
            'averageDailyVolume3Month': 1050000,
            'fiftyTwoWeekLow': 80.0,
            'fiftyTwoWeekHigh': 120.0,
            'fiftyDayAverage': 95.0,
            'twoHundredDayAverage': 90.0,
            'beta': 1.2,
            'priceToBook': 2.5,
            'enterpriseValue': 1200000000,
            'profitMargins': 0.15,
            'grossMargins': 0.45,
            'operatingMargins': 0.25,
            'returnOnAssets': 0.08,
            'returnOnEquity': 0.12,
            'freeCashflow': 100000000,
            'totalCash': 200000000,
            'totalDebt': 50000000,
            'earningsGrowth': 0.10,
            'revenueGrowth': 0.08
        }
        
        with patch('backend.agent.data_processor.DB_PATH', self.test_db_path):
            save_stock_info_to_db(mock_info)
        
        # Verify data was saved
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM stocks WHERE ticker = ?", ('1234.T',))
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == '1234.T'  # ticker
        assert result[1] == 'Test Company'  # longName
        assert result[2] == 'Technology'  # sector
        
        conn.close()
    
    def test_save_stock_info_handles_missing_data(self):
        """Test that save_stock_info_to_db handles missing data gracefully."""
        init_db(self.test_db_path)
        
        # Mock stock info with minimal data
        mock_info = {
            'symbol': '5678.T',
            'longName': 'Minimal Company'
        }
        
        with patch('backend.agent.data_processor.DB_PATH', self.test_db_path):
            save_stock_info_to_db(mock_info)
        
        # Verify data was saved with None values for missing fields
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ticker, longName, sector FROM stocks WHERE ticker = ?", ('5678.T',))
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == '5678.T'
        assert result[1] == 'Minimal Company'
        assert result[2] is None  # sector should be None
        
        conn.close()


class TestDataFetching:
    """Test TSE data fetching and processing functions."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_stocks.db")
    
    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('backend.agent.data_processor.requests.get')
    def test_download_tse_listed_stocks_success(self, mock_get):
        """Test successful download of TSE listed stocks."""
        # Mock the webpage response
        mock_html = """
        <html>
            <body>
                <a href="test_stocks.xlsx">Download Excel</a>
            </body>
        </html>
        """
        mock_response = Mock()
        mock_response.content = mock_html.encode('utf-8')
        mock_response.raise_for_status.return_value = None
        
        # Mock the Excel file response
        mock_excel_response = Mock()
        mock_excel_response.content = b'fake excel content'
        mock_excel_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_response, mock_excel_response]
        
        with patch('backend.agent.data_processor.DATA_DIR', self.temp_dir):
            result = download_tse_listed_stocks()
        
        assert result is not None
        assert result.endswith('tse_listed_stocks.xlsx')
        assert os.path.exists(result)
    
    @patch('backend.agent.data_processor.requests.get')
    def test_download_tse_listed_stocks_failure(self, mock_get):
        """Test handling of download failure."""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Network error")
        
        with patch('backend.agent.data_processor.DATA_DIR', self.temp_dir):
            result = download_tse_listed_stocks()
        
        assert result is None
    
    @patch('backend.agent.data_processor.download_tse_listed_stocks')
    @patch('backend.agent.data_processor.save_stock_info_to_db')
    @patch('backend.agent.data_processor.yf.Ticker')
    @patch('backend.agent.data_processor.pd.read_excel')
    def test_fetch_and_store_tse_data(self, mock_read_excel, mock_ticker, mock_save, mock_download):
        """Test fetching and storing TSE data."""
        # Mock the Excel file download
        mock_download.return_value = "/fake/path/stocks.xlsx"
        
        # Mock the Excel data
        mock_df = pd.DataFrame({
            'コード': ['1234', '5678'],
            '銘柄名': ['Company A', 'Company B'],
            '33業種区分': ['Technology', 'Finance'],
            '市場・商品区分': ['Prime', 'Standard']
        })
        mock_read_excel.return_value = mock_df
        
        # Mock yfinance ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'symbol': '1234.T',
            'longName': 'Company A',
            'sector': 'Technology'
        }
        mock_ticker.return_value = mock_ticker_instance
        
        with patch('backend.agent.data_processor.time.sleep'):  # Skip sleep for testing
            fetch_and_store_tse_data(max_workers=1, delay=0)
        
        # Verify that save_stock_info_to_db was called
        assert mock_save.call_count == 2  # Called for both stocks
    
    @patch('backend.agent.data_processor.download_tse_listed_stocks')
    def test_fetch_and_store_tse_data_no_file(self, mock_download):
        """Test handling when Excel file download fails."""
        mock_download.return_value = None
        
        # Should return early without error
        fetch_and_store_tse_data()
        
        # No assertions needed - just verify it doesn't crash


class TestTSEDataProcessor:
    """Test TSEDataProcessor class functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_stocks.db")
    
    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('backend.agent.data_processor.init_db')
    @patch('backend.agent.data_processor.os.makedirs')
    def test_tse_data_processor_init(self, mock_makedirs, mock_init_db):
        """Test TSEDataProcessor initialization."""
        processor = TSEDataProcessor(max_workers=2, rate_limit_delay=0.5)
        
        assert processor.max_workers == 2
        assert processor.rate_limit_delay == 0.5
        
        # Verify that directories are created and database is initialized
        mock_makedirs.assert_called_once()
        mock_init_db.assert_called_once()
    
    @patch('backend.agent.data_processor.fetch_and_store_tse_data')
    def test_tse_data_processor_run(self, mock_fetch):
        """Test TSEDataProcessor run method."""
        with patch('backend.agent.data_processor.init_db'), patch('backend.agent.data_processor.os.makedirs'):
            processor = TSEDataProcessor(max_workers=2, rate_limit_delay=0.5)
            processor.run()
        
        # Verify that fetch_and_store_tse_data was called with correct parameters
        mock_fetch.assert_called_once_with(max_workers=2, delay=0.5)


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_stocks.db")
    
    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('backend.agent.data_processor.download_tse_listed_stocks')
    @patch('backend.agent.data_processor.yf.Ticker')
    @patch('backend.agent.data_processor.pd.read_excel')
    def test_complete_workflow(self, mock_read_excel, mock_ticker, mock_download):
        """Test the complete workflow from initialization to data storage."""
        # Mock the Excel file download
        mock_download.return_value = "/fake/path/stocks.xlsx"
        
        # Mock the Excel data
        mock_df = pd.DataFrame({
            'コード': ['1234'],
            '銘柄名': ['Test Company'],
            '33業種区分': ['Technology'],
            '市場・商品区分': ['Prime']
        })
        mock_read_excel.return_value = mock_df
        
        # Mock yfinance ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'symbol': '1234.T',
            'longName': 'Test Company',
            'sector': 'Technology',
            'marketCap': 1000000000,
            'currentPrice': 100.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        with patch('backend.agent.data_processor.DATA_DIR', self.temp_dir):
            with patch('backend.agent.data_processor.DB_PATH', self.test_db_path):
                with patch('backend.agent.data_processor.time.sleep'):  # Skip sleep for testing
                    processor = TSEDataProcessor(max_workers=1, rate_limit_delay=0)
                    processor.run()
        
        # Verify data was saved to database
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stocks")
        count = cursor.fetchone()[0]
        
        assert count == 1
        
        cursor.execute("SELECT ticker, longName, sector FROM stocks")
        result = cursor.fetchone()
        assert result[0] == '1234.T'
        assert result[1] == 'Test Company'
        assert result[2] == 'Technology'
        
        conn.close()


if __name__ == "__main__":
    pytest.main([__file__])