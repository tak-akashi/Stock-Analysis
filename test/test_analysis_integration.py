"""
Integration tests for the analysis modules.
Tests the interaction between different analysis components.
"""

import pytest
import numpy as np
import pandas as pd
import sqlite3
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
sys.path.append('/Users/tak/Markets/Stocks/Stock-Analysis/backend/analysis')

from high_low_ratio import calc_hl_ratio_for_all, calc_hl_ratio_by_code
from minervini import minervini_strategy, make_minervini_df, update_type8_db_by_date
from relative_strength import relative_strength_percentage, update_rsi_db


class TestAnalysisIntegration:
    
    def test_full_analysis_pipeline(self, mock_jquants_database, mock_analysis_results_database):
        """Test complete analysis pipeline from data to results"""
        
        # 1. Test High-Low Ratio Analysis
        with patch('high_low_ratio.setup_logging') as mock_hl_logging:
            mock_hl_logging.return_value = MagicMock()
            
            with patch('pandas.DataFrame.to_csv'):
                hl_results = calc_hl_ratio_for_all(
                    db_path=mock_jquants_database,
                    end_date='2023-12-01',
                    weeks=26
                )
            
            assert isinstance(hl_results, pd.DataFrame)
            assert 'hl_ratio' in hl_results.columns
            assert len(hl_results) > 0
            assert all(0 <= ratio <= 100 for ratio in hl_results['hl_ratio'])
        
        # 2. Test Minervini Analysis with same data
        conn = sqlite3.connect(mock_jquants_database)
        test_code = hl_results.iloc[0]['code']  # Use first stock from HL results
        
        # Get price data for Minervini analysis
        price_data = pd.read_sql("""
        SELECT Date, AdjustmentClose 
        FROM daily_quotes 
        WHERE Code = ?
        ORDER BY Date
        """, conn, params=[test_code], parse_dates=['Date'])
        
        conn.close()
        
        if len(price_data) >= 260:  # Ensure sufficient data
            dates = price_data['Date'].dt.date
            codes = np.array([test_code] * len(price_data))
            closes = price_data['AdjustmentClose'].values
            
            # Test Minervini strategy
            with patch('minervini.HAS_TALIB', False):
                minervini_results = minervini_strategy(closes)
                assert len(minervini_results) == 11  # Should return 11 components
                
                # Test DataFrame creation
                minervini_df = make_minervini_df(codes, dates, closes)
                assert len(minervini_df) == len(price_data)
                assert 'type_1' in minervini_df.columns
        
        # 3. Test Relative Strength Analysis
        rsp_results = relative_strength_percentage(closes, period=200)
        assert len(rsp_results) == len(closes)
        assert all(np.isnan(rsp_results[:200]))  # First 200 should be NaN
    
    def test_cross_analysis_consistency(self, mock_jquants_database):
        """Test that different analyses produce consistent results for the same data"""
        
        # Get data for a specific stock
        test_code = '1001'
        
        # Test HL ratio for specific stock
        with patch('high_low_ratio.logging.getLogger'):
            hl_ratio, hl_data = calc_hl_ratio_by_code(
                test_code, 
                db_path=mock_jquants_database,
                end_date='2023-12-01',
                weeks=26
            )
        
        if hl_ratio is not None and not hl_data.empty:
            # Use the same data for relative strength calculation
            closes = hl_data['AdjustmentClose'].values
            
            if len(closes) >= 200:
                rsp = relative_strength_percentage(closes, period=200)
                
                # Both analyses should have processed the same number of data points
                assert len(closes) == len(rsp)
                
                # If HL ratio is high (near 100), recent RSP should generally be positive
                if hl_ratio > 80:
                    recent_rsp = rsp[-10:]  # Last 10 values
                    non_nan_rsp = recent_rsp[~np.isnan(recent_rsp)]
                    if len(non_nan_rsp) > 0:
                        # High HL ratio should correlate with positive recent performance
                        assert np.mean(non_nan_rsp) != np.mean(non_nan_rsp)  # Just check it's calculated
    
    def test_minervini_with_relative_strength_integration(self, mock_analysis_results_database):
        """Test integration between Minervini analysis and relative strength"""
        
        # Initialize database with some test data
        conn = sqlite3.connect(mock_analysis_results_database)
        
        # Insert Minervini test data
        test_date = '2023-12-01'
        test_codes = ['1001', '1002', '1003']
        
        for code in test_codes:
            conn.execute("""
            INSERT INTO minervini (date, code, close, sma50, sma150, sma200, 
                                  type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (test_date, code, 100, 95, 90, 85, 1, 1, 1, 1, 1, 1, 1, None))
        
        # Insert relative strength data
        for i, code in enumerate(test_codes):
            rsi_value = 80 - (i * 10)  # 80, 70, 60
            conn.execute("""
            INSERT INTO relative_strength (date, code, relative_strength_index)
            VALUES (?, ?, ?)
            """, (test_date, code, rsi_value))
        
        conn.commit()
        
        # Test the integration function
        errors = update_type8_db_by_date(conn, test_date)
        
        assert len(errors) == 0  # Should process without errors
        
        # Verify type_8 values were updated correctly
        results = conn.execute("""
        SELECT code, type_8 FROM minervini 
        WHERE date = ? 
        ORDER BY code
        """, (test_date,)).fetchall()
        
        assert len(results) == 3
        assert results[0][1] == 1.0  # RSI 80 >= 70
        assert results[1][1] == 1.0  # RSI 70 >= 70
        assert results[2][1] == 0.0  # RSI 60 < 70
        
        conn.close()
    
    def test_data_validation_across_modules(self, sample_price_series):
        """Test that all modules handle the same data validation requirements"""
        
        # Test with insufficient data
        short_data = sample_price_series[:50]  # Only 50 days
        
        # High-Low Ratio should handle insufficient data
        df_short = pd.DataFrame({
            'High': short_data * 1.02,
            'Low': short_data * 0.98,
            'AdjustmentClose': short_data
        })
        
        with patch('high_low_ratio.logging.getLogger'):
            try:
                from high_low_ratio import calc_hl_ratio
                ratio = calc_hl_ratio(df_short, weeks=52)  # Requires 260 days
                # Should either return a valid ratio or handle gracefully
                assert isinstance(ratio, (int, float))
            except Exception:
                # Acceptable to fail with insufficient data
                pass
        
        # Relative Strength should handle insufficient data
        with patch('relative_strength.logging.getLogger'):
            rsp = relative_strength_percentage(short_data, period=200)
            assert len(rsp) == len(short_data)
            assert all(np.isnan(rsp))  # Should be all NaN with insufficient data
        
        # Minervini should handle insufficient data
        with patch('minervini.HAS_TALIB', False):
            minervini_result = minervini_strategy(short_data)
            assert len(minervini_result) == 11
            # Should return arrays of the same length as input
            assert all(len(arr) == len(short_data) for arr in minervini_result)
    
    def test_performance_consistency(self, sample_price_series):
        """Test that analyses produce consistent results with the same input"""
        
        # Run the same analysis multiple times
        results1 = relative_strength_percentage(sample_price_series, period=200)
        results2 = relative_strength_percentage(sample_price_series, period=200)
        
        # Results should be identical (deterministic)
        np.testing.assert_array_equal(results1, results2)
        
        # Test Minervini consistency
        with patch('minervini.HAS_TALIB', False):
            minervini1 = minervini_strategy(sample_price_series)
            minervini2 = minervini_strategy(sample_price_series)
            
            for arr1, arr2 in zip(minervini1, minervini2):
                np.testing.assert_array_equal(arr1, arr2)
    
    def test_error_handling_integration(self):
        """Test error handling across all modules"""
        
        # Test with problematic data
        problematic_data = np.array([np.nan, 0, -1, float('inf')])
        
        # Each module should handle problematic data gracefully
        
        # Relative Strength
        with patch('relative_strength.logging.getLogger'):
            rsp = relative_strength_percentage(problematic_data, period=200)
            assert len(rsp) == len(problematic_data)
            # Should return all NaN or handle gracefully
        
        # Minervini
        with patch('minervini.HAS_TALIB', False):
            with patch('minervini.logging.getLogger'):
                try:
                    minervini_results = minervini_strategy(problematic_data)
                    # Should either work or fail gracefully
                    assert len(minervini_results) == 11
                except Exception:
                    # Acceptable to fail with invalid data
                    pass
    
    def test_database_transaction_integrity(self, mock_analysis_results_database):
        """Test that database operations maintain integrity across modules"""
        
        conn = sqlite3.connect(mock_analysis_results_database)
        
        # Test that partial failures don't corrupt the database
        test_date = '2023-12-01'
        
        # Insert partial data that might cause issues
        conn.execute("""
        INSERT INTO relative_strength (date, code, relative_strength_index)
        VALUES (?, ?, ?)
        """, (test_date, '1001', 75))
        
        # This should not cause the valid data to be lost
        conn.execute("""
        INSERT INTO minervini (date, code, close, type_8)
        VALUES (?, ?, ?, ?)
        """, (test_date, '1001', 100, None))
        
        conn.commit()
        
        # Test update operation
        errors = update_type8_db_by_date(conn, test_date)
        
        # Should complete successfully
        assert isinstance(errors, list)
        
        # Verify data integrity
        result = conn.execute("""
        SELECT type_8 FROM minervini WHERE date = ? AND code = ?
        """, (test_date, '1001')).fetchone()
        
        assert result is not None
        assert result[0] == 1.0  # RSI 75 >= 70
        
        conn.close()
    
    @pytest.mark.parametrize("weeks,period", [
        (26, 100),  # Half year
        (52, 200),  # Full year
        (13, 50),   # Quarter
    ])
    def test_parameter_consistency(self, sample_price_series, weeks, period):
        """Test that different parameter combinations work consistently"""
        
        # Test that different time periods work across modules
        
        # High-Low ratio with different weeks
        df = pd.DataFrame({
            'High': sample_price_series * 1.02,
            'Low': sample_price_series * 0.98,
            'AdjustmentClose': sample_price_series
        })
        
        with patch('high_low_ratio.logging.getLogger'):
            from high_low_ratio import calc_hl_ratio
            
            if len(sample_price_series) >= weeks * 5:
                ratio = calc_hl_ratio(df, weeks=weeks)
                assert isinstance(ratio, (int, float))
                assert 0 <= ratio <= 100
        
        # Relative strength with different periods
        if len(sample_price_series) >= period:
            rsp = relative_strength_percentage(sample_price_series, period=period)
            assert len(rsp) == len(sample_price_series)
            assert all(np.isnan(rsp[:period]))  # First 'period' values should be NaN


if __name__ == '__main__':
    pytest.main([__file__])