"""
Test script to validate that optimizations maintain calculation accuracy.
Compares results from original and optimized implementations.
"""

import os
import sys
import logging
import time
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import original implementations
from backend.analysis import high_low_ratio, relative_strength, minervini

# Import optimized implementations
from backend.analysis import high_low_ratio_optimized, relative_strength_optimized, minervini_optimized

from backend.utils.parallel_processor import measure_performance


def setup_logging():
    """Setup logging configuration."""
    log_filename = f"optimization_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare two numpy arrays and return comparison statistics.
    
    Args:
        arr1: First array
        arr2: Second array
        tolerance: Tolerance for floating point comparison
        
    Returns:
        Dictionary with comparison statistics
    """
    if arr1.shape != arr2.shape:
        return {
            'arrays_equal': False,
            'shape_match': False,
            'arr1_shape': arr1.shape,
            'arr2_shape': arr2.shape
        }
    
    # Handle NaN values
    both_nan = np.isnan(arr1) & np.isnan(arr2)
    both_not_nan = ~np.isnan(arr1) & ~np.isnan(arr2)
    
    # Check if NaN patterns match
    nan_pattern_match = np.array_equal(np.isnan(arr1), np.isnan(arr2))
    
    # Compare non-NaN values
    if np.any(both_not_nan):
        diff = np.abs(arr1[both_not_nan] - arr2[both_not_nan])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        values_close = np.allclose(arr1[both_not_nan], arr2[both_not_nan], atol=tolerance)
    else:
        max_diff = 0
        mean_diff = 0
        values_close = True
    
    arrays_equal = nan_pattern_match and values_close
    
    return {
        'arrays_equal': arrays_equal,
        'shape_match': True,
        'nan_pattern_match': nan_pattern_match,
        'values_close': values_close,
        'max_difference': max_diff,
        'mean_difference': mean_diff,
        'tolerance': tolerance
    }


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare two DataFrames and return comparison statistics.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        tolerance: Tolerance for floating point comparison
        
    Returns:
        Dictionary with comparison statistics
    """
    if df1.shape != df2.shape:
        return {
            'dataframes_equal': False,
            'shape_match': False,
            'df1_shape': df1.shape,
            'df2_shape': df2.shape
        }
    
    if not df1.columns.equals(df2.columns):
        return {
            'dataframes_equal': False,
            'columns_match': False,
            'df1_columns': df1.columns.tolist(),
            'df2_columns': df2.columns.tolist()
        }
    
    column_results = {}
    overall_equal = True
    
    for col in df1.columns:
        if df1[col].dtype == 'object' or df2[col].dtype == 'object':
            # String comparison
            col_equal = df1[col].equals(df2[col])
        else:
            # Numeric comparison
            col_result = compare_arrays(df1[col].values, df2[col].values, tolerance)
            col_equal = col_result['arrays_equal']
            column_results[col] = col_result
        
        if not col_equal:
            overall_equal = False
    
    return {
        'dataframes_equal': overall_equal,
        'shape_match': True,
        'columns_match': True,
        'column_results': column_results
    }


@measure_performance
def test_high_low_ratio(logger: logging.Logger, test_codes: list, end_date: str, weeks: int = 52) -> Dict[str, Any]:
    """Test high-low ratio calculations."""
    logger.info("Testing High-Low Ratio calculations...")
    
    # Test original implementation
    logger.info("Running original implementation...")
    start_time = time.time()
    original_result = high_low_ratio.calc_hl_ratio_for_all(end_date=end_date, weeks=weeks)
    original_time = time.time() - start_time
    
    # Test optimized implementation
    logger.info("Running optimized implementation...")
    start_time = time.time()
    optimized_result = high_low_ratio_optimized.calc_hl_ratio_for_all_optimized(end_date=end_date, weeks=weeks)
    optimized_time = time.time() - start_time
    
    # Compare results
    if original_result.empty or optimized_result.empty:
        return {
            'test_name': 'high_low_ratio',
            'success': False,
            'error': 'One or both implementations returned empty results',
            'original_time': original_time,
            'optimized_time': optimized_time
        }
    
    # Sort both DataFrames by Code for comparison
    original_sorted = original_result.sort_values('Code').reset_index(drop=True)
    optimized_sorted = optimized_result.sort_values('Code').reset_index(drop=True)
    
    # Compare only common columns
    common_columns = set(original_sorted.columns) & set(optimized_sorted.columns)
    original_subset = original_sorted[list(common_columns)]
    optimized_subset = optimized_sorted[list(common_columns)]
    
    comparison = compare_dataframes(original_subset, optimized_subset)
    
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    
    return {
        'test_name': 'high_low_ratio',
        'success': comparison['dataframes_equal'],
        'comparison': comparison,
        'original_time': original_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'original_count': len(original_result),
        'optimized_count': len(optimized_result)
    }


def test_single_stock_calculations(logger: logging.Logger, test_code: str = "7203") -> Dict[str, Any]:
    """Test calculations for a single stock to ensure accuracy."""
    logger.info(f"Testing single stock calculations for {test_code}...")
    
    # Database paths
    jquants_db_path = "/Users/tak/Markets/Stocks/Stock-Analysis/data/jquants.db"
    
    results = {}
    
    try:
        # Get stock data
        with sqlite3.connect(jquants_db_path) as conn:
            stock_data = pd.read_sql(
                """
                SELECT Date, Code, High, Low, AdjustmentClose
                FROM daily_quotes
                WHERE Code = ?
                ORDER BY Date
                """,
                conn,
                params=(test_code,),
                parse_dates=['Date']
            )
        
        if stock_data.empty:
            return {
                'test_name': 'single_stock',
                'success': False,
                'error': f'No data found for stock {test_code}'
            }
        
        logger.info(f"Found {len(stock_data)} records for {test_code}")
        
        # Test High-Low ratio calculation
        end_date = stock_data['Date'].max().strftime('%Y-%m-%d')
        
        # Original implementation
        try:
            original_hl, _ = high_low_ratio.calc_hl_ratio_by_code(test_code, end_date=end_date, save_to_db=False)
        except Exception as e:
            logger.error(f"Error in original HL ratio calculation: {e}")
            original_hl = None
        
        # Optimized implementation
        try:
            optimized_hl, _ = high_low_ratio_optimized.calc_hl_ratio_by_code(test_code, end_date=end_date, save_to_db=False)
        except Exception as e:
            logger.error(f"Error in optimized HL ratio calculation: {e}")
            optimized_hl = None
        
        # Compare HL ratios
        if original_hl and optimized_hl:
            hl_diff = abs(original_hl['HlRatio'] - optimized_hl['HlRatio'])
            median_diff = abs(original_hl['MedianRatio'] - optimized_hl['MedianRatio'])
            
            results['hl_ratio'] = {
                'success': hl_diff < 1e-6 and median_diff < 1e-6,
                'original': original_hl,
                'optimized': optimized_hl,
                'hl_difference': hl_diff,
                'median_difference': median_diff
            }
        else:
            results['hl_ratio'] = {
                'success': False,
                'error': 'One or both HL ratio calculations failed'
            }
        
        # Test relative strength calculation
        if len(stock_data) >= 200:
            stock_data_indexed = stock_data.set_index('Date')
            close_prices = pd.to_numeric(stock_data_indexed['AdjustmentClose'], errors='coerce').ffill()
            
            # Original RSP
            try:
                original_rsp = relative_strength.relative_strength_percentage(close_prices.values)
            except Exception as e:
                logger.error(f"Error in original RSP calculation: {e}")
                original_rsp = None
            
            # Optimized RSP
            try:
                optimized_df = relative_strength_optimized.relative_strength_percentage_vectorized(
                    stock_data_indexed.copy()
                )
                optimized_rsp = optimized_df['RelativeStrengthPercentage'].values
            except Exception as e:
                logger.error(f"Error in optimized RSP calculation: {e}")
                optimized_rsp = None
            
            # Compare RSP values
            if original_rsp is not None and optimized_rsp is not None:
                rsp_comparison = compare_arrays(original_rsp, optimized_rsp)
                results['rsp'] = {
                    'success': rsp_comparison['arrays_equal'],
                    'comparison': rsp_comparison
                }
            else:
                results['rsp'] = {
                    'success': False,
                    'error': 'One or both RSP calculations failed'
                }
        
        return {
            'test_name': 'single_stock',
            'stock_code': test_code,
            'success': all(r.get('success', False) for r in results.values()),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error in single stock test: {e}")
        return {
            'test_name': 'single_stock',
            'success': False,
            'error': str(e)
        }


def run_performance_benchmark(logger: logging.Logger, modules: list = None) -> Dict[str, Any]:
    """Run performance benchmark comparing original vs optimized implementations."""
    logger.info("Running performance benchmark...")
    
    if modules is None:
        modules = ['hl_ratio']  # Start with just HL ratio for testing
    
    results = {}
    
    # Test parameters
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    test_codes = ["7203", "6758", "9984", "8306", "4563"]  # Sample of major stocks
    
    for module in modules:
        if module == 'hl_ratio':
            results[module] = test_high_low_ratio(logger, test_codes, end_date)
    
    return results


def main():
    """Main test function."""
    logger = setup_logging()
    logger.info("Starting optimization validation tests...")
    
    try:
        # Test 1: Single stock accuracy test
        logger.info("=" * 60)
        logger.info("TEST 1: Single Stock Accuracy Test")
        logger.info("=" * 60)
        
        single_stock_result = test_single_stock_calculations(logger)
        
        if single_stock_result['success']:
            logger.info("✅ Single stock test PASSED")
        else:
            logger.error("❌ Single stock test FAILED")
            logger.error(f"Error: {single_stock_result.get('error', 'Unknown error')}")
        
        # Test 2: Performance benchmark
        logger.info("=" * 60)
        logger.info("TEST 2: Performance Benchmark")
        logger.info("=" * 60)
        
        benchmark_results = run_performance_benchmark(logger)
        
        # Summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        all_tests_passed = True
        
        for test_name, result in benchmark_results.items():
            if result['success']:
                speedup = result.get('speedup', 'N/A')
                logger.info(f"✅ {test_name}: PASSED (Speedup: {speedup:.2f}x)")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                all_tests_passed = False
        
        if single_stock_result['success']:
            logger.info("✅ Single stock accuracy: PASSED")
        else:
            logger.error("❌ Single stock accuracy: FAILED")
            all_tests_passed = False
        
        if all_tests_passed:
            logger.info("\n🎉 ALL TESTS PASSED! Optimizations are working correctly.")
        else:
            logger.error("\n⚠️  SOME TESTS FAILED! Please review the optimizations.")
        
        return all_tests_passed
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)