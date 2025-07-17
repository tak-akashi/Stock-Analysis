#!/usr/bin/env python3
"""
Test script for the optimized integrated analysis functions
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from backend.analysis.integrated_analysis import (
    get_comprehensive_analysis, 
    create_analysis_summary,
    check_database_coverage
)

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'test_integrated_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def test_comprehensive_analysis():
    """Test the optimized comprehensive analysis function"""
    logger = logging.getLogger(__name__)
    logger.info("Testing optimized comprehensive analysis...")
    
    # Get latest date
    results_db_path = os.path.join(project_root, "data", "analysis_results.db")
    
    if not os.path.exists(results_db_path):
        logger.error(f"Database not found: {results_db_path}")
        return False
    
    import sqlite3
    with sqlite3.connect(results_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM hl_ratio")
        latest_date = cursor.fetchone()[0]
    
    if not latest_date:
        logger.error("No data found in hl_ratio table")
        return False
    
    logger.info(f"Testing with date: {latest_date}")
    
    # Test comprehensive analysis
    start_time = time.time()
    df = get_comprehensive_analysis(latest_date)
    analysis_time = time.time() - start_time
    
    logger.info(f"Comprehensive analysis completed in {analysis_time:.2f} seconds")
    logger.info(f"Retrieved {len(df)} records")
    
    if not df.empty:
        # Check if composite scores were calculated
        if 'composite_score' in df.columns:
            logger.info(f"Composite scores: min={df['composite_score'].min():.2f}, max={df['composite_score'].max():.2f}")
        if 'minervini_score' in df.columns:
            logger.info(f"Minervini scores: min={df['minervini_score'].min():.2f}, max={df['minervini_score'].max():.2f}")
    
    return analysis_time < 30 and len(df) > 0

def test_analysis_summary():
    """Test the analysis summary function"""
    logger = logging.getLogger(__name__)
    logger.info("Testing analysis summary...")
    
    # Get latest date
    results_db_path = os.path.join(project_root, "data", "analysis_results.db")
    
    import sqlite3
    with sqlite3.connect(results_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(Date) FROM hl_ratio")
        latest_date = cursor.fetchone()[0]
    
    start_time = time.time()
    summary = create_analysis_summary(latest_date)
    summary_time = time.time() - start_time
    
    logger.info(f"Analysis summary completed in {summary_time:.2f} seconds")
    
    if summary:
        logger.info("Summary statistics:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
    
    return summary_time < 10 and len(summary) > 0

def test_database_coverage():
    """Test the database coverage function"""
    logger = logging.getLogger(__name__)
    logger.info("Testing database coverage check...")
    
    start_time = time.time()
    coverage = check_database_coverage()
    coverage_time = time.time() - start_time
    
    logger.info(f"Database coverage check completed in {coverage_time:.2f} seconds")
    
    if coverage:
        logger.info("Coverage statistics:")
        for key, value in coverage.items():
            logger.info(f"  {key}: {value}")
    
    return coverage_time < 5 and len(coverage) > 0

def main():
    """Main test function"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Testing OPTIMIZED integrated analysis functions")
    logger.info("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        # Test 1: Comprehensive analysis
        if test_comprehensive_analysis():
            logger.info("✅ Comprehensive analysis test PASSED")
            tests_passed += 1
        else:
            logger.error("❌ Comprehensive analysis test FAILED")
        
        # Test 2: Analysis summary
        if test_analysis_summary():
            logger.info("✅ Analysis summary test PASSED")
            tests_passed += 1
        else:
            logger.error("❌ Analysis summary test FAILED")
        
        # Test 3: Database coverage
        if test_database_coverage():
            logger.info("✅ Database coverage test PASSED")
            tests_passed += 1
        else:
            logger.error("❌ Database coverage test FAILED")
        
        logger.info("=" * 60)
        logger.info("INTEGRATED ANALYSIS OPTIMIZATION TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Tests passed: {tests_passed}/{total_tests}")
        
        if tests_passed == total_tests:
            logger.info("🎉 ALL TESTS PASSED! Integrated analysis optimization is successful.")
            return True
        else:
            logger.warning("⚠️  Some tests failed. Please review the optimization.")
            return False
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)