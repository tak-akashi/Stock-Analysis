#!/usr/bin/env python3
"""
Test script for the optimized RSI update function
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from backend.analysis.relative_strength import update_rsi_db

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'test_rsi_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Test the optimized RSI update function"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Testing OPTIMIZED RSI update function")
    logger.info("=" * 60)
    
    # Database path
    results_db_path = os.path.join(project_root, "data", "analysis_results.db")
    
    if not os.path.exists(results_db_path):
        logger.error(f"Database not found: {results_db_path}")
        return False
    
    try:
        # Test with last 5 dates
        logger.info("Starting RSI update test...")
        start_time = time.time()
        
        # Call the optimized function
        errors = update_rsi_db(
            result_db_path=results_db_path,
            date_list=None,  # Auto-detect dates
            period=-5
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("RSI UPDATE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"Errors: {errors}")
        
        if errors == 0:
            logger.info("✅ RSI update test PASSED!")
            logger.info(f"Performance: {elapsed_time:.2f} seconds (target: < 60 seconds)")
            return True
        else:
            logger.error(f"❌ RSI update test FAILED with {errors} errors")
            return False
            
    except Exception as e:
        logger.error(f"Error during RSI test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)