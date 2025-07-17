#!/usr/bin/env python3
"""
Test script for the optimized Type8 update function
"""

import os
import sys
import time
import logging
import sqlite3
from datetime import datetime, timedelta

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from backend.analysis.minervini import MinerviniConfig, MinerviniDatabase

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'test_type8_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Test the optimized Type8 update function"""
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Testing OPTIMIZED Type8 update function")
    logger.info("=" * 60)
    
    # Database path
    results_db_path = os.path.join(project_root, "data", "analysis_results.db")
    
    if not os.path.exists(results_db_path):
        logger.error(f"Database not found: {results_db_path}")
        return False
    
    try:
        # Get the latest date from the database
        with sqlite3.connect(results_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(Date) FROM relative_strength")
            latest_date = cursor.fetchone()[0]
        
        if not latest_date:
            logger.error("No data found in relative_strength table")
            return False
        
        # Test with just the latest date
        test_date_list = [latest_date]
        
        logger.info(f"Testing Type8 update for date: {latest_date}")
        start_time = time.time()
        
        # Initialize config and database processor
        config = MinerviniConfig()
        database = MinerviniDatabase(config)
        
        # Call the optimized function
        database.update_type8(
            dest_db_path=results_db_path,
            date_list=test_date_list,
            period=-1  # Just the latest date
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("TYPE8 UPDATE TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
        
        # Verify the update worked
        with sqlite3.connect(results_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM minervini 
                WHERE Date = ? AND Type_8 IS NOT NULL
            """, (latest_date,))
            updated_count = cursor.fetchone()[0]
        
        logger.info(f"Records with Type_8 values: {updated_count}")
        
        if elapsed_time < 60 and updated_count > 0:
            logger.info("✅ Type8 update test PASSED!")
            logger.info(f"Performance: {elapsed_time:.2f} seconds (target: < 60 seconds)")
            return True
        else:
            logger.error(f"❌ Type8 update test FAILED")
            logger.error(f"Time: {elapsed_time:.2f}s, Records: {updated_count}")
            return False
            
    except Exception as e:
        logger.error(f"Error during Type8 test: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)