#!/usr/bin/env python3
"""
Test script to verify update_rsi_db and update_type8_by_date functions
"""

import sys
import os
import sqlite3
from datetime import datetime, timedelta

# Add backend to path
sys.path.append('/Users/tak/Markets/Stocks/Stock-Analysis/backend')

from analysis.relative_strength import update_rsi_db
from analysis.minervini import update_type8_db_by_date

def check_database_status():
    """Check current database status"""
    db_path = "/Users/tak/Markets/Stocks/Stock-Analysis/data/analysis_results.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            # Check relative_strength table
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total, COUNT(RelativeStrengthIndex) as with_rsi FROM relative_strength")
            rs_total, rs_with_rsi = cursor.fetchone()
            
            # Check minervini table
            cursor.execute("SELECT COUNT(*) as total, COUNT(Type_8) as with_type8 FROM minervini")
            min_total, min_with_type8 = cursor.fetchone()
            
            # Get recent dates
            cursor.execute("SELECT DISTINCT Date FROM relative_strength ORDER BY Date DESC LIMIT 5")
            rs_dates = [row[0] for row in cursor.fetchall()]
            
            cursor.execute("SELECT DISTINCT Date FROM minervini ORDER BY Date DESC LIMIT 5")
            min_dates = [row[0] for row in cursor.fetchall()]
            
            print("=== Database Status ===")
            print(f"Relative Strength: {rs_total} total rows, {rs_with_rsi} with RSI ({rs_with_rsi/rs_total*100:.1f}%)")
            print(f"Minervini: {min_total} total rows, {min_with_type8} with Type_8 ({min_with_type8/min_total*100:.1f}% if total > 0)")
            print(f"Recent RS dates: {rs_dates}")
            print(f"Recent Minervini dates: {min_dates}")
            
            return rs_dates, min_dates, rs_with_rsi, min_with_type8
            
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None, None, 0, 0

def test_update_rsi_db():
    """Test the update_rsi_db function"""
    print("\n=== Testing update_rsi_db function ===")
    
    try:
        # Test with recent dates only (last 3 dates)
        errors = update_rsi_db(period=-3)
        print(f"update_rsi_db completed with {errors} errors")
        return errors == 0
        
    except Exception as e:
        print(f"Error in update_rsi_db: {e}")
        return False

def test_update_type8_by_date(test_date):
    """Test the update_type8_by_date function"""
    print(f"\n=== Testing update_type8_by_date function for {test_date} ===")
    
    try:
        db_path = "/Users/tak/Markets/Stocks/Stock-Analysis/data/analysis_results.db"
        with sqlite3.connect(db_path) as conn:
            errors = update_type8_db_by_date(conn, test_date)
            print(f"update_type8_by_date completed with {len(errors)} errors")
            if errors:
                print(f"First few errors: {errors[:3]}")
            return len(errors) == 0
            
    except Exception as e:
        print(f"Error in update_type8_by_date: {e}")
        return False

def main():
    print("Testing update functions...")
    
    # Check initial status
    rs_dates, min_dates, initial_rsi, initial_type8 = check_database_status()
    
    if not rs_dates:
        print("No data found in database or database locked")
        return
    
    # Test update_rsi_db
    rsi_success = test_update_rsi_db()
    
    # Check status after RSI update
    print("\n=== Status after RSI update ===")
    _, _, after_rsi, _ = check_database_status()
    print(f"RSI count changed from {initial_rsi} to {after_rsi}")
    
    # Test update_type8_by_date with the most recent date
    if rs_dates and rsi_success:
        test_date = rs_dates[0]  # Most recent date
        type8_success = test_update_type8_by_date(test_date)
        
        # Check final status
        print("\n=== Final Status ===")
        _, _, final_rsi, final_type8 = check_database_status()
        print(f"Type_8 count changed from {initial_type8} to {final_type8}")
        
        print(f"\n=== Summary ===")
        print(f"RSI update: {'SUCCESS' if rsi_success else 'FAILED'}")
        print(f"Type_8 update: {'SUCCESS' if type8_success else 'FAILED'}")
    else:
        print("Skipping Type_8 test due to RSI update failure or no dates")

if __name__ == "__main__":
    main()