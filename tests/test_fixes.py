#!/usr/bin/env python3
"""Test script to verify the fixes"""

import sqlite3
import sys
import os

# Add backend to path
sys.path.append('/Users/tak/Markets/Stocks/Stock-Analysis/backend')

from analysis.minervini import MinerviniConfig, MinerviniAnalyzer, MinerviniDatabase

def test_fixed_functions():
    """Test the fixed update_type8_by_date function"""
    db_path = "/Users/tak/Markets/Stocks/Stock-Analysis/data/analysis_results.db"
    
    try:
        # Setup
        config = MinerviniConfig()
        analyzer = MinerviniAnalyzer(config)
        database = MinerviniDatabase(config, analyzer)
        
        with sqlite3.connect(db_path) as conn:
            # Test with a date that has RSI data
            test_date = '2025-07-10'
            
            print(f"Testing improved update_type8_by_date for {test_date}")
            
            # Check before state
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM relative_strength WHERE Date = ? AND RelativeStrengthIndex IS NOT NULL", (test_date,))
            rsi_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM minervini WHERE Date LIKE ? AND Type_8 IS NOT NULL", (test_date + '%',))
            type8_before = cursor.fetchone()[0]
            
            print(f"Before: RSI records = {rsi_count}, Type_8 records = {type8_before}")
            
            # Run the fixed function
            errors = database.update_type8_by_date(conn, test_date)
            print(f"Function completed with {len(errors)} errors")
            
            # Check after state
            cursor.execute("SELECT COUNT(*) FROM minervini WHERE Date LIKE ? AND Type_8 IS NOT NULL", (test_date + '%',))
            type8_after = cursor.fetchone()[0]
            
            print(f"After: Type_8 records = {type8_after}")
            print(f"Improvement: +{type8_after - type8_before} records")
            
            # Sample verification
            cursor.execute("""
                SELECT rs.Code, rs.RelativeStrengthIndex, m.Type_8 
                FROM relative_strength rs 
                JOIN minervini m ON rs.Code = m.Code AND (rs.Date = substr(m.Date, 1, 10))
                WHERE rs.Date = ? AND rs.RelativeStrengthIndex IS NOT NULL 
                AND m.Type_8 IS NOT NULL
                LIMIT 5
            """, (test_date,))
            
            print("\nSample successful updates:")
            for row in cursor.fetchall():
                code, rsi, type8 = row
                print(f"  Code: {code}, RSI: {rsi:.1f}, Type_8: {type8}")
            
            if type8_after > type8_before:
                print(f"\n✓ Fix SUCCESSFUL: Improved Type_8 coverage")
            else:
                print(f"\n! Fix needs more work: No improvement seen")
                
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_fixed_functions()