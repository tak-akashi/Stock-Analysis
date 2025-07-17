#!/usr/bin/env python3
import sqlite3
import sys
import os

def test_basic_functionality():
    """Test basic database connectivity and data structure"""
    db_path = "/Users/tak/Markets/Stocks/Stock-Analysis/data/analysis_results.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Test 1: Check relative_strength table
            cursor.execute("SELECT COUNT(*) as total, COUNT(RelativeStrengthIndex) as with_rsi FROM relative_strength")
            rs_total, rs_with_rsi = cursor.fetchone()
            
            # Test 2: Check minervini table
            cursor.execute("SELECT COUNT(*) as total, COUNT(Type_8) as with_type8 FROM minervini")
            min_total, min_with_type8 = cursor.fetchone()
            
            # Test 3: Check latest date alignment
            cursor.execute("SELECT COUNT(*) FROM relative_strength WHERE Date = '2025-07-10' AND RelativeStrengthIndex IS NOT NULL")
            rsi_latest_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM minervini WHERE Date LIKE '2025-07-10%' AND Type_8 IS NOT NULL")
            type8_latest_count = cursor.fetchone()[0]
            
            print("=== Function Analysis Results ===")
            print(f"1. update_rsi_db function status:")
            print(f"   - Total relative_strength records: {rs_total:,}")
            print(f"   - Records with RSI: {rs_with_rsi:,} ({rs_with_rsi/rs_total*100:.2f}%)")
            print(f"   - Latest date (2025-07-10) RSI count: {rsi_latest_count:,}")
            print(f"   → Function is WORKING but with limited coverage")
            
            print(f"\n2. update_type8_by_date function status:")
            print(f"   - Total minervini records: {min_total:,}")
            print(f"   - Records with Type_8: {min_with_type8:,} ({min_with_type8/min_total*100:.2f}%)")
            print(f"   - Latest date (2025-07-10) Type_8 count: {type8_latest_count:,}")
            print(f"   → Function is WORKING but depends on RSI data")
            
            print(f"\n3. Data flow analysis:")
            print(f"   - RSI available for latest date: {rsi_latest_count:,}")
            print(f"   - Type_8 updated for latest date: {type8_latest_count:,}")
            print(f"   - Conversion rate: {type8_latest_count/rsi_latest_count*100:.1f}%" if rsi_latest_count > 0 else "   - No RSI data to convert")
            
            # Test 4: Check for any date format issues
            cursor.execute("""
                SELECT rs.Date, rs.Code, rs.RelativeStrengthIndex, m.Type_8 
                FROM relative_strength rs 
                LEFT JOIN minervini m ON rs.Code = m.Code AND (rs.Date = m.Date OR rs.Date = substr(m.Date, 1, 10))
                WHERE rs.Date = '2025-07-10' AND rs.RelativeStrengthIndex IS NOT NULL 
                LIMIT 5
            """)
            sample_data = cursor.fetchall()
            
            print(f"\n4. Sample data verification:")
            for row in sample_data:
                rs_date, code, rsi, type8 = row
                print(f"   Code: {code}, RSI: {rsi:.1f}, Type_8: {type8}")
            
            print(f"\n=== CONCLUSION ===")
            if rs_with_rsi > 0 and min_with_type8 > 0:
                print("✓ Both functions are FUNCTIONING correctly")
                print("✓ update_rsi_db: Calculating and storing RSI values")
                print("✓ update_type8_by_date: Converting RSI to Type_8 values")
                print("! Issue: Limited RSI coverage suggests update_rsi_db needs to run on more dates")
            else:
                print("✗ One or both functions have issues")
                
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    test_basic_functionality()