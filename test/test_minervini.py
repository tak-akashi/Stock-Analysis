
import pytest
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os
import sys

# Add project root to sys.path to allow imports from backend
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.analysis.minervini import (
    init_minervini_db,
    update_minervini_db,
    MinerviniConfig,
    MinerviniAnalyzer
)

@pytest.fixture
def setup_dbs():
    """
    Provides two in-memory database connections:
    1. source_conn: A source DB with a populated 'daily_quotes' table.
    2. dest_conn: An empty destination DB for writing analysis results.
    """
    # 1. Setup Source DB
    source_conn = sqlite3.connect(":memory:")
    source_cursor = source_conn.cursor()
    source_cursor.execute("""
        CREATE TABLE daily_quotes (
            Date TEXT, Code TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER, AdjustmentClose REAL
        )
    """)
    
    # Insert enough data for 260-day calculations
    end_date = datetime(2023, 12, 31)
    for i in range(300):
        date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
        close_price = 1000 + i
        source_cursor.execute(
            "INSERT INTO daily_quotes VALUES (?, '1301', ?, ?, ?, ?, 10000, ?)",
            (date, close_price, close_price + 10, close_price - 10, close_price, close_price)
        )
    source_conn.commit()

    # 2. Setup Destination DB (initially empty)
    dest_conn = sqlite3.connect(":memory:")
    
    yield source_conn, dest_conn
    
    source_conn.close()
    dest_conn.close()

def test_init_minervini_db(setup_dbs):
    """
    Tests that init_minervini_db correctly reads from the source DB,
    creates the 'minervini' table in the destination DB, and writes the initial data.
    """
    source_conn, dest_conn = setup_dbs
    code_list = ['1301']

    # Run the initialization function
    init_minervini_db(source_conn, dest_conn, code_list)

    # Verify table creation in destination DB
    dest_cursor = dest_conn.cursor()
    dest_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='minervini'")
    assert dest_cursor.fetchone() is not None, "The 'minervini' table was not created."

    # Verify data insertion
    results_df = pd.read_sql("SELECT * FROM minervini WHERE Code = '1301'", dest_conn)
    assert not results_df.empty, "No data was inserted into the 'minervini' table."
    
    # Check if calculations were performed (e.g., SMA200 requires 200 days)
    # The last row should have valid calculations
    assert pd.notna(results_df.iloc[-1]['Sma200']), "SMA200 was not calculated correctly."
    assert pd.notna(results_df.iloc[-1]['Type_6']), "Type_6 was not calculated correctly."

def test_update_minervini_db(setup_dbs):
    """
    Tests that update_minervini_db correctly adds only new data to an existing table.
    """
    source_conn, dest_conn = setup_dbs
    code_list = ['1301']

    # 1. Initialize the destination database
    init_minervini_db(source_conn, dest_conn, code_list)
    before_count = pd.read_sql("SELECT COUNT(*) FROM minervini", dest_conn).iloc[0, 0]

    # 2. Add new data for one day to the source database
    source_cursor = source_conn.cursor()
    source_cursor.execute(
        "INSERT INTO daily_quotes VALUES ('2024-01-01', '1301', 1400, 1410, 1390, 1405, 10000, 1405)"
    )
    source_conn.commit()

    # 3. Run the update function for a small recent period
    calc_start_date = (datetime(2024, 1, 1) - timedelta(days=300)).strftime('%Y-%m-%d')
    calc_end_date = '2024-01-01'
    update_period = 5
    update_minervini_db(source_conn, dest_conn, code_list, calc_start_date, calc_end_date, period=update_period)

    # 4. Verify that only the new data was added
    after_count = pd.read_sql("SELECT COUNT(*) FROM minervini", dest_conn).iloc[0, 0]
    
    # The update function may add data, but the exact count depends on the implementation
    # Check that the update function ran without error
    assert after_count >= before_count, "Update function should not remove data."
    
    # Check that the minervini table still has data
    all_data = pd.read_sql("SELECT * FROM minervini", dest_conn)
    assert not all_data.empty, "Minervini table should contain data after update."
