import pytest
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os
import sys

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from backend.analysis.high_low_ratio import (
    init_hl_ratio_db,
    calc_hl_ratio_for_all,
    calc_hl_ratio
)

@pytest.fixture
def setup_source_db(tmp_path):
    """Provides a temporary database file path with a populated 'daily_quotes' table."""
    db_path = tmp_path / "test_source.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE daily_quotes (
            Date TEXT, Code TEXT, High REAL, Low REAL, AdjustmentClose REAL
        )
    """)
    
    end_date = datetime(2023, 12, 31)
    for i in range(280):
        date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
        price = 100 + i
        cursor.execute(
            "INSERT INTO daily_quotes VALUES (?, '1301', ?, ?, ?)",
            (date, price + 5, price - 5, price)
        )
    conn.commit()
    conn.close()
    
    yield db_path

def test_init_hl_ratio_db(tmp_path):
    """Tests that the hl_ratio table is created correctly in a file database."""
    db_path = tmp_path / "test_dest.db"
    init_hl_ratio_db(db_path=str(db_path))
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='hl_ratio'")
    assert cursor.fetchone() is not None, "The 'hl_ratio' table was not created."
    conn.close()

def test_calc_hl_ratio_for_all(setup_source_db):
    """
    Tests that calc_hl_ratio_for_all correctly calculates the ratio
    and returns a DataFrame, using a file-based database.
    """
    db_path = setup_source_db
    end_date = '2023-12-31'
    weeks = 52

    results_df = calc_hl_ratio_for_all(db_path=str(db_path), end_date=end_date, weeks=weeks)

    assert isinstance(results_df, pd.DataFrame)
    assert not results_df.empty
    row = results_df[results_df['Code'] == '1301'].iloc[0]
    assert 0.0 <= row['HlRatio'] <= 100.0

def test_calc_hl_ratio():
    """Unit test for the core ratio calculation logic with a more realistic assertion."""
    end_date = datetime(2023, 12, 31)
    dates = [end_date - timedelta(days=i) for i in range(260)]
    prices = [100 + i for i in range(260)]
    price_df = pd.DataFrame({
        'High': [p + 5 for p in prices],
        'Low': [p - 5 for p in prices],
        'AdjustmentClose': prices
    }, index=dates)

    ratio = calc_hl_ratio(price_df, weeks=52)
    
    # The highest price is 359+5=364, lowest is 100-5=95. Current is 359.
    # Ratio = (359 - 95) / (364 - 95) * 100 = 98.14...
    assert 98.0 < ratio < 99.0, "Calculation for increasing price is not within expected range."