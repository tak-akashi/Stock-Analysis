
import pytest
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from unittest.mock import patch

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import the main function to be tested
from scripts.run_daily_analysis import run_daily_analysis
from backend.analysis.minervini import MinerviniConfig

@pytest.fixture
def setup_test_environment(tmp_path):
    """
    Sets up a temporary directory structure and mock databases for a full integration test.
    This fixture simulates the real project structure and data flow.
    """
    # 1. Create temporary directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # 2. Define mock database paths
    jquants_db_path = data_dir / "jquants.db"
    results_db_path = data_dir / "analysis_results.db"

    # 3. Populate the mock jquants.db (source)
    source_conn = sqlite3.connect(jquants_db_path)
    source_cursor = source_conn.cursor()
    source_cursor.execute("""
        CREATE TABLE daily_quotes (
            Date TEXT, Code TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER, AdjustmentClose REAL
        )
    """)
    end_date = datetime(2023, 12, 31)
    for i in range(300):
        date = (end_date - timedelta(days=i)).strftime('%Y-%m-%d')
        price = 1000 + i
        source_cursor.execute(
            "INSERT INTO daily_quotes VALUES (?, '9999', ?, ?, ?, ?, 10000, ?)",
            (date, price, price + 10, price - 10, price, price)
        )
    source_conn.commit()
    source_conn.close()

    # 4. Patch MinerviniConfig to use the temporary paths
    with patch('backend.analysis.minervini.MinerviniConfig') as mock_config:
        mock_instance = mock_config.return_value
        mock_instance.base_dir = tmp_path
        mock_instance.data_dir = data_dir
        mock_instance.logs_dir = logs_dir
        mock_instance.output_dir = output_dir
        mock_instance.jquants_db_path = jquants_db_path
        mock_instance.results_db_path = results_db_path
        
        yield # The test runs here


def test_run_daily_analysis_full_flow(setup_test_environment):
    """
    Tests the entire daily analysis workflow from start to finish.
    Scenario:
    1. Run the analysis for the first time, verifying that all tables are created and populated.
    2. Add new data to the source DB.
    3. Run the analysis again, verifying that the results are updated correctly.
    """
    # --- 1. First Run (Initialization) ---
    with patch('scripts.run_daily_analysis.datetime') as mock_datetime:
        # Mock the date to a fixed point for reproducible results
        mock_datetime.now.return_value = datetime(2023, 12, 31)
        
        success = run_daily_analysis()
        assert success, "run_daily_analysis reported a failure on the first run."

    # Verify the results of the first run
    config = MinerviniConfig() # This will be the patched config
    results_conn = sqlite3.connect(config.results_db_path)
    
    # Check that all tables were created
    for table_name in ['relative_strength', 'minervini', 'hl_ratio']:
        df = pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'", results_conn)
        assert not df.empty, f"Table '{table_name}' was not created in the results database."

    # Check that data was populated
    minervini_df = pd.read_sql("SELECT * FROM minervini", results_conn)
    assert not minervini_df.empty
    initial_minervini_count = len(minervini_df)

    hl_ratio_df = pd.read_sql("SELECT * FROM hl_ratio", results_conn)
    assert not hl_ratio_df.empty

    results_conn.close()

    # --- 2. Second Run (Update) ---
    
    # Add new data to the source jquants.db
    source_conn = sqlite3.connect(config.jquants_db_path)
    source_cursor = source_conn.cursor()
    source_cursor.execute(
        "INSERT INTO daily_quotes VALUES ('2024-01-01', '9999', 1400, 1410, 1390, 1405, 10000, 1405)"
    )
    source_conn.commit()
    source_conn.close()

    # Run the analysis again for the new date
    with patch('scripts.run_daily_analysis.datetime') as mock_datetime:
        mock_datetime.now.return_value = datetime(2024, 1, 1)
        
        success = run_daily_analysis()
        assert success, "run_daily_analysis reported a failure on the second run."

    # Verify the results of the second run
    results_conn = sqlite3.connect(config.results_db_path)
    
    # Minervini should have new rows added
    updated_minervini_df = pd.read_sql("SELECT * FROM minervini", results_conn)
    assert len(updated_minervini_df) > initial_minervini_count, "Minervini table was not updated."

    # hl_ratio table should be replaced with new data for the new date
    updated_hl_ratio_df = pd.read_sql("SELECT * FROM hl_ratio", results_conn)
    assert not updated_hl_ratio_df.empty
    assert updated_hl_ratio_df.iloc[0]['Date'] == '2024-01-01', "hl_ratio table was not replaced with new date's data."

    results_conn.close()
