"""
Pytest tests for chart_classification.py
"""

import os
import sqlite3
import sys
from unittest.mock import MagicMock, patch, call
import argparse # Added for test_main_argparse

import numpy as np
import pandas as pd
import pytest

# Add the backend directory to the path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend/analysis')))

# Now import the module
from chart_classification import ChartClassifier, get_all_tickers, init_results_db, save_result_to_db, main_sample, main_full_run, main # Import main

# --- Fixtures ---

@pytest.fixture
def mock_db_connections(mocker):
    """Mocks all database interactions (read and write)."""
    # Mock for sqlite3.connect
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.execute.return_value = None # Ensure execute returns something
    mock_conn.commit.return_value = None # Mock commit as well

    # Explicitly set __enter__ and __exit__ for the context manager
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = False # Indicate no exception was handled

    mocker.patch('sqlite3.connect', return_value=mock_conn)

    # Mock for pandas.read_sql_query
    mock_df_stock = pd.DataFrame({
        'Date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=300)),
        'AdjustmentClose': np.linspace(100, 150, 300)
    })
    mock_df_master = pd.DataFrame({'Code': ["101", "102", "103"]})

    def mock_read_sql_query(query, conn, params=None, parse_dates=None):
        if "FROM daily_quotes" in query:
            return mock_df_stock
        elif "FROM master" in query:
            return mock_df_master
        return pd.DataFrame() # Default empty dataframe

    mocker.patch('pandas.read_sql_query', side_effect=mock_read_sql_query)

    return mock_conn

@pytest.fixture
def classifier_instance(mock_db_connections):
    """Returns a standard ChartClassifier instance for testing."""
    # The mock_db_connections fixture already sets up the necessary mocks
    return ChartClassifier(ticker="99999", window=30)

# --- Test Cases ---

def test_chart_classifier_initialization(classifier_instance):
    """Test if the ChartClassifier initializes correctly."""
    assert classifier_instance.ticker == "99999"
    assert classifier_instance.window == 30
    assert len(classifier_instance.price_data) > 0
    assert len(classifier_instance.templates_manual) == 9
    assert '上昇' in classifier_instance.templates_manual

def test_initialization_not_enough_data(mock_db_connections):
    """Test that initialization raises ValueError if there is not enough data."""
    # Override the mock_df_stock for this specific test
    # We need to patch pandas.read_sql_query specifically for this test
    with patch('pandas.read_sql_query', return_value=pd.DataFrame({
        'Date': pd.to_datetime(pd.date_range(start='2024-01-01', periods=10)),
        'AdjustmentClose': np.linspace(100, 150, 10)
    })):
        with pytest.raises(ValueError, match="Not enough data for ticker"):
            ChartClassifier(ticker="12345", window=20)

def test_normalize():
    """Test the static _normalize method."""
    arr = np.array([10, 20, 30, 40, 50])
    normalized = ChartClassifier._normalize(arr)
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)
    assert np.allclose(normalized, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

def test_classify_latest(classifier_instance):
    """Test the classification of the latest window of data."""
    # Create price data that perfectly matches the '上昇' pattern
    perfect_rise = np.linspace(100, 200, 30)
    classifier_instance.price_data = pd.Series(perfect_rise)
    
    label, score = classifier_instance.classify_latest()
    
    assert label == "上昇"
    assert np.isclose(score, 1.0)

def test_save_classification_plot(mocker, classifier_instance):
    """Test that the plot saving function calls the correct file system and plotting methods."""
    mock_makedirs = mocker.patch('os.makedirs')
    mock_savefig = mocker.patch('matplotlib.pyplot.savefig')
    mocker.patch('matplotlib.pyplot.close') # Don't need to test this, just mock it

    output_dir = "/tmp/test_output"
    classifier_instance.save_classification_plot("上昇", 0.95, output_dir)

    mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
    expected_path = os.path.join(output_dir, "99999_window30_上昇.png")
    mock_savefig.assert_called_once_with(expected_path)


# --- Test Database and Main Functions ---

def test_get_all_tickers(mock_db_connections):
    """Test fetching all tickers from the master DB."""
    # mock_db_connections already sets up pandas.read_sql_query for master DB
    tickers = get_all_tickers("dummy/master.db")
    assert tickers == ["101", "102", "103"]

def test_init_results_db(mock_db_connections):
    """Test the initialization of the results database."""
    init_results_db("dummy/results.db")
    cursor = mock_db_connections.cursor()
    # Check that the CREATE TABLE query was executed
    cursor.execute.assert_called_once()
    assert "CREATE TABLE IF NOT EXISTS" in cursor.execute.call_args[0][0]

def test_save_result_to_db(mock_db_connections):
    """Test saving a single result to the database."""
    save_result_to_db("dummy/results.db", "2024-07-11", "12345", 60, "調整", 0.88)
    cursor = mock_db_connections.cursor()
    cursor.execute.assert_called_once()
    sql, params = cursor.execute.call_args[0]
    assert "INSERT OR REPLACE INTO" in sql
    assert params == ("2024-07-11", "12345", 60, "調整", 0.88)

@patch('chart_classification.ChartClassifier')
def test_main_sample(MockClassifier, mock_db_connections):
    """Test the main_sample function to ensure it loops and calls correctly."""
    # Mock the instance methods
    mock_instance = MagicMock()
    mock_instance.classify_latest.return_value = ("上昇", 0.99)
    MockClassifier.return_value = mock_instance

    main_sample()

    # Check if the classifier was instantiated for all tickers and windows
    tickers = ["74530", "99840", "67580"]
    windows = [20, 60, 120, 240]
    assert MockClassifier.call_count == len(tickers) * len(windows)

    # Check if the plot saving method was called for each
    assert mock_instance.save_classification_plot.call_count == len(tickers) * len(windows)

@patch('chart_classification.get_all_tickers', return_value=["111", "222"])
@patch('chart_classification.save_result_to_db')
@patch('chart_classification.ChartClassifier')
def test_main_full_run(MockClassifier, mock_save_db, mock_get_tickers, mock_db_connections):
    """Test the main_full_run function."""
    mock_instance = MagicMock()
    mock_instance.classify_latest.return_value = ("下落", -0.95)
    MockClassifier.return_value = mock_instance

    main_full_run()

    # Check if it tried to get all tickers
    mock_get_tickers.assert_called_once()

    # Check if classifier and db save were called for all tickers and windows
    windows = [20, 60, 120, 240]
    assert MockClassifier.call_count == len(mock_get_tickers.return_value) * len(windows)
    assert mock_save_db.call_count == len(mock_get_tickers.return_value) * len(windows)

@pytest.mark.parametrize(
    "mode, expected_main_func_name",
    [('sample', 'main_sample'), ('full', 'main_full_run')]
)
def test_main_argparse_dispatch(mocker, mode, expected_main_func_name):
    """Test that the correct main function is called based on the --mode arg."""
    # Patch the actual main functions that `main()` will call
    mock_main_sample = mocker.patch('chart_classification.main_sample')
    mock_main_full_run = mocker.patch('chart_classification.main_full_run')

    # Simulate command-line arguments
    mocker.patch('sys.argv', ['script_name', '--mode', mode])

    # Call the top-level main function that parses args and dispatches
    main() # This is the actual main() from chart_classification.py

    if expected_main_func_name == 'main_sample':
        mock_main_sample.assert_called_once()
        mock_main_full_run.assert_not_called()
    else:
        mock_main_full_run.assert_called_once()
        mock_main_sample.assert_not_called()
