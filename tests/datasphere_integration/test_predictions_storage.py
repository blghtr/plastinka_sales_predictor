import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

from deployment.app.db.database import get_db_connection, execute_query
from deployment.app.db.schema import init_db, SCHEMA_SQL
from deployment.app.services.datasphere_service import save_predictions_to_db

# Sample data that mimics what get_predictions_df would produce
# Using string representations for quantile column names
SAMPLE_PREDICTIONS = {
    'barcode': ['123456789012', '123456789012', '987654321098', '987654321098', '555555555555'],
    'artist': ['Artist A', 'Artist A', 'Artist B', 'Artist B', 'Artist C'],
    'album': ['Album X', 'Album X', 'Album Y', 'Album Y', 'Album Z'],
    'cover_type': ['Standard', 'Standard', 'Deluxe', 'Deluxe', 'Limited'],
    'price_category': ['A', 'A', 'B', 'B', 'C'],
    'release_type': ['Studio', 'Studio', 'Live', 'Live', 'Compilation'],
    'recording_decade': ['2010s', '2010s', '2000s', '2000s', '1990s'],
    'release_decade': ['2020s', '2020s', '2010s', '2010s', '2000s'],
    'style': ['Rock', 'Rock', 'Pop', 'Pop', 'Jazz'],
    'record_year': [2015, 2015, 2007, 2007, 1995],
    '0.05': [10.5, 12.3, 5.2, 7.8, 3.1],  # String keys for quantiles
    '0.25': [15.2, 18.7, 8.9, 11.3, 5.7],
    '0.5': [21.4, 24.8, 12.6, 15.9, 7.5],
    '0.75': [28.3, 32.1, 17.8, 20.4, 10.2],
    '0.95': [35.7, 40.2, 23.1, 27.5, 15.8]
}

@pytest.fixture
def db_setup():
    # Create a temporary directory and database
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, 'test.db')
    
    # Initialize the database schema
    init_db(db_path)
    
    # Verify all necessary tables exist
    debug_conn = sqlite3.connect(db_path)
    debug_cursor = debug_conn.cursor()
    debug_cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('dim_multiindex_mapping', 'fact_predictions', 'prediction_results', 'models', 'jobs')")
    table_count = debug_cursor.fetchone()[0]
    debug_conn.close()
    
    # Ensure all required tables are created
    assert table_count == 5, "Database should have all required tables"
    
    # Create a temporary predictions CSV file
    predictions_path = os.path.join(temp_dir.name, 'predictions.csv')
    
    # Create the DataFrame with string column names for quantiles
    df = pd.DataFrame(SAMPLE_PREDICTIONS)
    # Save to CSV
    df.to_csv(predictions_path, index=False)
    
    # Mock job_id and model_id
    job_id = "test_job_123"
    model_id = "test_model_456"
    
    # Override the database path used by get_db_connection
    original_db_path = os.environ.get('DATABASE_PATH', None)
    os.environ['DATABASE_PATH'] = db_path
    
    # Create a connection to the test database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Insert test job
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "prediction", "completed", datetime.now().isoformat(), datetime.now().isoformat())
    )
    
    # Insert test model
    cursor.execute(
        "INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)",
        (model_id, job_id, "/test/path/model.onnx", datetime.now().isoformat())
    )
    
    conn.commit()
    conn.close()
    
    # Return setup data as a dictionary
    yield {
        "temp_dir": temp_dir,
        "db_path": db_path,
        "predictions_path": predictions_path,
        "job_id": job_id,
        "model_id": model_id,
        "original_db_path": original_db_path
    }
    
    # Teardown - restore original environment variable
    if original_db_path is not None:
        os.environ['DATABASE_PATH'] = original_db_path
    else:
        os.environ.pop('DATABASE_PATH', None)
        
    # Make sure to close any possible open connections to the DB before cleanup
    try:
        conn = sqlite3.connect(db_path)
        conn.close()
    except Exception:
        pass
        
    # Clean up temporary directory and its contents
    try:
        temp_dir.cleanup()
    except PermissionError:
        # On Windows, we might not be able to delete the file immediately
        # Just continue
        pass

@patch('deployment.app.db.database.get_db_connection')
def test_save_predictions_to_db(mock_get_db, db_setup):
    """Test saving predictions from CSV to database"""
    # Set up the mock to use our test database
    conn = sqlite3.connect(db_setup["db_path"])
    conn.row_factory = sqlite3.Row
    
    # Verify database connection
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fact_predictions'")
    fact_predictions_exists = cur.fetchone()
    assert fact_predictions_exists is not None, "Table fact_predictions should exist"
    
    # Configure mock
    mock_get_db.return_value = conn
    
    # Call the function we want to test with direct DB connection
    result = save_predictions_to_db(
        predictions_path=db_setup["predictions_path"],
        job_id=db_setup["job_id"],
        model_id=db_setup["model_id"],
        direct_db_connection=conn
    )
    
    # Verify the return value contains a valid result_id
    assert result is not None
    assert "result_id" in result
    assert "predictions_count" in result
    assert result["predictions_count"] == len(SAMPLE_PREDICTIONS["barcode"])
    
    # Connect to the database and verify the data was saved correctly
    cursor = conn.cursor()
    
    # Check the prediction_results table
    cursor.execute("SELECT * FROM prediction_results WHERE job_id = ?", (db_setup["job_id"],))
    prediction_result = cursor.fetchone()
    
    assert prediction_result is not None, "A prediction_result record should be created"
    assert prediction_result["model_id"] == db_setup["model_id"]
    assert prediction_result["result_id"] == result["result_id"]
    
    # Check that the predictions table exists and has the right data
    cursor.execute("SELECT COUNT(*) as count FROM fact_predictions")
    count = cursor.fetchone()["count"]
    assert count == len(SAMPLE_PREDICTIONS["barcode"])
    
    # Check specific records match
    cursor.execute("""
        SELECT p.*, m.barcode, m.artist, m.album FROM fact_predictions p
        JOIN dim_multiindex_mapping m ON p.multiindex_id = m.multiindex_id
        WHERE m.barcode = ? AND m.artist = ? AND m.album = ?
    """, (SAMPLE_PREDICTIONS["barcode"][0], SAMPLE_PREDICTIONS["artist"][0], SAMPLE_PREDICTIONS["album"][0]))
    record = cursor.fetchone()
    
    assert record is not None
    assert pytest.approx(float(record["quantile_05"]), 0.01) == SAMPLE_PREDICTIONS['0.05'][0]
    assert pytest.approx(float(record["quantile_25"]), 0.01) == SAMPLE_PREDICTIONS['0.25'][0]
    assert pytest.approx(float(record["quantile_50"]), 0.01) == SAMPLE_PREDICTIONS['0.5'][0]
    assert pytest.approx(float(record["quantile_75"]), 0.01) == SAMPLE_PREDICTIONS['0.75'][0]
    assert pytest.approx(float(record["quantile_95"]), 0.01) == SAMPLE_PREDICTIONS['0.95'][0]
    
    conn.close()
    
@patch('deployment.app.db.database.get_db_connection')
def test_save_predictions_to_db_invalid_path(mock_get_db, db_setup):
    """Test handling of invalid prediction file path"""
    # Set up the mock to use our test database
    conn = sqlite3.connect(db_setup["db_path"])
    conn.row_factory = sqlite3.Row
    mock_get_db.return_value = conn
    
    with pytest.raises(FileNotFoundError):
        save_predictions_to_db(
            predictions_path="/nonexistent/path/predictions.csv",
            job_id=db_setup["job_id"],
            model_id=db_setup["model_id"],
            direct_db_connection=conn
        )
    conn.close()
        
@patch('deployment.app.db.database.get_db_connection')
def test_save_predictions_to_db_invalid_format(mock_get_db, db_setup):
    """Test handling of invalid prediction file format"""
    # Set up the mock to use our test database
    conn = sqlite3.connect(db_setup["db_path"])
    conn.row_factory = sqlite3.Row
    mock_get_db.return_value = conn
    
    # Create a file with invalid format
    invalid_path = os.path.join(db_setup["temp_dir"].name, 'invalid.csv')
    with open(invalid_path, 'w') as f:
        f.write("This is not a valid CSV file")
        
    with pytest.raises(ValueError):
        save_predictions_to_db(
            predictions_path=invalid_path,
            job_id=db_setup["job_id"],
            model_id=db_setup["model_id"],
            direct_db_connection=conn
        )
    conn.close()

@patch('deployment.app.db.database.get_db_connection')
def test_save_predictions_to_db_missing_columns(mock_get_db, db_setup):
    """Test handling of prediction file with missing required columns"""
    # Set up the mock to use our test database
    conn = sqlite3.connect(db_setup["db_path"])
    conn.row_factory = sqlite3.Row
    mock_get_db.return_value = conn
    
    # Create a CSV with missing quantile columns
    # Start with a copy of our sample data
    missing_data = SAMPLE_PREDICTIONS.copy()
    # Remove some columns
    del missing_data['0.05']
    del missing_data['0.25']
    
    # Create and save the DataFrame
    missing_cols_df = pd.DataFrame(missing_data)
    missing_cols_path = os.path.join(db_setup["temp_dir"].name, 'missing_cols.csv')
    missing_cols_df.to_csv(missing_cols_path, index=False)
    
    with pytest.raises(ValueError):
        save_predictions_to_db(
            predictions_path=missing_cols_path,
            job_id=db_setup["job_id"],
            model_id=db_setup["model_id"],
            direct_db_connection=conn
        )
    conn.close() 