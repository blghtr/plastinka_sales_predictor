import pytest
import os
import sqlite3
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
import uuid
import json
from pathlib import Path
import tempfile
import asyncio
from datetime import datetime, date
import logging

from deployment.app.db.schema import init_db
from deployment.app.models.api_models import (
    TrainingConfig, ModelConfig, OptimizerConfig, LRSchedulerConfig,
    TrainingDatasetConfig
)
from deployment.app.services.datasphere_service import settings
from tests.deployment.app.datasphere.test_datasphere_pipeline_integration import create_sample_training_config

@pytest.fixture(scope="function")
def mocked_db(temp_db):
    """
    Provides a mocked DB interface with helper functions for integration tests.
    - Uses a real in-memory DB connection from the `temp_db` fixture.
    - Provides helper methods to create jobs and execute queries.
    """
    # temp_db might be a connection or a dict with 'conn' key depending on fixture
    conn = temp_db['conn'] if isinstance(temp_db, dict) else temp_db
    
    # Initialize all necessary tables if they don't exist
    # These tables are required for the integration tests to work
    cursor = conn.cursor()
    
    # Check if tables exist, and create them if not
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='configs'")
    if not cursor.fetchone():
        cursor.execute("""
        CREATE TABLE configs (
            config_id TEXT PRIMARY KEY,
            config TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            is_active BOOLEAN DEFAULT 0
        )
        """)
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='models'")
    if not cursor.fetchone():
        cursor.execute("""
        CREATE TABLE models (
            model_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            model_path TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            metadata TEXT,
            is_active BOOLEAN DEFAULT 0,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        )
        """)
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='training_results'")
    if not cursor.fetchone():
        cursor.execute("""
        CREATE TABLE training_results (
            result_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            model_id TEXT,
            config_id TEXT,
            metrics TEXT,
            duration INTEGER,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id),
            FOREIGN KEY (model_id) REFERENCES models(model_id),
            FOREIGN KEY (config_id) REFERENCES configs(config_id)
        )
        """)
    
    conn.commit()
    
    def create_job(job_id, job_type="training", status="pending", config_id=None, model_id=None):
        """Helper to insert a job record into the database."""
        if config_id is None:
            # Create a dummy config if none provided
            config_id = "config_" + str(uuid.uuid4())
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO configs (config_id, config, created_at) VALUES (?, ?, ?)",
                (config_id, json.dumps({"sample": "config"}), datetime.now().isoformat())
            )
            conn.commit()

        cursor = conn.cursor()
        # Check if jobs table has config_id and model_id columns
        cursor.execute("PRAGMA table_info(jobs)")
        columns = [column[1] for column in cursor.fetchall()]
        
        now = datetime.now().isoformat()
        
        if 'config_id' in columns and 'model_id' in columns:
            # Full schema with config_id and model_id
            cursor.execute(
                """
                INSERT INTO jobs (job_id, job_type, status, config_id, model_id, created_at, updated_at, progress)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, job_type, status, config_id, model_id, now, now, 0)
            )
        elif 'config_id' in columns:
            # Schema with config_id but no model_id
            cursor.execute(
                """
                INSERT INTO jobs (job_id, job_type, status, config_id, created_at, updated_at, progress)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, job_type, status, config_id, now, now, 0)
            )
        else:
            # Minimal schema (from test_database_setup.py)
            cursor.execute(
                """
                INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, progress)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, job_type, status, now, now, 0)
            )
        
        conn.commit()
        return job_id

    def get_job_status_history(job_id):
        """Get job status history for a specific job."""
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT status FROM job_status_history WHERE job_id = ? ORDER BY id", (job_id,))
            rows = cursor.fetchall()
            # Convert rows to a list of status values based on row structure
            if rows and isinstance(rows[0], dict):
                return [row['status'] for row in rows]
            elif rows and isinstance(rows[0], sqlite3.Row):
                return [row['status'] for row in rows]
            elif rows:
                # If rows are tuples, assume status is the first column
                return [row[0] for row in rows]
            return []
        except Exception as e:
            print(f"Error getting job status history: {e}")
            return []

    def create_job_status_history(job_id, status, status_message=None):
        """Create a job status history entry."""
        cursor = conn.cursor()
        try:
            # Check if the job_status_history table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='job_status_history'")
            if not cursor.fetchone():
                # Create the job_status_history table with the correct schema
                cursor.execute("""
                CREATE TABLE job_status_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL
                )
                """)
            
            # Get current time for updated_at
            now = datetime.now().isoformat()
            
            # Insert the history record with updated_at
            cursor.execute("""
            INSERT INTO job_status_history (job_id, status, status_message, updated_at)
            VALUES (?, ?, ?, ?)
            """, (job_id, status, status_message, now))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            print(f"Error creating job status history: {e}")
            return False

    def create_model(job_id, model_path="/fake/path/model.onnx"):
        """Create a model record that can be referenced by training_results."""
        model_id = "model_" + str(uuid.uuid4())
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)",
            (model_id, job_id, model_path, datetime.now().isoformat())
        )
        conn.commit()
        return model_id

    def create_training_result(job_id, model_id, config_id, metrics=None):
        """Create a training result record."""
        if metrics is None:
            metrics = {"mape": 10.5}
        
        result_id = str(uuid.uuid4())
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics, duration)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (result_id, job_id, model_id, config_id, json.dumps(metrics), 20)
        )
        conn.commit()
        return result_id

    def query_handler(query, params=(), fetchall=False):
        """Handles both regular SQL and special-cased queries from tests."""
        if query == 'job_status_history':
             # This is a special case from the test, not a real query
             # The job_id should be passed in params
             job_id_param = params[0] if params else None
             if not job_id_param:
                 raise ValueError("job_id must be provided to fetch status history.")
             statuses = get_job_status_history(job_id_param)
             return statuses
        return execute_query(query, params, fetchall)

    def execute_query(query, params=(), fetchall=False):
        """Helper to execute a query against the test database."""
        cursor = conn.cursor()
        # As temp_db returns a connection with Row factory, we can treat rows as dicts
        cursor.execute(query, params)
        if fetchall:
            return cursor.fetchall()
        return cursor.fetchone()

    yield {
        "conn": conn,
        "create_job": create_job,
        "execute_query": query_handler,
        "create_job_status_history": create_job_status_history,
        "create_model": create_model,
        "create_training_result": create_training_result
    }

def json_default_serializer(obj):
    """
    JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# Constants for tests
TEST_MODEL_ID = "model_" + str(uuid.uuid4())

# Sample predictions data for tests
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
    '0.05': [10.5, 12.3, 5.2, 7.8, 3.1],
    '0.25': [15.2, 18.7, 8.9, 11.3, 5.7],
    '0.5': [21.4, 24.8, 12.6, 15.9, 7.5],
    '0.75': [28.3, 32.1, 17.8, 20.4, 10.2],
    '0.95': [35.7, 40.2, 23.1, 27.5, 15.8]
}

# Helper function to create a complete TrainingParams object
def create_training_params(base_params=None):
    """
    Creates a complete TrainingParams object with all required fields.
    
    Args:
        base_params: Optional dictionary with parameters to use as a base
        
    Returns:
        A valid TrainingParams object
    """
    base_params = base_params or {}
    
    # Create model config
    model_config = ModelConfig(
        num_encoder_layers=3,
        num_decoder_layers=2,
        decoder_output_dim=128,
        temporal_width_past=12,
        temporal_width_future=6,
        temporal_hidden_size_past=64,
        temporal_hidden_size_future=64,
        temporal_decoder_hidden=128,
        batch_size=base_params.get('batch_size', 32),
        dropout=base_params.get('dropout', 0.2),
        use_reversible_instance_norm=True,
        use_layer_norm=True
    )
    
    # Create optimizer config
    optimizer_config = OptimizerConfig(
        lr=base_params.get('learning_rate', 0.001),
        weight_decay=0.0001
    )
    
    # Create LR scheduler config
    lr_shed_config = LRSchedulerConfig(
        T_0=10,
        T_mult=2
    )
    
    # Create training dataset config
    train_ds_config = TrainingDatasetConfig(
        alpha=0.05,
        span=12
    )
    
    # Create complete TrainingParams
    return TrainingConfig(
        nn_model_config=model_config,
        optimizer_config=optimizer_config,
        lr_shed_config=lr_shed_config,
        train_ds_config=train_ds_config,
        lags=12,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )

# ==============================================
# Pytest fixtures
# ==============================================

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_datasphere(fs, monkeypatch):
    """
    Мокирует DataSphere окружение для интеграционных тестов.
    Создает фейковую файловую систему и настройки.
    """
    # Create fake directories in pyfakefs
    input_dir = "/fake/datasphere_input"
    output_dir = "/fake/datasphere_output"
    fake_config_path = "/fake/datasphere_job/config.yaml"
    
    fs.makedirs(input_dir, exist_ok=True)
    fs.makedirs(output_dir, exist_ok=True)
    fs.makedirs("/fake/datasphere_job", exist_ok=True)
    fs.create_file(fake_config_path, contents="name: test_job\ntype: python")
    
    # Create a mock settings object with the computed properties overridden
    from deployment.app.config import AppSettings
    from unittest.mock import PropertyMock
    
    # Create a mock settings object
    mock_settings = MagicMock(spec=AppSettings)
    
    # Set up the computed properties as PropertyMock objects
    type(mock_settings).datasphere_input_dir = PropertyMock(return_value=input_dir)
    type(mock_settings).datasphere_output_dir = PropertyMock(return_value=output_dir)
    type(mock_settings).datasphere_job_config_path = PropertyMock(return_value=fake_config_path)
    
    # Set up other necessary attributes
    mock_settings.project_root_dir = "/fake/project"
    mock_settings.datasphere = MagicMock()
    mock_settings.datasphere.max_polls = 5
    mock_settings.datasphere.poll_interval = 1.0
    mock_settings.datasphere.client = {
        "project_id": "test-project",
        "folder_id": "test-folder",
        "oauth_token": "test-token"
    }
    
    # Patch the settings in the modules that use them
    monkeypatch.setattr('deployment.app.services.datasphere_service.settings', mock_settings)
    monkeypatch.setattr('deployment.app.config.settings', mock_settings)
    
    # Mock DataSphere service functions that the integration tests expect
    mock_save_model = AsyncMock(return_value="model_" + str(uuid.uuid4()))
    mock_save_predictions = MagicMock(return_value={"result_id": "test-result", "predictions_count": 5})
    mock_get_datasets = MagicMock()
    mock_update_job_status = MagicMock()
    
    # Mock DataSphere client
    mock_client = MagicMock()
    mock_client.submit_job.return_value = "ds_job_" + str(uuid.uuid4())
    mock_client.get_job_status.return_value = "COMPLETED"
    
    def download_results_side_effect(job_id, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        fs.create_file(os.path.join(output_dir, "metrics.json"), contents='{"mape": 15.3}')
        fs.create_file(os.path.join(output_dir, "model.onnx"), contents="dummy onnx model data")
        fs.create_file(os.path.join(output_dir, "predictions.csv"), contents="header1,header2\nvalue1,value2\n")
    
    mock_client.download_job_results.side_effect = download_results_side_effect
    
    # Apply the mocks
    monkeypatch.setattr('deployment.app.services.datasphere_service.save_model_file_and_db', mock_save_model)
    monkeypatch.setattr('deployment.app.services.datasphere_service.save_predictions_to_db', mock_save_predictions)
    monkeypatch.setattr('deployment.app.services.datasphere_service.get_datasets', mock_get_datasets)
    monkeypatch.setattr('deployment.app.services.datasphere_service.update_job_status', mock_update_job_status)
    monkeypatch.setattr('deployment.app.services.datasphere_service._initialize_datasphere_client', AsyncMock(return_value=mock_client))
    
    # Mock dataset preparation and verification functions for integration tests
    monkeypatch.setattr('deployment.app.services.datasphere_service._prepare_job_datasets', AsyncMock())
    monkeypatch.setattr('deployment.app.services.datasphere_service._verify_datasphere_job_inputs', AsyncMock())
    
    yield {
        'settings': mock_settings,
        'input_dir': input_dir,
        'output_dir': output_dir,
        'config_path': fake_config_path,
        'fs': fs,
        'save_model_file_and_db': mock_save_model,
        'save_predictions_to_db': mock_save_predictions,
        'get_datasets': mock_get_datasets,
        'update_job_status': mock_update_job_status,
        'client': mock_client
    }

# ==============================================
# Utility functions for assertions
# ==============================================

def verify_predictions_saved(connection, result, expected_data):
    """
    Проверяет, что предсказания корректно сохранены в базе данных.
    
    Args:
        connection: Соединение с БД
        result: Результат сохранения предсказаний
        expected_data: Ожидаемые данные (словарь как SAMPLE_PREDICTIONS)
    """
    cursor = connection.cursor()
    
    # Проверяем, что результат содержит валидный result_id
    assert result is not None
    assert "result_id" in result
    assert "predictions_count" in result
    assert result["predictions_count"] == len(expected_data["barcode"])
    
    # Проверяем таблицу prediction_results
    cursor.execute("SELECT * FROM prediction_results WHERE result_id = ?", (result["result_id"],))
    prediction_result = cursor.fetchone()
    assert prediction_result is not None
    
    # Проверяем, что в fact_predictions сохранены все предсказания
    cursor.execute("SELECT COUNT(*) as count FROM fact_predictions WHERE result_id = ?", (result["result_id"],))
    count = cursor.fetchone()["count"]
    assert count == len(expected_data["barcode"])
    
    # Проверяем значения квантилей для первой записи
    cursor.execute("""
        SELECT p.*, m.barcode, m.artist, m.album FROM fact_predictions p
        JOIN dim_multiindex_mapping m ON p.multiindex_id = m.multiindex_id
        WHERE m.barcode = ? AND m.artist = ? AND m.album = ?
        AND p.result_id = ?
    """, (expected_data["barcode"][0], expected_data["artist"][0], expected_data["album"][0], result["result_id"]))
    
    record = cursor.fetchone()
    assert record is not None
    
    # Проверяем каждую квантиль
    quantile_map = {
        '05': '0.05',
        '25': '0.25',
        '50': '0.5',
        '75': '0.75',
        '95': '0.95',
    }
    for q_db, q_data in quantile_map.items():
        db_value = float(record[f"quantile_{q_db}"])
        expected_value = expected_data[q_data][0]
        assert abs(db_value - expected_value) < 1e-6 

@pytest.fixture
def sample_predictions_data():
    """Provides sample predictions data as a dictionary."""
    return SAMPLE_PREDICTIONS.copy()

@pytest.fixture
def temp_db(sample_predictions_data):
    """
    Creates a temporary database and a predictions CSV file for testing.
    This fixture provides a more specific setup for tests that need to
    read a predictions file and write to a database.
    """
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, 'test_predictions.db')
    predictions_path = os.path.join(temp_dir.name, 'predictions.csv')
    
    # Create and save the predictions CSV
    pd.DataFrame(sample_predictions_data).to_csv(predictions_path, index=False)
    
    # Initialize the database
    conn = sqlite3.connect(db_path)
    init_db(connection=conn)
    conn.row_factory = sqlite3.Row
    
    job_id = "job-" + str(uuid.uuid4())
    model_id = "model-" + str(uuid.uuid4())
    
    # Pre-populate jobs and models tables since they are foreign keys
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "prediction", "running", datetime.now().isoformat(), datetime.now().isoformat())
    )
    cursor.execute(
        "INSERT INTO models (model_id, job_id, model_path, is_active, created_at) VALUES (?, ?, ?, ?, ?)",
        (model_id, job_id, "/fake/path/model.onnx", True, datetime.now().isoformat())
    )
    conn.commit()

    db_info = {
        "temp_dir": temp_dir,
        "db_path": db_path,
        "predictions_path": predictions_path,
        "job_id": job_id,
        "model_id": model_id,
        "conn": conn
    }
    
    yield db_info
    
    # Teardown
    conn.close()
    temp_dir.cleanup() 