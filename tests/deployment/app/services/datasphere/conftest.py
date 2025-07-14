import gc
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from deployment.app.db.schema import init_db
from deployment.app.models.api_models import (
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingDatasetConfig,
)
from tests.deployment.app.services.conftest import temp_workspace, mock_datasphere_env


@pytest.fixture(scope="function")
def mocked_db(temp_db):
    """
    Provides a mocked DB interface with helper functions for integration tests.
    - Uses a real in-memory DB connection from the `temp_db` fixture.
    - Provides helper methods to create jobs and execute queries.
    - **Note:** This fixture relies on `temp_db` for schema initialization and `row_factory` settings, 
      ensuring that its internal helper functions (like `get_job`) return dictionary-like objects.
    """
    conn = temp_db["conn"] if isinstance(temp_db, dict) else temp_db

    # Initialize all necessary tables if they don't exist
    init_db(connection=conn)
    conn.commit()

    # These tables are required for the integration tests to work
    # The 'cursor' variable is only used within nested functions now, so it should be defined there.

    def create_job(
        job_id, job_type="training", status="pending", config_id=None, model_id=None
    ):
        """Helper to insert a job record into the database."""
        # Create a new cursor for this function
        cursor = conn.cursor()

        if config_id is None:
            # Create a dummy config if none provided
            config_id = "config_" + str(uuid.uuid4())
            cursor.execute(
                "INSERT OR IGNORE INTO configs (config_id, config, created_at) VALUES (?, ?, ?)",
                (
                    config_id,
                    json.dumps({"sample": "config"}),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

        # Re-check columns after schema initialization
        cursor.execute("PRAGMA table_info(jobs)")
        columns = [column["name"] for column in cursor.fetchall()]

        now = datetime.now().isoformat()

        # Build query dynamically based on available columns
        query = "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at"

        # Add optional columns if they exist in the schema
        if "config_id" in columns and config_id is not None:
            query += ", config_id"
        if "model_id" in columns and model_id is not None:
            query += ", model_id"

        query += ") VALUES (?, ?, ?, ?, ?"

        # Add placeholders for optional columns
        params = [job_id, job_type, status, now, now]
        if "config_id" in columns and config_id is not None:
            query += ", ?"
            params.append(config_id)
        if "model_id" in columns and model_id is not None:
            query += ", ?"
            params.append(model_id)

        query += ")"

        # Execute with dynamic parameters
        cursor.execute(query, params)
        conn.commit()

        return job_id

    def update_job_status(job_id, status, progress=None, error_message=None, status_message=None, connection=None):
        """Если connection не передан, используется conn из фикстуры."""
        if connection is None:
            connection = conn
        """Update a job's status."""
        cursor = connection.cursor()
        now = datetime.now().isoformat()
        cursor.execute(
            """
            UPDATE jobs
            SET status = ?, updated_at = ?, progress = COALESCE(?, progress), error_message = COALESCE(?, error_message)
            WHERE job_id = ?
            """,
            (status, now, progress, error_message, job_id),
        )
        conn.commit()

        # Add to job status history if provided
        if status_message:
            create_job_status_history(job_id, status, status_message)

        return True

    def get_job_status_history(job_id):
        """Get job status history for a job, ordered by updated_at."""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, status_message, updated_at FROM job_status_history WHERE job_id = ? ORDER BY updated_at",
            (job_id,),
        )
        return cursor.fetchall()

    def create_job_status_history(job_id, status, status_message=None):
        """Create a job status history entry."""
        cursor = conn.cursor()
        try:
            # Check if the job_status_history table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='job_status_history'"
            )
            if not cursor.fetchone():
                # Create the job_status_history table with the correct schema
                cursor.execute("""
                CREATE TABLE job_status_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_message TEXT,
                    updated_at TEXT NOT NULL
                )
                """
                )

            # Get current time for updated_at
            now = datetime.now().isoformat()

            # Insert the history record with updated_at
            cursor.execute(
                """
            INSERT INTO job_status_history (job_id, status, status_message, updated_at)
            VALUES (?, ?, ?, ?)
            """,
                (job_id, status, status_message, now),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            return False

    def create_model(model_id, job_id, model_path="/fake/path/model.onnx", created_at=None, metadata=None, is_active=False, connection=None):
        """Если connection не передан, используется conn из фикстуры."""
        if connection is None:
            connection = conn
        """Create a model record that can be referenced by training_results, with explicit model_id and job_id."""
        if created_at is None:
            created_at = datetime.now().isoformat()
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO models (model_id, job_id, model_path, created_at, is_active) VALUES (?, ?, ?, ?, ?)",
            (model_id, job_id, model_path, created_at, is_active),
        )
        conn.commit()
        return model_id

    def create_training_result(job_id, model_id, config_id, metrics=None, duration=None, **kwargs):
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
            (result_id, job_id, model_id, config_id, json.dumps(metrics), 20),
        )
        conn.commit()
        return result_id
    
    def get_job(job_id, connection=None):
        """Если connection не передан, используется conn из фикстуры."""
        if connection is None:
            connection = conn
        """Get a job by ID."""
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return cursor.fetchone()

    def query_handler(query, params=(), fetchall=False, connection=None):
        """Handles both regular SQL and special-cased queries from tests."""
        if query == "job_status_history":
            # This is a special case from the test, not a real query
            # The job_id should be passed in params
            job_id_param = params[0] if params else None
            if not job_id_param:
                raise ValueError("job_id must be provided to fetch status history.")
            statuses = get_job_status_history(job_id_param)
            return statuses
        return execute_query(query, params, fetchall, connection=connection)

    def execute_query(query, params=(), fetchall=False, connection=None):
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
        "create_training_result": create_training_result,
        "get_job": get_job,  # Add the get_job function
        "update_job_status": update_job_status,  # Add update_job_status function
    }


def json_default_serializer(obj):
    """
    JSON serializer for objects not serializable by default json code
    """
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# Constants for tests
TEST_MODEL_ID = "model_" + str(uuid.uuid4())

# Sample predictions data for tests
SAMPLE_PREDICTIONS = {
    "barcode": [
        "123456789012",
        "234567890123",
        "987654321098",
        "876543210987",
        "555555555555",
    ],
    "artist": ["Artist A", "Artist A2", "Artist B", "Artist B2", "Artist C"],
    "album": ["Album X", "Album X2", "Album Y", "Album Y2", "Album Z"],
    "cover_type": ["Standard", "Deluxe", "Deluxe", "Standard", "Limited"],
    "price_category": ["A", "A", "B", "B", "C"],
    "release_type": ["Studio", "Studio", "Live", "Live", "Compilation"],
    "recording_decade": ["2010s", "2010s", "2000s", "2000s", "1990s"],
    "release_decade": ["2020s", "2020s", "2010s", "2010s", "2000s"],
    "style": ["Rock", "Rock", "Pop", "Pop", "Jazz"],
    "record_year": [2015, 2016, 2007, 2008, 1995],
    "0.05": [10.5, 12.3, 5.2, 7.8, 3.1],
    "0.25": [15.2, 18.7, 8.9, 11.3, 5.7],
    "0.5": [21.4, 24.8, 12.6, 15.9, 7.5],
    "0.75": [28.3, 32.1, 17.8, 20.4, 10.2],
    "0.95": [35.7, 40.2, 23.1, 27.5, 15.8],
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
        batch_size=base_params.get("batch_size", 32),
        dropout=base_params.get("dropout", 0.2),
        use_reversible_instance_norm=True,
        use_layer_norm=True,
    )

    # Create optimizer config
    optimizer_config = OptimizerConfig(
        lr=base_params.get("learning_rate", 0.001), weight_decay=0.0001
    )

    # Create LR scheduler config
    lr_shed_config = LRSchedulerConfig(T_0=10, T_mult=2)

    # Create training dataset config
    train_ds_config = TrainingDatasetConfig(alpha=0.05, span=12)

    # Create complete TrainingParams
    return TrainingConfig(
        nn_model_config=model_config,
        optimizer_config=optimizer_config,
        lr_shed_config=lr_shed_config,
        train_ds_config=train_ds_config,
        lags=12,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    )


# ==============================================
# Pytest fixtures
# ==============================================

logger = logging.getLogger(__name__)

# REMOVED: Duplicate mock_service_env fixture
# This fixture was identical to the one in services/conftest.py but with function scope
# instead of session scope. To ensure consistent performance and avoid confusion,
# we now use the session-scoped version from services/conftest.py

# REMOVED: mock_datasphere fixture which used pyfakefs


@pytest.fixture
def mock_datasphere(monkeypatch):
    """
    Мокирует DataSphere окружение для интеграционных тестов.
    Создает фейковую файловую систему и настройки.
    """
    # Create fake directories in pyfakefs
    input_dir = "/fake/datasphere_input"
    output_dir = "/fake/datasphere_output"
    job_dir = "/fake/datasphere_jobs"
    fake_config_path = "/fake/datasphere_jobs/train/config.yaml"

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(os.path.join(job_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(job_dir, "tune"), exist_ok=True)
    with open(fake_config_path, "w") as f:
        f.write("name: test_job\ntype: python")

    # Create a mock settings object with the computed properties overridden
    from unittest.mock import PropertyMock

    from deployment.app.config import AppSettings

    # Create a mock settings object
    mock_settings = MagicMock(spec=AppSettings)

    # Set up the computed properties as PropertyMock objects
    type(mock_settings).datasphere_input_dir = PropertyMock(return_value=input_dir)
    type(mock_settings).datasphere_output_dir = PropertyMock(return_value=output_dir)
    type(mock_settings).datasphere_job_config_path = PropertyMock(
        return_value=fake_config_path
    )
    type(mock_settings).datasphere_job_dir = PropertyMock(return_value=job_dir)
    type(mock_settings).datasphere_job_train_dir = PropertyMock(return_value=os.path.join(job_dir, "train"))
    type(mock_settings).datasphere_job_tune_dir = PropertyMock(return_value=os.path.join(job_dir, "tune"))

    # Set up other necessary attributes
    mock_settings.project_root_dir = "/fake/project"
    mock_settings.datasphere = MagicMock()
    mock_settings.datasphere.max_polls = 5
    mock_settings.datasphere.poll_interval = 1.0

    # Fix timeout values to be actual integers (not MagicMock objects)
    mock_settings.datasphere.client_submit_timeout_seconds = 60
    mock_settings.datasphere.client_status_timeout_seconds = 30
    mock_settings.datasphere.client_download_timeout_seconds = 600
    mock_settings.datasphere.client_init_timeout_seconds = 60
    mock_settings.datasphere.client_cancel_timeout_seconds = 60

    mock_settings.datasphere.client = {
        "project_id": "test-project",
        "folder_id": "test-folder",
        "oauth_token": "test-token",
    }

    # Patch the settings in the modules that use them
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service.get_settings", lambda: mock_settings
    )

    # Mock DataSphere service functions that the integration tests expect
    mock_save_model = AsyncMock(return_value="model_" + str(uuid.uuid4()))
    mock_save_predictions = MagicMock(
        return_value={"result_id": "test-result", "predictions_count": 5}
    )
    mock_get_datasets = MagicMock()
    mock_update_job_status = MagicMock()

    # Mock DataSphere client
    mock_client = MagicMock()
    mock_client.submit_job.return_value = "ds_job_" + str(uuid.uuid4())
    mock_client.get_job_status.return_value = "COMPLETED"

    def download_results_side_effect(job_id, output_dir, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            f.write('{"mape": 15.3}')
        with open(os.path.join(output_dir, "model.onnx"), "w") as f:
            f.write("dummy onnx model data")
        with open(os.path.join(output_dir, "predictions.csv"), "w") as f:
            f.write("header1,header2\nvalue1,value2\n")

    mock_client.download_job_results.side_effect = download_results_side_effect

    # Apply the mocks
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service.save_model_file_and_db",
        mock_save_model,
    )
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service.save_predictions_to_db",
        mock_save_predictions,
    )
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service.get_datasets", mock_get_datasets
    )
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service.update_job_status",
        mock_update_job_status,
    )
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service._initialize_datasphere_client",
        AsyncMock(return_value=mock_client),
    )

    # Mock dataset preparation and verification functions for integration tests
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service._prepare_job_datasets", AsyncMock()
    )
    monkeypatch.setattr(
        "deployment.app.services.datasphere_service._verify_datasphere_job_inputs",
        AsyncMock(),
    )

    yield {
        "settings": mock_settings,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "config_path": fake_config_path,
        "save_model_file_and_db": mock_save_model,
        "save_predictions_to_db": mock_save_predictions,
        "get_datasets": mock_get_datasets,
        "update_job_status": mock_update_job_status,
        "client": mock_client,
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
    cursor.execute(
        "SELECT * FROM prediction_results WHERE result_id = ?", (result["result_id"],)
    )
    prediction_result = cursor.fetchone()
    assert prediction_result is not None

    # Проверяем, что в fact_predictions сохранены все предсказания
    cursor.execute(
        "SELECT COUNT(*) as count FROM fact_predictions WHERE result_id = ?",
        (result["result_id"],),
    )
    count = cursor.fetchone()["count"]
    assert count == len(expected_data["barcode"])

    # Проверяем значения квантилей для первой записи
    cursor.execute(
        """
        SELECT p.*, m.barcode, m.artist, m.album FROM fact_predictions p
        JOIN dim_multiindex_mapping m ON p.multiindex_id = m.multiindex_id
        WHERE m.barcode = ? AND m.artist = ? AND m.album = ?
        AND p.result_id = ?
    """,
        (
            expected_data["barcode"][0],
            expected_data["artist"][0],
            expected_data["album"][0],
            result["result_id"],
        ),
    )

    record = cursor.fetchone()
    assert record is not None

    # Проверяем каждую квантиль
    quantile_map = {
        "05": "0.05",
        "25": "0.25",
        "50": "0.5",
        "75": "0.75",
        "95": "0.95",
    }
    for q_db, q_data in quantile_map.items():
        db_value = float(record[f"quantile_{q_db}"])
        expected_value = expected_data[q_data][0]
        assert abs(db_value - expected_value) < 1e-6


@pytest.fixture
def sample_predictions_data():
    """Provides sample predictions data as a dictionary."""
    return SAMPLE_PREDICTIONS.copy()


# Добавляем dict_factory для row_factory

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


@pytest.fixture
def temp_db(sample_predictions_data):
    """
    Creates a temporary database and a predictions CSV file for testing.
    This fixture provides a more specific setup for tests that need to
    read a predictions file and write to a database.

    CRITICAL FIXES APPLIED:
    - Uses context manager pattern for proper resource management
    - Forces garbage collection before cleanup to release Windows file locks
    - Uses manual file removal instead of TemporaryDirectory for better control
    - Implements retry logic for Windows file locking issues
    """
    # Create temporary directory manually for better control
    temp_dir_path = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir_path, "test_predictions.db")
    predictions_path = os.path.join(temp_dir_path, "predictions.csv")

    # Create and save the predictions CSV
    pd.DataFrame(sample_predictions_data).to_csv(predictions_path, index=False)

    # Initialize the database with context manager
    conn = sqlite3.connect(db_path)
    try:
        init_db(connection=conn)
        conn.row_factory = dict_factory

        job_id = "job-" + str(uuid.uuid4())
        model_id = "model-" + str(uuid.uuid4())

        # Pre-populate jobs and models tables since they are foreign keys
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (
                job_id,
                "prediction",
                "running",
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )
        cursor.execute(
            "INSERT INTO models (model_id, job_id, model_path, is_active, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                model_id,
                job_id,
                "/fake/path/model.onnx",
                True,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()

        db_info = {
            "temp_dir_path": temp_dir_path,  # Changed from temp_dir object to path
            "db_path": db_path,
            "predictions_path": predictions_path,
            "job_id": job_id,
            "model_id": model_id,
            "conn": conn,
        }

        yield db_info
    finally:
        conn.close()
        # Force garbage collection to release file handles on Windows
        gc.collect()
        # Clean up temporary directory with retry logic
        cleanup_with_retry(temp_dir_path)

def cleanup_with_retry(path, max_retries=3):
    for i in range(max_retries):
        try:
            shutil.rmtree(path)
            break
        except OSError as e:
            if i < max_retries - 1:
                time.sleep(0.1 * (2 ** i))  # Exponential backoff
            else:
                logger.warning(f"Failed to remove directory {path} after {max_retries} retries: {e}")


@pytest.fixture(autouse=True, scope='function')
def insert_minimal_fact_data(mocked_db):
    """
    Заполняет fact_sales, fact_stock, fact_stock_changes минимум 13 уникальными датами для lags=12.
    Использует conn из mocked_db.
    """
    conn = mocked_db["conn"] if isinstance(mocked_db, dict) and "conn" in mocked_db else mocked_db
    today = datetime.today().date()
    base_barcode = "1234567890"
    base_artist = "Test Artist"
    base_album = "Test Album"
    base_cover = "CD"
    base_price = "Standard"
    base_release_type = "Studio"
    base_recording_decade = "2010s"
    base_release_decade = "2010s"
    base_style = "Rock"
    base_year = 2015
    # Получить или создать multiindex_id
    from deployment.app.db.database import get_or_create_multiindex_id
    multiindex_id = get_or_create_multiindex_id(
        base_barcode, base_artist, base_album, base_cover, base_price,
        base_release_type, base_recording_decade, base_release_decade, base_style, base_year,
        connection=conn
    )
    from dateutil.relativedelta import relativedelta
    # Генерируем 13 месяцев назад от текущего месяца
    base_date = today.replace(day=1) - relativedelta(months=13)
    
    # Generate stock data for only the very first month (base_date)
    stock_date_str = base_date.strftime("%Y-%m-01")
    stock_rows = [
        (multiindex_id, stock_date_str, 100)
    ] # Single data point for stock

    # Generate sales and changes data starting from the month after base_date
    sales_rows = []
    changes_rows = []
    for i in range(1, 16):
        d = base_date + relativedelta(months=i)
        date_str = d.strftime("%Y-%m-01")
        sales_rows.append((multiindex_id, date_str, 10 + i))
        changes_rows.append((multiindex_id, date_str, 1))

    conn.executemany(
        "INSERT INTO fact_sales (multiindex_id, data_date, value) VALUES (?, ?, ?)",
        sales_rows
        )
    conn.executemany(
        "INSERT INTO fact_stock (multiindex_id, data_date, value) VALUES (?, ?, ?)",
        stock_rows
    )
    conn.executemany(
        "INSERT INTO fact_stock_changes (multiindex_id, data_date, value) VALUES (?, ?, ?)",
        changes_rows
        )
    conn.commit()

@pytest.fixture
def mock_init_db():
    """Mock для init_db (schema initialization)."""
    return MagicMock()

@pytest.fixture
def mock_get_db():
    """Mock для get_db (возвращает MagicMock с нужным интерфейсом)."""
    mock = MagicMock()
    mock.cursor.return_value = MagicMock()
    return mock
