import gc
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.models.api_models import (
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingDatasetConfig,
)


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


def verify_predictions_saved(dal, result, expected_data):
    """
    Проверяет, что предсказания корректно сохранены в базе данных.

    Args:
        dal: DataAccessLayer instance
        result: Результат сохранения предсказаний
        expected_data: Ожидаемые данные (словарь как SAMPLE_PREDICTIONS)
    """
    # Проверяем, что результат содержит валидный result_id
    assert result is not None
    assert "result_id" in result
    assert "predictions_count" in result
    assert result["predictions_count"] == len(expected_data["barcode"])

    # Проверяем таблицу prediction_results
    prediction_result = dal.execute_raw_query(
        "SELECT * FROM prediction_results WHERE result_id = ?",
        (result["result_id"],)
    )
    assert prediction_result is not None

    # Проверяем, что в fact_predictions сохранены все предсказания
    count_result = dal.execute_raw_query(
        "SELECT COUNT(*) as count FROM fact_predictions WHERE result_id = ?",
        (result["result_id"],),
    )
    count = count_result["count"]
    assert count == len(expected_data["barcode"])

    # Проверяем значения квантилей для первой записи
    record = dal.execute_raw_query(
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
def temp_db(sample_predictions_data, in_memory_db):
    """
    Creates a temporary database and a predictions CSV file for testing.
    This fixture provides a more specific setup for tests that need to
    read a predictions file and write to a database.

    CRITICAL FIXES APPLIED:
    - Uses the existing in_memory_db fixture for proper database initialization
    - Uses context manager pattern for proper resource management
    - Forces garbage collection before cleanup to release Windows file locks
    - Uses manual file removal instead of TemporaryDirectory for better control
    - Implements retry logic for Windows file locking issues
    """
    # Create temporary directory manually for better control
    temp_dir_path = tempfile.mkdtemp()
    predictions_path = os.path.join(temp_dir_path, "predictions.csv")

    # Create and save the predictions CSV
    pd.DataFrame(sample_predictions_data).to_csv(predictions_path, index=False)

    # Use the existing in_memory_db fixture for database operations
    dal = in_memory_db

    # Create a job and model record for the predictions
    try:
            model_id = "model-" + str(uuid.uuid4())

            # Pre-populate jobs and models tables since they are foreign keys
            # Добавляю параметры с prediction_month
            job_parameters = json.dumps({"prediction_month": "2023-10-01"})
            job_id = dal.create_job(
                job_type="prediction",
                parameters=job_parameters,
                status="running",
            )
            dal.create_model_record(
                model_id=model_id,
                job_id=job_id,
                model_path="/fake/path/model.onnx",
                created_at=datetime.now(),
                is_active=True,
            )

            db_info = {
                "temp_dir_path": temp_dir_path,  # Changed from temp_dir object to path
                "predictions_path": predictions_path,
                "job_id": job_id,
                "model_id": model_id,
                "dal": dal,
            }

            yield db_info
    finally:
        # dal manages connection closing
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
def insert_minimal_fact_data(in_memory_db: DataAccessLayer):
    """
    Заполняет fact_sales, fact_stock, fact_stock_changes минимум 13 уникальными датами для lags=12.
    Использует dal из in_memory_db.
    """
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
    multiindex_id = in_memory_db.get_or_create_multiindex_id(
        base_barcode, base_artist, base_album, base_cover, base_price,
        base_release_type, base_recording_decade, base_release_decade, base_style, base_year,
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

    import pandas as pd

    from deployment.app.db.feature_storage import SQLFeatureStore

    feature_store = SQLFeatureStore(dal=in_memory_db)

    def create_df(data):
        df = pd.DataFrame(data, columns=["multiindex_id", "data_date", "value"])
        df["data_date"] = pd.to_datetime(df["data_date"])
        pivot_df = df.pivot_table(index="data_date", columns="multiindex_id", values="value")

        from deployment.app.db.schema import MULTIINDEX_NAMES
        # Create a mapping from multiindex_id to the full multiindex tuple
        multiindex_id_to_tuple = {}
        for multiindex_id_val in pivot_df.columns:
            # Retrieve the full multiindex tuple from the DAL
            # Assuming dal.get_multiindex_mapping_by_id exists or can be mocked
            # For now, create a dummy tuple that matches the expected structure
            # In a real scenario, you'd fetch this from the DB or ensure it's created
            multiindex_tuple = (
                str(multiindex_id_val), # barcode
                base_artist, # artist
                base_album, # album
                base_cover, # cover_type
                base_price, # price_category
                base_release_type, # release_type
                base_recording_decade, # recording_decade
                base_release_decade, # release_decade
                base_style, # style
                base_year # record_year
            )
            multiindex_id_to_tuple[multiindex_id_val] = multiindex_tuple

        # Map the pivot_df columns (multiindex_ids) to the full multiindex tuples
        pivot_df.columns = pd.MultiIndex.from_tuples(
            [multiindex_id_to_tuple[col] for col in pivot_df.columns],
            names=MULTIINDEX_NAMES
        )
        return pivot_df

    feature_store.save_features({"sales": create_df(sales_rows)})
    feature_store.save_features({"stock": create_df(stock_rows)})
    feature_store.save_features({"change": create_df(changes_rows)})

@pytest.fixture
def mock_init_db():
    """Mock для init_db (schema initialization)."""
    return MagicMock()


@pytest.fixture
def mocked_db(in_memory_db):
    """
    Creates a mocked database fixture that provides all the necessary methods
    and objects for datasphere pipeline integration tests.

    Returns a dictionary-like object with:
    - create_job: method to create jobs
    - create_model: method to create models
    - conn: database connection
    - dal: DataAccessLayer instance
    - execute_query: method to execute queries
    - create_job_status_history: method to create job status history
    """
    # Create a mock object that behaves like a dictionary
    mock_db = MagicMock()

    # Set up the database connection
    mock_db["conn"] = in_memory_db._connection

    # Set up the DAL
    mock_db["dal"] = in_memory_db

    # Create job method
    def create_job(job_id):
        return in_memory_db.create_job(
            job_type="prediction",
            parameters='{"prediction_month": "2023-10-01"}',
            status="running"
        )
    mock_db["create_job"] = create_job

    # Create model method
    def create_model(model_id, job_id):
        return in_memory_db.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path="/fake/path/model.onnx",
            created_at=datetime.now(),
            is_active=True
        )
    mock_db["create_model"] = create_model

    # Execute query method
    def execute_query(query, params=None):
        cursor = in_memory_db.get_connection().cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()
    mock_db["execute_query"] = execute_query

    # Create job status history method
    def create_job_status_history(job_id, status, progress=0, error_message=None):
        # Actually create job status history records in the database
        cursor = in_memory_db._connection.cursor()
        cursor.execute(
            "INSERT INTO job_status_history (job_id, status, progress, error_message, created_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, status, progress, error_message, datetime.now().isoformat())
        )
        in_memory_db._connection.commit()
        return True
    mock_db["create_job_status_history"] = create_job_status_history

    return mock_db

@pytest.fixture
def mock_get_db():
    """Mock для get_db (возвращает MagicMock с нужным интерфейсом)."""
    mock = MagicMock()
    mock.cursor.return_value = MagicMock()
    return mock
