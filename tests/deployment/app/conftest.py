"""
Фикстуры и моки для тестов deployment/app

- temp_db, temp_db_with_data, isolated_db_session — создают временные SQLite БД с полной схемой и cleanup.
- in_memory_db — для мокирования БД в интеграционных тестах.
- mocked_db — предоставляет моки для всех основных операций с БД, гарантирует передачу connection между слоями.
- Все фикстуры имеют scope='function' для предотвращения state leakage.
- pyfakefs и патчинг aiofiles используются только в тестах, где это необходимо.
- Все временные файлы и директории удаляются после теста.
- Моки DataSphere и DAL возвращают структуры, строго соответствующие контракту теста.
"""

import gc
import os
import shutil
import tempfile
import uuid
from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from passlib.context import CryptContext

from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.db.feature_storage import SQLFeatureStore
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


@pytest.fixture
def mock_active_config():
    """
    Создает фиктивный набор активных параметров для тестирования.
    """
    valid_training_params = {
        "config_id": "default-active-config-id",
        "parameters": {
            "model_config": {
                "num_encoder_layers": 3,
                "num_decoder_layers": 2,
                "decoder_output_dim": 128,
                "temporal_width_past": 12,
                "temporal_width_future": 6,
                "temporal_hidden_size_past": 64,
                "temporal_hidden_size_future": 64,
                "temporal_decoder_hidden": 128,
                "batch_size": 32,
                "dropout": 0.2,
                "use_reversible_instance_norm": True,
                "use_layer_norm": True,
            },
            "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
            "lr_shed_config": {"T_0": 10, "T_mult": 2},
            "train_ds_config": {"alpha": 0.05, "span": 12},
            "lags": 12,
            "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
            "model_id": "default_model",
            "additional_params": {
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2022-12-31",
            },
        },
    }
    return valid_training_params


@pytest.fixture
def sample_model_data():
    """
    Sample model data for testing
    """
    return {
        "model_id": str(uuid.uuid4()),
        "job_id": str(uuid.uuid4()),
        "model_path": "/path/to/model.onnx",
        "created_at": datetime.now(),
        "metadata": {"framework": "pytorch", "version": "1.9.0"},
    }


@pytest.fixture
def sample_config(create_training_params_fn):
    """
    Sample config for testing, now returns a valid nested TrainingConfig structure.
    """
    # Use the helper from the root conftest to create a valid, nested config
    return create_training_params_fn().model_dump(mode="json")


@pytest.fixture
def in_memory_db():
    """
    Provides a DAL instance connected to a temporary, file-based SQLite DB.
    This solves the 'SQLite objects created in a thread...' issue by using a file,
    while ensuring complete test isolation.
    """
    # Create a temporary file that will be automatically deleted
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    # The DAL will initialize the schema in this new file
    dal = DataAccessLayer(db_path=db_path)
    try:
        yield dal
    finally:
        dal.close()
        # Ensure the file is cleaned up
        try:
            os.remove(db_path)
        except OSError:
            pass


@pytest.fixture
def file_based_db():
    """
    УНИФИЦИРОВАННАЯ фикстура для реальной временной БД на диске.
    Создает полнофункциональную БД с несколькими моделями, конфигами и результатами.
    Возвращает объект DataAccessLayer.
    """
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, "test.db")
    dal = DataAccessLayer(db_path=db_path)
    dal.init_db()

    try:
        # --- Параметр-сеты (configs) ---
        config_data_1 = TrainingConfig(
            nn_model_config=ModelConfig(
                num_encoder_layers=3,
                num_decoder_layers=2,
                decoder_output_dim=128,
                temporal_width_past=12,
                temporal_width_future=6,
                temporal_hidden_size_past=64,
                temporal_hidden_size_future=64,
                temporal_decoder_hidden=128,
                batch_size=32,
                dropout=0.2,
                use_reversible_instance_norm=True,
                use_layer_norm=True,
            ),
            optimizer_config=OptimizerConfig(lr=0.01, weight_decay=0.0001),
            lr_shed_config=LRSchedulerConfig(T_0=10, T_mult=2),
            train_ds_config=TrainingDatasetConfig(alpha=0.1, span=12),
            lags=12,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        ).model_dump(mode="json")
        config_data_2 = TrainingConfig(
            nn_model_config=ModelConfig(
                num_encoder_layers=4,
                num_decoder_layers=3,
                decoder_output_dim=256,
                temporal_width_past=24,
                temporal_width_future=12,
                temporal_hidden_size_past=128,
                temporal_hidden_size_future=128,
                temporal_decoder_hidden=256,
                batch_size=64,
                dropout=0.3,
                use_reversible_instance_norm=False,
                use_layer_norm=False,
            ),
            optimizer_config=OptimizerConfig(lr=0.005, weight_decay=0.00001),
            lr_shed_config=LRSchedulerConfig(T_0=20, T_mult=1),
            train_ds_config=TrainingDatasetConfig(alpha=0.01, span=24),
            lags=24,
            quantiles=[0.01, 0.5, 0.99]
        ).model_dump(mode="json")

        config_id_1 = dal.create_or_get_config(config_data_1, is_active=True)
        config_id_2 = dal.create_or_get_config(config_data_2, is_active=False)

        # --- Модели ---
        job_id = "job-test"
        model_id_1 = "model-1"
        model_id_2 = "model-2"
        model_path_1 = os.path.join(temp_dir.name, "model_1.onnx")
        model_path_2 = os.path.join(temp_dir.name, "model_2.onnx")
        with open(model_path_1, "w") as f:
            f.write("dummy model 1")
        with open(model_path_2, "w") as f:
            f.write("dummy model 2")

        dal.create_model_record(model_id_1, job_id, model_path_1, datetime.now(), is_active=True)
        dal.create_model_record(model_id_2, job_id, model_path_2, datetime.now(), is_active=False)

        # --- Результаты тренировки (для метрик) ---
        metrics_data_1 = {"mape": 9.9, "val_loss": 0.08}
        metrics_data_2 = {"mape": 9.8, "val_loss": 0.03}
        dal.create_training_result(job_id, model_id_1, config_id_1, metrics_data_1, None)
        dal.create_training_result(job_id, model_id_2, config_id_2, metrics_data_2, None)

        # --- Запись о job ---
        dal.create_job(job_id, "training", "completed")

        setup_data = {
            "temp_dir_path": temp_dir.name,
            "db_path": db_path,
            "dal": dal,
            "job_id": job_id,
            "model_id_1": model_id_1,
            "model_id_2": model_id_2,
            "config_id_1": config_id_1,
            "config_id_2": config_id_2,
        }

        yield setup_data

    finally:
        try:
            dal.close()
        except Exception:
            pass
        try:
            temp_dir.cleanup()
        except Exception:
            pass


@pytest.fixture
def sample_job_data():
    """
    Sample job data for testing
    """
    return {
        "job_id": str(uuid.uuid4()),
        "job_type": "training",
        "parameters": {"batch_size": 32, "learning_rate": 0.001},
    }




@pytest.fixture(scope="function", autouse=True)
def clean_retry_events_table(in_memory_db):
    """
    Autouse fixture to clean the retry_events table before and after each test.
    Ensures test isolation for operations that persist to this table.
    """
    # Use the DAL provided by in_memory_db
    dal = in_memory_db
    # Clean up before the test
    dal.execute_raw_query("DELETE FROM retry_events")
    yield # Yield control to the test function
    # Clean up after the test
    dal.execute_raw_query("DELETE FROM retry_events")



@pytest.fixture(scope="session", autouse=True)
def set_session_db_path(session_monkeypatch, tmp_path_factory):
    """
    Session-scoped fixture to set a temporary file-based database path for AppSettings.
    Ensures all database interactions use an isolated DB for the entire test session.
    """

    from deployment.app.config import get_settings

    # Use a real temporary directory for the session
    temp_dir = tmp_path_factory.mktemp("session_data")
    temp_dir / "session_test.db"
    models_dir = temp_dir / "models"
    logs_dir = temp_dir / "logs"
    temp_upload_dir = temp_dir / "temp_uploads"

    # Create directories
    models_dir.mkdir()
    logs_dir.mkdir()
    temp_upload_dir.mkdir()

    # The DAL will handle schema initialization when it's instantiated with db_path

    # --- Create a REAL, fully-populated AppSettings instance ---
    # This avoids all the AttributeError problems from an incomplete MagicMock.
    # We override only the paths and critical values for testing.

    # Use monkeypatch to set environment variables that AppSettings will read
    session_monkeypatch.setenv("DATA_ROOT_DIR", str(temp_dir))
    session_monkeypatch.setenv("DB_FILENAME", "session_test.db")
    session_monkeypatch.setenv("API_ADMIN_API_KEY_HASH", CryptContext(schemes=["bcrypt"]).hash("test_admin_token"))
    session_monkeypatch.setenv("API_X_API_KEY_HASH", CryptContext(schemes=["bcrypt"]).hash("test_x_api_key_conftest"))
    session_monkeypatch.setenv("METRIC_THESH_FOR_HEALTH_CHECK", "0.5") # Add missing health check metric

    # Clear the get_settings cache to force re-reading with our new env vars
    get_settings.cache_clear()

    # The original AppSettings will now load with our temporary paths and test keys.
    # No need to mock the class itself.

    # Patch configure_logging to prevent FileNotFoundError during tests
    session_monkeypatch.setattr("deployment.app.logger_config.configure_logging", MagicMock())

    yield

    # Cleanup: clear the cache again after the session
    get_settings.cache_clear()



@pytest.fixture
def temp_db():
    """
    Creates a temporary SQLite database file with the full schema and yields a dict with paths and a live DataAccessLayer instance.
    Cleans up after use.
    """
    import tempfile
    temp_dir_path = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir_path, "test_temp.db")
    dal = DataAccessLayer(db_path=db_path)
    try:
        yield {
            "temp_dir_path": temp_dir_path,
            "db_path": db_path,
            "dal": dal,
        }
    finally:
        dal.close()
        gc.collect()
        shutil.rmtree(temp_dir_path, ignore_errors=True)

@pytest.fixture
def temp_db_with_data(temp_db, sample_config):
    """
    Like temp_db, but pre-populates jobs, models, and configs. Provides all IDs needed for most test contracts.
    """
    dal = temp_db["dal"]
    model_id = str(uuid.uuid4())
    job_for_model_id = str(uuid.uuid4())

    # Create two jobs
    job_id_1 = dal.create_job(
        job_type="training",
        status="pending",
    )
    job_for_model_id = dal.create_job(
        job_type="training",
        status="pending",
    )
    # Create a model for job_for_model_id
    dal.create_model_record(
        model_id=model_id,
        job_id=job_for_model_id,
        model_path="/fake/path/model.onnx",
        created_at=datetime.now(),
        is_active=True,
    )
    # Create a config
    config_id = dal.create_or_get_config(
        sample_config,
        is_active=True,
    )
    temp_db["job_id"] = job_id_1
    temp_db["model_id"] = model_id
    temp_db["job_for_model_id"] = job_for_model_id
    temp_db["config_id"] = config_id
    yield temp_db

@pytest.fixture
def isolated_db_session():
    """
    Function-scoped fixture that creates a new, isolated SQLite DB file for each test and yields a dict with both the DataAccessLayer instance and the db_path.
    Use .get('dal') for DB operations, .get('db_path') for path-based tests.
    """
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    db_path = temp_file.name
    temp_file.close()
    dal = DataAccessLayer(db_path=db_path)
    try:
        yield {"dal": dal, "db_path": db_path}
    finally:
        dal.close()
        gc.collect()
        try:
            os.remove(db_path)
        except Exception:
            pass

@pytest.fixture
def db_with_sales_data(in_memory_db):
    """Provides a db with controlled sales data for date logic tests."""
    dal = in_memory_db

    # For simplicity, we'll manually insert into dim_multiindex_mapping.
    from deployment.app.db.database import get_or_create_multiindex_id
    get_or_create_multiindex_id(
        barcode="barcode1",
        artist="artist1",
        album="album1",
        cover_type="dummy", # Add dummy values for required fields
        price_category="dummy",
        release_type="dummy",
        recording_decade="dummy",
        release_decade="dummy",
        style="dummy",
        record_year=2000,
        connection=dal._connection, # Pass the connection
    )

    # Prepare sales data as a pandas DataFrame
    # Create the correct MultiIndex format for feature storage
    from deployment.app.db.schema import MULTIINDEX_NAMES

    # Create a tuple with all the required fields
    multiindex_tuple = ("barcode1", "artist1", "album1", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy", 2000)

    sales_data_list = [
        {"data_date": "2023-10-31", "value": 10},
        {"data_date": "2023-11-01", "value": 5},
        {"data_date": "2023-11-02", "value": 5},
        {"data_date": "2023-11-30", "value": 5},  # Add data for end of November to make it complete
    ]

    # Create a DataFrame with MultiIndex in index and DatetimeIndex in columns
    # This matches the expected format for _save_feature
    df_sales = pd.DataFrame(sales_data_list)
    df_sales["data_date"] = pd.to_datetime(df_sales["data_date"])
    
    # Create the correct format: MultiIndex in index, DatetimeIndex in columns
    pivot_df = pd.DataFrame(
        data=df_sales["value"].values.reshape(1, -1),
        index=pd.MultiIndex.from_tuples([multiindex_tuple], names=MULTIINDEX_NAMES),
        columns=df_sales["data_date"]
    )

    # Use SQLFeatureStore to save the features
    feature_store = SQLFeatureStore(dal=dal)
    feature_store.save_features({"sales": pivot_df})

    return dal

# NOTE: All DB fixtures yield dicts. Use .get('conn') for DB operations, .get('db_path') for path-based tests.
