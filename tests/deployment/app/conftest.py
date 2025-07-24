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

import json
import os
import sqlite3
import tempfile
import uuid
from datetime import date, datetime
from contextlib import contextmanager

import pytest
from unittest.mock import MagicMock, PropertyMock
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

from deployment.app.db.database import (
    dict_factory,
    DatabaseError,
    execute_query,
    execute_many,
    update_job_status,
    get_job,
    create_training_result,
    create_prediction_result,
    create_or_get_config as db_create_or_get_config,
    get_active_config,
    set_config_active,
    set_model_active,
    delete_model_record_and_file,
    get_active_model as db_get_active_model,
    get_all_models,
    get_recent_models,
    get_best_model_by_metric,
    get_top_configs,
    get_effective_config,
    auto_activate_best_config_if_enabled,
    auto_activate_best_model_if_enabled,
    list_jobs,
    update_processing_run,
    create_processing_run,
    get_prediction_result,
    get_training_results,
    get_report_result,
    get_data_upload_result,
    create_data_upload_result,
    get_or_create_multiindex_id,
    insert_retry_event,
    fetch_recent_retry_events,
    create_tuning_result,
    get_tuning_results,
)
from deployment.app.db.schema import init_db
from deployment.app.models.api_models import (
    LRSchedulerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingDatasetConfig,
    JobStatus # Import JobStatus
)

import shutil
import gc
import time
from types import SimpleNamespace


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
    УНИФИЦИРОВАННАЯ фикстура для SQLite в памяти.
    Создает и инициализирует схему БД в оперативной памяти (':memory:').
    Используется для быстрых unit-тестов.
    """
    conn = sqlite3.connect(":memory:")
    init_db(connection=conn)
    conn.row_factory = dict_factory

    def create_job_in_mock_db(
        job_type: str,
        parameters: dict = None,
        connection: sqlite3.Connection = conn,
        status: str = "pending",
    ) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        connection.execute(
            """
            INSERT INTO jobs (job_id, job_type, status, created_at, updated_at, parameters, progress)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job_type,
                status,
                now,
                now,
                json.dumps(parameters, default=json_default_serializer) if parameters else None,
                0,
            ),
        )
        connection.commit()
        return job_id

    def create_model_in_mock_db(
        job_id: str,
        model_path: str = "/mock/path/model.onnx",
        created_at: datetime = None,
        metadata: dict = None,
        is_active: bool = False,
        connection: sqlite3.Connection = conn,
    ) -> str:
        model_id = str(uuid.uuid4())
        now = created_at or datetime.now()
        connection.execute(
            """
            INSERT INTO models (model_id, job_id, model_path, created_at, metadata, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                job_id,
                model_path,
                now.isoformat(),
                json.dumps(metadata, default=json_default_serializer) if metadata else None,
                1 if is_active else 0,
            ),
        )
        connection.commit()
        return model_id

    def create_job_status_history_in_mock_db(
        job_id: str,
        status: str,
        status_message: str,
        progress: float = None,
        connection: sqlite3.Connection = conn,
    ) -> None:
        now = datetime.now().isoformat()
        connection.execute(
            """
            INSERT INTO job_status_history
            (job_id, status, status_message, progress, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, status, status_message, progress, now),
        )
        connection.commit()

    def create_config_in_mock_db(
        config_dict: dict,
        is_active: bool = False,
        source: str = None,
        connection: sqlite3.Connection = conn,
    ) -> str:
        config_json = json.dumps(config_dict, sort_keys=True, default=json_default_serializer)
        config_id = uuid.uuid4().hex # Using uuid for simplicity in mock, instead of md5 hash
        now = datetime.now().isoformat()
        connection.execute(
            """
            INSERT INTO configs (config_id, config, is_active, created_at, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (config_id, config_json, 1 if is_active else 0, now, source),
        )
        connection.commit()
        return config_id

    def get_job_from_mock_db(job_id: str, connection: sqlite3.Connection = conn) -> dict:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return cursor.fetchone()

    def get_model_from_mock_db(model_id: str, connection: sqlite3.Connection = conn) -> dict:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
        return cursor.fetchone()

    def get_config_from_mock_db(config_id: str, connection: sqlite3.Connection = conn) -> dict:
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM configs WHERE config_id = ?", (config_id,))
        return cursor.fetchone()

    def update_job_status_in_mock_db(
        job_id: str,
        status: str,
        progress: float = None,
        result_id: str = None,
        error_message: str = None,
        status_message: str = None,
        connection: sqlite3.Connection = conn,
    ) -> None:
        """Если connection не передан, используется conn из фикстуры."""
        if connection is None:
            connection = conn
        now = datetime.now().isoformat()
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,))
        if not cursor.fetchone():
            return
        update_query = """
            UPDATE jobs
            SET
                status = ?,
                updated_at = ?,
                progress = COALESCE(?, progress),
                result_id = COALESCE(?, result_id),
                error_message = COALESCE(?, error_message)
            WHERE job_id = ?
        """
        cursor.execute(update_query, (status, now, progress, result_id, error_message, job_id))

        history_message = (
            status_message if status_message else f"Status changed to: {status}"
        )
        history_query = """
            INSERT INTO job_status_history
            (job_id, status, status_message, progress, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """
        cursor.execute(history_query, (job_id, status, history_message, progress, now))
        connection.commit()


    yield {"conn": conn,
           "create_job": create_job_in_mock_db,
           "create_model": create_model_in_mock_db,
           "create_job_status_history": create_job_status_history_in_mock_db,
           "create_config": create_config_in_mock_db,
           "get_job_from_db": get_job_from_mock_db,
           "get_model_from_db": get_model_from_mock_db,
           "get_config_from_db": get_config_from_mock_db,
           "update_job_status": update_job_status_in_mock_db, # New: Add mock update_job_status
    }
    conn.close()


@pytest.fixture
def file_based_db():
    """
    УНИФИЦИРОВАННАЯ фикстура для реальной временной БД на диске.
    Создает полнофункциональную БД с несколькими моделями, конфигами и результатами.
    """
    temp_dir = tempfile.TemporaryDirectory()
    db_path = os.path.join(temp_dir.name, "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # --- Параметр-сеты (configs) ---
        config_id_1 = "param-1"
        config_id_2 = "param-2"
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

        cursor.execute(
            """INSERT INTO configs (config_id, parameters, is_active, created_at) VALUES (?, ?, ?, ?)""",
            (config_id_1, json.dumps(config_data_1), 1, datetime.now().isoformat()),
        )
        cursor.execute(
            """INSERT INTO configs (config_id, parameters, is_active, created_at) VALUES (?, ?, ?, ?)""",
            (config_id_2, json.dumps(config_data_2), 0, datetime.now().isoformat()),
        )

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

        cursor.execute(
            """INSERT INTO models (model_id, job_id, model_path, created_at, is_active) VALUES (?, ?, ?, ?, ?)""",
            (model_id_1, job_id, model_path_1, datetime.now().isoformat(), 1),
        )
        cursor.execute(
            """INSERT INTO models (model_id, job_id, model_path, created_at, is_active) VALUES (?, ?, ?, ?, ?)""",
            (model_id_2, job_id, model_path_2, datetime.now().isoformat(), 0),
        )

        # --- Результаты тренировки (для метрик) ---
        result_id_1 = "tr-result-1"
        result_id_2 = "tr-result-2"
        metrics_data_1 = {"mape": 9.9, "val_loss": 0.08}
        metrics_data_2 = {"mape": 9.8, "val_loss": 0.03}
        cursor.execute(
            """INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics) VALUES (?, ?, ?, ?, ?)""",
            (
                result_id_1,
                job_id,
                model_id_1,
                config_id_1,
                json.dumps(metrics_data_1, default=json_default_serializer),
            ),
        )
        cursor.execute(
            """INSERT INTO training_results (result_id, job_id, model_id, config_id, metrics) VALUES (?, ?, ?, ?, ?)""",
            (
                result_id_2,
                job_id,
                model_id_2,
                config_id_2,
                json.dumps(metrics_data_2, default=json_default_serializer),
            ),
        )

        # --- Запись о job ---
        cursor.execute(
            """INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)""",
            (
                job_id,
                "training",
                "completed",
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )

        conn.commit()

        setup_data = {
            "temp_dir_path": temp_dir.name,
            "db_path": db_path,
            "conn": conn,
            "job_id": job_id,
            "model_id_1": model_id_1,
            "model_id_2": model_id_2,
            "config_id_1": config_id_1,
            "config_id_2": config_id_2,
        }

        yield setup_data

    finally:
        try:
            conn.close()
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


@pytest.fixture
def mocked_db(in_memory_db, monkeypatch):
    """
    Mocks various database functions to use the in-memory SQLite database.
    This fixture provides a central point to control database interactions for tests.
    """
    conn = in_memory_db["conn"]

    # Patch database module functions to use the in-memory connection
    monkeypatch.setattr(
        "deployment.app.db.database.get_db_connection", lambda *args, **kwargs: conn
    )
    monkeypatch.setattr("deployment.app.db.database.execute_query",
                        lambda query, params=(), fetchall=False, connection=None:
                            execute_query(query, params, fetchall, connection or conn))
    monkeypatch.setattr("deployment.app.db.database.execute_many",
                        lambda query, params_list, connection=None:
                            execute_many(query, params_list, connection or conn))

    # Patch db_transaction context manager to use the in-memory connection
    @contextmanager
    def mock_db_transaction(db_path_or_conn=None):
        yield conn
        conn.commit()

    monkeypatch.setattr(
        "deployment.app.db.database.db_transaction", mock_db_transaction
    )

    # Return the wrapped functions from in_memory_db
    _ = {
        "conn": conn,
        "create_job": in_memory_db["create_job"],
        "update_job_status": in_memory_db["update_job_status"], # Use the mocked update_job_status
        "get_job": in_memory_db["get_job_from_db"], # Ensure this mapping is correct
        "create_model": in_memory_db["create_model"],
        "create_training_result": create_training_result, # Use real implementation
        "create_prediction_result": create_prediction_result,
        "create_or_get_config": db_create_or_get_config,
        "get_active_config": get_active_config,
        "set_config_active": set_config_active,
        "set_model_active": set_model_active,
        "delete_model_record_and_file": delete_model_record_and_file,
        "create_job_status_history": in_memory_db["create_job_status_history"],
        "execute_raw_query": execute_query,
        "get_model": in_memory_db["get_model_from_db"], # Exposing get_model_from_db as get_model for convenience in some tests
        "get_all_models": get_all_models, # Exposing get_all_models
        "get_recent_models": get_recent_models, # Exposing get_recent_models
        "get_best_model_by_metric": get_best_model_by_metric, # Exposing get_best_model_by_metric
        "get_top_configs": get_top_configs, # Exposing get_top_configs
        "get_effective_config": get_effective_config, # Exposing get_effective_config
        "auto_activate_best_config_if_enabled": auto_activate_best_config_if_enabled,
        "auto_activate_best_model_if_enabled": auto_activate_best_model_if_enabled,
        "list_jobs": list_jobs,
        "update_processing_run": update_processing_run,
        "create_processing_run": create_processing_run,
        "get_prediction_result": get_prediction_result,
        "get_training_results": get_training_results,
        "get_report_result": get_report_result,
        "get_data_upload_result": get_data_upload_result,
        "create_data_upload_result": create_data_upload_result,
        "get_or_create_multiindex_id": get_or_create_multiindex_id,
        "execute_many": execute_many,
        "insert_retry_event": insert_retry_event,
        "fetch_recent_retry_events": fetch_recent_retry_events,
        "create_tuning_result": create_tuning_result,
        "get_tuning_results": get_tuning_results,
}
    return _

@pytest.fixture(scope="function", autouse=True)
def clean_retry_events_table():
    """
    Autouse fixture to clean the retry_events table before and after each test.
    Ensures test isolation for operations that persist to this table.
    """
    from deployment.app.db.database import get_db_connection
    from deployment.app.db.schema import init_db # Import init_db

    conn = get_db_connection()
    try:
        init_db(connection=conn) # Ensure schema is initialized for this connection
        cursor = conn.cursor()
        cursor.execute("DELETE FROM retry_events")
        conn.commit()
        yield # Yield control to the test function
    finally:
        # Clean up after the test as well
        cursor = conn.cursor()
        cursor.execute("DELETE FROM retry_events")
        conn.commit()
        conn.close()


@pytest.fixture(scope="session", autouse=True)
def set_session_db_path(session_monkeypatch, tmp_path_factory):
    """
    Session-scoped fixture to set a temporary file-based database path for AppSettings.
    Ensures all database interactions use an isolated DB for the entire test session.
    """
    import os
    import sqlite3
    import tempfile
    from passlib.context import CryptContext
    from deployment.app.config import (
        AppSettings,
        APISettings,
        DatabaseSettings,
        DataSphereSettings,
        DataRetentionSettings,
        TuningSettings,
        get_settings,
    )
    from deployment.app.db.schema import init_db

    # Use a real temporary directory for the session
    temp_dir = tmp_path_factory.mktemp("session_data")
    session_db_path = temp_dir / "session_test.db"
    models_dir = temp_dir / "models"
    logs_dir = temp_dir / "logs"
    temp_upload_dir = temp_dir / "temp_uploads"
    
    # Create directories
    models_dir.mkdir()
    logs_dir.mkdir()
    temp_upload_dir.mkdir()

    # Initialize schema in this session-scoped temporary DB
    with sqlite3.connect(session_db_path) as conn:
        init_db(connection=conn)

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
    Creates a temporary SQLite database file with the full schema and yields a dict with paths and a live connection.
    Cleans up after use.
    """
    import tempfile
    temp_dir_path = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir_path, "test_temp.db")
    conn = sqlite3.connect(db_path)
    try:
        init_db(connection=conn)
        conn.row_factory = dict_factory
        yield {
            "temp_dir_path": temp_dir_path,
            "db_path": db_path,
            "conn": conn,
        }
    finally:
        conn.close()
        gc.collect()
        shutil.rmtree(temp_dir_path, ignore_errors=True)

@pytest.fixture
def temp_db_with_data(temp_db):
    """
    Like temp_db, but pre-populates jobs, models, and configs. Provides all IDs needed for most test contracts.
    """
    conn = temp_db["conn"]
    job_id = str(uuid.uuid4())
    model_id = str(uuid.uuid4())
    job_for_model_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    # Create two jobs
    conn.execute(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, "training", "pending", now, now),
    )
    conn.execute(
        "INSERT INTO jobs (job_id, job_type, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (job_for_model_id, "training", "pending", now, now),
    )
    # Create a model for job_for_model_id
    conn.execute(
        "INSERT INTO models (model_id, job_id, model_path, is_active, created_at) VALUES (?, ?, ?, ?, ?)",
        (model_id, job_for_model_id, "/fake/path/model.onnx", 1, now),
    )
    # Create a config
    config_dict = {"input_chunk_length": 12, "output_chunk_length": 6, "hidden_size": 64}
    config_json = json.dumps(config_dict)
    config_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO configs (config_id, config, is_active, created_at) VALUES (?, ?, ?, ?)",
        (config_id, config_json, 1, now),
    )
    conn.commit()
    temp_db["job_id"] = job_id
    temp_db["model_id"] = model_id
    temp_db["job_for_model_id"] = job_for_model_id
    temp_db["config_id"] = config_id
    yield temp_db

@pytest.fixture
def isolated_db_session():
    """
    Function-scoped fixture that creates a new, isolated SQLite DB file for each test and yields a dict with both the connection and the db_path.
    Use .get('conn') for DB operations, .get('db_path') for path-based tests.
    """
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    db_path = temp_file.name
    temp_file.close()
    conn = sqlite3.connect(db_path)
    try:
        init_db(connection=conn)
        conn.row_factory = dict_factory
        yield {"conn": conn, "db_path": db_path}
    finally:
        conn.close()
        gc.collect()
        try:
            os.remove(db_path)
        except Exception:
            pass

@pytest.fixture
def db_with_sales_data(in_memory_db):
    """Provides a db with controlled sales data for date logic tests."""
    conn = in_memory_db["conn"]

    # get_or_create_multiindex_id needs to be imported or mocked.
    # For simplicity, we'll manually insert into dim_multiindex_mapping.
    multiindex_id = 1
    conn.execute(
        """
        INSERT INTO dim_multiindex_mapping (multiindex_id, barcode, artist, album)
        VALUES (?, ?, ?, ?)
        """,
        (multiindex_id, "barcode1", "artist1", "album1"),
    )

    sales_data = [
        # --- 2023 ---
        (multiindex_id, "2023-10-31", 10),  # Full month
        (multiindex_id, "2023-11-01", 5),  # Incomplete month
        (multiindex_id, "2023-11-02", 5),
    ]

    conn.executemany(
        "INSERT INTO fact_sales (multiindex_id, data_date, value) VALUES (?, ?, ?)",
        sales_data,
    )
    conn.commit()
    return conn

# NOTE: All DB fixtures yield dicts. Use .get('conn') for DB operations, .get('db_path') for path-based tests.
