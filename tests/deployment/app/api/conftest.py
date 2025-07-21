import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime
import types
import os
from types import SimpleNamespace
import json

import pytest
from fastapi.testclient import TestClient

from deployment.app.db.schema import SCHEMA_SQL, init_db
from deployment.app.db.database import dict_factory, get_db_connection as original_get_db_connection

from deployment.app.dependencies import get_dal, get_dal_system, get_dal_for_general_user
from deployment.app.db.data_access_layer import DataAccessLayer, UserRoles

# Correctly resolve the project root (five levels up: api -> app -> deployment -> tests -> repo root)
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

# Disable pyfakefs for API tests explicitly if it's active
# This prevents the "TypeError: unsupported operand type(s) for |: 'type' and 'FakePathlibPathModule'"
# by ensuring pathlib.Path is not patched by pyfakefs
def pytest_configure(config):
    if hasattr(config, '_pyfakefs_patcher'):
        config._pyfakefs_patcher.stop()


# --- Mocks for API dependencies ---
@pytest.fixture(scope="session")
def mock_get_db_connection_for_api_tests(session_monkeypatch):
    """
    Patches get_db_connection to return an in-memory, schema-initialized database connection.
    This ensures API tests interact with a clean, isolated database.
    """
    def mock_db_connection(*args, **kwargs):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = dict_factory
        init_db(connection=conn)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.commit()
        return conn

    session_monkeypatch.setattr("deployment.app.db.database.get_db_connection", mock_db_connection)


@pytest.fixture(scope="function")
def mock_dal():
    """
    Мок DAL, возвращающий значения строго по контракту тестов (см. test_failure_contracts.md).
    """
    dal = MagicMock()

    # --- Список всех нужных таблиц для health check ---
    all_tables = [
        'jobs', 'models', 'configs', 'training_results', 'prediction_results',
        'job_status_history', 'dim_multiindex_mapping', 'fact_sales', 'fact_stock',
        'fact_prices', 'fact_stock_changes', 'fact_predictions',
        'processing_runs', 'data_upload_results', 'report_results'
    ]

    # --- create_job ---
    def create_job_side_effect(*args, **kwargs):
        # Если явно задан return_value через mock, возвращать его
        if hasattr(dal.create_job, 'return_value') and dal.create_job.return_value not in (None, create_job_side_effect):
            val = dal.create_job.return_value
            dal.create_job.return_value = create_job_side_effect  # сбросить, чтобы не влиять на другие тесты
            return val
        return kwargs.get("job_id", "test-job-id")
    dal.create_job.side_effect = create_job_side_effect

    # --- create_or_get_config ---
    dal.create_or_get_config.side_effect = lambda *args, **kwargs: kwargs.get("config_id", "uploaded-config")

    # --- create_model_record ---
    dal.create_model_record.return_value = None

    # --- get_active_model_primary_metric ---
    dal.get_active_model_primary_metric.return_value = 0.8  # Return a healthy metric value (above 0.5 threshold)

    # --- list_jobs ---
    dal.list_jobs.return_value = [
        {"job_id": "test-job-id", "job_type": "training", "status": "completed", "created_at": str(datetime.now()), "parameters": {}, "progress": 100},
        {"job_id": "test-job-id-2", "job_type": "data-upload", "status": "pending", "created_at": str(datetime.now()), "parameters": {}, "progress": 0},
    ]

    # --- get_configs ---
    dal.get_configs.return_value = [
        {"config_id": "cfg-1", "config": {"param": 1}, "created_at": str(datetime.now()), "is_active": True},
        {"config_id": "cfg-2", "config": {"param": 2}, "created_at": str(datetime.now()), "is_active": False},
    ]

    # --- get_recent_models ---
    dal.get_recent_models.return_value = [
        {"model_id": "model-1", "job_id": "test-job-id", "model_path": "/fake/model1.onnx", "created_at": str(datetime.now()), "metadata": {}, "is_active": True},
        {"model_id": "model-2", "job_id": "test-job-id-2", "model_path": "/fake/model2.onnx", "created_at": str(datetime.now()), "metadata": {}, "is_active": False},
    ]

    # --- get_all_models ---
    dal.get_all_models.return_value = [
        {"model_id": "model-1", "job_id": "test-job-id", "model_path": "/fake/model1.onnx", "created_at": str(datetime.now()), "metadata": {}, "is_active": True}
    ]

    # --- delete_configs_by_ids ---
    dal.delete_configs_by_ids.return_value = {"successful": 1, "failed": 0, "errors": []}

    # --- delete_models_by_ids ---
    dal.delete_models_by_ids.return_value = {"successful": 1, "failed": 0, "errors": []}

    # --- get_job ---
    def get_job_side_effect(job_id, **kwargs):
        # Если явно задан return_value через mock, возвращать его
        if hasattr(dal.get_job, 'return_value') and dal.get_job.return_value is not None:
            return dal.get_job.return_value
        if job_id in ("not-found", "non-existent", "invalid-id-format"):
            # Возвращаем None для несуществующих, но можно вернуть dict с нужными ключами, чтобы не было KeyError
            return None
        # Возвращаем полный dict для валидных job_id
        return {"job_id": job_id, "status": "completed", "job_type": "training", "parameters": {}, "progress": 100, "created_at": str(datetime.now())}
    dal.get_job.side_effect = get_job_side_effect

    # --- get_training_result ---
    training_base_data = {
        "job_id": "job1",
        "model_id": "model-abc",
        "config_id": "config1",
        "metrics": {"val_MIC": 0.9},
        "parameters": {"param1": "value1"},
        "duration": 120,
        "created_at": str(datetime.now()),
    }

    def get_training_results_side_effect_func(result_id=None, **kwargs):
        if result_id:
            if result_id == "not-found":
                return None
            return {**training_base_data, "result_id": result_id}
        return [{**training_base_data, "result_id": "result-1"}]
    dal.get_training_results.side_effect = get_training_results_side_effect_func

    # --- get_tuning_result ---
    tuning_base_data = {
        "job_id": "job-tune-1",
        "config_id": "config-tune-1",
        "metrics": {"val_loss": 0.1},
        "duration": 250,
        "created_at": str(datetime.now()),
    }

    def get_tuning_results_side_effect_func(result_id=None, **kwargs):
        if result_id:
            if result_id == "not-found":
                return None
            return {**tuning_base_data, "result_id": result_id}
        return [{**tuning_base_data, "result_id": "tuning-res-1"}]
    dal.get_tuning_results.side_effect = get_tuning_results_side_effect_func

    # --- get_report_result ---
    dal.get_report_result.return_value = {"result_id": "report-1", "report_type": "prediction_report", "prediction_month": "2023-01-01", "records_count": 100, "csv_data": "header1,header2\nvalue1,value2", "has_enriched_metrics": True, "enriched_columns": "[]", "generated_at": str(datetime.now()), "filters_applied": "{}", "parameters": "{}", "output_path": "/fake/path"}

    # --- get_active_model ---
    dal.get_active_model.return_value = {"model_id": "model-1", "model_path": "/fake/model1.onnx", "metadata": {}}

    # --- get_effective_config ---
    dal.get_effective_config.return_value = {"config_id": "cfg-1", "config": {"param": 1}}

    # --- get_active_config ---
    dal.get_active_config.return_value = {"config_id": "cfg-1", "config": {"param": 1}}

    # --- get_best_config_by_metric ---
    dal.get_best_config_by_metric.return_value = {"config_id": "cfg-1", "config": {"param": 1}, "metrics": {"val_MIC": 0.9}}

    # --- get_best_model_by_metric ---
    dal.get_best_model_by_metric.return_value = {"model_id": "model-1", "model_path": "/fake/model1.onnx", "metadata": {}, "metrics": {"val_MIC": 0.9}}

    # --- get_prediction_result ---
    dal.get_prediction_result.return_value = {"result_id": "pred-1", "summary_metrics": {"mae": 1.0}}

    # --- create_data_upload_result ---
    dal.create_data_upload_result.return_value = "result-upload-1"

    # --- create_training_result ---
    dal.create_training_result.return_value = "result-train-1"

    # --- create_prediction_result ---
    dal.create_prediction_result.return_value = "result-pred-1"

    # --- create_report_result ---
    dal.create_report_result.return_value = "result-report-1"

    # --- set_config_active ---
    dal.set_config_active.return_value = True

    # --- set_model_active ---
    dal.set_model_active.return_value = True

    # --- update_job_status ---
    dal.update_job_status.return_value = None

    # --- insert_retry_event ---
    dal.insert_retry_event.return_value = None

    # --- fetch_recent_retry_events ---
    dal.fetch_recent_retry_events.return_value = [
        {"timestamp": str(datetime.now()), "component": "test", "operation": "retry", "successful": 1}
    ]

    # --- execute_raw_query ---
    def execute_raw_query_side_effect(query, params=(), fetchall=False, connection=None):
        query_upper = query.upper()
        if "SELECT 1" in query_upper:
            return [{"1": 1}]
        
        if "SQLITE_MASTER" in query_upper and "NAME IN" in query_upper:
            # When check_database asks for specific tables, return them from all_tables
            # based on the parameters it passes.
            # This allows individual health tests to control which tables are "found"
            requested_tables = list(params)
            found_tables = [t for t in all_tables if t in requested_tables]
            return [{"name": t} for t in found_tables]
        
        # For monotonic checks, ensure 'data_date' is returned in correct format
        if ("FACT_SALES" in query_upper or "FACT_STOCK_CHANGES" in query_upper) and "DATA_DATE" in query_upper:
            if "DISTINCT DATA_DATE" in query_upper:
                # Return consecutive months without gaps for healthy scenario
                return [{"data_date": f"2024-{i:02d}-01"} for i in range(1, 4)]
        
        return []
    dal.execute_raw_query.side_effect = execute_raw_query_side_effect

    # --- execute_query_with_batching ---
    def execute_query_with_batching_side_effect(query_template, ids, batch_size=None, connection=None, fetchall=True, placeholder_name="placeholders"):
        """Mock for execute_query_with_batching that handles table existence checks."""
        query_upper = query_template.upper()
        
        if "SQLITE_MASTER" in query_upper and "NAME IN" in query_upper:
            # For table existence checks, return all requested tables that exist in all_tables
            requested_tables = list(ids)
            found_tables = [t for t in all_tables if t in requested_tables]
            return [{"name": t} for t in found_tables]
        
        # For other queries, return empty list
        return []
    dal.execute_query_with_batching.side_effect = execute_query_with_batching_side_effect

    # --- auto_activate_best_config_if_enabled ---
    dal.auto_activate_best_config_if_enabled.return_value = True

    # --- auto_activate_best_model_if_enabled ---
    dal.auto_activate_best_model_if_enabled.return_value = True

    # --- Ошибочные сценарии (side_effect) ---
    # Для тестов, где ожидается ошибка, можно добавить:
    # dal.create_job.side_effect = Exception("DB creation failed")
    # dal.get_job.side_effect = lambda job_id, **kwargs: None if job_id == "not-found" else {...}
    # и т.д.

    return dal

@pytest.fixture(scope="session")
def mock_run_cleanup_job_fixture():
    """Mock for run_cleanup_job function."""
    mock = MagicMock()
    mock.return_value = None
    return mock


@pytest.fixture(scope="session")
def mock_cleanup_old_predictions_fixture():
    """Mock for cleanup_old_predictions function."""
    mock = MagicMock()
    mock.return_value = 5  # Return some records removed count
    return mock


@pytest.fixture(scope="session")
def mock_cleanup_old_historical_data_fixture():
    """Mock for cleanup_old_historical_data function."""
    mock = MagicMock()
    mock.return_value = 3  # Return some records removed count
    return mock


@pytest.fixture(scope="session")
def mock_cleanup_old_models_fixture():
    """Mock for cleanup_old_models function."""
    mock = MagicMock()
    mock.return_value = 2  # Return some records removed count
    return mock


@pytest.fixture(scope="session")
def mock_db_conn_fixture():
    return MagicMock(spec=sqlite3.Connection)


# --- Main FastAPI Test Client Fixture ---


@pytest.fixture(scope="session")
def session_monkeypatch():
    """Session-scoped monkeypatch fixture."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="function", autouse=True)
def mock_settings(monkeypatch, tmp_path_factory):
    """
    Автоматически мокает get_settings для всех тестов API, чтобы возвращать структуру с вложенными атрибутами.
    """
    tmp_dir = tmp_path_factory.mktemp("mock_settings")
    models_dir = str(tmp_dir / "models")
    logs_dir = str(tmp_dir / "logs")
    temp_upload_dir = str(tmp_dir / "temp_uploads")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(temp_upload_dir, exist_ok=True)
    api = SimpleNamespace(log_level="INFO", allowed_origins=["*"], x_api_key="test-key")
    settings = SimpleNamespace(
        api=api,
        models_dir=models_dir,
        logs_dir=logs_dir,
        temp_upload_dir=temp_upload_dir,
        database_path=str(tmp_dir / "test.db"),
        default_metric="val_MIC",
        default_metric_higher_is_better=True,
        auto_select_best_configs=False,
        auto_select_best_model=False,
        sqlite_max_variables=900, # Added for batching fix
        metric_thesh_for_health_check=0.5, # Added for health check mock
    )
    monkeypatch.setattr("deployment.app.config.get_settings", lambda: settings)
    return settings


@pytest.fixture(scope="function")
def base_client(
    session_monkeypatch,
    mock_run_cleanup_job_fixture,
    mock_cleanup_old_predictions_fixture,
    mock_cleanup_old_historical_data_fixture,
    mock_cleanup_old_models_fixture,
    mock_get_db_connection_for_api_tests,
    mock_dal,
    mock_settings,  # <--- для порядка
):
    """
    Session-scoped FastAPI test client for maximum performance.
    This fixture is now localized to API tests to prevent conflicts.
    """
    # CRITICAL: Set environment variables BEFORE the app or settings are imported.
    session_monkeypatch.setenv("API_X_API_KEY", "test_x_api_key_conftest")
    session_monkeypatch.setenv("API_ADMIN_API_KEY", "test_admin_token")

    # Clear cached settings so that new environment variables are picked up
    from deployment.app.config import get_settings
    print("DIAG get_settings:", get_settings, "id:", id(get_settings), "module:", getattr(get_settings, "__module__", None), "type:", type(get_settings), "hasattr cache_clear:", hasattr(get_settings, "cache_clear"))
    if hasattr(get_settings, "cache_clear"):
        get_settings.cache_clear()

    # Set the encryption key for settings for the duration of the test session
    # settings = get_settings() # This line is removed as it's no longer needed after encryption removal

    # Replace unittest.mock.patch with monkeypatch.setattr
    session_monkeypatch.setattr("deployment.app.api.admin.run_cleanup_job", mock_run_cleanup_job_fixture)
    session_monkeypatch.setattr("deployment.app.api.admin.cleanup_old_predictions", mock_cleanup_old_predictions_fixture)
    session_monkeypatch.setattr("deployment.app.api.admin.cleanup_old_historical_data", mock_cleanup_old_historical_data_fixture)
    session_monkeypatch.setattr("deployment.app.api.admin.cleanup_old_models", mock_cleanup_old_models_fixture)


    from deployment.app.main import app
    from deployment.app.dependencies import get_dal as get_dal_dependency, get_dal_system as get_dal_system_dependency, get_dal_for_general_user as get_dal_for_general_user_dependency
    from deployment.app.utils.error_handling import configure_error_handlers
    configure_error_handlers(app)

    app.dependency_overrides.clear()
    # Override the get_dal dependency to return our mock
    app.dependency_overrides[get_dal_dependency] = lambda: mock_dal
    app.dependency_overrides[get_dal_system_dependency] = lambda: mock_dal # For health checks
    app.dependency_overrides[get_dal_for_general_user_dependency] = lambda: mock_dal

    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(base_client, mock_settings):  # <--- для порядка
    return base_client


# --- Fixture to reset mocks between tests ---


@pytest.fixture(scope="function", autouse=True)
def reset_api_mocks_between_tests(
    mock_run_cleanup_job_fixture,
    mock_cleanup_old_predictions_fixture,
    mock_cleanup_old_historical_data_fixture,
    mock_cleanup_old_models_fixture,
    mock_db_conn_fixture,
    mock_dal
):
    """
    Auto-use fixture that resets all session-scoped API mocks between tests.
    """
    # Сбросить return_value для create_job
    mock_dal.create_job.return_value = None
    mock_run_cleanup_job_fixture.reset_mock()
    mock_cleanup_old_predictions_fixture.reset_mock()
    mock_cleanup_old_historical_data_fixture.reset_mock()
    mock_cleanup_old_models_fixture.reset_mock()
    mock_db_conn_fixture.reset_mock()
    mock_dal.reset_mock()
    yield
    mock_run_cleanup_job_fixture.reset_mock()
    mock_cleanup_old_predictions_fixture.reset_mock()
    mock_cleanup_old_historical_data_fixture.reset_mock()
    mock_cleanup_old_models_fixture.reset_mock()
    mock_db_conn_fixture.reset_mock()
    mock_dal.reset_mock()


# --- Other Helper Fixtures for API tests ---


@pytest.fixture(scope="function")
def mock_x_api_key(monkeypatch):
    """A fixture to mock settings.api.x_api_key dynamically for tests."""

    def _set_x_api_key(value: str | None):
        from deployment.app.config import get_settings

        settings = get_settings()
        monkeypatch.setattr(settings.api, "x_api_key", value)

    yield _set_x_api_key


@pytest.fixture(scope="function", autouse=True)
def mock_retry_monitor_api(monkeypatch):
    """
    Auto-use fixture to mock retry_monitor for all API tests.
    """
    mock_module = MagicMock()
    mock_module.DEFAULT_PERSISTENCE_PATH = None
    mock_module.record_retry = MagicMock()
    mock_module.get_retry_statistics = MagicMock(return_value={"total_retries": 0})
    mock_module.reset_retry_statistics = MagicMock(return_value={})
    monkeypatch.setitem(sys.modules, "deployment.app.utils.retry_monitor", mock_module)
    yield mock_module
