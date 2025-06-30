import gc
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Добавляем корневую директорию проекта в sys.path для исправления импортов
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Импорты schema, auth, config остаются здесь, так как они нужны для определения моков
from deployment.app.db.schema import SCHEMA_SQL

# from deployment.app.services.auth import get_admin_user, security # Removed - will be imported in fixture
# from deployment.app.config import settings # Removed - will be imported in fixture

# Импортируем функции-провайдеры зависимостей из admin.py, чтобы использовать их как ключи в overrides
# Этот блок больше не нужен, так как зависимости изменились
# from deployment.app.api.admin import (
#     get_run_cleanup_job_dependency,
#     get_cleanup_old_predictions_dependency,
#     get_cleanup_old_historical_data_dependency,
#     get_cleanup_old_models_dependency,
#     get_db_connection_dependency
# )

# Добавляем session-scoped кэш для FastAPI app
_cached_app = None


def get_cached_app():
    """Get or create cached FastAPI app instance for performance optimization."""
    global _cached_app
    if _cached_app is None:
        # Import and create the app with mocked dependencies
        from deployment.app.main import app

        _cached_app = app
    return _cached_app


@pytest.fixture(scope="session")
def cached_fastapi_app(pytestconfig):
    """Session-scoped cached FastAPI app using pytest cache."""
    cache_key = "fastapi_app/instance"
    app_instance = pytestconfig.cache.get(cache_key, None)

    if app_instance is None:
        print("Creating new FastAPI app instance...")
        from deployment.app.main import app

        app_instance = app
        pytestconfig.cache.set(cache_key, app_instance)
    else:
        print("Using cached FastAPI app instance...")

    return app_instance


# --- Фикстуры для моков (остаются session-scoped, если не требуют сброса состояния между тестами)
# --- или function-scoped, если требуют ---
@pytest.fixture(scope="session")
def mock_run_cleanup_job_fixture():
    """Mock for run_cleanup_job function."""
    mock = MagicMock()
    mock.return_value = None  # run_cleanup_job doesn't return anything
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
    # This mock is for dependencies that expect a connection object.
    # If they perform operations, this mock needs to be configured.
    return MagicMock(spec=sqlite3.Connection)


@pytest.fixture
def test_db_path():
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)  # Close the file descriptor, we just need the path
    yield db_path
    # Cleanup the database file after the test
    try:
        os.unlink(db_path)
    except OSError:
        pass  # Ignore errors if file already removed or not created


@pytest.fixture
def isolated_db_session(monkeypatch):
    """
    Provides a completely isolated database session for a single test.

    - Creates a temporary database file.
    - Uses monkeypatch to safely set environment variables (auto-cleanup)
    - Initializes database schema for the isolated database
    - REMOVED: Dangerous global module deletion that causes test pollution

    Yields:
        str: The file path to the temporary database.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = Path(temp_dir) / "test_isolated.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

        from deployment.app.db.database import dict_factory
        from deployment.app.db.schema import init_db

        init_db(str(db_path))

        original_sqlite3_connect = sqlite3.connect

        def patched_connect(*args, **kwargs):
            conn = original_sqlite3_connect(*args, **kwargs)
            conn.row_factory = dict_factory
            return conn

        monkeypatch.setattr("sqlite3.connect", patched_connect)

        yield str(db_path)

    finally:
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_db():
    """
    Provides a clean, in-memory SQLite database with the schema initialized.
    The connection is yielded to the test and automatically closed.
    """
    try:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.commit()
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="function")
def temp_db_with_data(temp_db):
    """
    Provides a clean, in-memory SQLite database populated with a standard
    set of test data for integration testing. Follows foreign key constraints.
    """
    conn = temp_db
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    config_id = str(uuid.uuid4())
    config_params = {"param1": 100, "param2": "value"}
    cursor.execute(
        "INSERT INTO configs (config_id, config, created_at) VALUES (?, ?, ?)",
        (config_id, json.dumps(config_params), now),
    )

    job_for_model_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, config_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job_for_model_id, "training", "completed", config_id, now, now),
    )

    model_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)",
        (model_id, job_for_model_id, "/path/to/model", now),
    )

    job_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, config_id, model_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (job_id, "prediction", "running", config_id, model_id, now, now),
    )

    job_id_2 = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, config_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id_2, "prediction", "pending", config_id, now, now),
    )

    conn.commit()

    return {
        "conn": conn,
        "config_id": config_id,
        "model_id": model_id,
        "job_id": job_id,
        "job_id_2": job_id_2,
        "job_for_model_id": job_for_model_id,
    }


# This is the main fixture for API tests
@pytest.fixture(scope="function", autouse=True)
def reset_mocks_between_tests(
    mock_run_cleanup_job_fixture,
    mock_cleanup_old_predictions_fixture,
    mock_cleanup_old_historical_data_fixture,
    mock_cleanup_old_models_fixture,
    mock_db_conn_fixture,
):
    """
    Auto-use fixture that resets all module-scoped mocks between tests.
    This ensures test isolation while keeping performance benefits.
    """
    # Reset all mocks before each test
    mock_run_cleanup_job_fixture.reset_mock()
    mock_cleanup_old_predictions_fixture.reset_mock()
    mock_cleanup_old_historical_data_fixture.reset_mock()
    mock_cleanup_old_models_fixture.reset_mock()
    mock_db_conn_fixture.reset_mock()

    yield

    # Optionally reset after test as well for extra safety
    mock_run_cleanup_job_fixture.reset_mock()
    mock_cleanup_old_predictions_fixture.reset_mock()
    mock_cleanup_old_historical_data_fixture.reset_mock()
    mock_cleanup_old_models_fixture.reset_mock()
    mock_db_conn_fixture.reset_mock()


@pytest.fixture(scope="session")
def session_monkeypatch():
    """Session-scoped monkeypatch fixture."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def base_client(
    session_monkeypatch,
    mock_run_cleanup_job_fixture,
    mock_cleanup_old_predictions_fixture,
    mock_cleanup_old_historical_data_fixture,
    mock_cleanup_old_models_fixture,
):
    """
    Session-scoped FastAPI test client for maximum performance.

    PERFORMANCE OPTIMIZATION:
    - Session scope: FastAPI app initialized only ONCE per test session
    - All expensive setup (imports, dependency injection) happens once
    - Shared across ALL tests in the session

    ISOLATION MAINTAINED:
    - reset_mocks_between_tests fixture ensures mocks are reset between tests
    - Database operations use isolated test databases
    - No state pollution between tests
    """
    # CRITICAL FIX: Set environment variables BEFORE importing the app
    # This ensures that the app's settings are loaded with the test values.
    session_monkeypatch.setenv("API_X_API_KEY", "test_x_api_key_conftest")
    session_monkeypatch.setenv("API_API_KEY", "test_token")

    # Patch the functions in the module where they are *used* (the admin API module)
    # This is crucial because of how they are imported and used with Depends(lambda:...).
    with (
        patch("deployment.app.api.admin.run_cleanup_job", mock_run_cleanup_job_fixture),
        patch(
            "deployment.app.api.admin.cleanup_old_predictions",
            mock_cleanup_old_predictions_fixture,
        ),
        patch(
            "deployment.app.api.admin.cleanup_old_historical_data",
            mock_cleanup_old_historical_data_fixture,
        ),
        patch(
            "deployment.app.api.admin.cleanup_old_models",
            mock_cleanup_old_models_fixture,
        ),
    ):
        # Import and create the app with mocked dependencies
        # Import settings and main app *after* monkeypatching
        from deployment.app.config import get_config_values, get_settings
        from deployment.app.main import app

        # CRITICAL FIX: Clear caches after setting environment variables
        # The @lru_cache decorator means settings won't be re-evaluated without explicit cache clearing
        get_settings.cache_clear()
        get_config_values.cache_clear()

        # CRITICAL FIX: Clear the dependency overrides on the app object
        # to ensure a clean state for the test client.
        app.dependency_overrides.clear()  # Clear any old overrides

        # CRITICAL FIX: Removed incorrect dependency overrides.
        # The authentication should now work via the correctly configured settings.
        # def override_bearer_auth():
        #     return HTTPAuthorizationCredentials(scheme="Bearer", credentials="test_token")
        #
        # def override_api_key():
        #     return "test_x_api_key_conftest"
        #
        # app.dependency_overrides[HTTPAuthorizationCredentials] = override_bearer_auth
        # app.dependency_overrides[APIKeyHeader(name="X-API-Key")] = override_api_key

        yield TestClient(app)

        # Cleanup
        app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(base_client):
    """
    Function-scoped client that reuses the module-scoped base_client.
    Combined with reset_mocks_between_tests, this provides both performance
    and test isolation.
    """
    return base_client


@pytest.fixture(scope="function")
def mock_x_api_key(monkeypatch):
    """
    A fixture to mock settings.api.x_api_key dynamically for tests.
    Yields a callable that takes the desired value for x_api_key.
    This ensures that the settings object used by the app instance is correctly modified.
    """

    def _set_x_api_key(value: str | None):
        # Import get_settings here to avoid circular dependencies
        from deployment.app.config import get_settings

        settings = get_settings()
        # Use monkeypatch to safely set the attribute for the duration of the test
        monkeypatch.setattr(settings.api, "x_api_key", value)
        print(f"[DEBUG mock_x_api_key] Set settings.api.x_api_key to: {value}")

    yield _set_x_api_key

    print("[DEBUG mock_x_ap_key] Fixture teardown.")


@pytest.fixture(scope="function", autouse=True)
def mock_retry_monitor_global(monkeypatch):
    """
    Global fixture to ensure retry_monitor is properly mocked.
    This fixture is automatically used for all tests to prevent retry_monitor
    from using real files or persistent storage.
    """
    # Create a comprehensive mock
    mock_module = MagicMock()
    mock_module.DEFAULT_PERSISTENCE_PATH = None
    mock_module.record_retry = MagicMock()
    mock_module.get_retry_statistics = MagicMock(
        return_value={
            "total_retries": 0,
            "successful_retries": 0,
            "exhausted_retries": 0,
            "successful_after_retry": 0,
            "high_failure_operations": [],
            "alerted_operations": [],
            "alert_thresholds": {},
            "operation_stats": {},
            "exception_stats": {},
            "timestamp": "2021-01-01T00:00:00",
        }
    )
    mock_module.reset_retry_statistics = MagicMock(return_value={})

    # SAFE APPROACH: Use monkeypatch to replace the module instead of deleting from sys.modules
    # This follows Testing Policy guidelines for safe test isolation
    monkeypatch.setitem(sys.modules, "deployment.app.utils.retry_monitor", mock_module)

    yield mock_module
