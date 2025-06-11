import os
import pytest
import json
import sqlite3
import tempfile
import pandas as pd
from datetime import datetime
import uuid
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials
import numpy as np
from fastapi import Depends, HTTPException, status, Security
import importlib
import shutil
import gc

# Добавляем корневую директорию проекта в sys.path для исправления импортов
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Импорты schema, auth, config остаются здесь, так как они нужны для определения моков
from deployment.app.db.schema import SCHEMA_SQL
from deployment.app.db.data_retention import (
    run_cleanup_job,
    cleanup_old_predictions,
    cleanup_old_historical_data,
    cleanup_old_models
)
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

# --- Фикстуры для моков (остаются session-scoped, если не требуют сброса состояния между тестами)
# --- или function-scoped, если требуют --- 
@pytest.fixture(scope="function")
def mock_run_cleanup_job_fixture():
    return MagicMock()

@pytest.fixture(scope="function")
def mock_cleanup_old_predictions_fixture():
    return MagicMock()

@pytest.fixture(scope="function")
def mock_cleanup_old_historical_data_fixture():
    return MagicMock()

@pytest.fixture(scope="function")
def mock_cleanup_old_models_fixture():
    return MagicMock()

@pytest.fixture(scope="function")
def mock_db_conn_fixture():
    # This mock is for dependencies that expect a connection object.
    # If they perform operations, this mock needs to be configured.
    return MagicMock(spec=sqlite3.Connection)

@pytest.fixture
def test_db_path():
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd) # Close the file descriptor, we just need the path
    yield db_path
    # Cleanup the database file after the test
    try:
        os.unlink(db_path)
    except OSError:
        pass # Ignore errors if file already removed or not created

@pytest.fixture
def isolated_db_session(monkeypatch):
    """
    Provides a completely isolated database session for a single test.

    - Creates a temporary database file.
    - Sets the DATABASE_URL environment variable to point to this file.
    - Ensures that the application's config and database modules are reloaded
      to use this new database.
    - Cleans up the temporary database file and directory afterward.
    
    Yields:
        str: The file path to the temporary database.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        db_path = Path(temp_dir) / "test_isolated.db"
        
        # Ensure modules are gone before we start
        if "deployment.app.config" in sys.modules:
            del sys.modules["deployment.app.config"]
        if "deployment.app.db.database" in sys.modules:
            del sys.modules["deployment.app.db.database"]

        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

        # Import fresh modules that will use the monkeypatched env var
        from deployment.app.config import settings
        from deployment.app.db.schema import init_db
        from deployment.app.db.database import dict_factory

        # Initialize the schema in the new database
        init_db(str(db_path))
        
        # Ensure that get_db_connection uses the dict_factory
        # This is the key fix - patch sqlite3.connect to set row_factory
        original_sqlite3_connect = sqlite3.connect
        
        def patched_connect(*args, **kwargs):
            conn = original_sqlite3_connect(*args, **kwargs)
            conn.row_factory = dict_factory
            return conn
            
        monkeypatch.setattr('sqlite3.connect', patched_connect)

        yield str(db_path)

    finally:
        # Force garbage collection to release file handles, especially on Windows
        gc.collect()
        # Use shutil.rmtree with ignore_errors=True for robust cleanup
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
    
    Yields:
        dict: A dictionary containing the connection object and the IDs of
              the created test data.
    """
    conn = temp_db
    cursor = conn.cursor()
    
    # --- Create Test Data in Correct Order ---
    now = datetime.now().isoformat()
    
    # 1. Config record
    config_id = str(uuid.uuid4())
    config_params = {"param1": 100, "param2": "value"}
    cursor.execute(
        "INSERT INTO configs (config_id, config, created_at) VALUES (?, ?, ?)",
        (config_id, json.dumps(config_params), now)
    )
    
    # 2. A job that will be used to create a model
    job_for_model_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, config_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job_for_model_id, "training", "completed", config_id, now, now)
    )

    # 3. Model record (depends on a job)
    model_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO models (model_id, job_id, model_path, created_at) VALUES (?, ?, ?, ?)",
        (model_id, job_for_model_id, "/path/to/model", now)
    )

    # 4. A main job for testing (depends on config and model)
    job_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, config_id, model_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (job_id, "prediction", "running", config_id, model_id, now, now)
    )
    
    # 5. Another job for variety
    job_id_2 = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO jobs (job_id, job_type, status, config_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (job_id_2, "prediction", "pending", config_id, now, now)
    )

    conn.commit()
    
    test_data_ids = {
        "conn": conn,
        "config_id": config_id,
        "model_id": model_id,
        "job_id": job_id,
        "job_id_2": job_id_2,
        "job_for_model_id": job_for_model_id
    }
    
    return test_data_ids


# This is the main fixture for API tests
@pytest.fixture
def client(
    monkeypatch,
    mock_run_cleanup_job_fixture, 
    mock_cleanup_old_predictions_fixture,
    mock_cleanup_old_historical_data_fixture,
    mock_cleanup_old_models_fixture,
    mock_db_conn_fixture,
    test_db_path
):
    # 1. Patch environment variables for DATABASE_URL *before* importing the app
    #    This ensures that when `settings` is initialized (possibly at app import time),
    #    it picks up the correct database URL for the test session.
    patched_db_url = f"sqlite:///{test_db_path}"
    print(f"[DEBUG] Setting DATABASE_URL to: {patched_db_url}")
    print(f"[DEBUG] Original DATABASE_URL: {os.environ.get('DATABASE_URL')}")
    print(f"[DEBUG] Original DATABASE_PATH: {os.environ.get('DATABASE_PATH')}")
    
    monkeypatch.setenv("DATABASE_URL", patched_db_url)
    monkeypatch.setenv("API_API_KEY", "test_token")
    monkeypatch.setenv("API_X_API_KEY", "test_x_api_key_conftest")

    # --- NEW: Patch data_retention functions at the source BEFORE app import ---
    # This is the key fix. We replace the functions in their original module.
    # When the admin router imports them, it will get our mocks.
    monkeypatch.setattr(
        "deployment.app.db.data_retention.run_cleanup_job", 
        mock_run_cleanup_job_fixture
    )
    monkeypatch.setattr(
        "deployment.app.db.data_retention.cleanup_old_predictions", 
        mock_cleanup_old_predictions_fixture
    )
    monkeypatch.setattr(
        "deployment.app.db.data_retention.cleanup_old_historical_data", 
        mock_cleanup_old_historical_data_fixture
    )
    monkeypatch.setattr(
        "deployment.app.db.data_retention.cleanup_old_models", 
        mock_cleanup_old_models_fixture
    )
    # The DB connection dependency is different, it's yielded by a function.
    # We will still use dependency_overrides for this one as it's cleaner.
    # --- END NEW ---

    # API_API_KEY is also patched here for consistency
    with patch.dict(os.environ, {}, clear=True): # Clear os.environ to ensure our monkeypatched vars are used
        monkeypatch.setenv("DATABASE_URL", patched_db_url)
        monkeypatch.setenv("API_API_KEY", "test_token")
        monkeypatch.setenv("API_X_API_KEY", "test_x_api_key_conftest")

        print(f"[DEBUG] Inside patch.dict: DATABASE_URL={os.environ.get('DATABASE_URL')}, API_X_API_KEY={os.environ.get('API_X_API_KEY')}")
        
        # 2. Explicitly initialize the schema on the test_db_path.
        # This database file is the one the application will try to connect to
        # if any real (unmocked) database operations occur.
        conn_for_schema_init = sqlite3.connect(test_db_path)
        try:
            print(f"[DEBUG] Initializing schema on: {test_db_path}")
            conn_for_schema_init.executescript(SCHEMA_SQL)
            conn_for_schema_init.commit()
            
            # Verify schema was created properly
            cursor = conn_for_schema_init.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"[DEBUG] Tables created: {tables}")
            
        except sqlite3.Error as e:
            # Using pytest.fail to make it clear that schema initialization is critical
            pytest.fail(f"Failed to initialize schema on {test_db_path}: {e}")
        finally:
            conn_for_schema_init.close()

        # 3. Ensure a fresh import of app and settings by deleting from sys.modules.
        # This will force modules to be reloaded from scratch, picking up new env vars.
        if 'deployment.app.main' in sys.modules:
            del sys.modules['deployment.app.main']
        if 'deployment.app.config' in sys.modules:
            del sys.modules['deployment.app.config']
        if 'deployment.app.services.auth' in sys.modules:
            del sys.modules['deployment.app.services.auth']
        if 'deployment.app.api.health' in sys.modules:
            del sys.modules['deployment.app.api.health']
        if 'deployment.app.api.jobs' in sys.modules:
            del sys.modules['deployment.app.api.jobs']
        if 'deployment.app.api.models_configs' in sys.modules:
            del sys.modules['deployment.app.api.models_configs']
        if 'deployment.app.api.admin' in sys.modules:
            del sys.modules['deployment.app.api.admin']
        if 'deployment.app.utils.error_handling' in sys.modules:
            del sys.modules['deployment.app.utils.error_handling']

        print("[DEBUG conftest] All relevant modules deleted from sys.modules.")
        sys.stdout.flush()

        # Now, import config and immediately update its global 'settings' instance
        import deployment.app.config as app_config
        # Re-instantiate settings object within the config module's scope.
        # This is the key: replace the global settings object that other modules will import.
        app_config.settings = app_config.AppSettings()
        
        # Patch the data_root_dir to control where database_path computes to
        expected_db_path = patched_db_url.replace("sqlite:///", "")
        # Extract the directory part (remove /database/plastinka.db from the path)
        # expected_db_path will be something like C:\Users\Gamer\AppData\Local\Temp\tmpccymf3fo.db
        # We need to set data_root_dir so that data_root_dir/database/plastinka.db == expected_db_path
        import tempfile
        test_data_root = os.path.join(tempfile.gettempdir(), f"plastinka_test_{os.path.basename(expected_db_path).split('.')[0]}")
        # Create the database subdirectory
        db_dir = os.path.join(test_data_root, "database")
        os.makedirs(db_dir, exist_ok=True)
        # Copy the test database to the expected location within the test data root
        target_db_path = os.path.join(db_dir, app_config.settings.db.filename)
        shutil.copy2(expected_db_path, target_db_path)
        
        # Now set data_root_dir to our test data root
        monkeypatch.setattr(app_config.settings, 'data_root_dir', test_data_root)
        
        print(f"[DEBUG conftest] Set data_root_dir to: {test_data_root}")
        print(f"[DEBUG conftest] Expected database_path: {target_db_path}")
        print(f"[DEBUG conftest] Actual database_path: {app_config.settings.database_path}")
        
        # Verify that the newly created settings object has the correct values
        print(f"[DEBUG conftest] New app_config.settings.api.x_api_key: {app_config.settings.api.x_api_key}")
        print(f"[DEBUG conftest] New app_config.settings.api.api_key: {app_config.settings.api.api_key}")
        
        # Now import app, auth, etc. They will import the *updated* app_config.settings
        from deployment.app.main import app
        from deployment.app.services.auth import get_admin_user, bearer_scheme, api_key_header_scheme, get_current_api_key_validated # Import api_key_header_scheme and get_current_api_key_validated
        from deployment.app.api.admin import get_db_conn_and_close # Import the new DB dependency

        print(f"[DEBUG] app_config.settings.database_url: {app_config.settings.database_url}")

        # Assertions to confirm settings are correctly picked up
        # Note: database_url will now point to the computed path within our test data root
        expected_computed_db_url = f"sqlite:///{target_db_path}"
        assert app_config.settings.database_url == expected_computed_db_url, f"Expected DB URL {expected_computed_db_url}, got {app_config.settings.database_url}"
        assert app_config.settings.api.api_key == "test_token", f"Expected API_API_KEY 'test_token', got {app_config.settings.api.api_key}"
        assert app_config.settings.api.x_api_key == "test_x_api_key_conftest", f"Expected API_X_API_KEY 'test_x_api_key_conftest', got {app_config.settings.api.x_api_key}"
        print("[DEBUG conftest] All settings assertions passed. Yielding client to test...")
        sys.stdout.flush()

        async def mock_get_admin_user_override(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
            # print("MOCK get_admin_user called") # For debugging
            if credentials.scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme. Bearer token required",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            # Use the settings value which is confirmed to be patched
            if credentials.credentials != app_config.settings.api.api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return {
                "user_type": "admin",
                "roles": ["admin"],
                "permissions": ["manage_data_retention", "manage_models", "manage_jobs"]
            }

        async def mock_get_current_api_key_validated_override(api_key: str = Security(api_key_header_scheme)) -> bool:
            print(f"[DEBUG MOCK] mock_get_current_api_key_validated_override called with api_key: {api_key}")
            print(f"[DEBUG MOCK] app_config.settings.api.x_api_key: {app_config.settings.api.x_api_key}")
            if not app_config.settings.api.x_api_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="X-API-Key authentication is not configured on the server."
                )
            if not api_key: # This case is handled by auto_error=True in APIKeyHeader if we set it
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not authenticated: X-API-Key header is missing."
                )
                
            if api_key != app_config.settings.api.x_api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid X-API-Key."
                )
            return True

        original_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[get_admin_user] = mock_get_admin_user_override
        # Add override for get_current_api_key_validated
        app.dependency_overrides[get_current_api_key_validated] = mock_get_current_api_key_validated_override

        # The data_retention functions are now patched at the source, so we don't need these overrides.
        # app.dependency_overrides[run_cleanup_job] = mock_run_cleanup_job_fixture
        # app.dependency_overrides[cleanup_old_predictions] = mock_cleanup_old_predictions_fixture
        # app.dependency_overrides[cleanup_old_historical_data] = mock_cleanup_old_historical_data_fixture
        # app.dependency_overrides[cleanup_old_models] = mock_cleanup_old_models_fixture
        
        # We still need to override the get_db_conn_and_close dependency in the admin router
        app.dependency_overrides[get_db_conn_and_close] = lambda: mock_db_conn_fixture

        print("[DEBUG conftest] About to create TestClient(app)...")
        sys.stdout.flush()
        # Create TestClient with the app instance that has overrides applied
        with TestClient(app) as c:
            print("[DEBUG conftest] TestClient(app) context entered. Yielding client...")
            sys.stdout.flush()
            yield c # Yield the client for use in tests

        print("[DEBUG conftest] Client yielded, test finished. Starting client fixture cleanup...")
        sys.stdout.flush()

        # Restore original dependency overrides after the test
        app.dependency_overrides = original_overrides
        print("[DEBUG conftest] Dependency overrides restored.")
        sys.stdout.flush()
        # Explicitly close the TestClient if needed (FastAPI TestClient's 'with' statement handles this)
        # print("[DEBUG conftest] TestClient context manager exited. App should be torn down.")
        # sys.stdout.flush()

        print("[DEBUG conftest] Client fixture cleanup complete.")
        sys.stdout.flush()

@pytest.fixture(scope="function")
def mock_x_api_key(monkeypatch):
    """
    A fixture to mock settings.api.x_api_key dynamically for tests.
    Yields a callable that takes the desired value for x_api_key.
    This ensures that the settings object used by the app instance is correctly modified.
    """
    import deployment.app.config as current_app_config
    # No need to store original_x_api_key explicitly, monkeypatch handles restore for setattr.
    
    def _set_x_api_key(value: str | None):
        # Use setattr on the live settings object within the app_config module
        monkeypatch.setattr(current_app_config.settings.api, "x_api_key", value)
        print(f"[DEBUG mock_x_api_key] Set current_app_config.settings.api.x_api_key to: {value}")

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
    mock_module.get_retry_statistics = MagicMock(return_value={
        "total_retries": 0,
        "successful_retries": 0,
        "exhausted_retries": 0,
        "successful_after_retry": 0,
        "high_failure_operations": [],
        "alerted_operations": [],
        "alert_thresholds": {},
        "operation_stats": {},
        "exception_stats": {},
        "timestamp": "2021-01-01T00:00:00"
    })
    mock_module.reset_retry_statistics = MagicMock(return_value={})
    
    # Ensure this mock is always applied before any imports happen
    monkeypatch.setitem(sys.modules, "deployment.app.utils.retry_monitor", mock_module)
    
    # Also delete any existing import to force re-import
    if "deployment.app.utils.retry_monitor" in sys.modules:
        del sys.modules["deployment.app.utils.retry_monitor"]
        
    yield mock_module