import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
import tempfile
import sqlite3
from pathlib import Path
import sys
from unittest.mock import MagicMock
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials


# Добавляем корневую директорию проекта в sys.path для исправления импортов
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Импорты schema, auth, config остаются здесь, так как они нужны для определения моков
from deployment.app.db.schema import SCHEMA_SQL
# from deployment.app.services.auth import get_admin_user, security # Removed - will be imported in fixture
# from deployment.app.config import settings # Removed - will be imported in fixture

# Импортируем функции-провайдеры зависимостей из admin.py, чтобы использовать их как ключи в overrides
# from deployment.app.api.admin import ( # Removed - will be imported in fixture
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
def client(
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
    print(f"\n[DEBUG] Setting DATABASE_URL to: {patched_db_url}")
    print(f"[DEBUG] Original DATABASE_URL: {os.environ.get('DATABASE_URL')}")
    print(f"[DEBUG] Original DATABASE_PATH: {os.environ.get('DATABASE_PATH')}")
    
    # API_API_KEY is also patched here for consistency
    with patch.dict(os.environ, {
        "DATABASE_URL": patched_db_url,
        "API_API_KEY": "test_token", # For bearer admin auth
        "API_X_API_KEY": "test_x_api_key_conftest" # For X-API-Key header auth
    }):
        print(f"[DEBUG] Inside patch.dict: DATABASE_URL={os.environ.get('DATABASE_URL')}, API_X_API_KEY={os.environ.get('API_X_API_KEY')}")
        
        # Force reload the config module to make sure it picks up the new DATABASE_URL
        import importlib
        if 'deployment.app.config' in sys.modules:
            print("[DEBUG] Reloading deployment.app.config module")
            importlib.reload(sys.modules['deployment.app.config'])
        
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

        # 3. Now import the app and other necessary modules.
        #    By doing this after patching os.environ, internal configurations
        #    (like `settings` in `deployment.app.config`) should use the patched values.
        from deployment.app.main import app
        from deployment.app.services.auth import get_admin_user, bearer_scheme # Renamed from security
        from deployment.app.api.admin import (
            get_run_cleanup_job_dependency,
            get_cleanup_old_predictions_dependency,
            get_cleanup_old_historical_data_dependency,
            get_cleanup_old_models_dependency,
            get_db_connection_dependency
        )
        # Import settings here to ensure it's the version affected by the patch.dict
        from deployment.app.config import settings

        # Ensure the settings object reflects the patched DATABASE_URL and API_KEY
        print(f"[DEBUG] settings.db.url: {settings.db.url}")
        print(f"[DEBUG] settings.db.path: {settings.db.path}")
        print(f"[DEBUG] settings.api.api_key: {settings.api.api_key}")
        print(f"[DEBUG] settings.api.x_api_key: {settings.api.x_api_key}")
        
        assert settings.db.url == patched_db_url, f"Expected DB URL {patched_db_url}, got {settings.db.url}"
        assert settings.api.api_key == "test_token", f"Expected API_API_KEY 'test_token', got {settings.api.api_key}"
        assert settings.api.x_api_key == "test_x_api_key_conftest", f"Expected API_X_API_KEY 'test_x_api_key_conftest', got {settings.api.x_api_key}"


        async def mock_get_admin_user_override(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
            # print("MOCK get_admin_user called") # For debugging
            if credentials.scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme. Bearer token required",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            # Use the settings value which is confirmed to be patched
            if credentials.credentials != settings.api.api_key: 
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

        async def override_provider_for_run_cleanup(): return mock_run_cleanup_job_fixture
        async def override_provider_for_cleanup_predictions(): return mock_cleanup_old_predictions_fixture
        async def override_provider_for_cleanup_historical(): return mock_cleanup_old_historical_data_fixture
        async def override_provider_for_cleanup_models(): return mock_cleanup_old_models_fixture
        async def override_provider_for_db_conn(): return mock_db_conn_fixture

        original_overrides = app.dependency_overrides.copy()
        app.dependency_overrides[get_admin_user] = mock_get_admin_user_override
        app.dependency_overrides[get_run_cleanup_job_dependency] = override_provider_for_run_cleanup
        app.dependency_overrides[get_cleanup_old_predictions_dependency] = override_provider_for_cleanup_predictions
        app.dependency_overrides[get_cleanup_old_historical_data_dependency] = override_provider_for_cleanup_historical
        app.dependency_overrides[get_cleanup_old_models_dependency] = override_provider_for_cleanup_models
        app.dependency_overrides[get_db_connection_dependency] = override_provider_for_db_conn
        
        # Create TestClient with the app instance that has overrides applied
        with TestClient(app) as c:
            yield c # Yield the client for use in tests
        
        # Restore original dependency overrides after the test
        app.dependency_overrides = original_overrides