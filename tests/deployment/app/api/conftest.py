import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Correctly resolve the project root (five levels up: api -> app -> deployment -> tests -> repo root)
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

# Disable pyfakefs for API tests explicitly if it's active
# This prevents the "TypeError: unsupported operand type(s) for |: 'type' and 'FakePathlibPathModule'"
# by ensuring pathlib.Path is not patched by pyfakefs
def pytest_configure(config):
    if hasattr(config, '_pyfakefs_patcher'):
        config._pyfakefs_patcher.stop()
    # Patch configure_logging to prevent FileNotFoundError during tests
    from unittest.mock import MagicMock

    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    mp.setattr("deployment.app.logger_config.configure_logging", MagicMock())
    config.add_cleanup(mp.undo)


# --- Mocks for API dependencies ---



@pytest.fixture(scope="function")
def api_client(
    session_monkeypatch,
    dal,
):
    """
    Provides a FastAPI TestClient with a fully initialized PostgreSQL database for API tests.
    This fixture ensures that each test runs with a clean, isolated database.
    Uses async DAL from tests/deployment/app/db/conftest.py.
    """
    # Set necessary environment variables for authentication
    admin_raw = "test_admin_token"
    admin_hash = CryptContext(schemes=["bcrypt"]).hash(admin_raw)
    session_monkeypatch.setenv("API_ADMIN_API_KEY_HASH", admin_hash)

    x_api_hash = CryptContext(schemes=["bcrypt"]).hash("test_x_api_key_conftest")
    session_monkeypatch.setenv("API_X_API_KEY_HASH", x_api_hash)

    # Patch logger to avoid file system access during tests
    session_monkeypatch.setattr("deployment.app.logger_config.configure_logging", lambda: None)

    # Mock database initialization to prevent startup errors
    # The app tries to initialize DB pool on startup, but we use test fixtures instead
    async def mock_init_db_pool():
        return None
    
    async def mock_close_db_pool():
        pass
    
    async def mock_init_postgres_schema(pool):
        return True
    
    # Replace lifespan with empty function to prevent DB initialization
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def empty_lifespan(app):
        # No-op lifespan for tests
        yield
    
    session_monkeypatch.setattr("deployment.app.db.connection.init_db_pool", mock_init_db_pool)
    session_monkeypatch.setattr("deployment.app.db.connection.close_db_pool", mock_close_db_pool)
    session_monkeypatch.setattr("deployment.app.db.schema_postgresql.init_postgres_schema", mock_init_postgres_schema)
    
    # Import app and dependencies BEFORE patching lifespan to avoid import-time execution
    from deployment.app.dependencies import (
        get_dal,
        get_dal_for_general_user,
        get_dal_for_admin_user,
        get_dal_system,
    )
    from deployment.app.main import app
    
    # Patch lifespan AFTER app is imported to avoid execution during import
    session_monkeypatch.setattr("deployment.app.main.lifespan", empty_lifespan)
    # Also patch the app's lifespan attribute directly
    app.router.lifespan_context = empty_lifespan

    # Override dependencies to use the *same* isolated async DAL instance for the whole test
    # Dependencies return AsyncGenerator, so we need to create async generator functions
    async def override_get_dal():
        yield dal
    
    async def override_get_dal_system():
        yield dal
    
    async def override_get_dal_for_general_user(api_key_valid=None):
        yield dal
    
    async def override_get_dal_for_admin_user(admin_token=None):
        yield dal
    
    app.dependency_overrides[get_dal] = override_get_dal
    app.dependency_overrides[get_dal_system] = override_get_dal_system
    app.dependency_overrides[get_dal_for_general_user] = override_get_dal_for_general_user
    app.dependency_overrides[get_dal_for_admin_user] = override_get_dal_for_admin_user

    # Yield the test api_client
    # Note: lifespan is mocked to be empty, so no DB initialization happens
    with TestClient(app) as api_client:
        yield api_client

    # Clean up dependency overrides after the test
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def async_api_client(
    session_monkeypatch,
    dal,
):
    """
    Provides an httpx.AsyncClient for async API tests.
    This fixture ensures proper async/await handling and event loop compatibility.
    Uses ASGITransport to directly test the FastAPI app without network.
    """
    # Set necessary environment variables for authentication
    admin_raw = "test_admin_token"
    admin_hash = CryptContext(schemes=["bcrypt"]).hash(admin_raw)
    session_monkeypatch.setenv("API_ADMIN_API_KEY_HASH", admin_hash)

    x_api_hash = CryptContext(schemes=["bcrypt"]).hash("test_x_api_key_conftest")
    session_monkeypatch.setenv("API_X_API_KEY_HASH", x_api_hash)

    # Patch logger to avoid file system access during tests
    session_monkeypatch.setattr("deployment.app.logger_config.configure_logging", lambda: None)

    # Mock database initialization to prevent startup errors
    async def mock_init_db_pool():
        return None
    
    async def mock_close_db_pool():
        pass
    
    async def mock_init_postgres_schema(pool):
        return True
    
    # Replace lifespan with empty function to prevent DB initialization
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def empty_lifespan(app):
        # No-op lifespan for tests
        yield
    
    session_monkeypatch.setattr("deployment.app.db.connection.init_db_pool", mock_init_db_pool)
    session_monkeypatch.setattr("deployment.app.db.connection.close_db_pool", mock_close_db_pool)
    session_monkeypatch.setattr("deployment.app.db.schema_postgresql.init_postgres_schema", mock_init_postgres_schema)
    
    # Import app and dependencies BEFORE patching lifespan to avoid import-time execution
    from deployment.app.dependencies import (
        get_dal,
        get_dal_for_general_user,
        get_dal_for_admin_user,
        get_dal_system,
    )
    from deployment.app.main import app
    
    # Patch lifespan AFTER app is imported to avoid execution during import
    session_monkeypatch.setattr("deployment.app.main.lifespan", empty_lifespan)
    # Also patch the app's lifespan attribute directly
    app.router.lifespan_context = empty_lifespan

    # Override dependencies to use the *same* isolated async DAL instance for the whole test
    async def override_get_dal():
        yield dal
    
    async def override_get_dal_system():
        yield dal
    
    async def override_get_dal_for_general_user(api_key_valid=None):
        yield dal
    
    async def override_get_dal_for_admin_user(admin_token=None):
        yield dal
    
    app.dependency_overrides[get_dal] = override_get_dal
    app.dependency_overrides[get_dal_system] = override_get_dal_system
    app.dependency_overrides[get_dal_for_general_user] = override_get_dal_for_general_user
    app.dependency_overrides[get_dal_for_admin_user] = override_get_dal_for_admin_user

    # Create async client with ASGITransport for proper async handling
    from httpx import ASGITransport, AsyncClient
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client

    # Clean up dependency overrides after the test
    app.dependency_overrides.clear()




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
