import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
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
    in_memory_db,
):
    """
    Provides a FastAPI TestClient with a fully initialized in-memory database for API tests.
    This fixture ensures that each test runs with a clean, isolated database.
    """
    # Set necessary environment variables for authentication
    admin_raw = "test_admin_token"
    admin_hash = CryptContext(schemes=["bcrypt"]).hash(admin_raw)
    session_monkeypatch.setenv("API_ADMIN_API_KEY_HASH", admin_hash)

    x_api_hash = CryptContext(schemes=["bcrypt"]).hash("test_x_api_key_conftest")
    session_monkeypatch.setenv("API_X_API_KEY_HASH", x_api_hash)

    # Patch logger to avoid file system access during tests
    session_monkeypatch.setattr("deployment.app.logger_config.configure_logging", lambda: None)

    # Import app and dependencies within the fixture to use the patched environment
    from deployment.app.dependencies import (
        get_dal,
        get_dal_for_general_user,
        get_dal_for_admin_user,
        get_dal_system,
    )
    from deployment.app.main import app

    # Override dependencies to use the *same* isolated in-memory DAL instance for the whole test
    app.dependency_overrides[get_dal] = lambda: in_memory_db
    app.dependency_overrides[get_dal_system] = lambda: in_memory_db
    app.dependency_overrides[get_dal_for_general_user] = lambda: in_memory_db
    app.dependency_overrides[get_dal_for_admin_user] = lambda: in_memory_db

    # Yield the test api_client
    with TestClient(app) as api_client:
        yield api_client

    # Clean up dependency overrides after the test
    app.dependency_overrides.clear()

    # Explicitly close DAL connections after each test
    # This is crucial for in-memory SQLite to prevent resource leaks
    # and ensure a clean state for the next test.
    # The DAL factory creates a new connection for each request, so we need to close them.
    # This might require modifying the DAL factory to return a context manager
    # or to have a way to track and close all connections created during a test.
    # For now, this is a placeholder for explicit cleanup if needed.




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
