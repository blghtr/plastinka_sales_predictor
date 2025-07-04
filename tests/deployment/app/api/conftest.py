import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Correctly resolve the project root (five levels up: api -> app -> deployment -> tests -> repo root)
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))


# --- Mocks for API dependencies ---
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
    mock.return_value = 5
    return mock


@pytest.fixture(scope="session")
def mock_cleanup_old_historical_data_fixture():
    """Mock for cleanup_old_historical_data function."""
    mock = MagicMock()
    mock.return_value = 3
    return mock


@pytest.fixture(scope="session")
def mock_cleanup_old_models_fixture():
    """Mock for cleanup_old_models function."""
    mock = MagicMock()
    mock.return_value = 2
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
    This fixture is now localized to API tests to prevent conflicts.
    """
    session_monkeypatch.setenv("API_X_API_KEY", "test_x_api_key_conftest")
    session_monkeypatch.setenv("API_API_KEY", "test_token")

    # Clear cached settings so that new environment variables are picked up
    from deployment.app.config import get_settings
    get_settings.cache_clear()

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
        from deployment.app.main import app

        app.dependency_overrides.clear()
        yield TestClient(app)
        app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def client(base_client):
    """Function-scoped client that reuses the session-scoped base_client."""
    return base_client


# --- Fixture to reset mocks between tests ---


@pytest.fixture(scope="function", autouse=True)
def reset_api_mocks_between_tests(
    mock_run_cleanup_job_fixture,
    mock_cleanup_old_predictions_fixture,
    mock_cleanup_old_historical_data_fixture,
    mock_cleanup_old_models_fixture,
    mock_db_conn_fixture,
):
    """
    Auto-use fixture that resets all session-scoped API mocks between tests.
    """
    mock_run_cleanup_job_fixture.reset_mock()
    mock_cleanup_old_predictions_fixture.reset_mock()
    mock_cleanup_old_historical_data_fixture.reset_mock()
    mock_cleanup_old_models_fixture.reset_mock()
    mock_db_conn_fixture.reset_mock()
    yield
    mock_run_cleanup_job_fixture.reset_mock()
    mock_cleanup_old_predictions_fixture.reset_mock()
    mock_cleanup_old_historical_data_fixture.reset_mock()
    mock_cleanup_old_models_fixture.reset_mock()
    mock_db_conn_fixture.reset_mock()


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
