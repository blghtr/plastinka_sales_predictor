import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import time
from datetime import datetime

# Assuming your FastAPI app instance is created in deployment/app/main.py
# Adjust the import path if necessary
from deployment.app.main import app 
from deployment.app.api.health import (
    HealthResponse, 
    SystemStatsResponse, 
    RetryStatsResponse, 
    ComponentHealth, 
    check_database, 
    check_environment, 
    start_time
)
from deployment.app.config import settings # Needed for DB path in check_database test

# Fixture for the TestClient
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# --- Test Helper Functions ---

@patch("app.api.health.sqlite3")
def test_check_database_healthy(mock_sqlite3):
    """Test check_database returns healthy when connection and tables are ok."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    # Simulate finding all required tables
    mock_cursor.fetchall.return_value = [('cloud_functions',), ('cloud_function_executions',), ('cloud_storage_objects',)]
    mock_sqlite3.connect.return_value = mock_conn
    
    result = check_database()
    
    assert result.status == "healthy"
    mock_sqlite3.connect.assert_called_once_with(settings.db.path)
    mock_cursor.execute.assert_any_call("SELECT 1")
    mock_cursor.execute.assert_any_call(
        "SELECT name FROM sqlite_master WHERE type='table' AND ("
        "name='cloud_functions' OR "
        "name='cloud_function_executions' OR "
        "name='cloud_storage_objects'"
        ")"
    )
    mock_conn.close.assert_called_once()

@patch("app.api.health.sqlite3")
def test_check_database_degraded_missing_tables(mock_sqlite3):
    """Test check_database returns degraded if tables are missing."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    # Simulate finding only one table
    mock_cursor.fetchall.return_value = [('cloud_functions',)]
    mock_sqlite3.connect.return_value = mock_conn
    
    result = check_database()
    
    assert result.status == "degraded"
    assert "missing_tables" in result.details
    assert set(result.details["missing_tables"]) == {'cloud_function_executions', 'cloud_storage_objects'}
    mock_conn.close.assert_called_once()

@patch("app.api.health.sqlite3")
def test_check_database_unhealthy_connection_error(mock_sqlite3):
    """Test check_database returns unhealthy on connection error."""
    mock_sqlite3.connect.side_effect = Exception("Connection failed")
    
    result = check_database()
    
    assert result.status == "unhealthy"
    assert "error" in result.details
    assert result.details["error"] == "Connection failed"

@patch.dict(os.environ, {"YANDEX_CLOUD_ACCESS_KEY": "key1", "YANDEX_CLOUD_SECRET_KEY": "key2", "YANDEX_CLOUD_FOLDER_ID": "folder", "YANDEX_CLOUD_API_KEY": "api_key", "CLOUD_CALLBACK_AUTH_TOKEN": "token"})
def test_check_environment_healthy():
    """Test check_environment returns healthy when all vars are set."""
    result = check_environment()
    assert result.status == "healthy"
    assert result.details == {}

@patch.dict(os.environ, {"YANDEX_CLOUD_ACCESS_KEY": "key1", "YANDEX_CLOUD_SECRET_KEY": "key2"}, clear=True) # Clear others
def test_check_environment_degraded():
    """Test check_environment returns degraded when vars are missing."""
    result = check_environment()
    assert result.status == "degraded"
    assert "missing_variables" in result.details
    assert len(result.details["missing_variables"]) == 3
    assert "YANDEX_CLOUD_FOLDER_ID (Cloud Folder ID)" in result.details["missing_variables"]

# --- Test API Endpoints ---

@patch("app.api.health.check_database")
@patch("app.api.health.check_environment")
def test_health_check_endpoint_healthy(mock_check_env, mock_check_db, client):
    """Test /health endpoint returns healthy status."""
    mock_check_db.return_value = ComponentHealth(status="healthy")
    mock_check_env.return_value = ComponentHealth(status="healthy")
    
    response = client.get("/health/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["components"]["api"]["status"] == "healthy"
    assert data["components"]["database"]["status"] == "healthy"
    assert data["components"]["config"]["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0
    mock_check_db.assert_called_once()
    mock_check_env.assert_called_once()

@patch("app.api.health.check_database")
@patch("app.api.health.check_environment")
def test_health_check_endpoint_unhealthy_db(mock_check_env, mock_check_db, client):
    """Test /health endpoint returns unhealthy if database is unhealthy."""
    mock_check_db.return_value = ComponentHealth(status="unhealthy", details={"error": "DB down"})
    mock_check_env.return_value = ComponentHealth(status="healthy")
    
    response = client.get("/health/")
    
    assert response.status_code == 200 # Health check itself should succeed
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["components"]["database"]["status"] == "unhealthy"
    assert data["components"]["database"]["details"] == {"error": "DB down"}
    assert data["components"]["config"]["status"] == "healthy"

@patch("app.api.health.check_database")
@patch("app.api.health.check_environment")
def test_health_check_endpoint_degraded_config(mock_check_env, mock_check_db, client):
    """Test /health endpoint returns degraded if config is degraded."""
    mock_check_db.return_value = ComponentHealth(status="healthy")
    mock_check_env.return_value = ComponentHealth(status="degraded", details={"missing": ["VAR"]})
    
    response = client.get("/health/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["components"]["database"]["status"] == "healthy"
    assert data["components"]["config"]["status"] == "degraded"
    assert data["components"]["config"]["details"] == {"missing": ["VAR"]}

@patch("app.api.health.psutil")
def test_system_stats_endpoint(mock_psutil, client):
    """Test /health/system endpoint returns system statistics."""
    mock_process = MagicMock()
    mock_process.memory_info.return_value.rss = 50 * 1024 * 1024 # 50 MB
    mock_process.open_files.return_value = ["file1", "file2"]
    mock_process.num_threads.return_value = 4
    
    mock_psutil.cpu_percent.return_value = 15.5
    mock_psutil.virtual_memory.return_value.percent = 60.2
    mock_psutil.disk_usage.return_value.percent = 75.0
    mock_psutil.Process.return_value = mock_process
    
    response = client.get("/health/system")
    
    assert response.status_code == 200
    data = response.json()
    assert data["cpu_percent"] == 15.5
    assert data["memory_percent"] == 60.2
    assert data["disk_usage_percent"] == 75.0
    assert data["process_memory_mb"] == 50.0
    assert data["open_files"] == 2
    assert data["active_threads"] == 4
    assert "timestamp" in data
    
    mock_psutil.Process.assert_called_once_with(os.getpid())

@patch("app.api.health.get_retry_statistics")
def test_retry_statistics_endpoint(mock_get_stats, client):
    """Test /health/retry-stats endpoint returns retry statistics."""
    sample_stats = {
        "total_retries": 10,
        "successful_retries": 5,
        "exhausted_retries": 2,
        "successful_after_retry": 3,
        "high_failure_operations": ["op1"],
        "operation_stats": {"op1": {"count": 5, "exceptions": {"TimeoutError": 2}}},
        "exception_stats": {"TimeoutError": {"count": 2, "operations": {"op1": 2}}},
        "timestamp": datetime.now().isoformat()
    }
    mock_get_stats.return_value = sample_stats
    
    response = client.get("/health/retry-stats")
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_retries"] == 10
    assert data["successful_retries"] == 5
    assert data["high_failure_operations"] == ["op1"]
    assert data["operation_stats"] == sample_stats["operation_stats"]
    assert data["exception_stats"] == sample_stats["exception_stats"]
    assert data["timestamp"] == sample_stats["timestamp"]
    mock_get_stats.assert_called_once()

@patch("app.api.health.reset_retry_statistics")
def test_reset_retry_stats_endpoint(mock_reset_stats, client):
    """Test /health/retry-stats/reset endpoint calls reset function."""
    response = client.post("/health/retry-stats/reset")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "Retry statistics reset successfully" in data["message"]
    mock_reset_stats.assert_called_once()

# Add more tests for edge cases if necessary
# e.g., what happens if psutil raises an exception?
# Consider testing the logger calls if important. 