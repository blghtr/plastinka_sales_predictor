import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, PropertyMock
import os
import time
from datetime import datetime
from fastapi import HTTPException, status # Added for direct exception raising
import importlib
import sqlite3
import sys

# Assuming your FastAPI app instance is created in deployment/app/main.py
# Adjust the import path if necessary
from deployment.app.main import app
from deployment.app.api.health import (
    HealthResponse,
    SystemStatsResponse,
    RetryStatsResponse,
    ComponentHealth,
    check_environment,
    start_time,
    get_retry_statistics,
    reset_retry_statistics
)
from deployment.app.config import settings # Needed for DB path in check_database test

# TEST_X_API_KEY = "test_x_api_key_conftest" # Removed: Managed by conftest.py and direct usage

# --- Factory functions for pre-configured mocks ---
def make_mock_returning_healthy_status():
    m = MagicMock()
    m.return_value = ComponentHealth(status="healthy", details={"mock_type": "healthy"})
    return m

def make_mock_returning_unhealthy_status():
    m = MagicMock()
    m.return_value = ComponentHealth(status="unhealthy", details={"error": "mocked unhealthy"})
    return m

def make_mock_returning_degraded_status():
    m = MagicMock()
    m.return_value = ComponentHealth(status="degraded", details={"error": "mocked degraded"})
    return m

def get_mock_retry_stats_data():
    """Returns a dictionary conforming to RetryStatsResponse."""
    return {
        "total_retries": 10,
        "successful_retries": 5,
        "exhausted_retries": 2,
        "successful_after_retry": 3,
        "high_failure_operations": ["operation_A", "operation_B"],
        "operation_stats": {"op_X": {"total": 5, "failed": 1}},
        "exception_stats": {"ErrA": {"count": 1, "last_message": "Failed A"}},
        "timestamp": datetime.now().isoformat()
    }

# Using the global client fixture from conftest.py

# --- Test Helper Functions ---

def test_check_database_healthy():
    """Test check_database returns healthy when connection and tables are ok."""
    # Set up the mocks
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Simulate finding all required tables
    mock_cursor.fetchall.return_value = [
        ('jobs',), ('models',), ('configs',), ('training_results',),
        ('prediction_results',), ('job_status_history',), ('dim_multiindex_mapping',),
        ('fact_sales',), ('dim_price_categories',), ('dim_styles',),
        ('fact_stock',), ('fact_prices',), ('fact_stock_changes',),
        ('fact_predictions',), ('sqlite_sequence',), ('processing_runs',),
        ('data_upload_results',), ('report_results',)
    ]
    
    # Use a direct patch of the sqlite3.connect function within the health module
    with patch('deployment.app.api.health.sqlite3.connect', return_value=mock_conn) as mock_connect:
        # Call the function directly
        from deployment.app.api.health import check_database
        result = check_database()
        
        # Assertions
        assert result.status == "healthy"
        # Verify that connect was called exactly once (without checking the specific path)
        # since the path computation may vary between test environments
        mock_connect.assert_called_once()
        # Verify it was called with a string ending in '.db' 
        call_args = mock_connect.call_args[0][0]
        assert isinstance(call_args, str)
        assert call_args.endswith('.db')
        mock_conn.cursor.assert_called_once()
        # The function makes two execute calls: one for basic connectivity check and one for table check
        assert mock_cursor.execute.call_count == 2
        # First call is the basic connectivity check
        assert mock_cursor.execute.call_args_list[0][0][0] == 'SELECT 1'
        # Second call is to check for required tables
        assert "SELECT name FROM sqlite_master WHERE type='table'" in mock_cursor.execute.call_args_list[1][0][0]
        mock_cursor.fetchall.assert_called_once()
        mock_conn.close.assert_called_once()

def test_check_database_degraded_missing_tables():
    """Test check_database returns degraded if tables are missing."""
    # Set up the mocks
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Simulate finding only one table
    mock_cursor.fetchall.return_value = [('jobs',)]
    
    # Use a direct patch of the sqlite3.connect function within the health module
    with patch('deployment.app.api.health.sqlite3.connect', return_value=mock_conn) as mock_connect:
        # Call the function directly
        from deployment.app.api.health import check_database
        result = check_database()
        
        # Assertions
        assert result.status == "degraded"
        assert "missing_tables" in result.details
        assert "jobs" not in result.details["missing_tables"]
        assert "models" in result.details["missing_tables"]
        # Verify that connect was called exactly once (without checking the specific path)
        # since the path computation may vary between test environments
        mock_connect.assert_called_once()
        # Verify it was called with a string ending in '.db' 
        call_args = mock_connect.call_args[0][0]
        assert isinstance(call_args, str)
        assert call_args.endswith('.db')
        mock_conn.close.assert_called_once()

def test_check_database_unhealthy_connection_error():
    """Test check_database returns unhealthy on connection error."""
    # Use a direct patch of the sqlite3.connect function within the health module
    with patch('deployment.app.api.health.sqlite3.connect', 
               side_effect=sqlite3.OperationalError("Connection failed")) as mock_connect:
        # Call the function directly
        from deployment.app.api.health import check_database
        result = check_database()
        
        # Assertions
        assert result.status == "unhealthy"
        assert "error" in result.details
        assert "Connection failed" in result.details["error"]
        # Verify that connect was called exactly once (without checking the specific path)
        # since the path computation may vary between test environments
        mock_connect.assert_called_once()
        # Verify it was called with a string ending in '.db' 
        call_args = mock_connect.call_args[0][0]
        assert isinstance(call_args, str)
        assert call_args.endswith('.db')

@patch.dict(os.environ, {
    "YANDEX_CLOUD_ACCESS_KEY": "key1_corrected", 
    "YANDEX_CLOUD_SECRET_KEY": "key2_corrected", 
    "YANDEX_CLOUD_FOLDER_ID": "folder_corrected", 
    "YANDEX_CLOUD_API_KEY": "api_key_corrected", 
    "CLOUD_CALLBACK_AUTH_TOKEN": "token_corrected" 
    # Removed S3_ENDPOINT_URL, S3_BUCKET_NAME as they are not in check_environment's required_vars
})
def test_check_environment_healthy():
    """Test check_environment returns healthy when all vars are set."""
    result = check_environment()
    assert result.status == "healthy"
    assert result.details == {}

@patch.dict(os.environ, {"YANDEX_CLOUD_ACCESS_KEY_ID": "key1", "YANDEX_CLOUD_SECRET_ACCESS_KEY": "key2"}, clear=True) # Clear others
def test_check_environment_degraded():
    """Test check_environment returns degraded when vars are missing."""
    result = check_environment()
    assert result.status == "degraded"
    assert "missing_variables" in result.details
    
    # Expected missing variables with descriptions
    expected_missing = [
        "YANDEX_CLOUD_FOLDER_ID (Cloud Folder ID)",
        "YANDEX_CLOUD_API_KEY (Cloud API Key)",
        "CLOUD_CALLBACK_AUTH_TOKEN (Cloud Callback Authentication Token)",
        # The following were expected by check_environment but not provided by the old mock:
        "YANDEX_CLOUD_ACCESS_KEY (Cloud Storage Access Key)",
        "YANDEX_CLOUD_SECRET_KEY (Cloud Storage Secret Key)"
    ]
    # Check that the expected missing variables are a subset of the actual missing ones
    # This is more robust if check_environment adds more checks later
    actual_missing = result.details["missing_variables"]
    assert len(actual_missing) >= 3 # Check a minimum number based on expectations
    for var_desc in expected_missing:
        assert var_desc in actual_missing, \
               f"Expected '{var_desc}' to be in missing variables: {actual_missing}"

# --- Test API Endpoints ---

@patch("deployment.app.api.health.check_database", new_callable=make_mock_returning_healthy_status)
@patch("deployment.app.api.health.check_environment", new_callable=make_mock_returning_healthy_status)
def test_health_check_endpoint_healthy(mock_check_env, mock_check_db, client):
    """Test /health endpoint returns healthy status."""
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

@patch("deployment.app.api.health.check_database", new_callable=make_mock_returning_unhealthy_status)
@patch("deployment.app.api.health.check_environment", new_callable=make_mock_returning_healthy_status)
def test_health_check_endpoint_unhealthy_db(mock_check_env, mock_check_db, client):
    """Test /health endpoint returns unhealthy if database is unhealthy."""
    response = client.get("/health/")
    assert response.status_code == 503 # Service Unavailable if critical component is unhealthy
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["components"]["database"]["status"] == "unhealthy"
    assert data["components"]["database"]["details"]["error"] == "mocked unhealthy"
    mock_check_db.assert_called_once()
    mock_check_env.assert_called_once()

@patch("deployment.app.api.health.check_database", new_callable=make_mock_returning_healthy_status)
@patch("deployment.app.api.health.check_environment", new_callable=make_mock_returning_degraded_status)
def test_health_check_endpoint_degraded_env(mock_check_env, mock_check_db, client):
    """Test /health endpoint returns degraded if environment is degraded."""
    response = client.get("/health/")
    assert response.status_code == 200 # Degraded is not a 5xx error
    data = response.json()
    assert data["status"] == "degraded"
    assert data["components"]["config"]["status"] == "degraded"
    assert data["components"]["config"]["details"]["error"] == "mocked degraded"
    mock_check_db.assert_called_once()
    mock_check_env.assert_called_once()

@patch("deployment.app.api.health.psutil")
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
    
    response = client.get("/health/system", headers={"X-API-Key": "test_x_api_key_conftest"}) # Use direct string literal

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

def test_system_stats_unauthorized_missing_key(client, mock_x_api_key):
    """Test /health/system returns 401 if X-API-Key header is missing but expected."""
    mock_x_api_key("test_x_api_key_conftest") # Ensure server expects a key
    response = client.get("/health/system") # No X-API-Key header
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing." in response.json()["error"]["message"]

def test_system_stats_unauthorized_invalid_key(client, mock_x_api_key):
    """Test /health/system returns 401 if X-API-Key header is invalid."""
    mock_x_api_key("test_x_api_key_conftest") # Ensure server expects a specific key
    response = client.get("/health/system", headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key." in response.json()["error"]["message"]

def test_system_stats_server_key_not_configured(client, mock_x_api_key):
    """Test /health/system returns 500 if X-API-Key authentication is not configured on server."""
    mock_x_api_key(None) # Simulate server not having key configured
    response = client.get("/health/system", headers={"X-API-Key": "any_key_does_not_matter"}) # Key is present but server doesn't expect one
    
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server." in response.json()["error"]["message"]

@patch("deployment.app.api.health.get_retry_statistics", side_effect=get_mock_retry_stats_data)
def test_retry_statistics_endpoint(mock_get_stats, client):
    """Test /health/retry-stats endpoint returns retry statistics."""
    response = client.get("/health/retry-stats", headers={"X-API-Key": "test_x_api_key_conftest"}) # Use direct string literal
    assert response.status_code == 200
    data = response.json()
    assert data["total_retries"] == 10
    assert data["successful_retries"] == 5
    assert data["exhausted_retries"] == 2
    assert data["successful_after_retry"] == 3
    assert "timestamp" in data
    mock_get_stats.assert_called_once()

def test_retry_statistics_unauthorized_missing_key(client, mock_x_api_key):
    """Test /health/retry-stats returns 401 if X-API-Key header is missing but expected."""
    mock_x_api_key("test_x_api_key_conftest") # Ensure server expects a key
    response = client.get("/health/retry-stats") # No X-API-Key header
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing." in response.json()["error"]["message"]

def test_retry_statistics_unauthorized_invalid_key(client, mock_x_api_key):
    """Test /health/retry-stats returns 401 if X-API-Key header is invalid."""
    mock_x_api_key("test_x_api_key_conftest") # Ensure server expects a specific key
    response = client.get("/health/retry-stats", headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key." in response.json()["error"]["message"]

def test_retry_statistics_server_key_not_configured(client, mock_x_api_key):
    """Test /health/retry-stats returns 500 if X-API-Key authentication is not configured on server."""
    mock_x_api_key(None) # Simulate server not having key configured
    response = client.get("/health/retry-stats", headers={"X-API-Key": "any_key_does_not_matter"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server." in response.json()["error"]["message"]

@patch("deployment.app.api.health.reset_retry_statistics")
def test_reset_retry_stats_endpoint(mock_reset_stats, client):
    """Test /health/retry-stats/reset endpoint resets statistics."""
    response = client.post("/health/retry-stats/reset", headers={"X-API-Key": "test_x_api_key_conftest"}) # Use direct string literal
    assert response.status_code == 200
    assert response.json()["message"] == "Retry statistics reset successfully"
    mock_reset_stats.assert_called_once()

def test_reset_retry_stats_unauthorized_missing_key(client, mock_x_api_key):
    """Test /health/retry-stats/reset returns 401 if X-API-Key header is missing but expected."""
    mock_x_api_key("test_x_api_key_conftest") # Ensure server expects a key
    response = client.post("/health/retry-stats/reset") # No X-API-Key header
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing." in response.json()["error"]["message"]

def test_reset_retry_stats_unauthorized_invalid_key(client, mock_x_api_key):
    """Test /health/retry-stats/reset returns 401 if X-API-Key header is invalid."""
    mock_x_api_key("test_x_api_key_conftest") # Ensure server expects a specific key
    response = client.post("/health/retry-stats/reset", headers={"X-API-Key": "invalid_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key." in response.json()["error"]["message"]

def test_reset_retry_stats_server_key_not_configured(client, mock_x_api_key):
    """Test /health/retry-stats/reset returns 500 if X-API-Key authentication is not configured on server."""
    mock_x_api_key(None) # Simulate server not having key configured
    response = client.post("/health/retry-stats/reset", headers={"X-API-Key": "any_key_does_not_matter"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server." in response.json()["error"]["message"]

# Add more tests for edge cases if necessary
# e.g., what happens if psutil raises an exception?
# Consider testing the logger calls if important.

# def test_placeholder_health():
#    assert True