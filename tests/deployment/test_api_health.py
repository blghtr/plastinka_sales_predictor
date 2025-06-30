"""
Comprehensive tests for deployment.app.api.health

This test suite covers all health check endpoints and component health functions
with comprehensive mocking of external dependencies. Tests are organized by endpoint 
groups and include both success and failure scenarios.

Testing Approach:
- Mock all external dependencies (database connections, environment variables, psutil)
- Test health check endpoints with different component states (healthy/degraded/unhealthy)
- Test authentication and authorization scenarios for protected endpoints
- Test system statistics endpoint functionality with psutil mocking
- Test retry statistics management (get/reset operations)  
- Test individual component health check functions in isolation
- Verify proper HTTP status codes and response formats
- Test error handling scenarios and edge cases

All external imports and dependencies are mocked to ensure test isolation.
"""

import os
import sqlite3
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from deployment.app.api.health import (
    ComponentHealth,
    HealthResponse,
    RetryStatsResponse,
    SystemStatsResponse,
    check_environment,
)
from deployment.app.main import app

TEST_X_API_KEY = "test_x_api_key_conftest"


class TestHealthCheckEndpoint:
    """Test suite for /health endpoint."""

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.check_environment")
    def test_health_check_endpoint_healthy(self, mock_check_env, mock_check_db, client):
        """Test /health endpoint returns healthy status when all components are healthy."""
        # Arrange
        mock_check_db.return_value = ComponentHealth(status="healthy", details={"mock_type": "healthy"})
        mock_check_env.return_value = ComponentHealth(status="healthy", details={"mock_type": "healthy"})

        # Act
        response = client.get("/health/")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "api" in data["components"]
        assert "database" in data["components"]
        assert "config" in data["components"]
        assert data["components"]["api"]["status"] == "healthy"
        assert data["components"]["database"]["status"] == "healthy"
        assert data["components"]["config"]["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        mock_check_db.assert_called_once()
        mock_check_env.assert_called_once()

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.check_environment")
    def test_health_check_endpoint_unhealthy_db(self, mock_check_env, mock_check_db, client):
        """Test /health endpoint returns unhealthy status when database is unhealthy."""
        # Arrange
        mock_check_db.return_value = ComponentHealth(status="unhealthy", details={"error": "mocked unhealthy"})
        mock_check_env.return_value = ComponentHealth(status="healthy", details={"mock_type": "healthy"})

        # Act
        response = client.get("/health/")

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["components"]["database"]["status"] == "unhealthy"
        assert data["components"]["config"]["status"] == "healthy"
        mock_check_db.assert_called_once()
        mock_check_env.assert_called_once()

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.check_environment")
    def test_health_check_endpoint_degraded_env(self, mock_check_env, mock_check_db, client):
        """Test /health endpoint returns degraded status when environment is degraded."""
        # Arrange
        mock_check_db.return_value = ComponentHealth(status="healthy", details={"mock_type": "healthy"})
        mock_check_env.return_value = ComponentHealth(status="degraded", details={"error": "mocked degraded"})

        # Act
        response = client.get("/health/")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["database"]["status"] == "healthy"
        assert data["components"]["config"]["status"] == "degraded"
        mock_check_db.assert_called_once()
        mock_check_env.assert_called_once()


class TestSystemStatsEndpoint:
    """Test suite for /system/stats endpoint."""

    @patch("deployment.app.api.health.psutil")
    def test_system_stats_endpoint_success(self, mock_psutil, client):
        """Test successful retrieval of system statistics."""
        # Arrange
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 50 * 1024 * 1024  # 50 MB
        mock_process.open_files.return_value = ["file1", "file2"]
        mock_process.num_threads.return_value = 4

        mock_psutil.cpu_percent.return_value = 15.5
        mock_psutil.virtual_memory.return_value.percent = 60.2
        mock_psutil.disk_usage.return_value.percent = 75.0
        mock_psutil.Process.return_value = mock_process

        # Act
        response = client.get("/health/system", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
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
        mock_psutil.cpu_percent.assert_called_once()
        mock_psutil.virtual_memory.assert_called_once()
        mock_psutil.disk_usage.assert_called_once()

    def test_system_stats_unauthorized_missing_key(self, client):
        """Test system stats endpoint fails with 401 if X-API-Key header is missing."""
        # Act
        response = client.get("/health/system")

        # Assert
        assert response.status_code == 401

    def test_system_stats_unauthorized_invalid_key(self, client):
        """Test system stats endpoint fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.get("/health/system", headers={"X-API-Key": "wrong_key"})

        # Assert
        assert response.status_code == 401

    def test_system_stats_server_key_not_configured(self, client):
        """Test system stats endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange - Override the dependency to simulate server key not configured
        from fastapi import Security
        from fastapi.security import APIKeyHeader

        from deployment.app.services.auth import get_current_api_key_validated

        api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

        async def mock_server_key_not_configured(api_key: str = Security(api_key_header_scheme)) -> bool:
            # Simulate the case where server X-API-Key is None/not configured
            if api_key is None:
                raise HTTPException(status_code=401, detail="X-API-Key header required")
            # This simulates the 500 error when server key is not configured
            raise HTTPException(status_code=500, detail="Server X-API-Key not configured")

        # Temporarily override the dependency for this test
        original_override = app.dependency_overrides.get(get_current_api_key_validated)
        app.dependency_overrides[get_current_api_key_validated] = mock_server_key_not_configured

        try:
            # Act
            response = client.get("/health/system", headers={"X-API-Key": TEST_X_API_KEY})

            # Assert
            assert response.status_code == 500
        finally:
            # Restore original override
            if original_override:
                app.dependency_overrides[get_current_api_key_validated] = original_override
            else:
                app.dependency_overrides.pop(get_current_api_key_validated, None)


class TestRetryStatsEndpoint:
    """Test suite for retry statistics endpoints."""

    def get_mock_retry_stats_data(self):
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

    @patch("deployment.app.api.health.get_retry_statistics")
    def test_retry_statistics_endpoint_success(self, mock_get_stats, client):
        """Test successful retrieval of retry statistics."""
        # Arrange
        mock_get_stats.return_value = self.get_mock_retry_stats_data()

        # Act
        response = client.get("/health/retry-stats", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["total_retries"] == 10
        assert data["successful_retries"] == 5
        assert data["exhausted_retries"] == 2
        assert data["successful_after_retry"] == 3
        assert "operation_A" in data["high_failure_operations"]
        assert "operation_B" in data["high_failure_operations"]
        assert "op_X" in data["operation_stats"]
        assert "ErrA" in data["exception_stats"]
        assert "timestamp" in data
        mock_get_stats.assert_called_once()

    def test_retry_statistics_unauthorized_missing_key(self, client):
        """Test retry stats endpoint fails with 401 if X-API-Key header is missing."""
        # Act
        response = client.get("/health/retry-stats")

        # Assert
        assert response.status_code == 401

    def test_retry_statistics_unauthorized_invalid_key(self, client):
        """Test retry stats endpoint fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.get("/health/retry-stats", headers={"X-API-Key": "wrong_key"})

        # Assert
        assert response.status_code == 401

    def test_retry_statistics_server_key_not_configured(self, client):
        """Test retry stats endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange - Override the dependency to simulate server key not configured
        from fastapi import Security
        from fastapi.security import APIKeyHeader

        from deployment.app.services.auth import get_current_api_key_validated

        api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

        async def mock_server_key_not_configured(api_key: str = Security(api_key_header_scheme)) -> bool:
            # Simulate the case where server X-API-Key is None/not configured
            if api_key is None:
                raise HTTPException(status_code=401, detail="X-API-Key header required")
            # This simulates the 500 error when server key is not configured
            raise HTTPException(status_code=500, detail="Server X-API-Key not configured")

        # Temporarily override the dependency for this test
        original_override = app.dependency_overrides.get(get_current_api_key_validated)
        app.dependency_overrides[get_current_api_key_validated] = mock_server_key_not_configured

        try:
            # Act
            response = client.get("/health/retry-stats", headers={"X-API-Key": TEST_X_API_KEY})

            # Assert
            assert response.status_code == 500
        finally:
            # Restore original override
            if original_override:
                app.dependency_overrides[get_current_api_key_validated] = original_override
            else:
                app.dependency_overrides.pop(get_current_api_key_validated, None)

    @patch("deployment.app.api.health.reset_retry_statistics")
    def test_reset_retry_stats_endpoint_success(self, mock_reset_stats, client):
        """Test successful reset of retry statistics."""
        # Act
        response = client.post("/health/retry-stats/reset", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Retry statistics reset successfully"
        mock_reset_stats.assert_called_once()

    def test_reset_retry_stats_unauthorized_missing_key(self, client):
        """Test reset retry stats endpoint fails with 401 if X-API-Key header is missing."""
        # Act
        response = client.post("/health/retry-stats/reset")

        # Assert
        assert response.status_code == 401

    def test_reset_retry_stats_unauthorized_invalid_key(self, client):
        """Test reset retry stats endpoint fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.post("/health/retry-stats/reset", headers={"X-API-Key": "wrong_key"})

        # Assert
        assert response.status_code == 401

    def test_reset_retry_stats_server_key_not_configured(self, client):
        """Test reset retry stats endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange - Override the dependency to simulate server key not configured
        from fastapi import Security
        from fastapi.security import APIKeyHeader

        from deployment.app.services.auth import get_current_api_key_validated

        api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

        async def mock_server_key_not_configured(api_key: str = Security(api_key_header_scheme)) -> bool:
            # Simulate the case where server X-API-Key is None/not configured
            if api_key is None:
                raise HTTPException(status_code=401, detail="X-API-Key header required")
            # This simulates the 500 error when server key is not configured
            raise HTTPException(status_code=500, detail="Server X-API-Key not configured")

        # Temporarily override the dependency for this test
        original_override = app.dependency_overrides.get(get_current_api_key_validated)
        app.dependency_overrides[get_current_api_key_validated] = mock_server_key_not_configured

        try:
            # Act
            response = client.post("/health/retry-stats/reset", headers={"X-API-Key": TEST_X_API_KEY})

            # Assert
            assert response.status_code == 500
        finally:
            # Restore original override
            if original_override:
                app.dependency_overrides[get_current_api_key_validated] = original_override
            else:
                app.dependency_overrides.pop(get_current_api_key_validated, None)


class TestComponentHealthChecks:
    """Test suite for individual component health check functions."""

    def test_check_database_healthy(self):
        """Test check_database returns healthy when connection and tables are ok."""
        # Arrange
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

        # Act & Assert
        with patch('deployment.app.api.health.sqlite3.connect', return_value=mock_conn) as mock_connect:
            from deployment.app.api.health import check_database
            result = check_database()

            assert result.status == "healthy"
            mock_connect.assert_called_once()
            call_args = mock_connect.call_args[0][0]
            assert isinstance(call_args, str)
            assert call_args.endswith('.db')
            mock_conn.cursor.assert_called_once()
            assert mock_cursor.execute.call_count == 2
            assert mock_cursor.execute.call_args_list[0][0][0] == 'SELECT 1'
            assert "SELECT name FROM sqlite_master WHERE type='table'" in mock_cursor.execute.call_args_list[1][0][0]
            mock_cursor.fetchall.assert_called_once()
            mock_conn.close.assert_called_once()

    def test_check_database_degraded_missing_tables(self):
        """Test check_database returns degraded if tables are missing."""
        # Arrange
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Simulate finding only one table
        mock_cursor.fetchall.return_value = [('jobs',)]

        # Act & Assert
        with patch('deployment.app.api.health.sqlite3.connect', return_value=mock_conn) as mock_connect:
            from deployment.app.api.health import check_database
            result = check_database()

            assert result.status == "degraded"
            assert "missing_tables" in result.details
            assert "jobs" not in result.details["missing_tables"]
            assert "models" in result.details["missing_tables"]
            mock_connect.assert_called_once()
            call_args = mock_connect.call_args[0][0]
            assert isinstance(call_args, str)
            assert call_args.endswith('.db')
            mock_conn.close.assert_called_once()

    def test_check_database_unhealthy_connection_error(self):
        """Test check_database returns unhealthy on connection error."""
        # Act & Assert
        with patch('deployment.app.api.health.sqlite3.connect',
                   side_effect=sqlite3.OperationalError("Connection failed")) as mock_connect:
            from deployment.app.api.health import check_database
            result = check_database()

            assert result.status == "unhealthy"
            assert "error" in result.details
            assert "Connection failed" in result.details["error"]
            mock_connect.assert_called_once()
            call_args = mock_connect.call_args[0][0]
            assert isinstance(call_args, str)
            assert call_args.endswith('.db')

    @patch.dict(os.environ, {
        "DATASPHERE_PROJECT_ID": "project123",
        "DATASPHERE_FOLDER_ID": "folder456", 
        "API_X_API_KEY": "api_key_789",
        "CALLBACK_AUTH_TOKEN": "token_abc",
        "DATASPHERE_OAUTH_TOKEN": "oauth_token_def"
    })
    def test_check_environment_healthy(self):
        """Test check_environment returns healthy when all vars are set."""
        # Act
        result = check_environment()

        # Assert
        assert result.status == "healthy"
        assert result.details == {}

    @patch.dict(os.environ, {
        "DATASPHERE_PROJECT_ID": "project123",
        "DATASPHERE_FOLDER_ID": "folder456", 
        "API_X_API_KEY": "api_key_789",
        "CALLBACK_AUTH_TOKEN": "token_abc",
        "DATASPHERE_YC_PROFILE": "default"
    })
    def test_check_environment_healthy_with_yc_profile(self):
        """Test check_environment returns healthy when using YC profile for DataSphere auth."""
        # Act
        result = check_environment()

        # Assert
        assert result.status == "healthy"
        assert result.details == {}

    @patch.dict(os.environ, {}, clear=True)
    def test_check_environment_degraded(self):
        """Test check_environment returns degraded when vars are missing."""
        # Act
        result = check_environment()

        # Assert
        assert result.status == "degraded"
        assert "missing_variables" in result.details

        # Expected missing variables with descriptions
        expected_missing = [
            "DATASPHERE_PROJECT_ID (DataSphere Project ID)",
            "DATASPHERE_FOLDER_ID (Yandex Cloud Folder ID)", 
            "API_X_API_KEY (API Authentication Key)",
            "CALLBACK_AUTH_TOKEN (Cloud Callback Authentication Token)",
            "DATASPHERE_OAUTH_TOKEN or DATASPHERE_YC_PROFILE (DataSphere Authentication)"
        ]

        actual_missing = result.details["missing_variables"]
        assert len(actual_missing) == 5
        for var_desc in expected_missing:
            assert var_desc in actual_missing, \
                   f"Expected '{var_desc}' to be in missing variables: {actual_missing}"


class TestIntegration:
    """Integration tests for module imports and configuration."""

    def test_module_imports_successfully(self):
        """Test that the health module can be imported without errors."""
        # Act & Assert
        try:
            import deployment.app.api.health
            # Test that key components are available
            assert hasattr(deployment.app.api.health, 'check_database')
            assert hasattr(deployment.app.api.health, 'check_environment')
            assert hasattr(deployment.app.api.health, 'get_retry_statistics')
            assert hasattr(deployment.app.api.health, 'reset_retry_statistics')
        except ImportError as e:
            pytest.fail(f"Failed to import health module: {e}")

    def test_models_defined(self):
        """Test that all expected response models are defined."""
        # Act & Assert
        # Test basic model instantiation
        component_health = ComponentHealth(status="healthy", details={})
        assert component_health.status == "healthy"
        assert component_health.details == {}

        # Test model attributes exist
        assert hasattr(HealthResponse, 'model_fields')
        assert hasattr(SystemStatsResponse, 'model_fields')
        assert hasattr(RetryStatsResponse, 'model_fields')

    def test_router_configuration(self):
        """Test that the health router is properly configured."""
        # Act & Assert
        from deployment.app.api.health import router

        # Test router exists and has expected attributes
        assert router is not None
        assert hasattr(router, 'routes')
        assert len(router.routes) > 0

        # Test that expected routes exist
        route_paths = [route.path for route in router.routes]
        expected_paths = ["/", "/system", "/retry-stats", "/retry-stats/reset"]

        for expected_path in expected_paths:
            assert any(path.endswith(expected_path) for path in route_paths), \
                   f"Expected path ending with '{expected_path}' not found in {route_paths}"
