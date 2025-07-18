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
from fastapi import status

from deployment.app.api.health import (
    HealthResponse,
    RetryStatsResponse,
    SystemStatsResponse,
)
from deployment.app.utils.environment import ComponentHealth, get_environment_status
from deployment.app.main import app
from deployment.app.services.auth import get_current_api_key_validated

TEST_X_API_KEY = "test_x_api_key_conftest"


class TestHealthCheckEndpoint:
    """Test suite for /health endpoint."""

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.get_environment_status")
    def test_health_check_endpoint_healthy(self, mock_check_env, mock_check_db, client):
        """Test /health endpoint returns healthy status when all components are healthy."""
        # Arrange
        mock_check_db.return_value = ComponentHealth(
            status="healthy", details={"mock_type": "healthy"}
        )
        mock_check_env.return_value = ComponentHealth(
            status="healthy", details={"mock_type": "healthy"}
        )

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
    @patch("deployment.app.api.health.get_environment_status")
    def test_health_check_endpoint_unhealthy_db(
        self, mock_check_env, mock_check_db, client
    ):
        """Test /health endpoint returns unhealthy status when database is unhealthy and returns ErrorDetailResponse."""
        # Arrange
        mock_check_db.return_value = ComponentHealth(
            status="unhealthy", details={"error": "mocked unhealthy"}
        )
        mock_check_env.return_value = ComponentHealth(
            status="healthy", details={"mock_type": "healthy"}
        )

        # Act
        response = client.get("/health/")

        # Assert
        assert response.status_code == 503
        data = response.json()
        # Error handler wraps the HTTPException detail as a string, need to parse it
        import ast
        error_detail = ast.literal_eval(data["error"]["message"])
        assert error_detail["message"] == "API is unhealthy."
        assert error_detail["code"] == "service_unavailable"
        assert error_detail["status_code"] == 503
        assert "details" in error_detail
        assert error_detail["details"]["database"]["status"] == "unhealthy"
        assert error_detail["details"]["database"]["details"]["error"] == "mocked unhealthy"
        assert error_detail["details"]["config"]["status"] == "healthy"
        mock_check_db.assert_called_once()
        mock_check_env.assert_called_once()

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.get_environment_status")
    def test_health_check_endpoint_degraded_env(
        self, mock_check_env, mock_check_db, client
    ):
        """Test /health endpoint returns degraded status when environment is degraded."""
        # Arrange
        mock_check_db.return_value = ComponentHealth(
            status="healthy", details={"mock_type": "healthy"}
        )
        mock_check_env.return_value = ComponentHealth(
            status="degraded", details={"error": "mocked degraded"}
        )

        # Act
        response = client.get("/health/")

        # Assert
        assert response.status_code == 200 # Degraded status typically returns 200
        data = response.json()
        # Error handler wraps the HTTPException detail as a string, need to parse it
        import ast
        error_detail = ast.literal_eval(data["error"]["message"])
        assert error_detail["message"] == "API is degraded."
        assert error_detail["details"]["database"]["status"] == "healthy"
        assert error_detail["details"]["config"]["status"] == "degraded"
        mock_check_db.assert_called_once()
        mock_check_env.assert_called_once()

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.get_environment_status")
    def test_health_check_endpoint_unhealthy_missing_months(self, mock_check_env, mock_check_db, client):
        """Test /health endpoint returns unhealthy and details[missing_months] if months are missing and returns ErrorDetailResponse."""
        mock_check_db.return_value = ComponentHealth(
            status="unhealthy", details={"missing_months": {"fact_sales": ["2024-02-01"]}}
        )
        mock_check_env.return_value = ComponentHealth(
            status="healthy", details={}
        )
        response = client.get("/health/")
        assert response.status_code == 503
        data = response.json()
        # Error handler wraps the HTTPException detail as a string, need to parse it
        import ast
        error_detail = ast.literal_eval(data["error"]["message"])
        assert error_detail["message"] == "API is unhealthy."
        assert error_detail["code"] == "service_unavailable"
        assert error_detail["status_code"] == 503
        assert "details" in error_detail
        assert error_detail["details"]["database"]["status"] == "unhealthy"
        assert "missing_months" in error_detail["details"]["database"]["details"]
        assert "fact_sales" in error_detail["details"]["database"]["details"]["missing_months"]
        assert error_detail["details"]["database"]["details"]["missing_months"]["fact_sales"] == ["2024-02-01"]


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
        data = response.json()
        assert data["error"]["message"] == "Not authenticated: X-API-Key header is missing."

    def test_system_stats_unauthorized_invalid_key(self, client):
        """Test system stats endpoint fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.get("/health/system", headers={"X-API-Key": "wrong_key"})

        # Assert
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["message"] == "Invalid X-API-Key."

    def test_system_stats_server_key_not_configured(self, client):
        """Test system stats endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange - Override the dependency to simulate server key not configured
        def raise_server_error():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="X-API-Key authentication is not configured on the server.",
            )
        app.dependency_overrides[get_current_api_key_validated] = raise_server_error

        # Act
        response = client.get(
            "/health/system",
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["message"] == "X-API-Key authentication is not configured on the server."
        assert data["error"]["code"] == "http_500"
        
        # Cleanup
        app.dependency_overrides.clear()


class TestRetryStatsEndpoint:
    """Test suite for /health/retry-stats endpoint."""

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
            "timestamp": datetime.now().isoformat(),
        }

    @patch("deployment.app.api.health.get_retry_statistics")
    def test_retry_statistics_endpoint_success(self, mock_get_stats, client):
        """Test successful retrieval of retry statistics."""
        # Arrange
        mock_get_stats.return_value = self.get_mock_retry_stats_data()

        # Act
        response = client.get(
            "/health/retry-stats", headers={"X-API-Key": TEST_X_API_KEY}
        )

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
        data = response.json()
        assert data["error"]["message"] == "Not authenticated: X-API-Key header is missing."

    def test_retry_statistics_unauthorized_invalid_key(self, client):
        """Test retry stats endpoint fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.get("/health/retry-stats", headers={"X-API-Key": "wrong_key"})

        # Assert
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["message"] == "Invalid X-API-Key."

    def test_retry_statistics_server_key_not_configured(self, client):
        """Test retry stats endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        def raise_server_error():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="X-API-Key authentication is not configured on the server.",
            )
        app.dependency_overrides[get_current_api_key_validated] = raise_server_error

        # Act
        response = client.get(
            "/health/retry-stats",
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["message"] == "X-API-Key authentication is not configured on the server."
        assert data["error"]["code"] == "http_500"
        
        # Cleanup
        app.dependency_overrides.clear()

    @patch("deployment.app.api.health.reset_retry_statistics")
    def test_reset_retry_stats_endpoint_success(self, mock_reset_stats, client):
        """Test successful reset of retry statistics."""
        # Act
        response = client.post(
            "/health/retry-stats/reset", headers={"X-API-Key": TEST_X_API_KEY}
        )

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
        data = response.json()
        assert data["error"]["message"] == "Not authenticated: X-API-Key header is missing."

    def test_reset_retry_stats_unauthorized_invalid_key(self, client):
        """Test reset retry stats endpoint fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.post(
            "/health/retry-stats/reset", headers={"X-API-Key": "wrong_key"}
        )

        # Assert
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["message"] == "Invalid X-API-Key."

    def test_reset_retry_stats_server_key_not_configured(self, client):
        """Test reset retry stats endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        def raise_server_error():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="X-API-Key authentication is not configured on the server.",
            )
        app.dependency_overrides[get_current_api_key_validated] = raise_server_error

        # Act
        response = client.post(
            "/health/retry-stats/reset",
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["message"] == "X-API-Key authentication is not configured on the server."
        assert data["error"]["code"] == "http_500"
        
        # Cleanup
        app.dependency_overrides.clear()

    def test_retry_statistics_healthy(self, client, mock_dal, monkeypatch):
        """Test retry stats endpoint returns 200 with stats when X-API-Key is valid."""
        # Arrange
        monkeypatch.setitem(app.dependency_overrides, get_current_api_key_validated, lambda: True)
        mock_get_stats = MagicMock(return_value=self.get_mock_retry_stats_data())
        monkeypatch.setattr("deployment.app.api.health.get_retry_statistics", mock_get_stats)

        # Act
        response = client.get(
            "/health/retry-stats", headers={"X-API-Key": TEST_X_API_KEY}
        )

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

    def test_reset_retry_stats_success(self, client, mock_dal, monkeypatch):
        """Test reset retry stats endpoint returns 200 with success message when X-API-Key is valid."""
        # Arrange
        monkeypatch.setitem(app.dependency_overrides, get_current_api_key_validated, lambda: True)
        mock_reset_stats = MagicMock()
        monkeypatch.setattr("deployment.app.api.health.reset_retry_statistics", mock_reset_stats)

        # Act
        response = client.post(
            "/health/retry-stats/reset", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Retry statistics reset successfully"
        mock_reset_stats.assert_called_once()
        # Cleanup
        app.dependency_overrides = {}


class TestComponentHealthChecks:
    """Test suite for individual component health check functions."""

    def test_check_database_healthy(self, mock_dal):
        """Test check_database returns healthy when connection and tables are ok."""
        from deployment.app.api.health import check_database

        result = check_database(mock_dal)

        assert result.status == "healthy"
        mock_dal.execute_raw_query.assert_called()

    def test_check_database_degraded_missing_tables(self, mock_dal):
        """Test check_database returns degraded if tables are missing."""
        # Arrange
        def side_effect_for_missing_tables(query, params=(), fetchall=False, connection=None):
            if query == "SELECT 1":
                return [{"1": 1}]
            if "SELECT name FROM sqlite_master" in query:
                # Return only "jobs" table, missing all others
                return [{"name": "jobs"}]
            if "fact_sales" in query or "fact_stock_changes" in query:
                # Return empty for fact table queries since they don't exist
                return []
            return []

        mock_dal.execute_raw_query.side_effect = side_effect_for_missing_tables
        
        # Mock execute_query_with_batching to return the same result for table checks
        def batching_side_effect(query_template, ids, **kwargs):
            if "sqlite_master" in query_template.lower():
                # Return only "jobs" table from the requested tables
                return [{"name": "jobs"}] if "jobs" in ids else []
            return []
        
        mock_dal.execute_query_with_batching.side_effect = batching_side_effect

        # Act
        from deployment.app.api.health import check_database

        result = check_database(mock_dal)

        # Assert
        assert result.status == "degraded"
        assert "missing_tables" in result.details
        assert "jobs" not in result.details["missing_tables"]
        assert "models" in result.details["missing_tables"]
        mock_dal.execute_raw_query.assert_called()

    def test_check_database_unhealthy_connection_error(self, mock_dal):
        """Test check_database returns unhealthy on connection error."""
        # Arrange
        from deployment.app.db.database import DatabaseError
        mock_dal.execute_raw_query.side_effect = DatabaseError("Connection failed")

        # Act
        from deployment.app.api.health import check_database

        result = check_database(mock_dal)

        # Assert
        assert result.status == "unhealthy"
        assert "error" in result.details
        assert "Connection failed" in result.details["error"]
        mock_dal.execute_raw_query.assert_called_once()

    @patch('deployment.app.utils.environment.load_dotenv')
    def test_check_environment_healthy_with_dotenv(self, mock_load_dotenv, tmp_path):
        """Test get_environment_status returns healthy when vars are in .env file."""
        # Arrange
        dotenv_content = """
DATASPHERE_PROJECT_ID=project123
DATASPHERE_FOLDER_ID=folder456
API_X_API_KEY=api_key_789
API_ADMIN_API_KEY=admin_key_123
DATASPHERE_OAUTH_TOKEN=oauth_token_def
        """
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text(dotenv_content)

        # Simulate that load_dotenv sets the environment variables
        def dotenv_side_effect():
            os.environ['DATASPHERE_PROJECT_ID'] = 'project123'
            os.environ['DATASPHERE_FOLDER_ID'] = 'folder456'
            os.environ['API_X_API_KEY'] = 'api_key_789'
            os.environ['API_ADMIN_API_KEY'] = 'admin_key_123'
            os.environ['DATASPHERE_OAUTH_TOKEN'] = 'oauth_token_def'
            return True

        mock_load_dotenv.side_effect = dotenv_side_effect

        with patch.dict(os.environ, {}, clear=True):
            # Act
            result = get_environment_status()

        # Assert
        assert result.status == "healthy"
        assert result.details == {}
        mock_load_dotenv.assert_called_once()

    @patch('deployment.app.utils.environment.load_dotenv')
    @patch.dict(os.environ, {}, clear=True)
    def test_check_environment_degraded(self, mock_load_dotenv):
        """Test get_environment_status returns degraded when env vars are missing."""
        # Arrange
        mock_load_dotenv.return_value = False # Simulate .env not found or empty

        # Act
        result = get_environment_status()

        # Assert
        assert result.status == "degraded"
        assert "missing_variables" in result.details
        assert isinstance(result.details["missing_variables"], list)
        assert len(result.details["missing_variables"]) > 0
        # Check that the key names are present in the descriptive strings
        missing_vars_text = " ".join(result.details["missing_variables"])
        assert "API_X_API_KEY" in missing_vars_text
        assert "API_ADMIN_API_KEY" in missing_vars_text
        assert "DATASPHERE_PROJECT_ID" in missing_vars_text
        assert "DATASPHERE_FOLDER_ID" in missing_vars_text
        assert ("DATASPHERE_OAUTH_TOKEN" in missing_vars_text or "DATASPHERE_YC_PROFILE" in missing_vars_text)

    @patch('deployment.app.utils.environment.load_dotenv')
    def test_check_environment_healthy_with_legacy_api_key(self, mock_load_dotenv, tmp_path):
        """Test get_environment_status returns healthy when using legacy API_API_KEY."""
        # Arrange: Mock load_dotenv to prevent loading real .env file
        mock_load_dotenv.return_value = None
        
        # Arrange
        dotenv_content = """
DATASPHERE_PROJECT_ID=project123
DATASPHERE_FOLDER_ID=folder456
API_X_API_KEY=api_key_789
API_API_KEY=legacy_admin_key_123
DATASPHERE_OAUTH_TOKEN=oauth_token_def
        """
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text(dotenv_content)

        with patch.dict(os.environ, {
            'DATASPHERE_PROJECT_ID': 'project123',
            'DATASPHERE_FOLDER_ID': 'folder456',
            'API_X_API_KEY': 'api_key_789',
            'API_API_KEY': 'legacy_admin_key_123',  # Legacy key
            'DATASPHERE_OAUTH_TOKEN': 'oauth_token_def'
        }, clear=True):
            # Act
            result = get_environment_status()

        # Assert
        assert result.status == "healthy"
        assert result.details == {}

    @patch('deployment.app.utils.environment.load_dotenv')
    def test_check_environment_degraded_missing_auth(self, mock_load_dotenv, tmp_path):
        """Test get_environment_status returns degraded when DataSphere auth is missing."""
        # Arrange: Mock load_dotenv to prevent loading real .env file
        mock_load_dotenv.return_value = None
        
        # Arrange: Only patch environment, do not create .env file
        with patch.dict(os.environ, {
            'DATASPHERE_PROJECT_ID': 'project123',
            'DATASPHERE_FOLDER_ID': 'folder456',
            'API_X_API_KEY': 'api_key_789',
            'API_ADMIN_API_KEY': 'admin_key_123'
            # Missing DataSphere auth
        }, clear=True):
            # Act
            result = get_environment_status()

        # Assert
        assert result.status == "degraded"
        assert "missing_variables" in result.details
        missing_vars_text = " ".join(result.details["missing_variables"])
        assert "DATASPHERE_OAUTH_TOKEN" in missing_vars_text or "DATASPHERE_YC_PROFILE" in missing_vars_text

    @patch('deployment.app.utils.environment.load_dotenv')
    def test_check_environment_healthy_with_yc_profile(self, mock_load_dotenv, tmp_path):
        """Test get_environment_status returns healthy when using YC profile auth."""
        # Arrange: Mock load_dotenv to prevent loading real .env file
        mock_load_dotenv.return_value = None
        
        # Arrange
        dotenv_content = """
DATASPHERE_PROJECT_ID=project123
DATASPHERE_FOLDER_ID=folder456
API_X_API_KEY=api_key_789
API_ADMIN_API_KEY=admin_key_123
DATASPHERE_YC_PROFILE=default
        """
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text(dotenv_content)

        with patch.dict(os.environ, {
            'DATASPHERE_PROJECT_ID': 'project123',
            'DATASPHERE_FOLDER_ID': 'folder456',
            'API_X_API_KEY': 'api_key_789',
            'API_ADMIN_API_KEY': 'admin_key_123',
            'DATASPHERE_YC_PROFILE': 'default'  # YC profile auth
        }, clear=True):
            # Act
            result = get_environment_status()

        # Assert
        assert result.status == "healthy"
        assert result.details == {}

    def test_check_database_unhealthy_missing_tables(self, mock_dal):
        """Test check_database returns unhealthy if required tables are missing."""
        # Arrange
        # Simulate that the query for tables returns only one table.
        def side_effect_for_missing_tables(query, params=(), fetchall=False, connection=None):
            if query == "SELECT 1":
                return [{"1": 1}]
            if "SELECT name FROM sqlite_master" in query:
                # Return only "jobs" table, missing all others
                return [{"name": "jobs"}]
            if "fact_sales" in query or "fact_stock_changes" in query:
                # Return empty for fact table queries since they don't exist
                return []
            return []

        mock_dal.execute_raw_query.side_effect = side_effect_for_missing_tables
        
        # Mock execute_query_with_batching to return the same result for table checks
        def batching_side_effect(query_template, ids, **kwargs):
            if "sqlite_master" in query_template.lower():
                # Return only "jobs" table from the requested tables
                return [{"name": "jobs"}] if "jobs" in ids else []
            return []
        
        mock_dal.execute_query_with_batching.side_effect = batching_side_effect

        # Act
        from deployment.app.api.health import check_database

        result = check_database(mock_dal)

        # Assert
        assert result.status == "degraded"
        assert "missing_tables" in result.details
        assert "jobs" not in result.details["missing_tables"]
        assert "models" in result.details["missing_tables"]
        mock_dal.execute_raw_query.assert_called()

    def test_check_database_unhealthy_missing_months(self, mock_dal):
        """Test check_database returns unhealthy if there are missing months in fact_sales or fact_stock_changes."""

        def side_effect_for_missing_months(query, params=(), fetchall=False, connection=None):
            if query == "SELECT 1":
                return [{"1": 1}]
            if "SELECT name FROM sqlite_master" in query:
                return [{"name": table} for table in params]
            if "fact_sales" in query:
                # Simulate a gap in fact_sales
                return [{"data_date": "2024-01-01"}, {"data_date": "2024-03-01"}]
            if "fact_stock_changes" in query:
                # Simulate no gap in fact_stock_changes
                return [{"data_date": "2024-01-01"}, {"data_date": "2024-02-01"}, {"data_date": "2024-03-01"}]
            return []

        mock_dal.execute_raw_query.side_effect = side_effect_for_missing_months

        # Mock execute_query_with_batching to return all tables as existing
        def batching_side_effect(query_template, ids, **kwargs):
            if "sqlite_master" in query_template.lower():
                # Return all requested tables as existing
                return [{"name": table} for table in ids]
            return []
        
        mock_dal.execute_query_with_batching.side_effect = batching_side_effect

        from deployment.app.api.health import check_database
        result = check_database(mock_dal)
        assert result.status == "unhealthy"
        assert "missing_months_sales" in result.details
        assert result.details["missing_months_sales"] == ["2024-02-01"] # check_monotonic_months returns this format
        assert "missing_months_stock_changes" not in result.details

    def test_check_database_healthy_no_missing_months(self, mock_dal):
        """Test check_database returns healthy if all months are present in both tables."""

        def side_effect_for_healthy(query, params=(), fetchall=False, connection=None):
            if query == "SELECT 1":
                return [{"1": 1}]
            if "SELECT name FROM sqlite_master" in query:
                return [{"name": table} for table in params]
            if "fact_sales" in query:
                return [{"data_date": "2024-01-01"}, {"data_date": "2024-02-01"}, {"data_date": "2024-03-01"}]
            if "fact_stock_changes" in query:
                return [{"data_date": "2024-01-01"}, {"data_date": "2024-02-01"}, {"data_date": "2024-03-01"}]
            return []

        mock_dal.execute_raw_query.side_effect = side_effect_for_healthy

        # Mock execute_query_with_batching to return all tables as existing
        def batching_side_effect(query_template, ids, **kwargs):
            if "sqlite_master" in query_template.lower():
                # Return all requested tables as existing
                return [{"name": table} for table in ids]
            return []
        
        mock_dal.execute_query_with_batching.side_effect = batching_side_effect

        from deployment.app.api.health import check_database
        result = check_database(mock_dal)
        assert result.status == "healthy"


class TestIntegration:
    """Integration tests for module imports and configuration."""

    def test_module_imports_successfully(self):
        """Test that the health module can be imported without errors."""
        # Act & Assert
        try:
            import deployment.app.api.health

            # Test that key components are available
            assert hasattr(deployment.app.api.health, "check_database")
            # check_environment was moved to deployment.app.utils.environment as get_environment_status
            from deployment.app.utils.environment import get_environment_status
            assert callable(get_environment_status)
            assert hasattr(deployment.app.api.health, "get_retry_statistics")
            assert hasattr(deployment.app.api.health, "reset_retry_statistics")
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
        assert hasattr(HealthResponse, "model_fields")
        assert hasattr(SystemStatsResponse, "model_fields")
        assert hasattr(RetryStatsResponse, "model_fields")

    def test_router_configuration(self):
        """Test that the health router is properly configured."""
        # Act & Assert
        from deployment.app.api.health import router

        # Test router exists and has expected attributes
        assert router is not None
        assert hasattr(router, "routes")
        assert len(router.routes) > 0

        # Test that expected routes exist
        route_paths = [route.path for route in router.routes]
        expected_paths = ["/health", "/health/system", "/health/retry-stats", "/health/retry-stats/reset"]

        for expected_path in expected_paths:
            assert expected_path in route_paths, (
                f"Expected path '{expected_path}' not found in {route_paths}"
            )
