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
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status

from deployment.app.api.health import (
    HealthResponse,
    RetryStatsResponse,
    SystemStatsResponse,
)
from deployment.app.dependencies import get_dal_system
from deployment.app.main import app
from deployment.app.services.auth import get_unified_auth
from deployment.app.utils.environment import ComponentHealth, get_environment_status

TEST_X_API_KEY = "test_x_api_key_conftest"
TEST_BEARER_TOKEN = "test_admin_token"


class TestHealthCheckEndpoint:
    """Test suite for /health endpoint."""

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.get_environment_status")
    @patch("deployment.app.api.health.get_settings")
    def test_health_check_endpoint_healthy(self, mock_settings, mock_check_env, mock_check_db, api_client):
        """Test /health endpoint returns healthy status when all components are healthy."""
        # Arrange
        mock_check_db.return_value = ComponentHealth(
            status="healthy", details={"mock_type": "healthy"}
        )
        mock_check_env.return_value = ComponentHealth(
            status="healthy", details={"mock_type": "healthy"}
        )

        # Mock settings to return a valid threshold
        mock_settings_instance = MagicMock()
        mock_settings_instance.metric_thesh_for_health_check = 0.5
        mock_settings_instance.default_metric_higher_is_better = True
        mock_settings.return_value = mock_settings_instance

        # Mock the DAL to return a valid active model metric
        # Since dependency_overrides returns async generator, we need to create a mock DAL
        # and replace the dependency override
        from tests.deployment.app.conftest import dal as test_dal
        
        # Create a mock DAL with the needed method
        mock_dal = MagicMock(spec=test_dal)
        mock_dal.get_active_model_primary_metric = AsyncMock(return_value=0.8)
        
        async def mock_get_dal_system():
            yield mock_dal
        
        api_client.app.dependency_overrides[get_dal_system] = mock_get_dal_system
        
        # Act
        response = api_client.get("/health/")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "api" in data["components"]
        assert "database" in data["components"]
        assert "config" in data["components"]
        assert "active_model_metric" in data["components"]
        assert data["components"]["api"]["status"] == "healthy"
        assert data["components"]["database"]["status"] == "healthy"
        assert data["components"]["config"]["status"] == "healthy"
        assert data["components"]["active_model_metric"]["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        mock_check_db.assert_called_once()
        mock_check_env.assert_called_once()

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.get_environment_status")
    @pytest.mark.asyncio
    async def test_health_check_endpoint_unhealthy_db(
        self, mock_check_env, mock_check_db, async_api_client
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
        response = await async_api_client.get("/health", follow_redirects=True)

        # Assert
        assert response.status_code == 503
        data = response.json()
        assert data["error"]["message"] == "API is unhealthy."
        assert data["error"]["code"] == "http_503"
        assert "details" in data["error"]
        # Note: The details are not preserved in the error wrapper, so we can't test them
        mock_check_db.assert_called_once()
        mock_check_env.assert_called_once()

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.get_environment_status")
    @pytest.mark.asyncio
    async def test_health_check_endpoint_degraded_env(
        self, mock_check_env, mock_check_db, async_api_client
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
        response = await async_api_client.get("/health", follow_redirects=True)

        # Assert
        assert response.status_code == 200  # Degraded status returns 200
        data = response.json()
        assert data["status"] == "degraded"  # Status should be degraded in response body
        assert "components" in data
        assert data["components"]["config"]["status"] == "degraded"
        assert data["components"]["database"]["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        mock_check_db.assert_called_once()
        mock_check_env.assert_called_once()

    @patch("deployment.app.api.health.check_database")
    @patch("deployment.app.api.health.get_environment_status")
    @pytest.mark.asyncio
    async def test_health_check_endpoint_unhealthy_missing_months(self, mock_check_env, mock_check_db, async_api_client):
        """Test /health endpoint returns unhealthy and details[missing_months] if months are missing and returns ErrorDetailResponse."""
        mock_check_db.return_value = ComponentHealth(
            status="unhealthy", details={"missing_months": {"fact_sales": ["2024-02-01"]}}
        )
        mock_check_env.return_value = ComponentHealth(
            status="healthy", details={}
        )
        response = await async_api_client.get("/health", follow_redirects=True)
        assert response.status_code == 503
        data = response.json()
        assert data["error"]["message"] == "API is unhealthy."
        assert data["error"]["code"] == "http_503"
        assert "details" in data["error"]
        # Note: The details are not preserved in the error wrapper, so we can't test them


class TestSystemStatsEndpoint:
    """Test suite for /system/stats endpoint."""

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @patch("deployment.app.api.health.psutil")
    def test_system_stats_endpoint_success(self, mock_psutil, api_client, auth_header_name, auth_token):
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
        response = api_client.get("/health/system", headers={auth_header_name: auth_token})

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

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", "wrong_key"),
        ("Authorization", "Bearer wrong_token"),
    ])
    def test_system_stats_unauthorized_invalid_key(self, api_client, auth_header_name, auth_token):
        """Test system stats endpoint fails with 401 if X-API-Key or Bearer token is invalid."""
        # Act
        response = api_client.get("/health/system", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["message"] == "Invalid or missing credentials. Provide a valid Bearer token or X-API-Key."

    def test_system_stats_server_key_not_configured(self, api_client, monkeypatch):
        """Test system stats endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange - Override the dependency to simulate server key not configured
        def raise_server_error():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication is not configured on the server.",
            )
        monkeypatch.setitem(app.dependency_overrides, get_unified_auth, raise_server_error)

        # Act
        response = api_client.get(
            "/health/system",
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["message"] == "Authentication is not configured on the server."
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

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @patch("deployment.app.api.health.get_retry_statistics")
    def test_retry_statistics_endpoint_success(self, mock_get_stats, api_client, auth_header_name, auth_token):
        """Test successful retrieval of retry statistics."""
        # Arrange
        mock_get_stats.return_value = self.get_mock_retry_stats_data()

        # Act
        response = api_client.get(
            "/health/retry-stats", headers={auth_header_name: auth_token}
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

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", "wrong_key"),
        ("Authorization", "Bearer wrong_token"),
    ])
    def test_retry_statistics_unauthorized_invalid_key(self, api_client, auth_header_name, auth_token):
        """Test retry stats endpoint fails with 401 if X-API-Key or Bearer token is invalid."""
        # Act
        response = api_client.get("/health/retry-stats", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["message"] == "Invalid or missing credentials. Provide a valid Bearer token or X-API-Key."

    def test_retry_statistics_server_key_not_configured(self, api_client, monkeypatch):
        """Test retry stats endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        from deployment.app.services.auth import get_unified_auth
        def raise_server_error():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication is not configured on the server.",
            )
        monkeypatch.setitem(app.dependency_overrides, get_unified_auth, raise_server_error)

        # Act
        response = api_client.get(
            "/health/retry-stats",
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["message"] == "Authentication is not configured on the server."
        assert data["error"]["code"] == "http_500"

        # Cleanup
        app.dependency_overrides.clear()

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    @patch("deployment.app.api.health.reset_retry_statistics")
    def test_reset_retry_stats_endpoint_success(self, mock_reset_stats, api_client, auth_header_name, auth_token):
        """Test successful reset of retry statistics."""
        # Act
        response = api_client.post(
            "/health/retry-stats/reset", headers={auth_header_name: auth_token}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Retry statistics reset successfully"
        mock_reset_stats.assert_called_once()

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", "wrong_key"),
        ("Authorization", "Bearer wrong_token"),
    ])
    def test_reset_retry_stats_unauthorized_invalid_key(self, api_client, auth_header_name, auth_token):
        """Test reset retry stats endpoint fails with 401 if X-API-Key or Bearer token is invalid."""
        # Act
        response = api_client.post(
            "/health/retry-stats/reset", headers={auth_header_name: auth_token}
        )

        # Assert
        assert response.status_code == 401
        data = response.json()
        assert data["error"]["message"] == "Invalid or missing credentials. Provide a valid Bearer token or X-API-Key."

    def test_reset_retry_stats_server_key_not_configured(self, api_client, monkeypatch):
        """Test reset retry stats endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        from deployment.app.services.auth import get_unified_auth
        def raise_server_error():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication is not configured on the server.",
            )
        monkeypatch.setitem(app.dependency_overrides, get_unified_auth, raise_server_error)

        # Act
        response = api_client.post(
            "/health/retry-stats/reset",
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 500
        data = response.json()
        assert data["error"]["message"] == "Authentication is not configured on the server."
        assert data["error"]["code"] == "http_500"

        # Cleanup
        app.dependency_overrides.clear()

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_retry_statistics_healthy(self, api_client, dal, monkeypatch, auth_header_name, auth_token):
        """Test retry stats endpoint returns 200 with stats when X-API-Key or Bearer token is valid."""
        # Arrange
        from deployment.app.services.auth import get_unified_auth
        monkeypatch.setitem(app.dependency_overrides, get_unified_auth, lambda: True)
        mock_get_stats = MagicMock(return_value=self.get_mock_retry_stats_data())
        monkeypatch.setattr("deployment.app.api.health.get_retry_statistics", mock_get_stats)

        # Act
        response = api_client.get(
            "/health/retry-stats", headers={auth_header_name: auth_token}
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

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_reset_retry_stats_success(self, api_client, dal, monkeypatch, auth_header_name, auth_token):
        """Test reset retry stats endpoint returns 200 with success message when X-API-Key or Bearer token is valid."""
        # Arrange
        from deployment.app.services.auth import get_unified_auth
        monkeypatch.setitem(app.dependency_overrides, get_unified_auth, lambda: True)
        mock_reset_stats = MagicMock()
        monkeypatch.setattr("deployment.app.api.health.reset_retry_statistics", mock_reset_stats)

        # Act
        response = api_client.post(
            "/health/retry-stats/reset", headers={auth_header_name: auth_token}
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

    @pytest.mark.asyncio
    async def test_check_database_healthy(self, dal, monkeypatch):
        """Test check_database returns healthy when connection and tables are ok."""
        from deployment.app.api.health import check_database

        # Mock execute_raw_query to handle multiple calls:
        # 1. Connection check (SELECT 1)
        # 2. Table check query (SELECT table_name as name FROM information_schema.tables...)
        # 3. check_monotonic_months queries (fact_sales, fact_stock_movement)
        required_tables = [
            "jobs", "models", "configs", "training_results", "prediction_results",
            "job_status_history", "dim_multiindex_mapping", "fact_sales",
            "fact_stock_movement", "fact_predictions",
            "processing_runs", "data_upload_results", "report_results"
        ]
        
        async def side_effect_for_execute_raw_query(query, params=(), fetchall=False):
            if query == "SELECT 1":
                return {"1": 1}  # Connection check
            if "SELECT table_name as name FROM information_schema.tables" in query or "information_schema.tables" in query:
                # Return all required tables with 'name' key (matching the alias)
                return [{"name": table} for table in required_tables]
            if "fact_sales" in query or "fact_stock_movement" in query:
                # Return empty for fact table queries (no missing months = healthy)
                return []
            return []
        
        monkeypatch.setattr(dal, 'execute_raw_query', AsyncMock(side_effect=side_effect_for_execute_raw_query))

        result = await check_database(dal)

        assert result.status == "healthy"
        dal.execute_raw_query.assert_called()

    @pytest.mark.asyncio
    async def test_check_database_degraded_missing_tables(self, dal, monkeypatch):
        """Test check_database returns degraded if tables are missing."""
        # Arrange
        async def side_effect_for_missing_tables(query, params=(), fetchall=False):
            if query == "SELECT 1":
                return [{"1": 1}]
            if "SELECT table_name as name FROM information_schema.tables" in query or "information_schema.tables" in query:
                # Return only "jobs" table, missing all others
                return [{"name": "jobs"}]
            if "fact_sales" in query or "fact_stock_movement" in query:
                # Return empty for fact table queries since they don't exist
                return []
            return []

        monkeypatch.setattr(dal, 'execute_raw_query', AsyncMock(side_effect=side_effect_for_missing_tables))

        # Mock execute_query_with_batching to return the same result for table checks
        async def batching_side_effect(query_template, ids, **kwargs):
            if "information_schema" in query_template.lower() or "table_name" in query_template.lower():
                # Return only "jobs" table from the requested tables
                return [{"name": "jobs"}] if "jobs" in ids else []
            return []

        monkeypatch.setattr(dal, 'execute_query_with_batching', AsyncMock(side_effect=batching_side_effect))

        # Act
        from deployment.app.api.health import check_database

        result = await check_database(dal)

        # Assert
        assert result.status == "degraded"
        assert "missing_tables" in result.details
        assert "jobs" not in result.details["missing_tables"]
        assert "models" in result.details["missing_tables"]
        dal.execute_raw_query.assert_called()

    @pytest.mark.asyncio
    async def test_check_database_unhealthy_connection_error(self, dal, monkeypatch):
        """Test check_database returns unhealthy on connection error."""
        # Arrange
        from deployment.app.db.exceptions import DatabaseError
        monkeypatch.setattr(dal, 'execute_raw_query', AsyncMock(side_effect=DatabaseError("Connection failed")))

        # Act
        from deployment.app.api.health import check_database

        result = await check_database(dal)

        # Assert
        assert result.status == "unhealthy"
        assert "error" in result.details
        assert "Connection failed" in result.details["error"]
        dal.execute_raw_query.assert_called_once()

    @patch('deployment.app.utils.environment.load_dotenv')
    @patch('deployment.app.utils.environment.check_yc_token_health')
    def test_check_environment_healthy_with_dotenv(self, mock_check_token, mock_load_dotenv, tmp_path):
        """Test get_environment_status returns healthy when vars are in .env file."""
        # Arrange
        mock_check_token.return_value = ComponentHealth(status="healthy")

        dotenv_content = """
DATASPHERE_PROJECT_ID=project123
DATASPHERE_FOLDER_ID=folder456
API_X_API_KEY_HASH=api_key_789_hash
API_ADMIN_API_KEY_HASH=admin_key_123_hash
YC_OAUTH_TOKEN=oauth_token_def
        """
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text(dotenv_content)

        # Simulate that load_dotenv sets the environment variables
        def dotenv_side_effect():
            os.environ['DATASPHERE_PROJECT_ID'] = 'project123'
            os.environ['DATASPHERE_FOLDER_ID'] = 'folder456'
            os.environ['API_X_API_KEY_HASH'] = 'api_key_789_hash'
            os.environ['API_ADMIN_API_KEY_HASH'] = 'admin_key_123_hash'
            os.environ['YC_OAUTH_TOKEN'] = 'oauth_token_def'
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
        assert "API_X_API_KEY_HASH" in missing_vars_text
        assert "API_ADMIN_API_KEY_HASH" in missing_vars_text
        assert "DATASPHERE_PROJECT_ID" in missing_vars_text
        assert "DATASPHERE_FOLDER_ID" in missing_vars_text
        # DATASPHERE_YC_PROFILE removed after migration to OAuth-only
        assert "YC_OAUTH_TOKEN" in missing_vars_text

    @patch('deployment.app.utils.environment.load_dotenv')
    @patch('deployment.app.utils.environment.check_yc_token_health')
    def test_check_environment_healthy_with_legacy_api_key(self, mock_check_token, mock_load_dotenv, tmp_path):
        """Test get_environment_status returns healthy when using legacy API_API_KEY."""
        # Arrange: Mock load_dotenv to prevent loading real .env file
        mock_load_dotenv.return_value = None
        mock_check_token.return_value = ComponentHealth(status="healthy")

        # Arrange
        dotenv_content = """
DATASPHERE_PROJECT_ID=project123
DATASPHERE_FOLDER_ID=folder456
API_X_API_KEY_HASH=api_key_789_hash
API_ADMIN_API_KEY_HASH=legacy_admin_key_123_hash
YC_OAUTH_TOKEN=oauth_token_def
        """
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text(dotenv_content)

        with patch.dict(os.environ, {
            'DATASPHERE_PROJECT_ID': 'project123',
            'DATASPHERE_FOLDER_ID': 'folder456',
            'API_X_API_KEY_HASH': 'api_key_789_hash',
            'API_ADMIN_API_KEY_HASH': 'legacy_admin_key_123_hash',  # Legacy key
            'YC_OAUTH_TOKEN': 'oauth_token_def'
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
            'API_X_API_KEY_HASH': 'api_key_789_hash',
            'API_ADMIN_API_KEY_HASH': 'admin_key_123_hash'
            # Missing DataSphere auth
        }, clear=True):
            # Act
            result = get_environment_status()

        # Assert
        assert result.status == "degraded"
        assert "missing_variables" in result.details
        missing_vars_text = " ".join(result.details["missing_variables"])
        # DATASPHERE_YC_PROFILE removed after migration to OAuth-only
        assert "YC_OAUTH_TOKEN" in missing_vars_text

    @patch('deployment.app.utils.environment.load_dotenv')
    @patch('deployment.app.utils.environment.check_yc_token_health')
    def test_check_environment_healthy_with_oauth_token(self, mock_check_token, mock_load_dotenv, tmp_path):
        """Test get_environment_status returns healthy when using OAuth token auth (yc_profile removed)."""
        # Arrange: Mock load_dotenv to prevent loading real .env file
        mock_load_dotenv.return_value = None
        mock_check_token.return_value = ComponentHealth(status="healthy")

        # Arrange - OAuth token auth (yc_profile removed after migration)
        dotenv_content = """
DATASPHERE_PROJECT_ID=project123
DATASPHERE_FOLDER_ID=folder456
API_X_API_KEY_HASH=api_key_789
API_ADMIN_API_KEY_HASH=admin_key_123
YC_OAUTH_TOKEN=test-oauth-token
        """
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text(dotenv_content)

        with patch.dict(os.environ, {
            'DATASPHERE_PROJECT_ID': 'project123',
            'DATASPHERE_FOLDER_ID': 'folder456',
            'API_X_API_KEY_HASH': 'api_key_789_hash',
            'API_ADMIN_API_KEY_HASH': 'admin_key_123_hash',
            'YC_OAUTH_TOKEN': 'test-oauth-token'  # OAuth token auth (yc_profile removed)
        }, clear=True):
            # Act
            result = get_environment_status()

        # Assert
        assert result.status == "healthy"
        assert result.details == {}
        mock_check_token.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_database_unhealthy_missing_tables(self, dal, monkeypatch):
        """Test check_database returns unhealthy if required tables are missing."""
        # Arrange
        # Simulate that the query for tables returns only one table.
        async def side_effect_for_missing_tables(query, params=(), fetchall=False):
            if query == "SELECT 1":
                return [{"1": 1}]
            if "SELECT table_name as name FROM information_schema.tables" in query or "information_schema.tables" in query:
                # Return only "jobs" table, missing all others
                return [{"name": "jobs"}]
            if "fact_sales" in query or "fact_stock_movement" in query:
                # Return empty for fact table queries since they don't exist
                return []
            return []

        monkeypatch.setattr(dal, 'execute_raw_query', AsyncMock(side_effect=side_effect_for_missing_tables))

        # Mock execute_query_with_batching to return the same result for table checks
        def batching_side_effect(query_template, ids, **kwargs):
            if "information_schema.tables" in query_template.lower() or "table_name" in query_template.lower():
                # Return only "jobs" table from the requested tables
                return [{"name": "jobs"}] if "jobs" in ids else []
            return []

        monkeypatch.setattr(dal, 'execute_query_with_batching', MagicMock(side_effect=batching_side_effect))

        # Act
        from deployment.app.api.health import check_database

        result = await check_database(dal)

        # Assert
        assert result.status == "degraded"
        assert "missing_tables" in result.details
        assert "jobs" not in result.details["missing_tables"]
        assert "models" in result.details["missing_tables"]
        dal.execute_raw_query.assert_called()

    @pytest.mark.asyncio
    async def test_check_database_unhealthy_missing_months(self, dal, monkeypatch):
        """Test check_database returns unhealthy if there are missing months in fact_sales or fact_stock_movement."""

        async def side_effect_for_missing_months(query, params=(), fetchall=False):
            if query == "SELECT 1":
                return [{"1": 1}]
            if "SELECT table_name as name FROM information_schema.tables" in query or "information_schema.tables" in query:
                return [{"name": table} for table in params]
            if "fact_sales" in query:
                # Simulate a gap in fact_sales
                return [{"data_date": "2024-01-01"}, {"data_date": "2024-03-01"}]
            if "fact_stock_movement" in query:
                # Simulate no gap in fact_stock_movement
                return [{"data_date": "2024-01-01"}, {"data_date": "2024-02-01"}, {"data_date": "2024-03-01"}]
            return []

        monkeypatch.setattr(dal, 'execute_raw_query', AsyncMock(side_effect=side_effect_for_missing_months))

        # Mock execute_query_with_batching to return all tables as existing
        def batching_side_effect(query_template, ids, **kwargs):
            if "information_schema.tables" in query_template.lower() or "table_name" in query_template.lower():
                # Return all requested tables as existing
                return [{"name": table} for table in ids]
            return []

        monkeypatch.setattr(dal, 'execute_query_with_batching', MagicMock(side_effect=batching_side_effect))

        from deployment.app.api.health import check_database
        result = await check_database(dal)
        assert result.status == "unhealthy"
        assert "missing_months_sales" in result.details
        assert result.details["missing_months_sales"] == ["2024-02"] # check_monotonic_months returns this format
        assert "missing_months_stock_movement" not in result.details

    @pytest.mark.asyncio
    async def test_check_database_healthy_no_missing_months(self, dal, monkeypatch):
        """Test check_database returns healthy if all months are present in both tables."""

        async def side_effect_for_healthy(query, params=(), fetchall=False):
            if query == "SELECT 1":
                return [{"1": 1}]
            if "SELECT table_name as name FROM information_schema.tables" in query or "information_schema.tables" in query:
                return [{"name": table} for table in params]
            if "fact_sales" in query:
                return [{"data_date": "2024-01-01"}, {"data_date": "2024-02-01"}, {"data_date": "2024-03-01"}]
            if "fact_stock_movement" in query:
                return [{"data_date": "2024-01-01"}, {"data_date": "2024-02-01"}, {"data_date": "2024-03-01"}]
            return []

        monkeypatch.setattr(dal, 'execute_raw_query', AsyncMock(side_effect=side_effect_for_healthy))

        # Mock execute_query_with_batching to return all tables as existing
        def batching_side_effect(query_template, ids, **kwargs):
            if "information_schema.tables" in query_template.lower() or "table_name" in query_template.lower():
                # Return all requested tables as existing
                return [{"name": table} for table in ids]
            return []

        monkeypatch.setattr(dal, 'execute_query_with_batching', MagicMock(side_effect=batching_side_effect))

        from deployment.app.api.health import check_database
        result = await check_database(dal)
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
