"""
Comprehensive tests for deployment.app.api.admin

This test suite covers all admin data retention endpoints with comprehensive mocking
of external dependencies. Tests are organized by endpoint groups and include both
success and failure scenarios.

Testing Approach:
- Mock all external dependencies (database connections, cleanup functions, background tasks)
- Test admin data retention endpoints with different scenarios (success/failure/default params)
- Test authentication and authorization scenarios for all protected endpoints
- Test request validation and parameter handling (query params for days/counts)
- Test response formats and HTTP status codes
- Test background task triggering and cleanup function calls
- Test database connection handling and cleanup operations
- Verify proper service layer interaction and data retention logic

All external imports and dependencies are mocked to ensure test isolation.
The admin API uses Bearer token authentication rather than X-API-Key.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

TEST_BEARER_TOKEN = "test_admin_token"


@pytest.fixture
def admin_client(api_client: TestClient):
    """A test api_client with the database dependency overridden for admin endpoints."""
    # The api_client fixture already has the correct DAL dependency overrides
    yield api_client


class TestCleanupJobEndpoint:
    """Test suite for /admin/data-retention/cleanup endpoint."""

    @pytest.mark.asyncio
    async def test_trigger_cleanup_job_success(
        self, admin_client: TestClient, monkeypatch, dal
    ):
        """Test cleanup job endpoint successfully starts a background cleanup job."""
        # Arrange
        mock_run_cleanup = AsyncMock()
        monkeypatch.setattr("deployment.app.api.admin.run_cleanup_job", mock_run_cleanup)

        # Act
        with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
            response = admin_client.post(
                "/admin/data-retention/cleanup",
                headers={"Authorization": f"Bearer {TEST_BEARER_TOKEN}"},
            )

            # Assert
            assert response.status_code == 200
            assert "Data retention cleanup job started" in response.json()["message"]
            mock_add_task.assert_called_once_with(mock_run_cleanup, dal)

    def test_trigger_cleanup_unauthorized(self, admin_client: TestClient):
        """Test cleanup job endpoint fails without valid Bearer token."""
        # Act
        response = admin_client.post(
            "/admin/data-retention/cleanup",
            headers={"Authorization": "Bearer wrong_token"},
        )

        # Assert
        assert response.status_code == 401


class TestPredictionsCleanupEndpoint:
    """Test suite for /admin/data-retention/clean-predictions endpoint."""

    @pytest.mark.asyncio
    @patch("deployment.app.api.admin.cleanup_old_predictions")
    async def test_clean_predictions_success(
        self,
        mock_cleanup_old_predictions: AsyncMock,
        admin_client: TestClient,
        dal,
    ):
        """Test predictions cleanup endpoint successfully cleans old predictions."""
        # Arrange
        mock_cleanup_old_predictions.return_value = 5

        # Act
        response = admin_client.post(
            "/admin/data-retention/clean-predictions",
            params={"days_to_keep": 30},
            headers={"Authorization": f"Bearer {TEST_BEARER_TOKEN}"},
        )

        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ok"
        assert result["records_removed"] == 5
        assert result["days_kept"] == 30
        # cleanup_old_predictions is called with positional days_to_keep and keyword dal
        mock_cleanup_old_predictions.assert_called_once_with(
            30, dal=dal
        )

    @pytest.mark.asyncio
    @patch("deployment.app.api.admin.cleanup_old_predictions")
    async def test_clean_predictions_default_days(
        self,
        mock_cleanup_old_predictions: AsyncMock,
        admin_client: TestClient,
        dal,
    ):
        """Test predictions cleanup endpoint with default days parameter."""
        # Arrange
        mock_cleanup_old_predictions.return_value = 3

        # Act
        response = admin_client.post(
            "/admin/data-retention/clean-predictions",
            headers={"Authorization": f"Bearer {TEST_BEARER_TOKEN}"},
        )

        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ok"
        assert result["records_removed"] == 3
        assert result["days_kept"] is None
        mock_cleanup_old_predictions.assert_called_once_with(
            None, dal=dal
        )

    def test_clean_predictions_unauthorized(self, admin_client: TestClient):
        """Test predictions cleanup endpoint fails without valid Bearer token."""
        # Act
        response = admin_client.post(
            "/admin/data-retention/clean-predictions",
            headers={"Authorization": "Bearer wrong_token"},
        )

        # Assert
        assert response.status_code == 401


class TestHistoricalDataCleanupEndpoint:
    """Test suite for /admin/data-retention/clean-historical endpoint."""

    @pytest.mark.asyncio
    @patch("deployment.app.api.admin.cleanup_old_historical_data")
    async def test_clean_historical_data_success(
        self,
        mock_cleanup_old_historical_data: AsyncMock,
        admin_client: TestClient,
        dal,
    ):
        """Test historical data cleanup endpoint successfully cleans old data."""
        # Arrange
        mock_cleanup_old_historical_data.return_value = {
            "sales": 10,
            "stock_movement": 5,
        }

        # Act
        response = admin_client.post(
            "/admin/data-retention/clean-historical",
            params={"sales_days_to_keep": 60, "stock_days_to_keep": 90},
            headers={"Authorization": f"Bearer {TEST_BEARER_TOKEN}"},
        )

        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ok"
        mock_cleanup_old_historical_data.assert_called_once_with(
            60, 90, dal=dal
        )

    @pytest.mark.asyncio
    @patch("deployment.app.api.admin.cleanup_old_historical_data")
    async def test_clean_historical_data_default_days(
        self,
        mock_cleanup_old_historical_data: AsyncMock,
        admin_client: TestClient,
        dal,
    ):
        """Test historical data cleanup endpoint with default parameters."""
        # Arrange
        mock_cleanup_old_historical_data.return_value = {
            "sales": 8,
            "stock_movement": 6,
        }

        # Act
        response = admin_client.post(
            "/admin/data-retention/clean-historical",
            headers={"Authorization": f"Bearer {TEST_BEARER_TOKEN}"},
        )

        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ok"
        mock_cleanup_old_historical_data.assert_called_once_with(
            None, None, dal=dal
        )

    def test_clean_historical_data_unauthorized(self, admin_client: TestClient):
        """Test historical data cleanup endpoint fails without valid Bearer token."""
        # Act
        response = admin_client.post(
            "/admin/data-retention/clean-historical",
            headers={"Authorization": "Bearer wrong_token"},
        )

        # Assert
        assert response.status_code == 401


class TestModelsCleanupEndpoint:
    """Test suite for /admin/data-retention/clean-models endpoint."""

    @pytest.mark.asyncio
    @patch("deployment.app.api.admin.cleanup_old_models")
    async def test_clean_models_success(
        self,
        mock_cleanup_old_models: AsyncMock,
        admin_client: TestClient,
        dal,
    ):
        """Test models cleanup endpoint successfully cleans old models."""
        # Arrange
        mock_cleanup_old_models.return_value = ["model1", "model2", "model3"]

        # Act
        response = admin_client.post(
            "/admin/data-retention/clean-models",
            params={"models_to_keep": 5, "inactive_days_to_keep": 30},
            headers={"Authorization": f"Bearer {TEST_BEARER_TOKEN}"},
        )

        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ok"
        assert result["models_removed_count"] == 3
        assert result["models_kept"] == 5
        assert result["inactive_days_kept"] == 30
        mock_cleanup_old_models.assert_called_once_with(
            5, 30, dal=dal
        )

    @pytest.mark.asyncio
    @patch("deployment.app.api.admin.cleanup_old_models")
    async def test_clean_models_default_params(
        self,
        mock_cleanup_old_models: AsyncMock,
        admin_client: TestClient,
        dal,
    ):
        """Test models cleanup endpoint with default parameters."""
        # Arrange
        mock_cleanup_old_models.return_value = ["model4", "model5"]

        # Act
        response = admin_client.post(
            "/admin/data-retention/clean-models",
            headers={"Authorization": f"Bearer {TEST_BEARER_TOKEN}"},
        )

        # Assert
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "ok"
        assert result["models_removed_count"] == 2
        mock_cleanup_old_models.assert_called_once_with(
            None, None, dal=dal
        )

    def test_clean_models_unauthorized(self, admin_client: TestClient):
        """Test models cleanup endpoint fails without valid Bearer token."""
        # Act
        response = admin_client.post(
            "/admin/data-retention/clean-models",
            headers={"Authorization": "Bearer wrong_token"},
        )

        # Assert
        assert response.status_code == 401


class TestAuthenticationScenarios:
    """Test suite for authentication and authorization scenarios across all admin endpoints."""

    def test_all_endpoints_require_authorization(self, admin_client: TestClient):
        """Test all data retention endpoints fail without valid Bearer token."""
        # Arrange
        endpoints = [
            "/admin/data-retention/cleanup",
            "/admin/data-retention/clean-predictions",
            "/admin/data-retention/clean-historical",
            "/admin/data-retention/clean-models",
        ]

        # Act & Assert
        for endpoint in endpoints:
            response = admin_client.post(
                endpoint, headers={"Authorization": "Bearer wrong_token"}
            )
            assert response.status_code == 401, (
                f"Endpoint {endpoint} should be protected"
            )

    def test_endpoints_require_bearer_prefix(self, admin_client: TestClient):
        """Test endpoints fail with token but missing Bearer prefix."""
        # Arrange
        endpoints = [
            "/admin/data-retention/cleanup",
            "/admin/data-retention/clean-predictions",
            "/admin/data-retention/clean-historical",
            "/admin/data-retention/clean-models",
        ]

        # Act & Assert
        for endpoint in endpoints:
            response = admin_client.post(
                endpoint, headers={"Authorization": TEST_BEARER_TOKEN}
            )  # Missing "Bearer "
            assert response.status_code == 401, (
                f"Endpoint {endpoint} should return 401 Unauthorized for missing Bearer prefix"
            )

    def test_endpoints_require_authorization_header(self, admin_client: TestClient):
        """Test endpoints fail with missing Authorization header."""
        # Arrange
        endpoints = [
            "/admin/data-retention/cleanup",
            "/admin/data-retention/clean-predictions",
            "/admin/data-retention/clean-historical",
            "/admin/data-retention/clean-models",
        ]

        # Act & Assert
        for endpoint in endpoints:
            response = admin_client.post(endpoint)  # No Authorization header
            assert response.status_code == 401, (
                f"Endpoint {endpoint} should return 401 Unauthorized for missing Authorization header"
            )


class TestIntegration:
    """Integration tests for module imports and configuration."""

    def test_module_imports_successfully(self):
        """Test that the admin module can be imported without errors."""
        # Act & Assert
        try:
            import deployment.app.api.admin

            # Test that key components are available
            assert hasattr(deployment.app.api.admin, "router")
            # assert hasattr(deployment.app.api.admin, "run_cleanup_job") # Removed
            # assert hasattr(deployment.app.api.admin, "cleanup_old_predictions") # Removed
            # assert hasattr(deployment.app.api.admin, "cleanup_old_models") # Removed
            # assert hasattr(deployment.app.api.admin, "cleanup_old_historical_data") # Removed
        except ImportError as e:
            pytest.fail(f"Failed to import admin module: {e}")

    def test_data_retention_module_imports(self):
        """Test that the data retention module can be imported without errors."""
        # Act & Assert
        try:
            import deployment.app.db.data_retention

            # Test that key components are available
            assert hasattr(deployment.app.db.data_retention, "run_cleanup_job")
            assert hasattr(deployment.app.db.data_retention, "cleanup_old_predictions")
            assert hasattr(deployment.app.db.data_retention, "cleanup_old_models")
            assert hasattr(
                deployment.app.db.data_retention, "cleanup_old_historical_data"
            )
        except ImportError as e:
            pytest.fail(f"Failed to import data_retention module: {e}")

    def test_router_configuration(self):
        """Test that the admin router is properly configured."""
        # Act & Assert
        from deployment.app.api.admin import router

        # Test router exists and has expected attributes
        assert router is not None
        assert hasattr(router, "routes")
        assert len(router.routes) > 0

        # Test that expected routes exist
        route_paths = [route.path for route in router.routes]
        expected_paths = [
            "/data-retention/cleanup",
            "/data-retention/clean-predictions",
            "/data-retention/clean-historical",
            "/data-retention/clean-models",
        ]

        for expected_path in expected_paths:
            assert any(expected_path in path for path in route_paths), (
                f"Expected path containing '{expected_path}' not found in {route_paths}"
            )
