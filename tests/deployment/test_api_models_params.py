"""
Comprehensive tests for deployment.app.api.models_configs

This test suite covers all model configuration endpoints with comprehensive mocking
of external dependencies. Tests are organized by endpoint groups and include both 
success and failure scenarios.

Testing Approach:
- Mock all external dependencies (database operations, service layer calls)
- Test model configuration endpoints with different scenarios (success/not found/error)
- Test authentication and authorization scenarios for all protected endpoints
- Test request validation and parameter handling (query params, path params, JSON body)
- Test response formats and HTTP status codes
- Test error handling scenarios and edge cases
- Verify proper service layer interaction and database calls

All external imports and dependencies are mocked to ensure test isolation.
"""

import uuid
from unittest.mock import patch

import pytest
from fastapi import HTTPException

TEST_X_API_KEY = "test_x_api_key_conftest"


class TestActiveConfigEndpoint:
    """Test suite for /api/v1/models-configs/configs/active endpoint."""

    @patch("deployment.app.api.models_configs.get_active_config")
    def test_get_active_config_success(self, mock_get_active_config, client):
        """Test successful retrieval of active configuration."""
        # Arrange
        config_id = str(uuid.uuid4())
        mock_get_active_config.return_value = {
            "config_id": config_id,
            "configs": {
                "input_chunk_length": 12,
                "output_chunk_length": 6,
                "max_epochs": 10
            },
            "is_active": True
        }

        # Act
        response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["config_id"] == config_id
        assert data["configs"]["input_chunk_length"] == 12
        assert data["configs"]["output_chunk_length"] == 6
        assert data["configs"]["max_epochs"] == 10
        assert data["is_active"] is True
        mock_get_active_config.assert_called_once()

    @patch("deployment.app.api.models_configs.get_active_config", return_value=None)
    def test_get_active_config_not_found(self, mock_get_active_config, client):
        """Test 404 response when no active configuration exists."""
        # Act
        response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 404
        assert "No active config found" in response.text
        mock_get_active_config.assert_called_once()

    def test_get_active_config_unauthorized_missing_key(self, client):
        """Test active config endpoint fails with 401 if X-API-Key header is missing."""
        # Act
        response = client.get("/api/v1/models-configs/configs/active")

        # Assert
        assert response.status_code == 401

    def test_get_active_config_unauthorized_invalid_key(self, client):
        """Test active config endpoint fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": "wrong_key"})

        # Assert
        assert response.status_code == 401

    def test_get_active_config_server_key_not_configured(self, client, mock_x_api_key):
        """Test active config endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        mock_x_api_key(None)
        
        # Act
        response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 500


class TestActivateConfigEndpoint:
    """Test suite for /api/v1/models-configs/configs/{config_id}/set-active endpoint."""

    @patch("deployment.app.api.models_configs.set_config_active", return_value=True)
    def test_activate_config_success(self, mock_set_active, client):
        """Test successful activation of a configuration."""
        # Arrange
        config_id = str(uuid.uuid4())

        # Act
        response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert f"Config {config_id} set as active" in data["message"]
        mock_set_active.assert_called_once_with(config_id)

    @patch("deployment.app.api.models_configs.set_config_active", return_value=False)
    def test_activate_config_not_found(self, mock_set_active, client):
        """Test 404 response when configuration to activate doesn't exist."""
        # Arrange
        config_id = str(uuid.uuid4())

        # Act
        response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 404
        assert f"Config {config_id} not found" in response.text
        mock_set_active.assert_called_once_with(config_id)

    def test_activate_config_unauthorized_missing_key(self, client):
        """Test activate config endpoint fails with 401 if X-API-Key header is missing."""
        # Arrange
        config_id = str(uuid.uuid4())

        # Act
        response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active")

        # Assert
        assert response.status_code == 401

    def test_activate_config_unauthorized_invalid_key(self, client):
        """Test activate config endpoint fails with 401 if X-API-Key is invalid."""
        # Arrange
        config_id = str(uuid.uuid4())

        # Act
        response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": "wrong_key"})

        # Assert
        assert response.status_code == 401

    def test_activate_config_server_key_not_configured(self, client, mock_x_api_key):
        """Test activate config endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        config_id = str(uuid.uuid4())
        mock_x_api_key(None)

        # Act
        response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 500


class TestBestConfigEndpoint:
    """Test suite for /api/v1/models-configs/configs/best endpoint."""

    @patch("deployment.app.api.models_configs.get_best_config_by_metric")
    @patch("deployment.app.api.models_configs.get_settings")
    def test_get_best_config_custom_metric(self, mock_get_settings, mock_get_best, client):
        """Test retrieval of best configuration with custom metric parameters."""
        # Arrange
        config_id = str(uuid.uuid4())
        mock_get_best.return_value = {
            "config_id": config_id,
            "configs": {
                "input_chunk_length": 24,
                "output_chunk_length": 12
            },
            "is_active": False
        }

        # Act
        response = client.get("/api/v1/models-configs/configs/best?metric_name=mae&higher_is_better=false", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["config_id"] == config_id
        assert data["configs"]["input_chunk_length"] == 24
        assert data["configs"]["output_chunk_length"] == 12
        assert data["is_active"] is False
        mock_get_best.assert_called_once_with("mae", False)

    @patch("deployment.app.api.models_configs.get_best_config_by_metric")
    @patch("deployment.app.api.models_configs.get_settings")
    def test_get_best_config_default_metric(self, mock_get_settings, mock_get_best, client):
        """Test retrieval of best configuration with default metric from settings."""
        # Arrange
        config_id = str(uuid.uuid4())
        mock_get_settings.return_value.default_metric = "mape"
        mock_get_settings.return_value.default_metric_higher_is_better = False
        mock_get_best.return_value = {
            "config_id": config_id,
            "configs": {"batch_size": 32},
            "is_active": False
        }

        # Act
        response = client.get("/api/v1/models-configs/configs/best", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["config_id"] == config_id
        assert data["configs"]["batch_size"] == 32
        mock_get_best.assert_called_once_with("mape", False)

    @patch("deployment.app.api.models_configs.get_best_config_by_metric", return_value=None)
    @patch("deployment.app.api.models_configs.get_settings")
    def test_get_best_config_not_found(self, mock_get_settings, mock_get_best, client):
        """Test 404 response when no configurations with the metric exist."""
        # Arrange
        mock_get_settings.return_value.default_metric = "rmse"
        mock_get_settings.return_value.default_metric_higher_is_better = False

        # Act
        response = client.get("/api/v1/models-configs/configs/best", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 404
        assert "No configs found with metric" in response.text
        mock_get_best.assert_called_once_with("rmse", False)

    def test_get_best_config_unauthorized_missing_key(self, client):
        """Test best config endpoint fails with 401 if X-API-Key header is missing."""
        # Act
        response = client.get("/api/v1/models-configs/configs/best")

        # Assert
        assert response.status_code == 401

    def test_get_best_config_unauthorized_invalid_key(self, client):
        """Test best config endpoint fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.get("/api/v1/models-configs/configs/best", headers={"X-API-Key": "wrong_key"})

        # Assert
        assert response.status_code == 401

    def test_get_best_config_server_key_not_configured(self, client, mock_x_api_key):
        """Test best config endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        mock_x_api_key(None)
        
        # Act
        response = client.get("/api/v1/models-configs/configs/best", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 500


class TestConfigListEndpoint:
    """Test suite for /api/v1/models-configs/configs endpoint (GET)."""

    @patch("deployment.app.api.models_configs.get_configs")
    def test_get_configs_success(self, mock_get_configs, client):
        """Test successful retrieval of configuration list."""
        # Arrange
        config_id_1 = str(uuid.uuid4())
        config_id_2 = str(uuid.uuid4())
        mock_get_configs.return_value = [
            {
                "config_id": config_id_1,
                "configs": {"input_chunk_length": 12, "output_chunk_length": 6},
                "is_active": True,
                "created_at": "2023-01-01T00:00:00Z"
            },
            {
                "config_id": config_id_2,
                "configs": {"input_chunk_length": 24, "output_chunk_length": 12},
                "is_active": False,
                "created_at": "2023-01-02T00:00:00Z"
            }
        ]

        # Act
        response = client.get("/api/v1/models-configs/configs", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["config_id"] == config_id_1
        assert data[0]["is_active"] is True
        assert data[1]["config_id"] == config_id_2
        assert data[1]["is_active"] is False
        mock_get_configs.assert_called_once()

    @patch("deployment.app.api.models_configs.get_configs", return_value=[])
    def test_get_configs_empty(self, mock_get_configs, client):
        """Test retrieval when no configurations exist."""
        # Act
        response = client.get("/api/v1/models-configs/configs", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data == []
        mock_get_configs.assert_called_once()

    def test_get_configs_unauthorized_missing_key(self, client):
        """Test config list endpoint fails with 401 if X-API-Key header is missing."""
        # Act
        response = client.get("/api/v1/models-configs/configs")

        # Assert
        assert response.status_code == 401

    def test_get_configs_unauthorized_invalid_key(self, client):
        """Test config list endpoint fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.get("/api/v1/models-configs/configs", headers={"X-API-Key": "wrong_key"})

        # Assert
        assert response.status_code == 401

    def test_get_configs_server_key_not_configured(self, client, mock_x_api_key):
        """Test config list endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        mock_x_api_key(None)
        
        # Act
        response = client.get("/api/v1/models-configs/configs", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 500


class TestConfigDeleteEndpoint:
    """Test suite for /api/v1/models-configs/configs/delete endpoint."""

    @patch("deployment.app.api.models_configs.delete_configs_by_ids")
    def test_delete_configs_success(self, mock_delete_configs, client):
        """Test successful deletion of configurations by IDs."""
        # Arrange
        config_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        mock_delete_configs.return_value = {"successful": 2, "failed": 0, "errors": []}
        request_data = {"ids": config_ids}

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/delete",
            headers={"X-API-Key": TEST_X_API_KEY},
            json=request_data
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["successful"] == 2
        assert data["failed"] == 0
        assert data["errors"] == []
        mock_delete_configs.assert_called_once_with(config_ids)

    def test_delete_configs_empty_ids(self, client):
        """Test deletion with empty config IDs list."""
        # Arrange
        request_data = {"ids": []}

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/delete",
            headers={"X-API-Key": TEST_X_API_KEY},
            json=request_data
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["successful"] == 0
        assert data["failed"] == 0
        assert "No IDs provided" in data["errors"][0]

    def test_delete_configs_unauthorized_missing_key(self, client):
        """Test delete configs endpoint fails with 401 if X-API-Key header is missing."""
        # Arrange
        request_data = {"ids": [str(uuid.uuid4())]}

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/delete",
            json=request_data
        )

        # Assert
        assert response.status_code == 401

    def test_delete_configs_unauthorized_invalid_key(self, client):
        """Test delete configs endpoint fails with 401 if X-API-Key is invalid."""
        # Arrange
        request_data = {"ids": [str(uuid.uuid4())]}

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/delete",
            headers={"X-API-Key": "wrong_key"},
            json=request_data
        )

        # Assert
        assert response.status_code == 401

    def test_delete_configs_server_key_not_configured(self, client, mock_x_api_key):
        """Test delete configs endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        request_data = {"ids": [str(uuid.uuid4())]}
        mock_x_api_key(None)

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/delete",
            headers={"X-API-Key": TEST_X_API_KEY},
            json=request_data
        )

        # Assert
        assert response.status_code == 500

    @patch("deployment.app.api.models_configs.delete_configs_by_ids", side_effect=HTTPException(status_code=500, detail="Simulated DB error from service"))
    def test_delete_configs_error(self, mock_delete_configs, client):
        """Test delete configs handles service layer errors properly."""
        # Arrange
        config_ids = [str(uuid.uuid4())]
        request_data = {"ids": config_ids}

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/delete",
            headers={"X-API-Key": TEST_X_API_KEY},
            json=request_data
        )

        # Assert
        assert response.status_code == 500
        mock_delete_configs.assert_called_once_with(config_ids)


class TestConfigUploadEndpoint:
    """Test suite for /api/v1/models-configs/configs/upload endpoint."""

    @patch("deployment.app.db.database.create_or_get_config")
    def test_upload_config_success(self, mock_create_or_get_config, client):
        """Test successful upload of new configuration."""
        # Arrange
        config_id = str(uuid.uuid4())
        mock_create_or_get_config.return_value = config_id
        request_data = {
            "json_payload": {"input_chunk_length": 18, "output_chunk_length": 9, "batch_size": 64},
            "is_active": False
        }

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/upload",
            headers={"X-API-Key": TEST_X_API_KEY},
            json=request_data
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["config_id"] == config_id
        assert data["configs"] == request_data["json_payload"]
        assert data["is_active"] is False
        mock_create_or_get_config.assert_called_once_with(request_data["json_payload"], is_active=False)

    @patch("deployment.app.db.database.create_or_get_config", side_effect=Exception("DB error"))
    def test_upload_config_db_error(self, mock_create_or_get_config, client):
        """Test upload config handles database errors properly."""
        # Arrange
        request_data = {
            "json_payload": {"input_chunk_length": 18, "output_chunk_length": 9},
            "is_active": False
        }

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/upload",
            headers={"X-API-Key": TEST_X_API_KEY},
            json=request_data
        )

        # Assert
        assert response.status_code == 500
        mock_create_or_get_config.assert_called_once_with(request_data["json_payload"], is_active=False)

    def test_upload_config_unauthorized_missing_key(self, client):
        """Test upload config endpoint fails with 401 if X-API-Key header is missing."""
        # Arrange
        request_data = {"json_payload": {"input_chunk_length": 18}, "is_active": False}

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/upload",
            json=request_data
        )

        # Assert
        assert response.status_code == 401

    def test_upload_config_unauthorized_invalid_key(self, client):
        """Test upload config endpoint fails with 401 if X-API-Key is invalid."""
        # Arrange
        request_data = {"json_payload": {"input_chunk_length": 18}, "is_active": False}

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/upload",
            headers={"X-API-Key": "wrong_key"},
            json=request_data
        )

        # Assert
        assert response.status_code == 401

    def test_upload_config_server_key_not_configured(self, client, mock_x_api_key):
        """Test upload config endpoint fails with 500 if server X-API-Key is not configured."""
        # Arrange
        request_data = {"json_payload": {"input_chunk_length": 18}, "is_active": False}
        mock_x_api_key(None)

        # Act
        response = client.post(
            "/api/v1/models-configs/configs/upload",
            headers={"X-API-Key": TEST_X_API_KEY},
            json=request_data
        )

        # Assert
        assert response.status_code == 500


class TestIntegration:
    """Integration tests for module imports and configuration."""

    def test_module_imports_successfully(self):
        """Test that the models_configs module can be imported without errors."""
        # Act & Assert
        try:
            import deployment.app.api.models_configs
            # Test that key components are available
            assert hasattr(deployment.app.api.models_configs, 'router')
            assert hasattr(deployment.app.api.models_configs, 'get_active_config')
            assert hasattr(deployment.app.api.models_configs, 'set_config_active')
            assert hasattr(deployment.app.api.models_configs, 'get_best_config_by_metric')
        except ImportError as e:
            pytest.fail(f"Failed to import models_configs module: {e}")

    def test_router_configuration(self):
        """Test that the models_configs router is properly configured."""
        # Act & Assert
        from deployment.app.api.models_configs import router

        # Test router exists and has expected attributes
        assert router is not None
        assert hasattr(router, 'routes')
        assert len(router.routes) > 0

        # Test that expected routes exist
        route_paths = [route.path for route in router.routes]
        expected_paths = ["/configs/active", "/configs/best", "/configs", "/configs/{config_id}/set-active", "/configs/delete", "/configs/upload"]

        for expected_path in expected_paths:
            assert any(expected_path in path for path in route_paths), \
                   f"Expected path containing '{expected_path}' not found in {route_paths}"

    def test_constants_defined(self):
        """Test that expected constants and configuration are defined."""
        # Act & Assert
        # Test that the module has the expected structure
        from deployment.app.api.models_configs import router
        assert router.prefix is not None
        assert router.tags is not None
