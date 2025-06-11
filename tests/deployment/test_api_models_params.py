import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException # Added for side_effect
from unittest.mock import patch
import json
import uuid
from pathlib import Path
import io
from fastapi import status
from deployment.app.api.models_configs import router as models_params_router

from deployment.app.main import app
from deployment.app.db.database import DatabaseError
from unittest.mock import PropertyMock # Added for PropertyMock

TEST_X_API_KEY = "test_x_api_key_value_models_params"

# --- Test /api/v1/model-params/configs/active ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.get_active_config")
def test_get_active_config_success(mock_get_active_config, mock_server_api_key, client):
    """Test successful retrieval of active config."""
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
    
    response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["config_id"] == config_id
    assert data["configs"]["input_chunk_length"] == 12
    assert data["configs"]["output_chunk_length"] == 6
    assert data["configs"]["max_epochs"] == 10
    assert data["is_active"] is True
    mock_get_active_config.assert_called_once()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.get_active_config", return_value=None)
def test_get_active_config_not_found(mock_get_active_config, mock_server_api_key, client):
    """Test 404 response when no active config exists."""
    response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 404
    assert "No active config found" in response.text
    mock_get_active_config.assert_called_once()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_active_config_unauthorized_missing_key(mock_server_api_key, client):
    """Test get active config fails with 401 if X-API-Key header is missing."""
    response = client.get("/api/v1/models-configs/configs/active") # No header
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_active_config_unauthorized_invalid_key(mock_server_api_key, client):
    """Test get active config fails with 401 if X-API-Key is invalid."""
    response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value="")) # Server key not configured
def test_get_active_config_server_key_not_configured(mock_server_api_key, client):
    """Test get active config fails with 500 if server X-API-Key is not configured."""
    response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/configs/{config_id}/set-active ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.set_config_active", return_value=True)
def test_activate_config_success(mock_set_active, mock_server_api_key, client):
    """Test successful activation of a config."""
    config_id = str(uuid.uuid4())
    
    response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert f"Config {config_id} set as active" in data["message"]
    mock_set_active.assert_called_once_with(config_id)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.set_config_active", return_value=False)
def test_activate_config_not_found(mock_set_active, mock_server_api_key, client):
    """Test 404 response when config to activate doesn't exist."""
    config_id = str(uuid.uuid4())
    
    response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 404
    assert f"Config {config_id} not found" in response.text
    mock_set_active.assert_called_once_with(config_id)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_activate_config_unauthorized_missing_key(mock_server_api_key, client):
    """Test activate config fails with 401 if X-API-Key header is missing."""
    config_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active") # No header
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_activate_config_unauthorized_invalid_key(mock_server_api_key, client):
    """Test activate config fails with 401 if X-API-Key is invalid."""
    config_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value="")) # Server key not configured
def test_activate_config_server_key_not_configured(mock_server_api_key, client):
    """Test activate config fails with 500 if server X-API-Key is not configured."""
    config_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/configs/best ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.get_best_config_by_metric")
@patch("deployment.app.api.models_configs.settings")
def test_get_best_config_custom_metric(mock_settings, mock_get_best, mock_server_api_key, client):
    """Test retrieval of best config with custom metric."""
    config_id = str(uuid.uuid4())
    mock_get_best.return_value = {
        "config_id": config_id,
        "configs": {
            "input_chunk_length": 24,
            "output_chunk_length": 12
        },
        "is_active": False
    }
    
    response = client.get("/api/v1/models-configs/configs/best?metric_name=mae&higher_is_better=false", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["config_id"] == config_id
    assert data["configs"]["input_chunk_length"] == 24
    assert data["configs"]["output_chunk_length"] == 12
    assert data["is_active"] is False
    mock_get_best.assert_called_once_with("mae", False)
    mock_settings.default_metric.assert_not_called()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.get_best_config_by_metric")
@patch("deployment.app.api.models_configs.settings")
def test_get_best_config_default_metric(mock_settings, mock_get_best, mock_server_api_key, client):
    """Test retrieval of best config with default metric from settings."""
    config_id = str(uuid.uuid4())
    mock_settings.default_metric = "mape"
    mock_settings.default_metric_higher_is_better = False
    mock_get_best.return_value = {
        "config_id": config_id,
        "configs": {"batch_size": 32},
        "is_active": False
    }
    
    response = client.get("/api/v1/models-configs/configs/best", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["config_id"] == config_id
    assert data["configs"]["batch_size"] == 32
    mock_get_best.assert_called_once_with("mape", False)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.get_best_config_by_metric", return_value=None)
@patch("deployment.app.api.models_configs.settings")
def test_get_best_config_not_found(mock_settings, mock_get_best, mock_server_api_key, client):
    """Test 404 response when no configs with the metric exist."""
    mock_settings.default_metric = "rmse"
    mock_settings.default_metric_higher_is_better = False
    
    response = client.get("/api/v1/models-configs/configs/best", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 404
    assert "No configs found with metric" in response.text
    mock_get_best.assert_called_once_with("rmse", False)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_best_config_unauthorized_missing_key(mock_server_api_key, client):
    response = client.get("/api/v1/models-configs/configs/best")
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_best_config_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.get("/api/v1/models-configs/configs/best", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_get_best_config_server_key_not_configured(mock_server_api_key, client):
    response = client.get("/api/v1/models-configs/configs/best", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/configs ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.get_configs")
def test_get_configs_success(mock_get_configs, mock_server_api_key, client):
    """Test successful retrieval of configs list."""
    config_id1 = str(uuid.uuid4())
    config_id2 = str(uuid.uuid4())
    mock_get_configs.return_value = [
        {
            "config_id": config_id1,
            "configs": {"input_chunk_length": 12},
            "is_active": True
        },
        {
            "config_id": config_id2,
            "configs": {"input_chunk_length": 24},
            "is_active": False
        }
    ]
    
    response = client.get("/api/v1/models-configs/configs?limit=10", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["config_id"] == config_id1
    assert data[0]["is_active"] is True
    assert data[1]["config_id"] == config_id2
    assert data[1]["is_active"] is False
    mock_get_configs.assert_called_once_with(limit=10)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.get_configs", return_value=[])
def test_get_configs_empty(mock_get_configs, mock_server_api_key, client):
    """Test empty response when no configs exist."""
    response = client.get("/api/v1/models-configs/configs", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    assert response.json() == []
    mock_get_configs.assert_called_once_with(limit=100)  # Default limit

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_configs_unauthorized_missing_key(mock_server_api_key, client):
    response = client.get("/api/v1/models-configs/configs")
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_configs_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.get("/api/v1/models-configs/configs", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_get_configs_server_key_not_configured(mock_server_api_key, client):
    response = client.get("/api/v1/models-configs/configs", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/configs/delete ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.delete_configs_by_ids")
def test_delete_configs_success(mock_delete_configs, mock_server_api_key, client):
    """Test successful deletion of configs."""
    mock_delete_configs.return_value = {
        "successful": 2,
        "failed": 1,
        "errors": ["Config xyz is active and cannot be deleted"]
    }
    
    response = client.post(
        "/api/v1/models-configs/configs/delete",
        json={"ids": ["id1", "id2", "id3"]},
        headers={"X-API-Key": TEST_X_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 2
    assert data["failed"] == 1
    assert len(data["errors"]) == 1
    assert "active and cannot be deleted" in data["errors"][0]
    mock_delete_configs.assert_called_once_with(["id1", "id2", "id3"])

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_delete_configs_empty_ids(mock_server_api_key, client):
    """Test response when no IDs provided for deletion."""
    response = client.post(
        "/api/v1/models-configs/configs/delete",
        json={"ids": []},
        headers={"X-API-Key": TEST_X_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 0
    assert data["failed"] == 0
    assert "No IDs provided" in data["errors"][0]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_delete_configs_unauthorized_missing_key(mock_server_api_key, client):
    response = client.post("/api/v1/models-configs/configs/delete", json={"ids": ["id1"]})
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_delete_configs_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.post("/api/v1/models-configs/configs/delete", json={"ids": ["id1"]}, headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_delete_configs_server_key_not_configured(mock_server_api_key, client):
    response = client.post("/api/v1/models-configs/configs/delete", json={"ids": ["id1"]}, headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_configs.delete_configs_by_ids", side_effect=HTTPException(status_code=500, detail="Simulated DB error from service"))
def test_delete_configs_error(mock_delete_configs, mock_server_api_key, client):
    """Test database error during config deletion."""
    response = client.post("/api/v1/models-configs/configs/delete", json={"ids": ["id1"]}, headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 500
    error_json = response.json()
    # Check the message from our custom error handler
    assert "Simulated DB error from service" in error_json.get("error", {}).get("message", "")
    mock_delete_configs.assert_called_once_with(["id1"])

# --- Test /api/v1/model-params/configs/upload ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.db.database.create_or_get_config")
def test_upload_config_success(mock_create_or_get_config, mock_server_api_key, client):
    """Test successful config upload."""
    test_payload = {"epochs": 10, "lr": 0.001}
    config_id = "test-config-id"
    mock_create_or_get_config.return_value = config_id
    
    response = client.post("/api/v1/models-configs/configs/upload", 
                           json={"json_payload": test_payload, "is_active": True},
                           headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.text}"
    
    data = response.json()
    assert data["config_id"] == config_id
    assert data["configs"] == test_payload
    assert data["is_active"] is True
    
    mock_create_or_get_config.assert_called_once_with(test_payload, is_active=True)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.db.database.create_or_get_config", side_effect=Exception("DB error"))
def test_upload_config_db_error(mock_create_or_get_config, mock_server_api_key, client):
    """Test 500 response on database error during config upload."""
    test_payload = {"epochs": 10, "lr": 0.001}
    
    response = client.post("/api/v1/models-configs/configs/upload",
                           json={"json_payload": test_payload, "is_active": False},
                           headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 500
    assert "Failed to upload config: DB error" in response.text
    mock_create_or_get_config.assert_called_once_with(test_payload, is_active=False)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_upload_config_unauthorized_missing_key(mock_server_api_key, client):
    """Test upload config fails with 401 if X-API-Key header is missing."""
    test_payload = {"epochs": 10, "lr": 0.001}
    response = client.post("/api/v1/models-configs/configs/upload", json={"json_payload": test_payload, "is_active": False})
    assert response.status_code == 401

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_upload_config_unauthorized_invalid_key(mock_server_api_key, client):
    """Test upload config fails with 401 if X-API-Key is invalid."""
    test_payload = {"epochs": 10, "lr": 0.001}
    response = client.post("/api/v1/models-configs/configs/upload", 
                           json={"json_payload": test_payload, "is_active": False},
                           headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_upload_config_server_key_not_configured(mock_server_api_key, client):
    """Test upload config fails with 500 if server X-API-Key is not configured."""
    test_payload = {"epochs": 10, "lr": 0.001}
    response = client.post("/api/v1/models-configs/configs/upload", 
                           json={"json_payload": test_payload, "is_active": False},
                           headers={"X-API-Key": "any_key"})
    assert response.status_code == 500