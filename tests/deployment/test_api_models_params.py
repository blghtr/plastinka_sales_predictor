import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException # Added for side_effect
from unittest.mock import patch, MagicMock
import json
import uuid
from pathlib import Path

from deployment.app.main import app
from deployment.app.db.database import DatabaseError
from unittest.mock import PropertyMock # Added for PropertyMock

TEST_X_API_KEY = "test_x_api_key_value_models_params"

# --- Test /api/v1/model-params/parameter-sets/active ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_active_parameter_set")
def test_get_active_parameter_set_success(mock_get_active_param_set, mock_server_api_key, client):
    """Test successful retrieval of active parameter set."""
    parameter_set_id = str(uuid.uuid4())
    mock_get_active_param_set.return_value = {
        "parameter_set_id": parameter_set_id,
        "parameters": {
            "input_chunk_length": 12,
            "output_chunk_length": 6,
            "max_epochs": 10
        },
        "is_active": True
    }
    
    response = client.get("/api/v1/model-params/parameter-sets/active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["parameter_set_id"] == parameter_set_id
    assert data["parameters"]["input_chunk_length"] == 12
    assert data["parameters"]["output_chunk_length"] == 6
    assert data["parameters"]["max_epochs"] == 10
    assert data["is_active"] is True
    mock_get_active_param_set.assert_called_once()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_active_parameter_set", return_value=None)
def test_get_active_parameter_set_not_found(mock_get_active_param_set, mock_server_api_key, client):
    """Test 404 response when no active parameter set exists."""
    response = client.get("/api/v1/model-params/parameter-sets/active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 404
    assert "No active parameter set found" in response.text
    mock_get_active_param_set.assert_called_once()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_active_parameter_set_unauthorized_missing_key(mock_server_api_key, client):
    """Test get active parameter set fails with 401 if X-API-Key header is missing."""
    response = client.get("/api/v1/model-params/parameter-sets/active") # No header
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_active_parameter_set_unauthorized_invalid_key(mock_server_api_key, client):
    """Test get active parameter set fails with 401 if X-API-Key is invalid."""
    response = client.get("/api/v1/model-params/parameter-sets/active", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value="")) # Server key not configured
def test_get_active_parameter_set_server_key_not_configured(mock_server_api_key, client):
    """Test get active parameter set fails with 500 if server X-API-Key is not configured."""
    response = client.get("/api/v1/model-params/parameter-sets/active", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/parameter-sets/{parameter_set_id}/set-active ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.set_parameter_set_active", return_value=True)
def test_activate_parameter_set_success(mock_set_active, mock_server_api_key, client):
    """Test successful activation of a parameter set."""
    parameter_set_id = str(uuid.uuid4())
    
    response = client.post(f"/api/v1/model-params/parameter-sets/{parameter_set_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert f"Parameter set {parameter_set_id} set as active" in data["message"]
    mock_set_active.assert_called_once_with(parameter_set_id)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.set_parameter_set_active", return_value=False)
def test_activate_parameter_set_not_found(mock_set_active, mock_server_api_key, client):
    """Test 404 response when parameter set to activate doesn't exist."""
    parameter_set_id = str(uuid.uuid4())
    
    response = client.post(f"/api/v1/model-params/parameter-sets/{parameter_set_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 404
    assert f"Parameter set {parameter_set_id} not found" in response.text
    mock_set_active.assert_called_once_with(parameter_set_id)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_activate_parameter_set_unauthorized_missing_key(mock_server_api_key, client):
    """Test activate parameter set fails with 401 if X-API-Key header is missing."""
    parameter_set_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/model-params/parameter-sets/{parameter_set_id}/set-active") # No header
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_activate_parameter_set_unauthorized_invalid_key(mock_server_api_key, client):
    """Test activate parameter set fails with 401 if X-API-Key is invalid."""
    parameter_set_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/model-params/parameter-sets/{parameter_set_id}/set-active", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value="")) # Server key not configured
def test_activate_parameter_set_server_key_not_configured(mock_server_api_key, client):
    """Test activate parameter set fails with 500 if server X-API-Key is not configured."""
    parameter_set_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/model-params/parameter-sets/{parameter_set_id}/set-active", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/parameter-sets/best ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_best_parameter_set_by_metric")
@patch("deployment.app.api.models_params.settings")
def test_get_best_parameter_set_custom_metric(mock_settings, mock_get_best, mock_server_api_key, client):
    """Test retrieval of best parameter set with custom metric."""
    parameter_set_id = str(uuid.uuid4())
    mock_get_best.return_value = {
        "parameter_set_id": parameter_set_id,
        "parameters": {
            "input_chunk_length": 24,
            "output_chunk_length": 12
        },
        "is_active": False
    }
    
    response = client.get("/api/v1/model-params/parameter-sets/best?metric_name=mae&higher_is_better=false", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["parameter_set_id"] == parameter_set_id
    assert data["parameters"]["input_chunk_length"] == 24
    assert data["parameters"]["output_chunk_length"] == 12
    assert data["is_active"] is False
    mock_get_best.assert_called_once_with("mae", False)
    mock_settings.default_metric.assert_not_called()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_best_parameter_set_by_metric")
@patch("deployment.app.api.models_params.settings")
def test_get_best_parameter_set_default_metric(mock_settings, mock_get_best, mock_server_api_key, client):
    """Test retrieval of best parameter set with default metric from settings."""
    parameter_set_id = str(uuid.uuid4())
    mock_settings.default_metric = "mape"
    mock_settings.default_metric_higher_is_better = False
    mock_get_best.return_value = {
        "parameter_set_id": parameter_set_id,
        "parameters": {"batch_size": 32},
        "is_active": False
    }
    
    response = client.get("/api/v1/model-params/parameter-sets/best", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["parameter_set_id"] == parameter_set_id
    assert data["parameters"]["batch_size"] == 32
    mock_get_best.assert_called_once_with("mape", False)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_best_parameter_set_by_metric", return_value=None)
@patch("deployment.app.api.models_params.settings")
def test_get_best_parameter_set_not_found(mock_settings, mock_get_best, mock_server_api_key, client):
    """Test 404 response when no parameter sets with the metric exist."""
    mock_settings.default_metric = "rmse"
    mock_settings.default_metric_higher_is_better = False
    
    response = client.get("/api/v1/model-params/parameter-sets/best", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 404
    assert "No parameter sets found with metric" in response.text
    mock_get_best.assert_called_once_with("rmse", False)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_best_parameter_set_unauthorized_missing_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/parameter-sets/best")
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_best_parameter_set_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/parameter-sets/best", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_get_best_parameter_set_server_key_not_configured(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/parameter-sets/best", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/parameter-sets ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_parameter_sets")
def test_get_parameter_sets_success(mock_get_params, mock_server_api_key, client):
    """Test successful retrieval of parameter sets list."""
    parameter_set_id1 = str(uuid.uuid4())
    parameter_set_id2 = str(uuid.uuid4())
    mock_get_params.return_value = [
        {
            "parameter_set_id": parameter_set_id1,
            "parameters": {"input_chunk_length": 12},
            "is_active": True
        },
        {
            "parameter_set_id": parameter_set_id2,
            "parameters": {"input_chunk_length": 24},
            "is_active": False
        }
    ]
    
    response = client.get("/api/v1/model-params/parameter-sets?limit=10", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["parameter_set_id"] == parameter_set_id1
    assert data[0]["is_active"] is True
    assert data[1]["parameter_set_id"] == parameter_set_id2
    assert data[1]["is_active"] is False
    mock_get_params.assert_called_once_with(limit=10)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_parameter_sets", return_value=[])
def test_get_parameter_sets_empty(mock_get_params, mock_server_api_key, client):
    """Test empty response when no parameter sets exist."""
    response = client.get("/api/v1/model-params/parameter-sets", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    assert response.json() == []
    mock_get_params.assert_called_once_with(limit=100)  # Default limit

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_parameter_sets_unauthorized_missing_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/parameter-sets")
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_parameter_sets_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/parameter-sets", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_get_parameter_sets_server_key_not_configured(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/parameter-sets", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/parameter-sets/delete ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.delete_parameter_sets_by_ids")
def test_delete_parameter_sets_success(mock_delete_params, mock_server_api_key, client):
    """Test successful deletion of parameter sets."""
    mock_delete_params.return_value = {
        "successful": 2,
        "failed": 1,
        "errors": ["Parameter set xyz is active and cannot be deleted"]
    }
    
    response = client.post(
        "/api/v1/model-params/parameter-sets/delete",
        json={"ids": ["id1", "id2", "id3"]},
        headers={"X-API-Key": TEST_X_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 2
    assert data["failed"] == 1
    assert len(data["errors"]) == 1
    assert "active and cannot be deleted" in data["errors"][0]
    mock_delete_params.assert_called_once_with(["id1", "id2", "id3"])

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_delete_parameter_sets_empty_ids(mock_server_api_key, client):
    """Test response when no IDs provided for deletion."""
    response = client.post(
        "/api/v1/model-params/parameter-sets/delete",
        json={"ids": []},
        headers={"X-API-Key": TEST_X_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 0
    assert data["failed"] == 0
    assert "No IDs provided" in data["errors"][0]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_delete_parameter_sets_unauthorized_missing_key(mock_server_api_key, client):
    response = client.post("/api/v1/model-params/parameter-sets/delete", json={"ids": ["id1"]})
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_delete_parameter_sets_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.post("/api/v1/model-params/parameter-sets/delete", json={"ids": ["id1"]}, headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_delete_parameter_sets_server_key_not_configured(mock_server_api_key, client):
    response = client.post("/api/v1/model-params/parameter-sets/delete", json={"ids": ["id1"]}, headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/models/active ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_active_model")
def test_get_active_model_success(mock_get_active_model, mock_server_api_key, client):
    """Test successful retrieval of active model."""
    model_id = str(uuid.uuid4())
    mock_get_active_model.return_value = {
        "model_id": model_id,
        "model_path": f"/models/{model_id}.pkl",
        "is_active": True,
        "metadata": {"trained_at": "2022-10-01T12:00:00Z"}
    }
    
    response = client.get("/api/v1/model-params/models/active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == model_id
    assert data["model_path"] == f"/models/{model_id}.pkl"
    assert data["is_active"] is True
    assert data["metadata"]["trained_at"] == "2022-10-01T12:00:00Z"
    mock_get_active_model.assert_called_once()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_active_model", return_value=None)
def test_get_active_model_not_found(mock_get_active_model, mock_server_api_key, client):
    """Test 404 response when no active model exists."""
    response = client.get("/api/v1/model-params/models/active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 404
    assert "No active model found" in response.text
    mock_get_active_model.assert_called_once()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_active_model_unauthorized_missing_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models/active")
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_active_model_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models/active", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_get_active_model_server_key_not_configured(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models/active", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/models/{model_id}/set-active ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.set_model_active", return_value=True)
def test_activate_model_success(mock_set_active, mock_server_api_key, client):
    """Test successful activation of a model."""
    model_id = str(uuid.uuid4())
    
    response = client.post(f"/api/v1/model-params/models/{model_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert f"Model {model_id} set as active" in data["message"]
    mock_set_active.assert_called_once_with(model_id)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.set_model_active", return_value=False)
def test_activate_model_not_found(mock_set_active, mock_server_api_key, client):
    """Test 404 response when model to activate doesn't exist."""
    model_id = str(uuid.uuid4())
    
    response = client.post(f"/api/v1/model-params/models/{model_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 404
    assert f"Model {model_id} not found" in response.text
    mock_set_active.assert_called_once_with(model_id)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_activate_model_unauthorized_missing_key(mock_server_api_key, client):
    model_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/model-params/models/{model_id}/set-active")
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_activate_model_unauthorized_invalid_key(mock_server_api_key, client):
    model_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/model-params/models/{model_id}/set-active", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_activate_model_server_key_not_configured(mock_server_api_key, client):
    model_id = str(uuid.uuid4())
    response = client.post(f"/api/v1/model-params/models/{model_id}/set-active", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/models/best ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_best_model_by_metric")
@patch("deployment.app.api.models_params.settings")
def test_get_best_model_custom_metric(mock_settings, mock_get_best, mock_server_api_key, client):
    """Test retrieval of best model with custom metric."""
    model_id = str(uuid.uuid4())
    mock_get_best.return_value = {
        "model_id": model_id,
        "model_path": f"/models/{model_id}.pkl",
        "is_active": False,
        "metadata": {"metric": {"rmse": 0.123}}
    }
    
    response = client.get("/api/v1/model-params/models/best?metric_name=rmse&higher_is_better=false", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == model_id
    assert data["model_path"] == f"/models/{model_id}.pkl"
    assert data["is_active"] is False
    assert data["metadata"]["metric"]["rmse"] == 0.123
    mock_get_best.assert_called_once_with("rmse", False)
    mock_settings.default_metric.assert_not_called()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_best_model_by_metric")
@patch("deployment.app.api.models_params.settings")
def test_get_best_model_default_metric(mock_settings, mock_get_best, mock_server_api_key, client):
    """Test retrieval of best model with default metric from settings."""
    model_id = str(uuid.uuid4())
    mock_settings.default_metric = "accuracy"
    mock_settings.default_metric_higher_is_better = True
    mock_get_best.return_value = {
        "model_id": model_id,
        "model_path": f"/models/{model_id}.pkl",
        "is_active": False,
        "metadata": {"metric": {"accuracy": 0.96}}
    }
    
    response = client.get("/api/v1/model-params/models/best", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == model_id
    assert data["metadata"]["metric"]["accuracy"] == 0.96
    mock_get_best.assert_called_once_with("accuracy", True)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_best_model_by_metric", return_value=None)
@patch("deployment.app.api.models_params.settings")
def test_get_best_model_not_found(mock_settings, mock_get_best, mock_server_api_key, client):
    """Test 404 response when no models with the metric exist."""
    mock_settings.default_metric = "rmse"
    mock_settings.default_metric_higher_is_better = False
    
    response = client.get("/api/v1/model-params/models/best", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 404
    assert "No models found with metric" in response.text
    mock_get_best.assert_called_once_with("rmse", False)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_best_model_unauthorized_missing_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models/best")
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_best_model_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models/best", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_get_best_model_server_key_not_configured(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models/best", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/models/recent ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_recent_models")
@patch("deployment.app.api.models_params.get_active_model")
def test_get_recent_models_success(mock_get_active, mock_get_recent, mock_server_api_key, client):
    """Test successful retrieval of recent models."""
    model_id1 = str(uuid.uuid4())
    model_id2 = str(uuid.uuid4())
    active_model_id = model_id1
    
    mock_get_recent.return_value = [
        (model_id1, "job1", f"/models/{model_id1}.pkl", "2022-10-01T12:00:00Z", '{"metric": {"rmse": 0.1}}'),
        (model_id2, "job2", f"/models/{model_id2}.pkl", "2022-09-28T10:30:00Z", '{"metric": {"rmse": 0.2}}')
    ]
    
    mock_get_active.return_value = {
        "model_id": active_model_id,
        "model_path": f"/models/{active_model_id}.pkl",
        "is_active": True
    }
    
    response = client.get("/api/v1/model-params/models/recent?limit=5", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["model_id"] == model_id1
    assert data[0]["is_active"] is True
    assert data[1]["model_id"] == model_id2
    assert data[1]["is_active"] is False
    mock_get_recent.assert_called_once_with(5)
    mock_get_active.assert_called_once()

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_recent_models", return_value=[])
def test_get_recent_models_empty(mock_get_recent, mock_server_api_key, client):
    """Test empty response when no models exist."""
    response = client.get("/api/v1/model-params/models/recent", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    assert response.json() == []
    mock_get_recent.assert_called_once_with(5)  # Default limit

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_recent_models_unauthorized_missing_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models/recent")
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_recent_models_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models/recent", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_get_recent_models_server_key_not_configured(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models/recent", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/models ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_all_models")
def test_get_all_models_success(mock_get_all, mock_server_api_key, client):
    """Test successful retrieval of all models."""
    model_id1 = str(uuid.uuid4())
    model_id2 = str(uuid.uuid4())
    
    mock_get_all.return_value = [
        {
            "model_id": model_id1,
            "model_path": f"/models/{model_id1}.pkl",
            "is_active": True,
            "metadata": {"created_at": "2022-10-01"}
        },
        {
            "model_id": model_id2,
            "model_path": f"/models/{model_id2}.pkl",
            "is_active": False,
            "metadata": {"created_at": "2022-09-28"}
        }
    ]
    
    response = client.get("/api/v1/model-params/models?limit=20", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["model_id"] == model_id1
    assert data[0]["is_active"] is True
    assert data[1]["model_id"] == model_id2
    assert data[1]["is_active"] is False
    mock_get_all.assert_called_once_with(limit=20)

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.get_all_models", return_value=[])
def test_get_all_models_empty(mock_get_all, mock_server_api_key, client):
    """Test empty response when no models exist."""
    response = client.get("/api/v1/model-params/models", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    assert response.json() == []
    mock_get_all.assert_called_once_with(limit=100)  # Default limit

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_all_models_unauthorized_missing_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models")
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_all_models_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_get_all_models_server_key_not_configured(mock_server_api_key, client):
    response = client.get("/api/v1/model-params/models", headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

# --- Test /api/v1/model-params/models/delete ---

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.delete_models_by_ids")
def test_delete_models_success(mock_delete_models, mock_server_api_key, client):
    """Test successful deletion of models."""
    mock_delete_models.return_value = {
        "successful": 2,
        "failed": 1,
        "errors": ["Model xyz is active and cannot be deleted"]
    }
    
    response = client.post(
        "/api/v1/model-params/models/delete",
        json={"ids": ["id1", "id2", "id3"]},
        headers={"X-API-Key": TEST_X_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 2
    assert data["failed"] == 1
    assert len(data["errors"]) == 1
    assert "active and cannot be deleted" in data["errors"][0]
    mock_delete_models.assert_called_once_with(["id1", "id2", "id3"])

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_delete_models_empty_ids(mock_server_api_key, client):
    """Test response when no IDs provided for deletion."""
    response = client.post(
        "/api/v1/model-params/models/delete",
        json={"ids": []},
        headers={"X-API-Key": TEST_X_API_KEY}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["successful"] == 0
    assert data["failed"] == 0
    assert "No IDs provided" in data["errors"][0]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_delete_models_unauthorized_missing_key(mock_server_api_key, client):
    response = client.post("/api/v1/model-params/models/delete", json={"ids": ["id1"]})
    assert response.status_code == 401
    assert "Not authenticated: X-API-Key header is missing" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_delete_models_unauthorized_invalid_key(mock_server_api_key, client):
    response = client.post("/api/v1/model-params/models/delete", json={"ids": ["id1"]}, headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401
    assert "Invalid X-API-Key" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_delete_models_server_key_not_configured(mock_server_api_key, client):
    response = client.post("/api/v1/model-params/models/delete", json={"ids": ["id1"]}, headers={"X-API-Key": "any_key"})
    assert response.status_code == 500
    assert "X-API-Key authentication is not configured on the server" in response.json()["error"]["message"]

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("deployment.app.api.models_params.delete_models_by_ids", side_effect=HTTPException(status_code=500, detail="Simulated DB error from service"))
def test_delete_models_error(mock_delete_models, mock_server_api_key, client):
    """Test error handling in delete models endpoint when service raises HTTPException."""
    response = client.post("/api/v1/model-params/models/delete", json={"ids": ["id1"]}, headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 500
    error_json = response.json()
    # Check the message from our custom error handler
    assert "Simulated DB error from service" in error_json.get("error", {}).get("message", "")
    mock_delete_models.assert_called_once_with(["id1"])