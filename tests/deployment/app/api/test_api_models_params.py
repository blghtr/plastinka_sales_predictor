import uuid
from unittest.mock import AsyncMock, patch
from io import BytesIO
import json
from datetime import datetime
import unittest
import os

import pytest
from fastapi import UploadFile

TEST_X_API_KEY = "test_x_api_key_conftest"

class TestConfigEndpoints:
    """Tests for config-related endpoints in /api/v1/models-configs."""

    def test_get_active_config_success(self, client, mock_dal):
        mock_dal.get_active_config.return_value = {"config_id": "active-config", "configs": {}, "is_active": True}
        response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        assert response.json()["config_id"] == "active-config"

    def test_get_active_config_not_found(self, client, mock_dal):
        mock_dal.get_active_config.return_value = None
        response = client.get("/api/v1/models-configs/configs/active", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 404

    def test_activate_config_success(self, client, mock_dal):
        config_id = "config-to-activate"
        mock_dal.set_config_active.return_value = True
        response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        mock_dal.set_config_active.assert_called_once_with(config_id)

    def test_activate_config_not_found(self, client, mock_dal):
        config_id = "non-existent-config"
        mock_dal.set_config_active.return_value = False
        response = client.post(f"/api/v1/models-configs/configs/{config_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 404

    def test_get_best_config(self, client, mock_dal):
        mock_dal.get_best_config_by_metric.return_value = {"config_id": "best-config", "configs": {"param": "value"}}
        response = client.get("/api/v1/models-configs/configs/best", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        assert response.json()["config_id"] == "best-config"

    def test_get_configs_list(self, client, mock_dal):
        mock_dal.get_configs.return_value = [
            {"config_id": "cfg1", "configs": {}, "created_at": "2023-01-01T00:00:00", "is_active": True},
            {"config_id": "cfg2", "configs": {}, "created_at": "2023-01-02T00:00:00", "is_active": False},
        ]
        response = client.get("/api/v1/models-configs/configs", headers={"X-API-Key": TEST_X_API_KEY})
        print("RESPONSE CONFIGS:", response.json())
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 2
        assert "configs" in response.json()[0]
        assert "config_id" in response.json()[0]

    def test_delete_configs(self, client, mock_dal):
        mock_dal.delete_configs_by_ids.return_value = {"successful": 1, "failed": 0, "errors": []}
        config_ids = [str(uuid.uuid4())]
        response = client.post("/api/v1/models-configs/configs/delete", json={"ids": config_ids}, headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        assert response.json()["successful"] == 1

    def test_upload_config(self, client, mock_dal):
        mock_dal.create_or_get_config.return_value = "uploaded-config"
        config_id = "uploaded-config"
        request_data = {"json_payload": {"p": 1}, "is_active": False, "source": "upload"}
        response = client.post("/api/v1/models-configs/configs/upload", json=request_data, headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        assert response.json()["config_id"] == config_id

class TestModelEndpoints:
    """Tests for model-related endpoints in /api/v1/models-configs."""

    def test_get_active_model(self, client, mock_dal):
        mock_dal.get_active_model.return_value = {"model_id": "active-model", "model_path": "/fake/path", "metadata": {}}
        response = client.get("/api/v1/models-configs/models/active", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        assert response.json()["model_id"] == "active-model"

    def test_activate_model(self, client, mock_dal):
        model_id = "model-to-activate"
        mock_dal.set_model_active.return_value = True
        response = client.post(f"/api/v1/models-configs/models/{model_id}/set-active", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        mock_dal.set_model_active.assert_called_once_with(model_id, deactivate_others=True)

    def test_get_best_model(self, client, mock_dal):
        mock_dal.get_best_model_by_metric.return_value = {"model_id": "best-model", "model_path": "/path", "metadata": {}, "metrics": {}}
        response = client.get("/api/v1/models-configs/models/best", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        assert response.json()["model_id"] == "best-model"

    def test_get_recent_models(self, client, mock_dal):
        mock_dal.get_recent_models.return_value = [
            ("m1", "j1", "/fake/path1", "2023-01-01T00:00:00", {}),
            ("m2", "j2", "/fake/path2", "2023-01-02T00:00:00", {}),
        ]
        mock_dal.get_active_model.return_value = {"model_id": "model2"}
        response = client.get("/api/v1/models-configs/models/recent", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) == 2

    def test_get_all_models(self, client, mock_dal):
        mock_dal.get_all_models.return_value = [
            {"model_id": "m1", "job_id": "j1", "model_path": "/fake/path1", "created_at": "2023-01-01T00:00:00", "metadata": {}, "is_active": True},
        ]
        response = client.get("/api/v1/models-configs/models", headers={"X-API-Key": TEST_X_API_KEY})
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_delete_models(self, client, mock_dal):
        def _side_effect(ids):
            print("MOCK CALLED WITH:", ids)
            return {
                "deleted_count": 1,
                "skipped_count": 0,
                "skipped_models": [],
                "failed_deletions": [],
            }
        mock_dal.delete_models_by_ids.side_effect = _side_effect
        model_ids = [str(uuid.uuid4())]
        response = client.post("/api/v1/models-configs/models/delete", json={"ids": model_ids}, headers={"X-API-Key": TEST_X_API_KEY})
        print("RESPONSE:", response.json())
        assert response.status_code == 200
        assert response.json()["successful"] == 1
        mock_dal.delete_models_by_ids.assert_called()

    def test_upload_model(self, monkeypatch, client, mock_dal):
        # 1. Mock settings
        mock_settings = unittest.mock.MagicMock()
        mock_settings.model_storage_dir = "/fake_storage"
        monkeypatch.setattr("deployment.app.api.models_configs.get_settings", lambda: mock_settings)

        # 2. Mock file system operations
        monkeypatch.setattr("os.path.join", lambda *args: "/".join(args))
        monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)
        mock_open = unittest.mock.mock_open()
        monkeypatch.setattr("builtins.open", mock_open)
        
        # 3. Prepare request data
        model_id = str(uuid.uuid4())
        dummy_content = b"dummy model content"
        files = {"model_file": ("model.onnx", BytesIO(dummy_content), "application/octet-stream")}
        data = {"model_id": model_id, "is_active": "false"}
        # Note: Not providing job_id, so the API will create a manual_upload job

        # 4. Make the request
        response = client.post(
            "/api/v1/models-configs/models/upload",
            files=files,
            data=data,
            headers={"X-API-Key": TEST_X_API_KEY}
        )
        
        # 5. Assertions
        assert response.status_code == 200
        mock_dal.create_model_record.assert_called_once()
        # Check that the file was "written" (use assert_any_call since mock_open is used as context manager)
        mock_open.assert_any_call(f"/fake_storage/{model_id}.onnx", "wb")
        # Check that write was called with the correct content
        # Note: We can't use assert_called_once_with because mock_open creates multiple mock instances
        handle = mock_open.return_value.__enter__.return_value
        handle.write.assert_called_with(dummy_content) 