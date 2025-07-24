import pytest
import json
import uuid
from unittest.mock import MagicMock

from deployment.app.db.database import DatabaseError
from deployment.app.models.api_models import TrainingResultResponse, TuningResultResponse

TEST_X_API_KEY = "test_x_api_key_conftest"
TEST_BEARER_TOKEN = "test_admin_token"


class TestResultsApi:
    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_get_training_results(self, client, mock_dal, auth_header_name, auth_token):
        # Arrange
        mock_dal.get_training_results.return_value = [
            {
                "result_id": "1",
                "job_id": "job1",
                "model_id": "model1",
                "config_id": "config1",
                "metrics": {"acc": 0.9},
                "duration": 100,
                "created_at": "2025-07-16T12:00:00",
            }
        ]

        # Act
        response = client.get("/api/v1/results/training", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["result_id"] == "result-1"

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_get_training_result_by_id(self, client, mock_dal, auth_header_name, auth_token):
        # Arrange
        result_id = "1"
        mock_dal.get_training_results.return_value = {
            "result_id": result_id,
            "job_id": "job1",
            "model_id": "model1",
            "config_id": "config1",
            "metrics": {"acc": 0.9},
            "duration": 100,
            "created_at": "2025-07-16T12:00:00",
        }

        # Act
        response = client.get(f"/api/v1/results/training/{result_id}", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["result_id"] == result_id

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_get_training_result_not_found(self, client, mock_dal, auth_header_name, auth_token):
        # Arrange
        result_id = "not-found"
        mock_dal.get_training_results.return_value = None

        # Act
        response = client.get(f"/api/v1/results/training/{result_id}", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 404

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_get_tuning_results(self, client, mock_dal, auth_header_name, auth_token):
        # Arrange
        mock_dal.get_tuning_results.return_value = [
            {
                "result_id": "1",
                "job_id": "job1",
                "config_id": "config1",
                "metrics": {"val_loss": 0.1},
                "duration": 200,
                "created_at": "2025-07-16T12:00:00",
            }
        ]

        # Act
        response = client.get("/api/v1/results/tuning", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["result_id"] == "tuning-res-1"

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_get_tuning_result_by_id(self, client, mock_dal, auth_header_name, auth_token):
        # Arrange
        result_id = "1"
        mock_dal.get_tuning_results.return_value = {
            "result_id": result_id,
            "job_id": "job1",
            "config_id": "config1",
            "metrics": {"val_loss": 0.1},
            "duration": 200,
            "created_at": "2025-07-16T12:00:00",
        }

        # Act
        response = client.get(f"/api/v1/results/tuning/{result_id}", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["result_id"] == result_id

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_get_tuning_result_not_found(self, client, mock_dal, auth_header_name, auth_token):
        # Arrange
        result_id = "not-found"
        mock_dal.get_tuning_results.return_value = None

        # Act
        response = client.get(f"/api/v1/results/tuning/{result_id}", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 404

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_get_tuning_results_with_query_params(self, client, mock_dal, auth_header_name, auth_token):
        # Arrange
        mock_dal.get_tuning_results.return_value = [
            {
                "result_id": "2",
                "job_id": "job2",
                "config_id": "config2",
                "metrics": {"val_loss": 0.05},
                "duration": 150,
                "created_at": "2025-07-16T11:00:00",
            },
            {
                "result_id": "1",
                "job_id": "job1",
                "config_id": "config1",
                "metrics": {"val_loss": 0.1},
                "duration": 200,
                "created_at": "2025-07-16T12:00:00",
            },
        ]

        # Act
        response = client.get(
            "/api/v1/results/tuning?metric_name=val_loss&higher_is_better=false&limit=2",
            headers={auth_header_name: auth_token},
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["result_id"] == "tuning-res-1"  # Assuming sorted by val_loss ascending
        mock_dal.get_tuning_results.assert_called_once_with(
            metric_name="val_loss", higher_is_better=False, limit=2, result_id=None
        )