import pytest
from fastapi import Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import json
from typing import Dict, Any, Callable
import sqlite3
from _pytest.logging import LogCaptureFixture

from deployment.app.main import app
from deployment.app.api.admin import (
    run_cleanup_job, 
    cleanup_old_predictions, 
    cleanup_old_models, 
    cleanup_old_historical_data,
    get_db_conn_and_close
)
from deployment.app.db.data_retention import (
    run_cleanup_job, 
    cleanup_old_predictions, 
    cleanup_old_models, 
    cleanup_old_historical_data
)

# Мок-фикстуры (mock_run_cleanup_job_fixture и т.д.) инжектируются pytest'ом из conftest.py
# по их именам, переданным в аргументы тестовых функций.

# --- Test admin endpoints ---

def test_trigger_cleanup_job_success(client: TestClient, mock_run_cleanup_job_fixture: MagicMock):
    """Test /admin/data-retention/cleanup endpoint successfully starts a cleanup job."""
    with patch("fastapi.BackgroundTasks.add_task") as mock_add_task:
        response = client.post("/admin/data-retention/cleanup", headers={"Authorization": "Bearer test_token"})

        assert response.status_code == 200
        assert "Data retention cleanup job started" in response.json()["message"]
        # The dependency override in conftest should inject the mock fixture
        mock_add_task.assert_called_once_with(mock_run_cleanup_job_fixture)


def test_trigger_cleanup_unauthorized(client: TestClient):
    """Test /admin/data-retention/cleanup endpoint fails without valid token."""
    response = client.post("/admin/data-retention/cleanup", headers={"Authorization": "Bearer wrong_token"})
    assert response.status_code == 401


def test_clean_predictions_success(client: TestClient, mock_cleanup_old_predictions_fixture: MagicMock, mock_db_conn_fixture: MagicMock):
    """Test /admin/data-retention/clean-predictions successfully cleans predictions."""
    mock_cleanup_old_predictions_fixture.return_value = 5

    response = client.post("/admin/data-retention/clean-predictions", params={"days_to_keep": 30}, headers={"Authorization": "Bearer test_token"})

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["records_removed"] == 5
    assert result["days_kept"] == 30
    mock_cleanup_old_predictions_fixture.assert_called_once_with(30, conn=mock_db_conn_fixture)


def test_clean_predictions_default_days(client: TestClient, mock_cleanup_old_predictions_fixture: MagicMock, mock_db_conn_fixture: MagicMock):
    """Test /admin/data-retention/clean-predictions with default days parameter."""
    mock_cleanup_old_predictions_fixture.return_value = 3

    response = client.post("/admin/data-retention/clean-predictions", headers={"Authorization": "Bearer test_token"})

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["records_removed"] == 3
    assert result["days_kept"] is None
    mock_cleanup_old_predictions_fixture.assert_called_once_with(None, conn=mock_db_conn_fixture)


def test_clean_historical_data_success(client: TestClient, mock_cleanup_old_historical_data_fixture: MagicMock, mock_db_conn_fixture: MagicMock):
    """Test /admin/data-retention/clean-historical successfully cleans historical data."""
    mock_cleanup_old_historical_data_fixture.return_value={
        "sales": 10, "stock": 5, "stock_changes": 3, "prices": 7
    }

    response = client.post(
        "/admin/data-retention/clean-historical",
        params={"sales_days_to_keep": 60, "stock_days_to_keep": 90},
        headers={"Authorization": "Bearer test_token"}
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["records_removed"]["total"] == 25
    assert result["sales_days_kept"] == 60
    assert result["stock_days_kept"] == 90
    mock_cleanup_old_historical_data_fixture.assert_called_once_with(60, 90, conn=mock_db_conn_fixture)


def test_clean_historical_data_default_days(client: TestClient, mock_cleanup_old_historical_data_fixture: MagicMock, mock_db_conn_fixture: MagicMock):
    """Test /admin/data-retention/clean-historical with default parameters."""
    mock_cleanup_old_historical_data_fixture.return_value={
        "sales": 8, "stock": 6, "stock_changes": 4, "prices": 2
    }

    response = client.post("/admin/data-retention/clean-historical", headers={"Authorization": "Bearer test_token"})

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["records_removed"]["total"] == 20
    mock_cleanup_old_historical_data_fixture.assert_called_once_with(None, None, conn=mock_db_conn_fixture)


def test_clean_models_success(client: TestClient, mock_cleanup_old_models_fixture: MagicMock, mock_db_conn_fixture: MagicMock):
    """Test /admin/data-retention/clean-models successfully cleans models."""
    mock_cleanup_old_models_fixture.return_value = ["model1", "model2", "model3"]

    response = client.post(
        "/admin/data-retention/clean-models",
        params={"models_to_keep": 5, "inactive_days_to_keep": 30},
        headers={"Authorization": "Bearer test_token"}
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["models_removed_count"] == 3
    assert result["models_kept"] == 5
    assert result["inactive_days_kept"] == 30
    mock_cleanup_old_models_fixture.assert_called_once_with(5, 30, conn=mock_db_conn_fixture)


def test_clean_models_default_params(client: TestClient, mock_cleanup_old_models_fixture: MagicMock, mock_db_conn_fixture: MagicMock):
    """Test /admin/data-retention/clean-models with default parameters."""
    mock_cleanup_old_models_fixture.return_value = ["model4", "model5"]
    
    response = client.post("/admin/data-retention/clean-models", headers={"Authorization": "Bearer test_token"})

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "ok"
    assert result["models_removed_count"] == 2
    mock_cleanup_old_models_fixture.assert_called_once_with(None, None, conn=mock_db_conn_fixture)


def test_clean_data_unauthorized(client: TestClient):
    """Test all data retention endpoints fail without a valid token."""
    endpoints = [
        "/admin/data-retention/clean-predictions",
        "/admin/data-retention/clean-historical",
        "/admin/data-retention/clean-models"
    ]
    for endpoint in endpoints:
        response = client.post(endpoint, headers={"Authorization": "Bearer wrong_token"})
        assert response.status_code == 401, f"Endpoint {endpoint} should be protected"