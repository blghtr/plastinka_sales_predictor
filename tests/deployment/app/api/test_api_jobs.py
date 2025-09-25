"""
Comprehensive tests for deployment.app.api.jobs

This test suite covers all API endpoints in the jobs router with comprehensive mocking
of external dependencies. Tests are organized by endpoint groups and include both success
and failure scenarios.

Testing Approach:
- Mock all service layer dependencies (create_job, get_job, list_jobs, etc.)
- Test FastAPI endpoints with TestClient
- Test authentication and authorization scenarios
- Test request/response validation
- Test error handling scenarios
- Test background task scheduling
- Test file upload functionality with pyfakefs
- Test database error propagation
- Verify proper HTTP status codes and response formats

All external imports and dependencies are mocked to ensure test isolation.
"""

import json
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest
from aiofiles.threadpool import AsyncBufferedIOBase, wrap
from pyfakefs.fake_file import FakeFileWrapper

from deployment.app.config import get_settings
from deployment.app.db.database import DatabaseError
from deployment.app.models.api_models import JobStatus, JobType
from fastapi import status as fastapi_status


# Helper for pyfakefs aiofiles compatibility
def pyfakefs_file_wrapper_for_aiofiles(pyfakefs_file_obj, *, loop=None, executor=None):
    return AsyncBufferedIOBase(pyfakefs_file_obj, loop=loop, executor=executor)


wrap.register(FakeFileWrapper)(pyfakefs_file_wrapper_for_aiofiles)
TEST_X_API_KEY = "test_x_api_key_conftest"
TEST_BEARER_TOKEN = "test_admin_token"


def assert_detail(response, expected_code=None, expected_message=None, expect_type=None):
    resp = response.json()
    detail = resp.get("detail")
    if detail is None:
        # fallback: сравниваем по response.text или коду
        assert response.status_code in (400, 401, 404, 422, 500)
        return
    if expect_type == dict:
        if not isinstance(detail, dict):
            raise AssertionError(f"detail is not dict: {detail} (type={type(detail)})")
        if expected_code:
            assert detail.get("code") == expected_code, f"Expected code {expected_code}, got {detail.get('code')} in {detail}"
        if expected_message:
            assert expected_message in detail.get("message", "")
    elif expect_type == str:
        if not isinstance(detail, str):
            raise AssertionError(f"detail is not str: {detail} (type={type(detail)})")
        if expected_message:
            assert expected_message in detail
    elif expect_type is list:
        if not isinstance(detail, list):
            raise AssertionError(f"detail is not list: {detail} (type={type(detail)})")
        assert any("msg" in err for err in detail)
    else:
        # Автоматический режим: как раньше
        if isinstance(detail, dict):
            if expected_code:
                assert detail.get("code") == expected_code
            if expected_message:
                assert expected_message in detail.get("message", "")
        elif isinstance(detail, str):
            if expected_message:
                assert expected_message in detail
        elif isinstance(detail, list):
            assert any("msg" in err for err in detail)

def get_full_mock_job_data(job_id, job_type, status):
    """Helper to create a complete job dictionary for mocking."""
    return {
        "job_id": job_id,
        "job_type": job_type.value if hasattr(job_type, 'value') else str(job_type),
        "status": status.value if hasattr(status, 'value') else str(status),
        "created_at": "2023-01-01T10:00:00",
        "updated_at": "2023-01-01T10:30:00",
        "parameters": {},
        "progress": 1.0 if (status == JobStatus.COMPLETED or (hasattr(status, 'value') and status.value == 'completed')) else 0.0,
        "result_id": "res-123" if (status == JobStatus.COMPLETED or (hasattr(status, 'value') and status.value == 'completed')) else None,
        "error_message": "An error occurred" if (status == JobStatus.FAILED or (hasattr(status, 'value') and status.value == 'failed')) else None,
    }

class TestDataUploadEndpoint:
    """Test suite for /api/v1/jobs/data-upload endpoint."""

    def test_create_data_upload_job_success(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test successful creation of a data upload job."""
        # Arrange


        # Mock background task and other dependencies
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)
        monkeypatch.setattr("deployment.app.utils.validation.validate_date_format", lambda x, y: (True, "2022-09-30"))
        monkeypatch.setattr("deployment.app.utils.validation.validate_data_file_upload", lambda x, y: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_data_file_content", lambda x, y, z: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_stock_file", lambda x, y: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_sales_file", lambda x, y: (True, None))

        # Mock file saving
        mock_save_file = AsyncMock(return_value=Path("/fake/path"))
        monkeypatch.setattr("deployment.app.api.jobs._save_uploaded_file", mock_save_file)

        stock_content = b"stock data"
        sales_content1 = b"sales data 1"

        files = [
            (
                "stock_file",
                ("stock.xlsx", BytesIO(stock_content), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
            (
                "sales_files",
                ("sales1.xlsx", BytesIO(sales_content1), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
        ]

        # Act
        response = api_client.post(
            "/api/v1/jobs/data-upload",
            files=files,
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        assert "job_id" in resp_data
        job_id = resp_data["job_id"]
        assert resp_data["status"] == JobStatus.PENDING.value

        # Verify the job was actually created in the database
        job_from_db = in_memory_db.get_job(job_id)
        assert job_from_db is not None
        assert job_from_db["job_id"] == job_id

        mock_add_task.assert_called_once()

    def test_create_data_upload_job_refractory_period_active(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test data upload job returns 429 when refractory period is active."""
        # Arrange
        # Mock the lock acquisition to return False (not acquired)
        monkeypatch.setattr(
            in_memory_db,
            "try_acquire_job_submission_lock",
            MagicMock(return_value=(False, 180))  # (acquired=False, retry_after=180)
        )

        # Mock other dependencies to avoid side effects
        monkeypatch.setattr("deployment.app.utils.validation.validate_date_format", lambda x, y: (True, "2022-09-30"))
        monkeypatch.setattr("deployment.app.utils.validation.validate_data_file_upload", lambda x, y: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_data_file_content", lambda x, y, z: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_stock_file", lambda x, y: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_sales_file", lambda x, y: (True, None))

        files = [
            (
                "stock_file",
                ("stock.xlsx", BytesIO(b"data"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
            (
                "sales_files",
                ("sales.xlsx", BytesIO(b"data"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
        ]

        # Act
        response = api_client.post(
            "/api/v1/jobs/data-upload",
            files=files,
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == fastapi_status.HTTP_429_TOO_MANY_REQUESTS
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "180"

        resp_data = response.json()
        assert resp_data["error"]["details"]["original_detail"]["code"] == "job_refractory_active"
        assert "similar data_upload job was submitted recently" in resp_data["error"]["details"]["original_detail"]["message"]
        assert resp_data["error"]["details"]["original_detail"]["retry_after_seconds"] == 180

        # Verify lock acquisition was called with correct parameters
        in_memory_db.try_acquire_job_submission_lock.assert_called_once()
        call_args = in_memory_db.try_acquire_job_submission_lock.call_args
        assert call_args[0][0] == JobType.DATA_UPLOAD.value  # job_type
        assert "stock_file" in call_args[0][1]  # parameters dict
        assert "sales_files" in call_args[0][1]

    def test_create_data_upload_job_refractory_period_acquired(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test data upload job succeeds when lock is acquired."""
        # Arrange
        # Mock the lock acquisition to return True (acquired)
        monkeypatch.setattr(
            in_memory_db, 
            "try_acquire_job_submission_lock", 
            MagicMock(return_value=(True, 0))  # (acquired=True, retry_after=0)
        )

        # Mock other dependencies
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)
        monkeypatch.setattr("deployment.app.utils.validation.validate_date_format", lambda x, y: (True, "2022-09-30"))
        monkeypatch.setattr("deployment.app.utils.validation.validate_data_file_upload", lambda x, y: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_data_file_content", lambda x, y, z: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_stock_file", lambda x, y: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_sales_file", lambda x, y: (True, None))

        mock_save_file = AsyncMock(return_value=Path("/fake/path"))
        monkeypatch.setattr("deployment.app.api.jobs._save_uploaded_file", mock_save_file)

        files = [
            (
                "stock_file",
                ("stock.xlsx", BytesIO(b"data"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
            (
                "sales_files",
                ("sales.xlsx", BytesIO(b"data"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
        ]

        # Act
        response = api_client.post(
            "/api/v1/jobs/data-upload",
            files=files,
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        assert "job_id" in resp_data
        assert resp_data["status"] == JobStatus.PENDING.value

        # Verify lock acquisition was called
        in_memory_db.try_acquire_job_submission_lock.assert_called_once()

    def test_create_data_upload_job_invalid_date(
        self, api_client, monkeypatch
    ):
        """Test data upload job creation fails with invalid date format and returns ErrorDetailResponse."""
        # Arrange
        monkeypatch.setattr("deployment.app.utils.validation.validate_date_format", lambda x: (False, None))

        files = [
            (
                "stock_file",
                ("stock.xlsx", BytesIO(b"data"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
            (
                "sales_files",
                ("sales.xlsx", BytesIO(b"data"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
        ]

        # Act
        response = api_client.post(
            "/api/v1/jobs/data-upload",
            files=files,
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 400
        assert_detail(response, expected_code="validation_error", expect_type=dict)


class TestTrainingJobEndpoint:
    """Test suite for /api/v1/jobs/training endpoint."""

    def test_create_training_job_success(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test successful creation of a training job."""
        # Arrange
        # 1. Create an active config in the database
        config_data = {
            "nn_model_config": {
                "num_encoder_layers": 1, "num_decoder_layers": 1, "decoder_output_dim": 1,
                "temporal_width_past": 1, "temporal_width_future": 1, "temporal_hidden_size_past": 1,
                "temporal_hidden_size_future": 1, "temporal_decoder_hidden": 1, "batch_size": 1,
                "dropout": 0.1, "use_reversible_instance_norm": False, "use_layer_norm": False
            },
            "optimizer_config": {"lr": 0.01, "weight_decay": 0.01},
            "lr_shed_config": {"T_0": 1, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 1},
            "lags": 1
        }
        config_id = in_memory_db.create_or_get_config(config_data, is_active=True)

        # 2. Mock background task
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)

        # Act
        response = api_client.post(
            "/api/v1/jobs/training",
            json={
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2023-01-31",
            },
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        job_id = resp_data["job_id"]
        assert resp_data["status"] == JobStatus.PENDING.value

        # Verify job creation in DB
        job_from_db = in_memory_db.get_job(job_id)
        assert job_from_db is not None
        job_from_db_params = json.loads(job_from_db["parameters"])
        assert job_from_db_params["config_id"] == config_id

        mock_add_task.assert_called_once()

    def test_create_training_job_db_error(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test training job creation fails with a database error and returns ErrorDetailResponse."""
        # Arrange
        # Create an active config first so the config validation passes
        config_data = {
            "nn_model_config": {
                "num_encoder_layers": 1, "num_decoder_layers": 1, "decoder_output_dim": 1,
                "temporal_width_past": 1, "temporal_width_future": 1, "temporal_hidden_size_past": 1,
                "temporal_hidden_size_future": 1, "temporal_decoder_hidden": 1, "batch_size": 1,
                "dropout": 0.1, "use_reversible_instance_norm": False, "use_layer_norm": False
            },
            "optimizer_config": {"lr": 0.01, "weight_decay": 0.01},
            "lr_shed_config": {"T_0": 1, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 1},
            "lags": 1
        }
        in_memory_db.create_or_get_config(config_data, is_active=True)

        # Simulate a DatabaseError when create_job is called
        monkeypatch.setattr(in_memory_db, "create_job", MagicMock(side_effect=DatabaseError("Simulated DB error")))

        # Act
        response = api_client.post(
            "/api/v1/jobs/training",
            json={"dataset_start_date": "2022-01-01", "dataset_end_date": "2022-12-31"},
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 500
        assert_detail(response, expected_code="internal_server_error", expect_type=dict)

    def test_create_training_job_no_active_config(
        self, api_client, in_memory_db
    ):
        """Test training job creation fails with 400 if no active config is found and returns ErrorDetailResponse."""
        # Arrange
        # Ensure no active config exists in the database (in_memory_db is clean by default)

        # Act
        response = api_client.post(
            "/api/v1/jobs/training",
            json={"dataset_start_date": "2022-01-01", "dataset_end_date": "2022-12-31"},
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 400
        assert_detail(response, expected_code="no_config_available", expect_type=dict)

    def test_create_training_job_invalid_date_range(
        self, api_client, in_memory_db
    ):
        """Test training job creation fails with 422 if the date range is invalid.
        NOTE: ValueError из валидатора Pydantic приводит к 422 Unprocessable Entity, а не 400.
        """
        # Arrange
        # Create an active config in the database
        config_data = {
            "nn_model_config": {
                "num_encoder_layers": 1, "num_decoder_layers": 1, "decoder_output_dim": 1,
                "temporal_width_past": 1, "temporal_width_future": 1, "temporal_hidden_size_past": 1,
                "temporal_hidden_size_future": 1, "temporal_decoder_hidden": 1, "batch_size": 1,
                "dropout": 0.1, "use_reversible_instance_norm": False, "use_layer_norm": False
            },
            "optimizer_config": {"lr": 0.01, "weight_decay": 0.01},
            "lr_shed_config": {"T_0": 1, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 1},
            "lags": 1
        }
        in_memory_db.create_or_get_config(config_data, is_active=True)

        # Act
        response = api_client.post(
            "/api/v1/jobs/training",
            json={
                "dataset_start_date": "2023-02-01",
                "dataset_end_date": "2023-01-31",
            },
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 422
        # The response structure is complex due to error handlers,
        # so we'll check for the message in the raw text.
        assert "Start date must be before or equal to end date" in response.text
        assert "value_error" in response.text

    def test_create_training_job_refractory_period_active(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test training job returns 429 when refractory period is active."""
        # Arrange
        # Create an active config first
        config_data = {
            "nn_model_config": {
                "num_encoder_layers": 1, "num_decoder_layers": 1, "decoder_output_dim": 1,
                "temporal_width_past": 1, "temporal_width_future": 1, "temporal_hidden_size_past": 1,
                "temporal_hidden_size_future": 1, "temporal_decoder_hidden": 1, "batch_size": 1,
                "dropout": 0.1, "use_reversible_instance_norm": False, "use_layer_norm": False
            },
            "optimizer_config": {"lr": 0.01, "weight_decay": 0.01},
            "lr_shed_config": {"T_0": 1, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 1},
            "lags": 1
        }
        in_memory_db.create_or_get_config(config_data, is_active=True)

        # Mock the lock acquisition to return False (not acquired)
        monkeypatch.setattr(
            in_memory_db, 
            "try_acquire_job_submission_lock", 
            MagicMock(return_value=(False, 300))  # (acquired=False, retry_after=300)
        )

        # Act
        response = api_client.post(
            "/api/v1/jobs/training",
            json={
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2023-01-31",
            },
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == fastapi_status.HTTP_429_TOO_MANY_REQUESTS
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "300"

        resp_data = response.json()
        assert resp_data["error"]["details"]["original_detail"]["code"] == "job_refractory_active"
        assert "similar training job was submitted recently" in resp_data["error"]["details"]["original_detail"]["message"]
        assert resp_data["error"]["details"]["original_detail"]["retry_after_seconds"] == 300

        # Verify lock acquisition was called with correct parameters
        in_memory_db.try_acquire_job_submission_lock.assert_called_once()
        call_args = in_memory_db.try_acquire_job_submission_lock.call_args
        assert call_args[0][0] == JobType.TRAINING.value  # job_type
        assert "config_id" in call_args[0][1]  # parameters dict
        assert "dataset_start_date" in call_args[0][1]
        assert "dataset_end_date" in call_args[0][1]

    def test_create_training_job_refractory_period_acquired(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test training job succeeds when lock is acquired."""
        # Arrange
        # Create an active config first
        config_data = {
            "nn_model_config": {
                "num_encoder_layers": 1, "num_decoder_layers": 1, "decoder_output_dim": 1,
                "temporal_width_past": 1, "temporal_width_future": 1, "temporal_hidden_size_past": 1,
                "temporal_hidden_size_future": 1, "temporal_decoder_hidden": 1, "batch_size": 1,
                "dropout": 0.1, "use_reversible_instance_norm": False, "use_layer_norm": False
            },
            "optimizer_config": {"lr": 0.01, "weight_decay": 0.01},
            "lr_shed_config": {"T_0": 1, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 1},
            "lags": 1
        }
        config_id = in_memory_db.create_or_get_config(config_data, is_active=True)

        # Mock the lock acquisition to return True (acquired)
        monkeypatch.setattr(
            in_memory_db, 
            "try_acquire_job_submission_lock", 
            MagicMock(return_value=(True, 0))  # (acquired=True, retry_after=0)
        )

        # Mock background task
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)

        # Act
        response = api_client.post(
            "/api/v1/jobs/training",
            json={
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2023-01-31",
            },
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        job_id = resp_data["job_id"]
        assert resp_data["status"] == JobStatus.PENDING.value

        # Verify job creation in DB
        job_from_db = in_memory_db.get_job(job_id)
        assert job_from_db is not None
        job_from_db_params = json.loads(job_from_db["parameters"])
        assert job_from_db_params["config_id"] == config_id

        # Verify lock acquisition was called
        in_memory_db.try_acquire_job_submission_lock.assert_called_once()
        mock_add_task.assert_called_once()


class TestReportJobEndpoint:
    """Test suite for /api/v1/jobs/reports endpoint."""

    def test_create_report_job_success(self, api_client, monkeypatch):
        """Test successful creation of a prediction report job."""
        # Arrange
        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_generate_report = MagicMock(return_value=mock_df)
        monkeypatch.setattr(
            "deployment.app.api.jobs.generate_report", mock_generate_report
        )

        params = {
            "report_type": "prediction_report",
            "prediction_month": "2023-01-01",
            "filters": {"artist": "test_artist"},
        }

        # Act
        response = api_client.post(
            "/api/v1/jobs/reports", json=params, headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()

        assert resp_data["report_type"] == "prediction_report"
        assert resp_data["prediction_month"] == "2023-01"
        assert resp_data["records_count"] == 2
        assert resp_data["csv_data"] == mock_df.to_csv(index=False)  # Match actual implementation
        assert "generated_at" in resp_data
        assert resp_data["filters_applied"] == {"artist": "test_artist"}

        mock_generate_report.assert_called_once()

    def test_create_report_job_no_month_provided_success(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test report job defaults to the latest month when none is provided."""
        # Arrange
        # Create a dummy prediction result in the database
        job_id = in_memory_db.create_job(JobType.PREDICTION, status=JobStatus.COMPLETED)
        model_id = "test-model-id"
        in_memory_db.create_model_record(model_id, job_id, "/path/to/model", datetime.now())
        in_memory_db.create_prediction_result(job_id=job_id, model_id=model_id, output_path="/fake/path/predictions.csv", summary_metrics={}, prediction_month=datetime(2023, 5, 1).date())

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_generate_report = MagicMock(return_value=mock_df)
        monkeypatch.setattr(
            "deployment.app.api.jobs.generate_report", mock_generate_report
        )

        params = {"report_type": "prediction_report"}  # No prediction_month

        # Act
        response = api_client.post(
            "/api/v1/jobs/reports", json=params, headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["prediction_month"] == "2023-05"
        mock_generate_report.assert_called_once()
        assert "dal" in mock_generate_report.call_args.kwargs
        assert mock_generate_report.call_args.kwargs["dal"] == in_memory_db

    def test_create_report_job_no_predictions_found(self, api_client, in_memory_db):
        """Test report job returns 404 if no predictions exist for the latest month."""
        # Arrange
        # Ensure no prediction results exist in the database (in_memory_db is clean by default)

        params = {"report_type": "prediction_report"}

        # Act
        response = api_client.post(
            "/api/v1/jobs/reports", json=params, headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 404
        assert_detail(response, expected_code="no_predictions_found", expect_type=dict)


class TestTuningJobEndpoint:
    """Test suite for /api/v1/jobs/tuning endpoint."""

    def test_create_tuning_job_refractory_period_active(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test tuning job returns 429 when refractory period is active."""
        # Arrange
        # Mock the lock acquisition to return False (not acquired)
        monkeypatch.setattr(
            in_memory_db,
            "try_acquire_job_submission_lock",
            MagicMock(return_value=(False, 600))  # (acquired=False, retry_after=600)
        )
        
        # Mock get_effective_config to return a valid config
        mock_config = {
            "config_id": "test-config-123",
            "config": {"test": "config"}
        }
        monkeypatch.setattr(
            in_memory_db,
            "get_effective_config",
            MagicMock(return_value=mock_config)
        )
        
        # Mock adjust_dataset_boundaries to return a valid date
        from datetime import date
        monkeypatch.setattr(
            in_memory_db,
            "adjust_dataset_boundaries",
            MagicMock(return_value=date(2023, 1, 31))
        )

        # Act
        response = api_client.post(
            "/api/v1/jobs/tuning",
            json={
                "mode": "lite",
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2023-01-31",
            },
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == fastapi_status.HTTP_429_TOO_MANY_REQUESTS
        assert "Retry-After" in response.headers
        assert response.headers["Retry-After"] == "600"

        resp_data = response.json()
        assert resp_data["error"]["details"]["original_detail"]["code"] == "job_refractory_active"
        assert "similar tuning job was submitted recently" in resp_data["error"]["details"]["original_detail"]["message"]
        assert resp_data["error"]["details"]["original_detail"]["retry_after_seconds"] == 600

        # Verify lock acquisition was called with correct parameters
        in_memory_db.try_acquire_job_submission_lock.assert_called_once()
        call_args = in_memory_db.try_acquire_job_submission_lock.call_args
        assert call_args[0][0] == JobType.TUNING.value  # job_type
        assert "mode" in call_args[0][1]  # parameters dict
        assert call_args[0][1]["mode"] == "lite"
        assert "dataset_start_date" in call_args[0][1]
        assert "dataset_end_date" in call_args[0][1]

    def test_create_tuning_job_refractory_period_acquired(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test tuning job succeeds when lock is acquired."""
        # Arrange
        # Mock the lock acquisition to return True (acquired)
        monkeypatch.setattr(
            in_memory_db, 
            "try_acquire_job_submission_lock", 
            MagicMock(return_value=(True, 0))  # (acquired=True, retry_after=0)
        )

        # Mock get_effective_config to return a valid config
        mock_config = {
            "config_id": "test-config-123",
            "config": {"test": "config"}
        }
        monkeypatch.setattr(
            in_memory_db,
            "get_effective_config",
            MagicMock(return_value=mock_config)
        )
        
        # Mock adjust_dataset_boundaries to return a valid date
        from datetime import date
        monkeypatch.setattr(
            in_memory_db,
            "adjust_dataset_boundaries",
            MagicMock(return_value=date(2023, 1, 31))
        )

        # Mock background task
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)

        # Act
        response = api_client.post(
            "/api/v1/jobs/tuning",
            json={
                "mode": "full",
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2023-01-31",
            },
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        job_id = resp_data["job_id"]
        assert resp_data["status"] == JobStatus.PENDING.value

        # Verify job creation in DB
        job_from_db = in_memory_db.get_job(job_id)
        assert job_from_db is not None
        job_from_db_params = json.loads(job_from_db["parameters"])
        assert job_from_db_params["mode"] == "full"

        # Verify lock acquisition was called
        in_memory_db.try_acquire_job_submission_lock.assert_called_once()
        mock_add_task.assert_called_once()

    def test_create_tuning_job_different_parameters_different_locks(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test that different tuning parameters create different locks."""
        # Arrange
        # Mock the lock acquisition to return True for both calls (different parameters)
        monkeypatch.setattr(
            in_memory_db, 
            "try_acquire_job_submission_lock", 
            MagicMock(return_value=(True, 0))  # (acquired=True, retry_after=0)
        )

        # Mock get_effective_config to return a valid config
        mock_config = {
            "config_id": "test-config-123",
            "config": {"test": "config"}
        }
        monkeypatch.setattr(
            in_memory_db,
            "get_effective_config",
            MagicMock(return_value=mock_config)
        )
        
        # Mock adjust_dataset_boundaries to return a valid date
        from datetime import date
        monkeypatch.setattr(
            in_memory_db,
            "adjust_dataset_boundaries",
            MagicMock(return_value=date(2023, 1, 31))
        )

        # Mock background task
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)

        # Act - First request with lite mode
        response1 = api_client.post(
            "/api/v1/jobs/tuning",
            json={
                "mode": "lite",
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2022-06-30",
            },
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Act - Second request with full mode (different parameters)
        response2 = api_client.post(
            "/api/v1/jobs/tuning",
            json={
                "mode": "full",
                "dataset_start_date": "2022-07-01",
                "dataset_end_date": "2022-12-31",
            },
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Verify both calls were made (different parameter hashes)
        assert in_memory_db.try_acquire_job_submission_lock.call_count == 2
        
        # Verify different parameters were passed
        calls = in_memory_db.try_acquire_job_submission_lock.call_args_list
        assert calls[0][0][0] == JobType.TUNING.value  # job_type
        assert calls[1][0][0] == JobType.TUNING.value  # job_type
        
        # Parameters should be different
        params1 = calls[0][0][1]
        params2 = calls[1][0][1]
        assert params1["mode"] == "lite"
        assert params2["mode"] == "full"
        assert params1["dataset_start_date"] != params2["dataset_start_date"]


class TestJobStatusEndpoint:
    """Test suite for job status retrieval."""
    def test_get_job_status_training_completed(
        self, api_client, in_memory_db
    ):
        """Test getting status of a completed training job."""
        # Arrange
        job_id = in_memory_db.create_job(JobType.TRAINING, status=JobStatus.COMPLETED)
        model_id = str(uuid.uuid4())
        config_data = {
            "nn_model_config": {
                "num_encoder_layers": 1, "num_decoder_layers": 1, "decoder_output_dim": 1,
                "temporal_width_past": 1, "temporal_width_future": 1, "temporal_hidden_size_past": 1,
                "temporal_hidden_size_future": 1, "temporal_decoder_hidden": 1, "batch_size": 1,
                "dropout": 0.1, "use_reversible_instance_norm": False, "use_layer_norm": False
            },
            "optimizer_config": {"lr": 0.01, "weight_decay": 0.01},
            "lr_shed_config": {"T_0": 1, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 1},
            "lags": 1
        }
        config_id = in_memory_db.create_or_get_config(config_data)
        # Create model record first to satisfy foreign key constraint
        in_memory_db.create_model_record(model_id, job_id, "/path/to/model", datetime.now())
        in_memory_db.create_training_result(
            job_id=job_id,
            model_id=model_id,
            config_id=config_id,
            metrics={"accuracy": 0.95},
            duration=120
        )

        # Act
        response = api_client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == JobStatus.COMPLETED.value
        assert "result" in data
        assert data["result"]["model_id"] == model_id



    def test_get_job_status_pending(self, api_client, in_memory_db):
        """Test getting status of a pending job."""
        # Arrange
        job_id = in_memory_db.create_job(JobType.DATA_UPLOAD, status=JobStatus.PENDING)

        # Act
        response = api_client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == JobStatus.PENDING.value
        assert data["result"] is None

    def test_get_job_status_result_db_error(
        self, api_client, in_memory_db, monkeypatch
    ):
        """Test job status handles database errors when fetching results and returns ErrorDetailResponse."""
        # Arrange
        job_id = in_memory_db.create_job(JobType.PREDICTION, status=JobStatus.COMPLETED)
        # Create a prediction result first so the job has a result
        model_id = "test-model-id"
        in_memory_db.create_model_record(model_id, job_id, "/path/to/model", datetime.now())
        in_memory_db.create_prediction_result(job_id=job_id, model_id=model_id, output_path="/fake/path/predictions.csv", summary_metrics={}, prediction_month=datetime(2023, 1, 1).date())

        # Now mock the get_prediction_result to raise an error
        # Mock the base function in database.py
        monkeypatch.setattr("deployment.app.db.database.get_prediction_result", MagicMock(side_effect=DatabaseError("Result error")))

        # Act
        response = api_client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 500
        assert_detail(response, expected_code="database_error", expect_type=dict)

    def test_get_job_status_not_found(self, api_client, in_memory_db):
        """Test job status returns 404 for non-existent job and returns ErrorDetailResponse."""
        # Arrange
        job_id = "non-existent"
        # in_memory_db is clean by default, so no job with this ID exists

        # Act
        response = api_client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 404
        assert_detail(response, expected_code="job_not_found")


class TestJobListingEndpoint:
    """Test suite for job listing functionality."""

    def test_list_jobs_no_filters(self, api_client, in_memory_db):
        """Test listing jobs without filters."""
        # Arrange
        # Create multiple jobs in the database
        in_memory_db.create_job(
            JobType.TRAINING,
            status=JobStatus.COMPLETED,
        )
        in_memory_db.create_job(
            JobType.TUNING,
            status=JobStatus.PENDING,
        )

        # Act
        response = api_client.get("/api/v1/jobs", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 2

        # Check that both job types are present (order may vary)
        job_types = [job["job_type"] for job in data["jobs"]]
        assert JobType.TRAINING.value in job_types
        assert JobType.TUNING.value in job_types

    def test_list_jobs_with_filters(self, api_client, in_memory_db):
        """Test listing jobs with filters applied."""
        # Arrange
        # Create jobs that match the filters
        in_memory_db.create_job(
            JobType.TRAINING,
            status=JobStatus.COMPLETED,
        )
        # Create a job that does not match the filters
        in_memory_db.create_job(
            JobType.PREDICTION,
            status=JobStatus.PENDING,
        )

        # Act
        response = api_client.get(
            "/api/v1/jobs?job_type=training&status=completed&limit=10",
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["job_type"] == JobType.TRAINING.value

    def test_list_jobs_db_error(self, api_client, in_memory_db, monkeypatch):
        """Test job listing handles database errors."""
        # Arrange
        monkeypatch.setattr(in_memory_db, "list_jobs", MagicMock(side_effect=DatabaseError("Simulated DB error")))

        # Act
        response = api_client.get("/api/v1/jobs", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 500
        assert_detail(response, expected_code="database_error")


class TestAuthenticationScenarios:
    """Test suite for API authentication."""

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_create_data_upload_job_success_with_unified_auth(self, api_client, in_memory_db, monkeypatch, auth_header_name, auth_token):
        """Test successful creation of a data upload job with either X-API-Key or Bearer token."""
        # Arrange
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)
        monkeypatch.setattr("deployment.app.utils.validation.validate_date_format", lambda x, y: (True, "2022-09-30"))
        monkeypatch.setattr("deployment.app.utils.validation.validate_data_file_upload", lambda x, y: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_data_file_content", lambda x, y, z: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_stock_file", lambda x, y: (True, None))
        monkeypatch.setattr("deployment.app.utils.validation.validate_sales_file", lambda x, y: (True, None))

        mock_save_file = AsyncMock(return_value=Path("/fake/path"))
        monkeypatch.setattr("deployment.app.api.jobs._save_uploaded_file", mock_save_file)

        stock_content = b"stock data"
        sales_content1 = b"sales data 1"

        files = [
            (
                "stock_file",
                ("stock.xlsx", BytesIO(stock_content), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
            (
                "sales_files",
                ("sales1.xlsx", BytesIO(sales_content1), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ),
        ]

        # Act
        response = api_client.post(
            "/api/v1/jobs/data-upload",
            files=files,
            headers={auth_header_name: auth_token},
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        job_id = resp_data["job_id"]
        assert resp_data["status"] == JobStatus.PENDING.value

        # Verify the job was actually created in the database
        job_from_db = in_memory_db.get_job(job_id)
        assert job_from_db is not None
        assert job_from_db["job_id"] == job_id

        mock_add_task.assert_called_once()

    def test_create_data_upload_job_unauthorized_missing_key(self, api_client):
        """Test data upload job fails with 401 if X-API-Key header is missing."""
        # Act
        response = api_client.post(
            "/api/v1/jobs/data-upload",
            files={"stock_file": ("f.csv", b"c")},
        )

        # Assert
        assert response.status_code == 401

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", "wrong_key"),
        ("Authorization", "Bearer wrong_token"),
    ])
    def test_create_data_upload_job_unauthorized_invalid_key(self, api_client, auth_header_name, auth_token):
        """Test data upload job fails with 401 if X-API-Key or Bearer token is invalid."""
        # Act
        response = api_client.post(
            "/api/v1/jobs/data-upload",
            files={"stock_file": ("f.csv", b"c")},
            headers={auth_header_name: auth_token},
        )

        # Assert
        assert response.status_code == 401

    def test_create_data_upload_job_server_key_not_configured(
        self, api_client, monkeypatch
    ):
        """Test data upload job fails with 500 if server X-API-Key is not configured."""
        # Arrange
        monkeypatch.setattr(get_settings().api, "x_api_key_hash", None)
        monkeypatch.setattr(get_settings().api, "admin_api_key_hash", None)

        # Act
        response = api_client.post(
            "/api/v1/jobs/data-upload",
            files={"stock_file": ("f.csv", b"c")},
            headers={"X-API-Key": "any_key"},
        )

        # Assert
        assert response.status_code == 401
        assert_detail(response, expect_type=str)

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_create_training_job_success_with_unified_auth(self, api_client, in_memory_db, monkeypatch, auth_header_name, auth_token):
        """Test successful creation of a training job with either X-API-Key or Bearer token."""
        # Arrange
        # Create an active config in the database
        config_data = {
            "nn_model_config": {
                "num_encoder_layers": 1, "num_decoder_layers": 1, "decoder_output_dim": 1,
                "temporal_width_past": 1, "temporal_width_future": 1, "temporal_hidden_size_past": 1,
                "temporal_hidden_size_future": 1, "temporal_decoder_hidden": 1, "batch_size": 1,
                "dropout": 0.1, "use_reversible_instance_norm": False, "use_layer_norm": False
            },
            "optimizer_config": {"lr": 0.01, "weight_decay": 0.01},
            "lr_shed_config": {"T_0": 1, "T_mult": 1},
            "train_ds_config": {"alpha": 0.1, "span": 1},
            "lags": 1
        }
        config_id = in_memory_db.create_or_get_config(config_data, is_active=True)

        # Mock background task
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)

        # Act
        response = api_client.post(
            "/api/v1/jobs/training",
            json={
                "dataset_start_date": "2022-01-01",
                "dataset_end_date": "2023-01-31",
            },
            headers={auth_header_name: auth_token},
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        job_id = resp_data["job_id"]
        assert resp_data["status"] == JobStatus.PENDING.value

        # Verify job creation in DB
        job_from_db = in_memory_db.get_job(job_id)
        assert job_from_db is not None
        job_from_db_params = json.loads(job_from_db["parameters"])
        assert job_from_db_params["config_id"] == config_id

        mock_add_task.assert_called_once()

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_create_report_job_success_with_unified_auth(self, api_client, in_memory_db, monkeypatch, auth_header_name, auth_token):
        """Test successful creation of a prediction report job with either X-API-Key or Bearer token."""
        # Arrange
        # Create a dummy prediction result in the database
        job_id = in_memory_db.create_job(JobType.PREDICTION, status=JobStatus.COMPLETED)
        model_id = "test-model-id"
        in_memory_db.create_model_record(model_id, job_id, "/path/to/model", datetime.now())
        in_memory_db.create_prediction_result(job_id=job_id, model_id=model_id, output_path="/fake/path/predictions.csv", summary_metrics={}, prediction_month=datetime(2023, 1, 1).date())

        mock_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_generate_report = MagicMock(return_value=mock_df)
        monkeypatch.setattr(
            "deployment.app.api.jobs.generate_report", mock_generate_report
        )

        params = {
            "report_type": "prediction_report",
            "prediction_month": "2023-01-01",
            "filters": {"artist": "test_artist"},
        }

        response = api_client.post(
            "/api/v1/jobs/reports", json=params, headers={auth_header_name: auth_token}
        )

        assert response.status_code == 200
        resp_data = response.json()

        assert resp_data["report_type"] == "prediction_report"
        assert resp_data["prediction_month"] == "2023-01"
        assert resp_data["records_count"] == 2
        assert resp_data["csv_data"] == mock_df.to_csv(index=False)
        assert "generated_at" in resp_data
        assert resp_data["filters_applied"] == {"artist": "test_artist"}

        mock_generate_report.assert_called_once()

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_get_job_status_training_completed_with_unified_auth(self, api_client, in_memory_db, auth_header_name, auth_token):
        """Test getting status of a completed training job with either X-API-Key or Bearer token."""
        # Arrange
        job_id = in_memory_db.create_job(JobType.TRAINING, status=JobStatus.COMPLETED)

        # Act
        response = api_client.get(f"/api/v1/jobs/{job_id}", headers={auth_header_name: auth_token})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == JobStatus.COMPLETED.value

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_list_jobs_no_filters_with_unified_auth(self, api_client, in_memory_db, auth_header_name, auth_token):
        """Test listing jobs without filters with either X-API-Key or Bearer token."""
        # Arrange
        # Create multiple jobs in the database
        in_memory_db.create_job(
            JobType.TRAINING,
            status=JobStatus.COMPLETED,
        )
        in_memory_db.create_job(
            JobType.TUNING,
            status=JobStatus.PENDING,
        )

        response = api_client.get("/api/v1/jobs", headers={auth_header_name: auth_token})

        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 2

        # Check that both job types are present (order may vary)
        job_types = [job["job_type"] for job in data["jobs"]]
        assert JobType.TRAINING.value in job_types
        assert JobType.TUNING.value in job_types

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_invalid_job_id_format_with_unified_auth(self, api_client, in_memory_db, auth_header_name, auth_token):
        """Test endpoints handle invalid job ID formats gracefully with either X-API-Key or Bearer token."""
        # Arrange
        # in_memory_db is clean by default, so no job with this ID exists

        response = api_client.get(
            "/api/v1/jobs/invalid-id-format", headers={auth_header_name: auth_token}
        )

        assert response.status_code == 404

    @pytest.mark.parametrize("auth_header_name, auth_token", [
        ("X-API-Key", TEST_X_API_KEY),
        ("Authorization", f"Bearer {TEST_BEARER_TOKEN}"),
    ])
    def test_invalid_query_parameters_with_unified_auth(self, api_client, auth_header_name, auth_token):
        """Test job listing handles invalid query parameters with either X-API-Key or Bearer token."""
        response = api_client.get(
            "/api/v1/jobs?job_type=invalid_type&status=invalid_status",
            headers={auth_header_name: auth_token},
        )

        assert response.status_code == 422
