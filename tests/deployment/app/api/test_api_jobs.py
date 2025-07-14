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
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from aiofiles.threadpool import AsyncBufferedIOBase, wrap
from pyfakefs.fake_file import FakeFileWrapper

from deployment.app.config import get_settings
from deployment.app.db.database import DatabaseError
from deployment.app.models.api_models import JobStatus, JobType


# Helper for pyfakefs aiofiles compatibility
def pyfakefs_file_wrapper_for_aiofiles(pyfakefs_file_obj, *, loop=None, executor=None):
    return AsyncBufferedIOBase(pyfakefs_file_obj, loop=loop, executor=executor)


wrap.register(FakeFileWrapper)(pyfakefs_file_wrapper_for_aiofiles)
TEST_X_API_KEY = "test_x_api_key_conftest"


def assert_detail(response, expected_code=None, expected_message=None, expect_type=None):
    resp = response.json()
    detail = resp.get("detail")
    if detail is None:
        # fallback: сравниваем по response.text или коду
        assert response.status_code in (400, 401, 404, 422, 500)
        return
    if expect_type == dict:
        if not isinstance(detail, dict):
            assert False, f"detail is not dict: {detail} (type={type(detail)})"
        if expected_code:
            assert detail.get("code") == expected_code, f"Expected code {expected_code}, got {detail.get('code')} in {detail}"
        if expected_message:
            assert expected_message in detail.get("message", "")
    elif expect_type == str:
        if not isinstance(detail, str):
            assert False, f"detail is not str: {detail} (type={type(detail)})"
        if expected_message:
            assert expected_message in detail
    elif expect_type == list:
        if not isinstance(detail, list):
            assert False, f"detail is not list: {detail} (type={type(detail)})"
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


class TestDataUploadEndpoint:
    """Test suite for /api/v1/jobs/data-upload endpoint."""

    def test_create_data_upload_job_success(
        self, client, mock_dal, monkeypatch
    ):
        """Test successful creation of a data upload job."""
        # Arrange
        job_id = str(uuid.uuid4())
        mock_dal.create_job.return_value = job_id

        # Mock background task and other dependencies
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)
        monkeypatch.setattr("deployment.app.api.jobs.validate_date_format", lambda x: (True, "2022-09-30"))
        monkeypatch.setattr("deployment.app.api.jobs.validate_data_file_upload", AsyncMock())
        monkeypatch.setattr("deployment.app.api.jobs.validate_stock_file", lambda x, y: (True, None))
        monkeypatch.setattr("deployment.app.api.jobs.validate_sales_file", lambda x, y: (True, None))

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
        data = {"cutoff_date": "30.09.2022"}

        # Act
        response = client.post(
            "/api/v1/jobs/data-upload",
            files=files,
            data=data,
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["job_id"] == job_id
        assert resp_data["status"] == JobStatus.PENDING.value
        mock_dal.create_job.assert_called_once()
        mock_add_task.assert_called_once()

    def test_create_data_upload_job_invalid_date(
        self, client, monkeypatch
    ):
        """Test data upload job creation fails with invalid date format and returns ErrorDetailResponse."""
        # Arrange
        monkeypatch.setattr("deployment.app.api.jobs.validate_date_format", lambda x: (False, None))
        
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
        data = {"cutoff_date": "30-09-2022"}

        # Act
        response = client.post(
            "/api/v1/jobs/data-upload",
            files=files,
            data=data,
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 400
        assert_detail(response, expected_code="validation_error", expect_type=dict)


class TestTrainingJobEndpoint:
    """Test suite for /api/v1/jobs/training endpoint."""

    def test_create_training_job_success(
        self, client, mock_dal, monkeypatch
    ):
        """Test successful creation of a training job."""
        # Arrange
        job_id = str(uuid.uuid4())
        mock_dal.create_job.return_value = job_id
        mock_dal.get_effective_config.return_value = {
            "config_id": "test-config",
            "config": {},
        }
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)

        # Act
        response = client.post(
            "/api/v1/jobs/training", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["job_id"] == job_id
        assert resp_data["status"] == JobStatus.PENDING.value
        mock_dal.create_job.assert_called_once()
        mock_add_task.assert_called_once()

    def test_create_training_job_db_error(
        self, client, mock_dal, monkeypatch
    ):
        """Test training job creation fails with a database error and returns ErrorDetailResponse."""
        # Arrange
        mock_dal.create_job.side_effect = DatabaseError("DB error")
        mock_dal.get_effective_config.return_value = {
            "config_id": "test-config",
            "config": {},
        }

        # Act
        response = client.post(
            "/api/v1/jobs/training", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 500
        assert_detail(response, expected_code="internal_server_error", expect_type=dict)

    def test_create_training_job_no_active_config(
        self, client, mock_dal
    ):
        """Test training job creation fails with 400 if no active config is found and returns ErrorDetailResponse."""
        # Arrange
        mock_dal.get_effective_config.side_effect = ValueError("No active config and no best config by metric available")

        # Act
        response = client.post(
            "/api/v1/jobs/training", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 400
        assert_detail(response, expected_code="invalid_input_data", expect_type=dict)


class TestReportJobEndpoint:
    """Test suite for /api/v1/jobs/reports endpoint."""

    def test_create_report_job_success(self, client, mock_dal, monkeypatch):
        """Test successful creation of a prediction report job."""
        # Arrange
        job_id = str(uuid.uuid4())
        mock_dal.create_job.return_value = job_id
        mock_add_task = MagicMock()
        monkeypatch.setattr("deployment.app.api.jobs.BackgroundTasks.add_task", mock_add_task)

        params = {
            "report_type": "prediction_report",
            "prediction_month": "2023-01-01T00:00:00Z",
        }

        # Act
        response = client.post(
            "/api/v1/jobs/reports", json=params, headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["job_id"] == job_id
        assert resp_data["status"] == JobStatus.PENDING.value

        # Assert create_job was called with processed data
        mock_dal.create_job.assert_called_once()
        call_args, call_kwargs = mock_dal.create_job.call_args
        assert call_kwargs["job_type"] == JobType.REPORT
        called_params = call_kwargs["parameters"]
        assert called_params["report_type"] == params["report_type"]
        assert called_params["prediction_month"] == datetime.fromisoformat(
            params["prediction_month"].replace("Z", "+00:00")
        )
        mock_add_task.assert_called_once()


class TestJobStatusEndpoint:
    """Test suite for job status retrieval."""

    def get_full_mock_job_data(self, job_id, job_type, status):
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

    def test_get_job_status_training_completed(
        self, client, mock_dal
    ):
        """Test getting status of a completed training job."""
        # Arrange
        job_id = "job-123"
        mock_dal.get_job.side_effect = lambda jid: self.get_full_mock_job_data(jid, JobType.TRAINING, JobStatus.COMPLETED)
        mock_dal.get_training_result.return_value = {
            "model_id": "model-abc",
            "metrics": json.dumps({"accuracy": 0.95}),
            "parameters": json.dumps({"param": "value"}),
            "duration": 120,
        }

        # Act
        response = client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == JobStatus.COMPLETED.value
        assert "result" in data
        assert data["result"]["model_id"] == "model-abc"

    def test_get_job_status_report_completed(
        self, client, mock_dal
    ):
        """Test getting status of a completed report job with all ReportResponse fields."""
        # Arrange
        job_id = "report-job-456"
        mock_dal.get_job.side_effect = lambda jid: self.get_full_mock_job_data(jid, JobType.REPORT, JobStatus.COMPLETED)
        mock_dal.get_report_result.return_value = {
            "report_type": "prediction_report",
            "prediction_month": "2023-01-01",
            "records_count": 100,
            "csv_data": "header1,header2\nvalue1,value2",
            "has_enriched_metrics": True,
            "enriched_columns": json.dumps(["enriched_col1", "enriched_col2"]),
            "generated_at": "2023-01-01T10:20:00",
            "filters_applied": json.dumps({"region": "Europe"}),
            "parameters": json.dumps({"model_id": "model-xyz"}),
            "output_path": "/path/to/report.csv",
        }

        # Act
        response = client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == JobStatus.COMPLETED.value
        assert "result" in data
        result = data["result"]
        assert result["report_type"] == "prediction_report"
        assert result["prediction_month"] == "2023-01-01"
        assert result["records_count"] == 100
        assert result["csv_data"] == "header1,header2\nvalue1,value2"
        assert result["has_enriched_metrics"] is True
        assert result["enriched_columns"] == ["enriched_col1", "enriched_col2"]
        assert result["generated_at"] == "2023-01-01T10:20:00"
        assert result["filters_applied"] == {"region": "Europe"}
        assert result["parameters"] == {"model_id": "model-xyz"}
        assert result["output_path"] == "/path/to/report.csv"

    def test_get_job_status_pending(self, client, mock_dal):
        """Test getting status of a pending job."""
        # Arrange
        job_id = "job-456"
        mock_dal.get_job.side_effect = lambda jid: self.get_full_mock_job_data(jid, JobType.DATA_UPLOAD, JobStatus.PENDING)

        # Act
        response = client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == JobStatus.PENDING.value
        # JobDetails model includes 'result' field set to None for non-completed jobs
        assert "result" in data
        assert data["result"] is None

    def test_get_job_status_result_db_error(
        self, client, mock_dal
    ):
        """Test job status handles database errors when fetching results and returns ErrorDetailResponse."""
        # Arrange
        job_id = "job-789"
        mock_dal.get_job.side_effect = lambda jid: self.get_full_mock_job_data(jid, JobType.PREDICTION, JobStatus.COMPLETED)
        mock_dal.get_prediction_result.side_effect = DatabaseError("Result error")

        # Act
        response = client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 500
        assert_detail(response, expected_code="database_error", expect_type=dict)

    def test_get_job_status_not_found(self, client, mock_dal):
        """Test job status returns 404 for non-existent job and returns ErrorDetailResponse."""
        # Arrange
        job_id = "non-existent"
        mock_dal.get_job.return_value = None

        # Act
        response = client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 404
        assert_detail(response, expected_code="job_not_found")


class TestJobListingEndpoint:
    """Test suite for job listing functionality."""

    def test_list_jobs_no_filters(self, client, mock_dal):
        """Test listing jobs without filters."""
        # Arrange
        mock_dal.list_jobs.return_value = [
            {
                "job_id": "job1",
                "job_type": "training",
                "status": "completed",
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T01:00:00",
                "progress": 100.0,
                "error_message": None,
            },
            {
                "job_id": "job2",
                "job_type": "tuning",
                "status": "pending",
                "created_at": "2023-01-02T00:00:00",
                "updated_at": "2023-01-02T01:00:00",
                "progress": 0.0,
                "error_message": None,
            },
        ]

        # Act
        response = client.get("/api/v1/jobs", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["jobs"][0]["job_id"] == "job1"
        assert data["jobs"][1]["job_id"] == "job2"
        mock_dal.list_jobs.assert_called_once_with(job_type=None, status=None, limit=100)

    def test_list_jobs_with_filters(self, client, mock_dal):
        """Test listing jobs with filters applied."""
        # Arrange
        mock_dal.list_jobs.return_value = [
            {
                "job_id": "job3",
                "job_type": "training",
                "status": "completed",
                "created_at": "2023-01-03T00:00:00",
                "updated_at": "2023-01-03T01:00:00",
                "progress": 100.0,
                "error_message": None,
            },
        ]

        # Act
        response = client.get(
            "/api/v1/jobs?job_type=training&status=completed&limit=10",
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["job_id"] == "job3"
        mock_dal.list_jobs.assert_called_once_with(
            job_type="training", status="completed", limit=10
        )

    def test_list_jobs_db_error(self, client, mock_dal):
        """Test job listing handles database errors."""
        # Arrange
        mock_dal.list_jobs.side_effect = DatabaseError("List error")
        
        # Act
        response = client.get("/api/v1/jobs", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 500
        assert_detail(response, expected_code="database_error")


class TestAuthenticationScenarios:
    """Test suite for API authentication."""

    def test_create_data_upload_job_unauthorized_missing_key(self, client):
        """Test data upload job fails with 401 if X-API-Key header is missing."""
        # Act
        response = client.post(
            "/api/v1/jobs/data-upload",
            files={"stock_file": ("f.csv", b"c")},
            data={"cutoff_date": "30.09.2022"},
        )

        # Assert
        assert response.status_code == 401

    def test_create_data_upload_job_unauthorized_invalid_key(self, client):
        """Test data upload job fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.post(
            "/api/v1/jobs/data-upload",
            files={"stock_file": ("f.csv", b"c")},
            data={"cutoff_date": "30.09.2022"},
            headers={"X-API-Key": "wrong_key"},
        )

        # Assert
        assert response.status_code == 401

    def test_create_data_upload_job_server_key_not_configured(
        self, client, monkeypatch
    ):
        """Test data upload job fails with 500 if server X-API-Key is not configured."""
        # Arrange
        monkeypatch.setattr(get_settings().api, "x_api_key", None)

        # Act
        response = client.post(
            "/api/v1/jobs/data-upload",
            files={"stock_file": ("f.csv", b"c")},
            data={"cutoff_date": "30.09.2022"},
            headers={"X-API-Key": "any_key"},
        )

        # Assert
        assert response.status_code == 500
        assert_detail(response, expect_type=str)


class TestValidationScenarios:
    """Test suite for request validation."""

    def test_invalid_job_id_format(self, client, mock_dal):
        """Test endpoints handle invalid job ID formats gracefully."""
        # Arrange
        mock_dal.get_job.return_value = None
        
        # Act
        response = client.get(
            "/api/v1/jobs/invalid-id-format", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        # The endpoint should handle this gracefully as 404
        assert response.status_code == 404

    def test_invalid_query_parameters(self, client):
        """Test job listing handles invalid query parameters."""
        # Act
        response = client.get(
            "/api/v1/jobs?job_type=invalid_type&status=invalid_status",
            headers={"X-API-Key": TEST_X_API_KEY},
        )

        # Assert
        # Should return validation error for invalid enum values
        assert response.status_code == 422


class TestIntegration:
    """Integration tests for the complete API functionality."""

    def test_module_imports_successfully(self):
        """Test that the jobs API module can be imported without errors."""
        # Act & Assert
        from deployment.app.api import jobs

        assert hasattr(jobs, "router")

    def test_constants_defined(self):
        """Test that all expected constants and dependencies are defined."""
        # Act & Assert
        from deployment.app.api.jobs import router
        from deployment.app.models.api_models import JobStatus, JobType

        assert JobStatus.PENDING is not None
        assert JobStatus.COMPLETED is not None
        assert JobType.DATA_UPLOAD is not None
        assert JobType.TRAINING is not None
        assert router is not None

    def test_fastapi_router_configuration(self):
        """Test that the FastAPI router is properly configured."""
        # Act & Assert
        from deployment.app.api.jobs import router

        # Verify router has expected routes
        route_paths = [route.path for route in router.routes]
        expected_paths = ["/data-upload", "/training", "/reports", "/{job_id}"]

        for expected_path in expected_paths:
            assert any(expected_path in path for path in route_paths), (
                f"Expected path {expected_path} not found in routes"
            )
