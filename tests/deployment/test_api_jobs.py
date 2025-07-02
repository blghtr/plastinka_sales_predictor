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


class TestDataUploadEndpoint:
    """Test suite for /api/v1/jobs/data-upload endpoint."""

    @patch("aiofiles.threadpool.sync_open", new_callable=MagicMock)
    @patch("deployment.app.api.jobs.BackgroundTasks.add_task")
    @patch("deployment.app.api.jobs.create_job")
    @patch("deployment.app.api.jobs.validate_sales_file", return_value=(True, None))
    @patch("deployment.app.api.jobs.validate_stock_file", return_value=(True, None))
    @patch("deployment.app.api.jobs.validate_data_file_upload", new_callable=AsyncMock)
    @patch(
        "deployment.app.api.jobs.validate_date_format",
        return_value=(True, "2022-09-30"),
    )
    @patch("deployment.app.config.get_settings")
    async def test_create_data_upload_job_success(
        self,
        mock_get_settings,
        _mock_validate_date,
        _mock_validate_data,
        _mock_validate_stock,
        _mock_validate_sales,
        mock_create_job,
        mock_add_task,
        mock_sync_open,
        client,
        fs,
    ):
        """Test successful creation of a data upload job."""
        # Arrange
        get_settings.cache_clear()
        mock_settings_obj = MagicMock()
        mock_settings_obj.temp_upload_dir = "./temp_test_uploads"
        mock_get_settings.return_value = mock_settings_obj

        job_id = str(uuid.uuid4())
        mock_create_job.return_value = job_id

        stock_content = b"stock data"
        sales_content1 = b"sales data 1"
        sales_content2 = b"sales data 2"

        files = [
            (
                "stock_file",
                (
                    "stock.xlsx",
                    BytesIO(stock_content),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            ),
            (
                "sales_files",
                (
                    "sales1.xlsx",
                    BytesIO(sales_content1),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            ),
            (
                "sales_files",
                (
                    "sales2.xlsx",
                    BytesIO(sales_content2),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ),
            ),
        ]
        data = {"cutoff_date": "30.09.2022"}

        def sync_open_side_effect(file_path, mode, *args, **kwargs):
            return open(file_path, mode)

        mock_sync_open.side_effect = sync_open_side_effect

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
        mock_create_job.assert_called_once()
        mock_add_task.assert_called_once()

        # Cleanup
        get_settings.cache_clear()

    @patch("deployment.app.api.jobs.validate_date_format", return_value=(False, None))
    async def test_create_data_upload_job_invalid_date(
        self, _mock_validate_date, client
    ):
        """Test data upload job creation fails with invalid date format."""
        # Arrange
        files = {
            "stock_file": ("stock.xlsx", b"data"),
            "sales_files": ("sales.xlsx", b"data"),
        }
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
        assert "Invalid cutoff date format" in response.text


class TestTrainingJobEndpoint:
    """Test suite for /api/v1/jobs/training endpoint."""

    @patch("deployment.app.api.jobs.create_job")
    @patch("deployment.app.api.jobs.BackgroundTasks.add_task")
    @patch("deployment.app.api.jobs.get_effective_config")
    def test_create_training_job_success(
        self, mock_get_effective_config, mock_add_task, mock_create_job, client
    ):
        """Test successful creation of a training job."""
        # Arrange
        job_id = str(uuid.uuid4())
        mock_create_job.return_value = job_id
        mock_get_effective_config.return_value = {
            "config_id": "test-config",
            "config": {},
        }

        # Act
        response = client.post(
            "/api/v1/jobs/training", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["job_id"] == job_id
        assert resp_data["status"] == JobStatus.PENDING.value
        mock_create_job.assert_called_once()
        mock_add_task.assert_called_once()

    @patch("deployment.app.api.jobs.create_job", side_effect=DatabaseError("DB error"))
    @patch("deployment.app.api.jobs.get_effective_config")
    def test_create_training_job_db_error(
        self, mock_get_effective_config, mock_create_job, client
    ):
        """Test training job creation fails with a database error."""
        # Arrange
        mock_get_effective_config.return_value = {
            "config_id": "test-config",
            "config": {},
        }

        # Act
        response = client.post(
            "/api/v1/jobs/training", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 500
        # The training endpoint uses HTTPException which gets processed by http_exception_handler
        response_data = response.json()
        assert response_data["error"]["code"] == "http_500"
        assert (
            response_data["error"]["message"]
            == "An unexpected internal server error occurred while initiating the job."
        )

    @patch("deployment.app.api.jobs.get_effective_config", return_value=None)
    def test_create_training_job_no_active_config(
        self, mock_get_effective_config, client
    ):
        """Test training job creation fails with 400 if no active config is found."""
        # Act
        response = client.post(
            "/api/v1/jobs/training", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 400
        # Check the error message from the custom exception handler
        assert "No active config" in response.json()["error"]["message"]
        mock_get_effective_config.assert_called_once()


class TestReportJobEndpoint:
    """Test suite for /api/v1/jobs/reports endpoint."""

    @patch("deployment.app.api.jobs.create_job")
    @patch("deployment.app.api.jobs.BackgroundTasks.add_task")
    def test_create_report_job_success(self, mock_add_task, mock_create_job, client):
        """Test successful creation of a prediction report job."""
        # Arrange
        job_id = str(uuid.uuid4())
        mock_create_job.return_value = job_id
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
        mock_create_job.assert_called_once()
        call_args, call_kwargs = mock_create_job.call_args
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
            "job_type": job_type,
            "status": status,
            "created_at": "2023-01-01T10:00:00",
            "updated_at": "2023-01-01T10:30:00",
            "parameters": {},
            "progress": 1.0 if status == JobStatus.COMPLETED else 0.0,
            "result_id": "res-123" if status == JobStatus.COMPLETED else None,
            "error_message": "An error occurred"
            if status == JobStatus.FAILED
            else None,
        }

    @patch("deployment.app.api.jobs.get_job")
    @patch("deployment.app.api.jobs.get_training_result")
    def test_get_job_status_training_completed(
        self, mock_get_result, mock_get_job, client
    ):
        """Test getting status of a completed training job."""
        # Arrange
        job_id = "job-123"
        mock_get_job.return_value = self.get_full_mock_job_data(
            job_id, JobType.TRAINING, JobStatus.COMPLETED
        )
        mock_get_result.return_value = {
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

    @patch("deployment.app.api.jobs.get_job")
    def test_get_job_status_pending(self, mock_get_job, client):
        """Test getting status of a pending job."""
        # Arrange
        job_id = "job-456"
        mock_get_job.return_value = self.get_full_mock_job_data(
            job_id, JobType.DATA_UPLOAD, JobStatus.PENDING
        )

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

    @patch("deployment.app.api.jobs.get_job")
    @patch(
        "deployment.app.api.jobs.get_prediction_result",
        side_effect=DatabaseError("Result error"),
    )
    def test_get_job_status_result_db_error(
        self, mock_get_result, mock_get_job, client
    ):
        """Test job status handles database errors when fetching results."""
        # Arrange
        job_id = "job-789"
        mock_get_job.return_value = self.get_full_mock_job_data(
            job_id, JobType.PREDICTION, JobStatus.COMPLETED
        )

        # Act
        response = client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"

    @patch("deployment.app.api.jobs.get_job", return_value=None)
    def test_get_job_status_not_found(self, mock_get_job, client):
        """Test job status returns 404 for non-existent job."""
        # Arrange
        job_id = "non-existent"

        # Act
        response = client.get(
            f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        assert response.status_code == 404


class TestJobListingEndpoint:
    """Test suite for job listing functionality."""

    @patch("deployment.app.api.jobs.list_jobs")
    def test_list_jobs_no_filters(self, mock_list_jobs, client):
        """Test listing jobs without filters."""
        # Arrange
        mock_jobs = [
            {
                "job_id": "job1",
                "job_type": JobType.DATA_UPLOAD,
                "status": JobStatus.COMPLETED,
                "created_at": "2023-01-01T10:00:00",
                "updated_at": "2023-01-01T10:30:00",
                "progress": 1.0,
                "error_message": None,
            },
            {
                "job_id": "job2",
                "job_type": JobType.TRAINING,
                "status": JobStatus.PENDING,
                "created_at": "2023-01-01T11:00:00",
                "updated_at": "2023-01-01T11:30:00",
                "progress": 0.5,
                "error_message": None,
            },
        ]
        mock_list_jobs.return_value = mock_jobs

        # Act
        response = client.get("/api/v1/jobs", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["jobs"][0]["job_id"] == "job1"
        assert data["jobs"][1]["job_id"] == "job2"
        mock_list_jobs.assert_called_once_with(job_type=None, status=None, limit=100)

    @patch("deployment.app.api.jobs.list_jobs")
    def test_list_jobs_with_filters(self, mock_list_jobs, client):
        """Test listing jobs with filters applied."""
        # Arrange
        mock_jobs = [
            {
                "job_id": "job3",
                "job_type": JobType.TRAINING,
                "status": JobStatus.COMPLETED,
                "created_at": "2023-01-01T10:00:00",
                "updated_at": "2023-01-01T10:30:00",
                "progress": 1.0,
                "error_message": None,
            }
        ]
        mock_list_jobs.return_value = mock_jobs

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
        mock_list_jobs.assert_called_once_with(
            job_type="training", status="completed", limit=10
        )

    @patch("deployment.app.api.jobs.list_jobs", side_effect=DatabaseError("List error"))
    def test_list_jobs_db_error(self, mock_list_jobs, client):
        """Test job listing handles database errors."""
        # Act
        response = client.get("/api/v1/jobs", headers={"X-API-Key": TEST_X_API_KEY})

        # Assert
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "internal_error"


class TestAuthenticationScenarios:
    """Test suite for API authentication."""

    def test_create_data_upload_job_unauthorized_missing_key(self, client):
        """Test data upload job fails with 401 if X-API-Key header is missing."""
        # Act
        response = client.post(
            "/api/v1/jobs/data-upload",
            files={"f": "f"},
            data={"cutoff_date": "30.09.2022"},
        )

        # Assert
        assert response.status_code == 401

    def test_create_data_upload_job_unauthorized_invalid_key(self, client):
        """Test data upload job fails with 401 if X-API-Key is invalid."""
        # Act
        response = client.post(
            "/api/v1/jobs/data-upload",
            files={"f": "f"},
            data={"cutoff_date": "30.09.2022"},
            headers={"X-API-Key": "wrong_key"},
        )

        # Assert
        assert response.status_code == 401

    def test_create_data_upload_job_server_key_not_configured(
        self, client, mock_x_api_key
    ):
        """Test data upload job fails with 500 if server X-API-Key is not configured."""
        # Arrange
        mock_x_api_key(None)

        # Act
        response = client.post(
            "/api/v1/jobs/data-upload",
            files={"f": "f"},
            data={"cutoff_date": "30.09.2022"},
            headers={"X-API-Key": "any_key"},
        )

        # Assert
        assert response.status_code == 500
        assert (
            "X-API-Key authentication is not configured"
            in response.json()["error"]["message"]
        )


class TestValidationScenarios:
    """Test suite for request validation."""

    @patch("deployment.app.api.jobs.get_job", return_value=None)
    def test_invalid_job_id_format(self, mock_get_job, client):
        """Test endpoints handle invalid job ID formats gracefully."""
        # Act
        response = client.get(
            "/api/v1/jobs/invalid-id-format", headers={"X-API-Key": TEST_X_API_KEY}
        )

        # Assert
        # The endpoint should handle this gracefully, either as 404 or validation error
        assert response.status_code in [400, 404]

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
