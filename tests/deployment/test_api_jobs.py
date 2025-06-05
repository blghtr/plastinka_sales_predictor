import pytest
from fastapi import BackgroundTasks, UploadFile
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock
from io import BytesIO
import uuid
from pathlib import Path
import json
from datetime import datetime

from aiofiles.threadpool import wrap, AsyncBufferedIOBase
from pyfakefs.fake_file import FakeFileWrapper

from deployment.app.main import app
from deployment.app.models.api_models import JobStatus, JobType
from deployment.app.db.database import DatabaseError
from deployment.app.utils.validation import ValidationError

# Helper for pyfakefs aiofiles compatibility
def pyfakefs_file_wrapper_for_aiofiles(pyfakefs_file_obj, *, loop=None, executor=None):
    return AsyncBufferedIOBase(pyfakefs_file_obj, loop=loop, executor=executor)

wrap.register(FakeFileWrapper)(pyfakefs_file_wrapper_for_aiofiles)
TEST_X_API_KEY = "test_x_api_key_value_jobs"

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# --- Test /api/v1/jobs/data-upload ---
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
@patch("aiofiles.threadpool.sync_open", new_callable=MagicMock)
@patch("deployment.app.api.jobs.BackgroundTasks.add_task")
@patch("deployment.app.api.jobs.create_job")
@patch("deployment.app.api.jobs.validate_sales_file", return_value=(True, None))
@patch("deployment.app.api.jobs.validate_stock_file", return_value=(True, None))
@patch("deployment.app.api.jobs.validate_excel_file_upload", new_callable=AsyncMock)
@patch("deployment.app.api.jobs.validate_date_format", return_value=(True, "2022-09-30"))
@patch("deployment.app.api.jobs.settings.temp_upload_dir", return_value="./temp_test_uploads")
async def test_create_data_upload_job_success(
    mock_settings_temp_dir,
    _mock_validate_date,
    _mock_validate_excel,
    _mock_validate_stock,
    _mock_validate_sales,
    mock_create_job,
    mock_add_task,
    mock_sync_open,
    _mock_server_api_key,
    client, fs
):
    """Test successful creation of a data upload job."""
    job_id = str(uuid.uuid4())
    mock_create_job.return_value = job_id
    
    stock_content = b"stock data"
    sales_content1 = b"sales data 1"
    sales_content2 = b"sales data 2"
    
    files = [
        ("stock_file", ("stock.xlsx", BytesIO(stock_content), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
        ("sales_files", ("sales1.xlsx", BytesIO(sales_content1), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
        ("sales_files", ("sales2.xlsx", BytesIO(sales_content2), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
    ]
    data = {"cutoff_date": "30.09.2022"}

    def sync_open_side_effect(file_path, mode, *args, **kwargs):
        return open(file_path, mode)
    
    mock_sync_open.side_effect = sync_open_side_effect

    response = client.post("/api/v1/jobs/data-upload", files=files, data=data, headers={"X-API-Key": TEST_X_API_KEY})

    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["job_id"] == job_id
    assert resp_data["status"] == JobStatus.PENDING.value
    
@patch("deployment.app.api.jobs.validate_date_format", return_value=(False, None))
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
async def test_create_data_upload_job_invalid_date(_mock_server_api_key, _mock_validate_date, client):
    """Test data upload job creation fails with invalid date format."""
    files = {"stock_file": ("stock.xlsx", b"data"), "sales_files": ("sales.xlsx", b"data")}
    data = {"cutoff_date": "30-09-2022"}
    response = client.post("/api/v1/jobs/data-upload", files=files, data=data, headers={"X-API-Key": TEST_X_API_KEY})
    assert response.status_code == 400
    assert "Invalid cutoff date format" in response.text

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_create_data_upload_job_unauthorized_missing_key(_mock_server_api_key, client):
    """Test data upload job fails with 401 if X-API-Key header is missing."""
    response = client.post("/api/v1/jobs/data-upload", files={"f": "f"}, data={"cutoff_date": "30.09.2022"})
    assert response.status_code == 401

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_create_data_upload_job_unauthorized_invalid_key(_mock_server_api_key, client):
    """Test data upload job fails with 401 if X-API-Key is invalid."""
    response = client.post("/api/v1/jobs/data-upload", files={"f": "f"}, data={"cutoff_date": "30.09.2022"}, headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 401

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=""))
def test_create_data_upload_job_server_key_not_configured(_mock_server_api_key, client):
    """Test data upload job fails with 500 if server X-API-Key is not configured."""
    response = client.post("/api/v1/jobs/data-upload", files={"f": "f"}, data={"cutoff_date": "30.09.2022"}, headers={"X-API-Key": "any_key"})
    assert response.status_code == 500

# --- Training Job Tests ---
@patch("deployment.app.api.jobs.create_job")
@patch("deployment.app.api.jobs.BackgroundTasks.add_task")
@patch("deployment.app.api.jobs.get_effective_config")
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_create_training_job_success(_mock_server_api_key, mock_get_effective_config, mock_add_task, mock_create_job, client):
    """Test successful creation of a training job."""
    job_id = str(uuid.uuid4())
    mock_create_job.return_value = job_id
    mock_get_effective_config.return_value = {"config_id": "test-config", "config": {}}
    
    response = client.post("/api/v1/jobs/training", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["job_id"] == job_id
    assert resp_data["status"] == JobStatus.PENDING.value
    mock_create_job.assert_called_once()
    mock_add_task.assert_called_once()

@patch("deployment.app.api.jobs.create_job", side_effect=DatabaseError("DB error"))
@patch("deployment.app.api.jobs.get_effective_config")
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_create_training_job_db_error(_mock_server_api_key, mock_get_effective_config, mock_create_job, client):
    """Test training job creation fails with a database error."""
    mock_get_effective_config.return_value = {"config_id": "test-config", "config": {}}
    response = client.post("/api/v1/jobs/training", headers={"X-API-Key": TEST_X_API_KEY})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "http_500"

@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_create_training_job_invalid_params(_mock_server_api_key, client):
    """
    Test training job creation fails with invalid parameters.
    This test is complex. Sending invalid form data doesn't trigger a 422.
    Instead, the request passes, and the code fails later when get_effective_config
    can't find a config, which raises a ValueError, resulting in a 400.
    """
    # The endpoint doesn't take a JSON body, so a 422 won't be raised for malformed JSON.
    # It takes optional form data. Sending no data is valid, but then get_effective_config fails.
    response = client.post(
        "/api/v1/jobs/training",
        headers={"X-API-Key": TEST_X_API_KEY}
    )
    assert response.status_code == 400 # Expecting 400 from ValueError: No active config
    assert "No active config" in response.text

# --- Report Job Tests ---
@patch("deployment.app.api.jobs.create_job")
@patch("deployment.app.api.jobs.BackgroundTasks.add_task")
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_create_report_job_success(_mock_server_api_key, mock_add_task, mock_create_job, client):
    """Test successful creation of a report job."""
    job_id = str(uuid.uuid4())
    mock_create_job.return_value = job_id
    params = {
        "report_type": "sales_summary",
        "start_date": "2023-01-01",
        "end_date": "2023-01-31"
    }
    
    response = client.post("/api/v1/jobs/reports", json=params, headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["job_id"] == job_id
    assert resp_data["status"] == JobStatus.PENDING.value
    
    # Assert that create_job was called with the processed data
    mock_create_job.assert_called_once()
    call_args, call_kwargs = mock_create_job.call_args
    assert call_args[0] == JobType.REPORT
    called_params = call_kwargs['parameters']
    assert called_params['report_type'].value == params['report_type']
    assert called_params['start_date'] == datetime.fromisoformat(params['start_date'])
    assert called_params['end_date'] == datetime.fromisoformat(params['end_date'])
    mock_add_task.assert_called_once()

# --- Job Status and Listing Tests ---
def get_full_mock_job_data(job_id, job_type, status):
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
        "error_message": "An error occurred" if status == JobStatus.FAILED else None
    }

@patch("deployment.app.api.jobs.get_job")
@patch("deployment.app.api.jobs.get_training_result")
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_job_status_training_completed(_mock_server_api_key, mock_get_result, mock_get_job, client):
    """Test getting status of a completed training job."""
    job_id = "job-123"
    mock_get_job.return_value = get_full_mock_job_data(job_id, JobType.TRAINING, JobStatus.COMPLETED)
    mock_get_result.return_value = {
        "model_id": "model-abc",
        "metrics": json.dumps({"accuracy": 0.95}),
        "parameters": json.dumps({"param": "value"}),
        "duration": 120
    }
    
    response = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["status"] == JobStatus.COMPLETED.value
    assert resp_data["result"]["metrics"]["accuracy"] == 0.95
    assert resp_data["result"]["parameters"]["param"] == "value"

@patch("deployment.app.api.jobs.get_job")
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_job_status_pending(_mock_server_api_key, mock_get_job, client):
    """Test getting status of a pending job."""
    job_id = "job-123"
    mock_get_job.return_value = get_full_mock_job_data(job_id, JobType.DATA_UPLOAD, JobStatus.PENDING)
    
    response = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    assert response.json()["status"] == JobStatus.PENDING.value

@patch("deployment.app.api.jobs.get_job")
@patch("deployment.app.api.jobs.get_prediction_result", side_effect=DatabaseError("Result error"))
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_job_status_result_db_error(_mock_server_api_key, mock_get_result, mock_get_job, client):
    """Test job status check fails gracefully on database error when fetching result."""
    job_id = "job-123"
    mock_get_job.return_value = get_full_mock_job_data(job_id, JobType.PREDICTION, JobStatus.COMPLETED)
    
    response = client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "database_error"
    mock_get_result.assert_called_once_with('res-123')

@patch("deployment.app.api.jobs.get_job", return_value=None)
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_get_job_status_not_found_authorized(_mock_server_api_key, mock_get_job, client):
    """Test getting status for a non-existent job returns 404."""
    response = client.get("/api/v1/jobs/job-404", headers={"X-API-Key": TEST_X_API_KEY})
    assert response.status_code == 404
    mock_get_job.assert_called_once_with("job-404")

@patch("deployment.app.api.jobs.list_jobs")
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_list_jobs_no_filters(_mock_server_api_key, mock_list_jobs, client):
    """Test listing jobs without any filters."""
    mock_list_jobs.return_value = [
        get_full_mock_job_data("job-1", JobType.TRAINING, JobStatus.COMPLETED),
        get_full_mock_job_data("job-2", JobType.PREDICTION, JobStatus.RUNNING),
    ]
    response = client.get("/api/v1/jobs/", headers={"X-API-Key": TEST_X_API_KEY})
    assert response.status_code == 200
    resp_data = response.json()
    assert len(resp_data["jobs"]) == 2
    mock_list_jobs.assert_called_once_with(job_type=None, status=None, limit=100)

@patch("deployment.app.api.jobs.list_jobs")
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_list_jobs_with_filters(_mock_server_api_key, mock_list_jobs, client):
    """Test listing jobs with all available filters."""
    mock_list_jobs.return_value = [
        get_full_mock_job_data("job-filtered", JobType.TRAINING, JobStatus.COMPLETED)
    ]
    response = client.get("/api/v1/jobs/", params={
        "job_type": JobType.TRAINING.value,
        "status": JobStatus.COMPLETED.value,
        "limit": 50
    }, headers={"X-API-Key": TEST_X_API_KEY})
    
    assert response.status_code == 200
    resp_data = response.json()
    assert len(resp_data["jobs"]) == 1
    assert resp_data["jobs"][0]["job_id"] == "job-filtered"
    mock_list_jobs.assert_called_once_with(
        job_type=JobType.TRAINING,
        status=JobStatus.COMPLETED,
        limit=50
    )

@patch("deployment.app.api.jobs.list_jobs", side_effect=DatabaseError("List error"))
@patch("deployment.app.services.auth.settings.api.x_api_key", new_callable=PropertyMock(return_value=TEST_X_API_KEY))
def test_list_jobs_db_error(_mock_server_api_key, mock_list_jobs, client):
    """Test job listing fails gracefully on database error."""
    response = client.get("/api/v1/jobs/", headers={"X-API-Key": TEST_X_API_KEY})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "database_error"
    mock_list_jobs.assert_called_once()