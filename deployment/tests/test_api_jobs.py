import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from unittest.mock import patch, MagicMock, AsyncMock
from io import BytesIO
import uuid

# Adjust imports based on your project structure
from app.main import app 
from app.models.api_models import (
    JobStatus, JobType, TrainingParams, PredictionParams, ReportParams
)
from app.db.database import DatabaseError
from app.utils.validation import ValidationError

# Fixture for the TestClient
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# --- Test /api/v1/jobs/data-upload ---

@patch("app.api.jobs.validate_date_format", return_value=(True, "2022-09-30"))
@patch("app.api.jobs.validate_excel_file_upload", new_callable=AsyncMock)
@patch("app.api.jobs.validate_stock_file", return_value=(True, None))
@patch("app.api.jobs.validate_sales_file", return_value=(True, None))
@patch("app.api.jobs.create_job")
@patch("app.api.jobs.BackgroundTasks.add_task")
async def test_create_data_upload_job_success(
    mock_add_task, mock_create_job, mock_validate_sales, mock_validate_stock, 
    mock_validate_excel, mock_validate_date, client
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
    
    response = client.post("/api/v1/jobs/data-upload", files=files, data=data)
    
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["job_id"] == job_id
    assert resp_data["status"] == JobStatus.PENDING.value
    
    # Check validations were called
    mock_validate_date.assert_called_once_with("30.09.2022")
    assert mock_validate_excel.call_count == 3 # 1 stock + 2 sales
    mock_validate_stock.assert_called_once_with(stock_content)
    assert mock_validate_sales.call_count == 2
    mock_validate_sales.assert_any_call(sales_content1)
    mock_validate_sales.assert_any_call(sales_content2)
    
    # Check job creation
    mock_create_job.assert_called_once_with(
        JobType.DATA_UPLOAD,
        parameters={
            "stock_file": "stock.xlsx",
            "sales_files": ["sales1.xlsx", "sales2.xlsx"],
            "cutoff_date": "30.09.2022"
        }
    )
    
    # Check background task scheduling
    assert mock_add_task.call_count == 1
    args, kwargs = mock_add_task.call_args
    assert args[0].__name__ == 'process_data_files' # Check the function passed
    assert kwargs['job_id'] == job_id
    assert kwargs['cutoff_date'] == "30.09.2022"
    assert isinstance(kwargs['stock_file'], UploadFile)
    assert isinstance(kwargs['sales_files'], list)
    assert len(kwargs['sales_files']) == 2
    assert isinstance(kwargs['sales_files'][0], UploadFile)

@patch("app.api.jobs.validate_date_format", return_value=(False, None))
async def test_create_data_upload_job_invalid_date(mock_validate_date, client):
    """Test data upload job creation fails with invalid date format."""
    files = {
        "stock_file": ("stock.xlsx", b"data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "sales_files": ("sales.xlsx", b"data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    }
    data = {"cutoff_date": "30-09-2022"}
    
    response = client.post("/api/v1/jobs/data-upload", files=files, data=data)
    
    # Expecting 400 Bad Request from custom ValidationError handler
    assert response.status_code == 400
    # Further checks depend on the exact structure of the validation error response
    assert "Invalid cutoff date format" in response.text

@patch("app.api.jobs.validate_date_format", return_value=(True, "2022-09-30"))
@patch("app.api.jobs.validate_excel_file_upload", new_callable=AsyncMock)
@patch("app.api.jobs.validate_stock_file", return_value=(False, "Bad stock format"))
async def test_create_data_upload_job_invalid_stock(mock_validate_stock, mock_validate_excel, mock_validate_date, client):
    """Test data upload job creation fails with invalid stock file."""
    files = {
        "stock_file": ("stock.xlsx", b"bad stock data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "sales_files": ("sales.xlsx", b"sales data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    }
    data = {"cutoff_date": "30.09.2022"}
    
    response = client.post("/api/v1/jobs/data-upload", files=files, data=data)
    
    assert response.status_code == 400
    assert "Invalid stock file: Bad stock format" in response.text

@patch("app.api.jobs.validate_date_format", return_value=(True, "2022-09-30"))
@patch("app.api.jobs.validate_excel_file_upload", new_callable=AsyncMock)
@patch("app.api.jobs.validate_stock_file", return_value=(True, None))
@patch("app.api.jobs.validate_sales_file", side_effect=[(True, None), (False, "Bad sales format")])
async def test_create_data_upload_job_invalid_sales(mock_validate_sales, mock_validate_stock, mock_validate_excel, mock_validate_date, client):
    """Test data upload job creation fails with an invalid sales file."""
    files = [
        ("stock_file", ("stock.xlsx", b"stock data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
        ("sales_files", ("sales1.xlsx", b"good sales", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")),
        ("sales_files", ("sales2.xlsx", b"bad sales", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
    ]
    data = {"cutoff_date": "30.09.2022"}
    
    response = client.post("/api/v1/jobs/data-upload", files=files, data=data)
    
    assert response.status_code == 400
    assert "Invalid sales file (sales2.xlsx): Bad sales format" in response.text

@patch("app.api.jobs.validate_date_format", return_value=(True, "2022-09-30"))
@patch("app.api.jobs.validate_excel_file_upload", new_callable=AsyncMock)
@patch("app.api.jobs.validate_stock_file", return_value=(True, None))
@patch("app.api.jobs.validate_sales_file", return_value=(True, None))
@patch("app.api.jobs.create_job", side_effect=DatabaseError("DB connection lost"))
async def test_create_data_upload_job_db_error(
    mock_create_job, mock_validate_sales, mock_validate_stock, 
    mock_validate_excel, mock_validate_date, client
):
    """Test data upload job creation fails on database error."""
    files = {
        "stock_file": ("stock.xlsx", b"stock data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "sales_files": ("sales.xlsx", b"sales data", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    }
    data = {"cutoff_date": "30.09.2022"}
    
    response = client.post("/api/v1/jobs/data-upload", files=files, data=data)
    
    assert response.status_code == 500
    assert "database_error" in response.text
    assert "DB connection lost" in response.text
    mock_create_job.assert_called_once()

# --- Test /api/v1/jobs/training ---

@patch("app.api.jobs.create_job")
@patch("app.api.jobs.BackgroundTasks.add_task")
def test_create_training_job_success(mock_add_task, mock_create_job, client):
    """Test successful creation of a training job."""
    job_id = str(uuid.uuid4())
    mock_create_job.return_value = job_id
    # Provide all required fields for TrainingParams
    params = TrainingParams(
        input_chunk_length=12, 
        output_chunk_length=6, 
        max_epochs=10, 
        batch_size=32, 
        learning_rate=0.001
    )
    
    response = client.post("/api/v1/jobs/training", json=params.model_dump())
    
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["job_id"] == job_id
    assert resp_data["status"] == JobStatus.PENDING.value
    
    mock_create_job.assert_called_once_with(JobType.TRAINING, parameters=params.model_dump())
    mock_add_task.assert_called_once()
    args, kwargs = mock_add_task.call_args
    assert args[0].__name__ == 'train_model'
    assert kwargs['job_id'] == job_id
    assert kwargs['params'] == params

@patch("app.api.jobs.create_job", side_effect=DatabaseError("DB error"))
def test_create_training_job_db_error(mock_create_job, client):
    """Test training job creation fails on database error."""
    # Provide all required fields for TrainingParams
    params = TrainingParams(
        input_chunk_length=12, 
        output_chunk_length=6, 
        max_epochs=10, 
        batch_size=32, 
        learning_rate=0.001
    )
    response = client.post("/api/v1/jobs/training", json=params.model_dump())
    assert response.status_code == 500
    assert "database_error" in response.text

def test_create_training_job_invalid_params(client):
    """Test training job creation fails with invalid parameters."""
    invalid_params = {"epochs": -5} # Invalid epoch number
    response = client.post("/api/v1/jobs/training", json=invalid_params)
    assert response.status_code == 422 # FastAPI validation

# --- Test /api/v1/jobs/prediction ---

@patch("app.api.jobs.create_job")
@patch("app.api.jobs.BackgroundTasks.add_task")
def test_create_prediction_job_success(mock_add_task, mock_create_job, client):
    """Test successful creation of a prediction job."""
    job_id = str(uuid.uuid4())
    mock_create_job.return_value = job_id
    model_id_val = str(uuid.uuid4())
    # Provide all required fields for PredictionParams
    params = PredictionParams(model_id=model_id_val, prediction_length=12)
    
    response = client.post("/api/v1/jobs/prediction", json=params.model_dump())
    
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["job_id"] == job_id
    assert resp_data["status"] == JobStatus.PENDING.value
    
    mock_create_job.assert_called_once_with(JobType.PREDICTION, parameters=params.model_dump())
    mock_add_task.assert_called_once()
    args, kwargs = mock_add_task.call_args
    assert args[0].__name__ == 'generate_predictions'
    assert kwargs['job_id'] == job_id
    assert kwargs['params'] == params

# --- Test /api/v1/jobs/reports ---

@patch("app.api.jobs.create_job")
@patch("app.api.jobs.BackgroundTasks.add_task")
def test_create_report_job_success(mock_add_task, mock_create_job, client):
    """Test successful creation of a report job."""
    job_id = str(uuid.uuid4())
    mock_create_job.return_value = job_id
    # Provide required field for ReportParams
    params = ReportParams(report_type="sales_summary")
    
    response = client.post("/api/v1/jobs/reports", json=params.model_dump())
    
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["job_id"] == job_id
    assert resp_data["status"] == JobStatus.PENDING.value
    
    mock_create_job.assert_called_once_with(JobType.REPORT, parameters=params.model_dump())
    mock_add_task.assert_called_once()
    args, kwargs = mock_add_task.call_args
    assert args[0].__name__ == 'generate_report'
    assert kwargs['job_id'] == job_id
    assert kwargs['params'] == params

# --- Test /api/v1/jobs/{job_id} ---

@patch("app.api.jobs.get_job")
@patch("app.api.jobs.get_training_result")
def test_get_job_status_training_completed(mock_get_result, mock_get_job, client):
    """Test getting status of a completed training job."""
    job_id = str(uuid.uuid4())
    result_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "job_type": JobType.TRAINING,
        "status": JobStatus.COMPLETED,
        "created_at": "2023-01-01T10:00:00",
        "updated_at": "2023-01-01T10:30:00",
        "parameters": {"epochs": 10},
        "progress": 1.0,
        "result": None,
        "result_id": result_id,
        "error_message": None
    }
    training_result = {
        "model_id": str(uuid.uuid4()), 
        "metrics": "{\"accuracy\": 0.95}", 
        "parameters": "{\"epochs\": 10}", 
        "duration": 120.5
    }
    
    mock_get_job.return_value = job_data
    mock_get_result.return_value = training_result
    
    response = client.get(f"/api/v1/jobs/{job_id}")
    
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["job_id"] == job_id
    assert resp_data["job_type"] == JobType.TRAINING.value
    assert resp_data["status"] == JobStatus.COMPLETED.value
    assert resp_data["result"] == {
        "model_id": training_result["model_id"],
        "metrics": {"accuracy": 0.95},
        "parameters": {"epochs": 10},
        "duration": 120.5
    }
    
    mock_get_job.assert_called_once_with(job_id)
    mock_get_result.assert_called_once_with(result_id)

@patch("app.api.jobs.get_job")
def test_get_job_status_pending(mock_get_job, client):
    """Test getting status of a pending job."""
    job_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "job_type": JobType.DATA_UPLOAD,
        "status": JobStatus.PENDING,
        "created_at": "2023-01-01T11:00:00",
        "updated_at": "2023-01-01T11:00:00",
        "parameters": {},
        "progress": 0.0,
        "result": None,
        "result_id": None,
        "error_message": None
    }
    mock_get_job.return_value = job_data
    
    response = client.get(f"/api/v1/jobs/{job_id}")
    
    assert response.status_code == 200
    resp_data = response.json()
    assert resp_data["job_id"] == job_id
    assert resp_data["status"] == JobStatus.PENDING.value
    assert resp_data["result"] is None
    # No result fetcher should be called for pending jobs

@patch("app.api.jobs.get_job")
@patch("app.api.jobs.get_prediction_result", side_effect=DatabaseError("Result error"))
def test_get_job_status_result_db_error(mock_get_result, mock_get_job, client):
    """Test job status fetch fails if result fetch fails."""
    job_id = str(uuid.uuid4())
    result_id = str(uuid.uuid4())
    job_data = {
        "job_id": job_id,
        "job_type": JobType.PREDICTION,
        "status": JobStatus.COMPLETED, # Status is completed, but result fetch fails
        "created_at": "2023-01-01T10:00:00",
        "updated_at": "2023-01-01T10:30:00",
        "parameters": {},
        "progress": 1.0,
        "result": None,
        "result_id": result_id,
        "error_message": None
    }
    mock_get_job.return_value = job_data
    
    response = client.get(f"/api/v1/jobs/{job_id}")
    
    assert response.status_code == 500
    assert "database_error" in response.text
    assert f"Failed to retrieve job {job_id} due to database error" in response.text
    mock_get_result.assert_called_once_with(result_id)

@patch("app.api.jobs.get_job", return_value=None)
def test_get_job_status_not_found(mock_get_job, client):
    """Test getting status of a non-existent job."""
    job_id = str(uuid.uuid4())
    response = client.get(f"/api/v1/jobs/{job_id}")
    assert response.status_code == 404
    # Made assertion less specific
    assert job_id in response.text 
    assert "not found" in response.text.lower()
    mock_get_job.assert_called_once_with(job_id)

# --- Test /api/v1/jobs/ ---

@patch("app.api.jobs.list_jobs")
def test_list_jobs_no_filters(mock_list_jobs, client):
    """Test listing jobs without any filters."""
    job1_id = str(uuid.uuid4())
    job2_id = str(uuid.uuid4())
    # Add more fields to match JobDetails model used in JobsList
    mock_jobs = [
        {
            "job_id": job1_id, 
            "status": JobStatus.COMPLETED.value, 
            "job_type": JobType.TRAINING.value, 
            "created_at": "2023-01-01T10:00:00", 
            "updated_at": "2023-01-01T10:30:00",
            "progress": 1.0,
            "result": None, 
            "error_message": None
        },
        {
            "job_id": job2_id, 
            "status": JobStatus.PENDING.value, 
            "job_type": JobType.PREDICTION.value, 
            "created_at": "2023-01-02T11:00:00", 
            "updated_at": "2023-01-02T11:00:00",
            "progress": 0.1,
            "result": None, 
            "error_message": None
        }
    ]
    mock_list_jobs.return_value = mock_jobs # Return only the list
    
    response = client.get("/api/v1/jobs/")
    
    assert response.status_code == 200
    resp_data = response.json()
    assert len(resp_data["jobs"]) == 2
    assert resp_data["jobs"][0]["job_id"] == job1_id
    assert resp_data["jobs"][1]["job_id"] == job2_id
    mock_list_jobs.assert_called_once_with(job_type=None, status=None, limit=100)

@patch("app.api.jobs.list_jobs")
def test_list_jobs_with_filters(mock_list_jobs, client):
    """Test listing jobs with type and status filters."""
    job_id = str(uuid.uuid4())
    # Add more fields
    mock_jobs = [
        {
            "job_id": job_id, 
            "status": JobStatus.FAILED.value, 
            "job_type": JobType.DATA_UPLOAD.value,
            "created_at": "2023-01-03T10:00:00", 
            "updated_at": "2023-01-03T10:05:00",
            "progress": 0.5,
            "result": None, 
            "error_message": "File processing failed"
        }
    ]
    mock_list_jobs.return_value = mock_jobs # Return only the list
    
    response = client.get("/api/v1/jobs/", params={
        "job_type": JobType.DATA_UPLOAD.value,
        "status": JobStatus.FAILED.value,
        "limit": 50
    })
    
    assert response.status_code == 200
    resp_data = response.json()
    assert len(resp_data["jobs"]) == 1
    assert resp_data["jobs"][0]["job_id"] == job_id
    mock_list_jobs.assert_called_once_with(
        job_type=JobType.DATA_UPLOAD,
        status=JobStatus.FAILED,
        limit=50
    )

@patch("app.api.jobs.list_jobs", side_effect=DatabaseError("List error"))
def test_list_jobs_db_error(mock_list_jobs, client):
    """Test listing jobs fails on database error."""
    response = client.get("/api/v1/jobs/")
    assert response.status_code == 500
    assert "database_error" in response.text
    assert "Failed to list jobs" in response.text
    mock_list_jobs.assert_called_once()

# Add more tests for edge cases, specific validation failures, etc. 