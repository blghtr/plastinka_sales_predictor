from unittest.mock import MagicMock, mock_open, patch

import pytest

from deployment.datasphere.client import DataSphereClient, DataSphereClientError

# Constants for testing
TEST_PROJECT_ID = "test-project-id"
TEST_FOLDER_ID = "test-folder-id"
TEST_JOB_ID = "test-job-id"

# Define mock objects for the DatasphereClient
class MockJob:
    def __init__(self, job_id="test-job-id", status="SUCCESS"):
        self.id = job_id
        self.status = status
        self.output_files = ["output1.txt", "output2.txt"]
        self.log_files = ["log1.txt"]
        self.diagnostic_files = ["diag1.txt"]

class MockOperation:
    def __init__(self, id="test-operation-id"):
        self.id = id

@pytest.fixture
def mock_datasphere_client():
    """Fixture to create a mock for the DatasphereClient class."""
    mock_client = MagicMock()
    # Configure default return values
    mock_client.create.return_value = TEST_JOB_ID
    mock_client.execute.return_value = (MockOperation(), MagicMock())
    mock_client.get.return_value = MockJob()
    mock_client.list.return_value = [MockJob(job_id="job1"), MockJob(job_id="job2")]
    mock_client.download_files.return_value = None
    mock_client.cancel.return_value = None
    return mock_client

@pytest.fixture
def client(mock_datasphere_client):
    """Fixture to create a DataSphereClient with mocked DatasphereClient."""
    with patch('deployment.datasphere.client.DatasphereClient', return_value=mock_datasphere_client):
        client = DataSphereClient(project_id=TEST_PROJECT_ID, folder_id=TEST_FOLDER_ID)
        # Client's internal _client is our mock
        assert client._client == mock_datasphere_client
        return client

# Test initialization
def test_init_success():
    """Test successful client initialization."""
    with patch('deployment.datasphere.client.DatasphereClient') as mock_client_class:
        mock_client_class.return_value = MagicMock()
        client = DataSphereClient(
            project_id=TEST_PROJECT_ID,
            folder_id=TEST_FOLDER_ID,
            oauth_token="fake-token",
            yc_profile="default"
        )
        # Check that DatasphereClient was initialized with correct params
        mock_client_class.assert_called_once_with(oauth_token="fake-token", yc_profile="default")
        assert client.project_id == TEST_PROJECT_ID
        assert client.folder_id == TEST_FOLDER_ID

def test_init_failure():
    """Test client initialization failure."""
    with patch('deployment.datasphere.client.DatasphereClient') as mock_client_class:
        mock_client_class.side_effect = Exception("Connection failed")
        with pytest.raises(DataSphereClientError, match="Failed to initialize official DataSphere client"):
            DataSphereClient(project_id=TEST_PROJECT_ID, folder_id=TEST_FOLDER_ID)

def test_init_missing_project_id():
    """Test that initialization fails with missing project_id."""
    with pytest.raises(ValueError, match="DataSphere project_id is required"):
        DataSphereClient(project_id="", folder_id=TEST_FOLDER_ID)

def test_init_missing_folder_id():
    """Test that initialization fails with missing folder_id."""
    with pytest.raises(ValueError, match="Yandex Cloud folder_id is required"):
        DataSphereClient(project_id=TEST_PROJECT_ID, folder_id="")

# Test submit_job method - Updated to match real implementation
@patch('deployment.datasphere.client.parse_config')
@patch('deployment.datasphere.client.define_py_env')
@patch('deployment.datasphere.client.prepare_inputs')
@patch('deployment.datasphere.client.prepare_local_modules')
@patch('deployment.datasphere.client.check_limits')
@patch('builtins.open', new_callable=mock_open, read_data='name: test-job\ncmd: python test.py\noutputs:\n  - output.txt: OUTPUT')
def test_submit_job_success(mock_open_file, mock_check_limits, mock_prepare_local_modules, mock_prepare_inputs, mock_define_py_env, mock_parse_config, client, mock_datasphere_client):
    """Test successful job submission."""
    # Configure mocks to match real implementation
    mock_config = MagicMock()
    mock_config.env.python = None  # No python env for simplicity
    mock_config.inputs = []
    mock_job_params = MagicMock()
    mock_config.get_job_params.return_value = mock_job_params
    mock_parse_config.return_value = mock_config

    mock_prepare_inputs.return_value = []
    mock_check_limits.return_value = None
    mock_define_py_env.return_value = None
    mock_prepare_local_modules.return_value = []

    # Call the method
    job_id = client.submit_job("path/to/config.yaml", "/tmp/test")

    # Verify calls - matching real implementation flow
    mock_open_file.assert_called_once_with("path/to/config.yaml")
    mock_parse_config.assert_called_once()
    mock_prepare_inputs.assert_called_once_with([], "/tmp/test")
    mock_check_limits.assert_called_once()

    # Verify the client methods were called
    mock_datasphere_client.create.assert_called_once()
    mock_datasphere_client.execute.assert_called_once_with(TEST_JOB_ID)

    # Verify result
    assert job_id == TEST_JOB_ID

@patch('builtins.open')
def test_submit_job_config_not_found(mock_open_file, client):
    """Test job submission with config file not found."""
    mock_open_file.side_effect = FileNotFoundError("File not found")

    with pytest.raises(DataSphereClientError, match="Invalid job configuration format"):
        client.submit_job("nonexistent.yaml", "/tmp/test")

@patch('deployment.datasphere.client.parse_config')
@patch('builtins.open', new_callable=mock_open, read_data='invalid: yaml: file:')
def test_submit_job_invalid_yaml(mock_open_file, mock_parse_config, client):
    """Test job submission with invalid YAML."""
    mock_parse_config.side_effect = ValueError("mapping values are not allowed")

    with pytest.raises(DataSphereClientError, match="Invalid job configuration format"):
        client.submit_job("invalid.yaml", "/tmp/test")

@patch('deployment.datasphere.client.parse_config')
@patch('deployment.datasphere.client.define_py_env')
@patch('deployment.datasphere.client.prepare_inputs')
@patch('deployment.datasphere.client.prepare_local_modules')
@patch('deployment.datasphere.client.check_limits')
@patch('builtins.open', new_callable=mock_open, read_data='name: test-job\ncmd: python test.py\noutputs:\n  - output.txt: OUTPUT')
def test_submit_job_create_failure(mock_open_file, mock_check_limits, mock_prepare_local_modules, mock_prepare_inputs, mock_define_py_env, mock_parse_config, client, mock_datasphere_client):
    """Test job submission with creation failure."""
    # Configure mocks
    mock_config = MagicMock()
    mock_config.env.python = None
    mock_config.inputs = []
    mock_job_params = MagicMock()
    mock_config.get_job_params.return_value = mock_job_params
    mock_parse_config.return_value = mock_config

    mock_prepare_inputs.return_value = []
    mock_check_limits.return_value = None
    mock_define_py_env.return_value = None
    mock_prepare_local_modules.return_value = []

    # Configure create to fail
    mock_datasphere_client.create.side_effect = Exception("Failed to create job")

    with pytest.raises(DataSphereClientError, match="Failed to submit job"):
        client.submit_job("path/to/config.yaml", "/tmp/test")

# Test get_job_status method
def test_get_job_status_success(client, mock_datasphere_client):
    """Test successfully getting job status."""
    status = client.get_job_status(TEST_JOB_ID)

    mock_datasphere_client.get.assert_called_once_with(TEST_JOB_ID)
    assert status == "success"  # Normalised to lowercase

def test_get_job_status_failure(client, mock_datasphere_client):
    """Test failure when getting job status."""
    mock_datasphere_client.get.side_effect = Exception("Job not found")

    with pytest.raises(DataSphereClientError, match="Failed to get job status"):
        client.get_job_status(TEST_JOB_ID)

# New numeric status mapping tests
def test_get_job_status_numeric_running(client, mock_datasphere_client):
    """Job.status is numeric 2 (EXECUTING) – should map to 'running'."""
    # Arrange: make underlying SDK return numeric 2 (EXECUTING)
    mock_job = MagicMock()
    mock_job.status = 2  # EXECUTING -> should map to "running"
    mock_datasphere_client.get.return_value = mock_job

    # Act
    status = client.get_job_status(TEST_JOB_ID)

    # Assert
    assert status == "running"

def test_get_job_status_numeric_failed(client, mock_datasphere_client):
    """Job.status is numeric 5 (ERROR) – should map to 'failed'."""
    # Arrange: make underlying SDK return numeric 5 (ERROR)
    mock_job = MagicMock()
    mock_job.status = 5  # ERROR -> should map to "failed"
    mock_datasphere_client.get.return_value = mock_job

    # Act
    status = client.get_job_status(TEST_JOB_ID)

    # Assert
    assert status == "failed"

def test_get_job_status_numeric_cancelled(client, mock_datasphere_client):
    """Job.status is numeric 6 (CANCELLED) – should map to 'cancelled'."""
    # Arrange: make underlying SDK return numeric 6 (CANCELLED)
    mock_job = MagicMock()
    mock_job.status = 6  # CANCELLED -> should map to "cancelled"
    mock_datasphere_client.get.return_value = mock_job

    # Act
    status = client.get_job_status(TEST_JOB_ID)

    # Assert
    assert status == "cancelled"

# Test download_job_results method
def test_download_job_results_success(client, mock_datasphere_client):
    """Test successfully downloading job results."""
    client.download_job_results(TEST_JOB_ID, "/tmp/output", with_logs=True, with_diagnostics=True)

    # Verify get was called to fetch job details
    mock_datasphere_client.get.assert_called_once_with(TEST_JOB_ID)
    # Verify download_files was called with collected files
    mock_datasphere_client.download_files.assert_called_once_with(
        TEST_JOB_ID, 
        ['output1.txt', 'output2.txt', 'log1.txt', 'diag1.txt'],  # All file types included
        "/tmp/output"
    )

def test_download_job_results_failure(client, mock_datasphere_client):
    """Test failure when downloading job results."""
    mock_datasphere_client.download_files.side_effect = Exception("Download failed")

    with pytest.raises(DataSphereClientError, match="Failed to download job files"):
        client.download_job_results(TEST_JOB_ID, "/tmp/output")

# Test cancel_job method
def test_cancel_job_success(client, mock_datasphere_client):
    """Test successfully cancelling a job."""
    client.cancel_job(TEST_JOB_ID, graceful=True)

    mock_datasphere_client.cancel.assert_called_once_with(TEST_JOB_ID, graceful=True)

def test_cancel_job_failure(client, mock_datasphere_client):
    """Test failure when cancelling a job."""
    mock_datasphere_client.cancel.side_effect = Exception("Cancel failed")

    with pytest.raises(DataSphereClientError, match="Failed to cancel job"):
        client.cancel_job(TEST_JOB_ID)

# Test list_jobs method
def test_list_jobs_success(client, mock_datasphere_client):
    """Test successfully listing jobs."""
    jobs = client.list_jobs()

    mock_datasphere_client.list.assert_called_once()
    assert len(jobs) == 2
    assert jobs[0].id == "job1"
    assert jobs[1].id == "job2"

def test_list_jobs_failure(client, mock_datasphere_client):
    """Test failure when listing jobs."""
    mock_datasphere_client.list.side_effect = Exception("List failed")

    with pytest.raises(DataSphereClientError, match="Failed to list jobs"):
        client.list_jobs()

# Test get_job method
def test_get_job_success(client, mock_datasphere_client):
    """Test successfully getting a job."""
    job = client.get_job(TEST_JOB_ID)

    mock_datasphere_client.get.assert_called_once_with(TEST_JOB_ID)
    assert job.id == TEST_JOB_ID

def test_get_job_failure(client, mock_datasphere_client):
    """Test failure when getting a job."""
    mock_datasphere_client.get.side_effect = Exception("Get failed")

    with pytest.raises(DataSphereClientError, match="Failed to get job"):
        client.get_job(TEST_JOB_ID)
