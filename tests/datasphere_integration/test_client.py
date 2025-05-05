import pytest
from unittest.mock import patch, MagicMock
import subprocess
import re # Import re

# Import the actual client and error classes
from deployment.app.datasphere_integration.client import DataSphereClient, DataSphereClientError

# Mock the SDK classes used by the client
# We need to mock where they are imported/used within the client module
# The client now imports `datasphere.sdk` directly
mock_sdk_module_path = 'deployment.app.datasphere_integration.client.datasphere.sdk'

# Define simplified mock objects for SDK components (adjust as needed)
class MockJob:
    def __init__(self, status):
        self.status = status

class MockJobWrapper:
    def __init__(self, job_id, status):
        self.id = job_id
        self.job = MockJob(status)
        # Add other methods/properties if the client uses them
        # e.g., self.op, self.execute_call, self.done

# Placeholder for actual protobuf enum if needed for type hints/checks
class MockJobStatus:
    EXECUTING = "EXECUTING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    # Add other statuses used


@pytest.fixture
def ds_client():
    """Provides a DataSphereClient instance for testing."""
    # Initialize client - it will attempt SDK init with default params (None)
    client = DataSphereClient(project_id="test_project", folder_id="test_folder")
    # We patch the SDK class, so the real init won't run in tests needing the mock.
    return client

@pytest.fixture
def ds_client_no_sdk_init():
    """Provides a DataSphereClient instance WITHOUT initializing the SDK.
       SDK initialization will happen using patched versions in specific tests.
    """
    # Temporarily prevent SDK import within the fixture scope if necessary,
    # although patching the class in the test should be sufficient.
    client = DataSphereClient(project_id="test_project", folder_id="test_folder")
    # Ensure the fixture doesn't hold a real SDK instance if init was attempted
    client._sdk = None
    return client

# Patch subprocess where it's used in the client module
@patch('deployment.app.datasphere_integration.client.subprocess.run')
def test_submit_job_cli_success_id_parsing(mock_subprocess_run, ds_client):
    """Test successful job submission using the 'datasphere' CLI, parsing ID from output."""
    config_path = "path/to/config.yaml"
    # params are ignored by the refactored method, so we don't pass them
    # params = {"PARAM1": "value1"}
    expected_job_id = "c1qabcdefg1234567890"

    # --- Mock different stdout formats --- #
    # 1. Explicit "Job ID: ..." format
    stdout_text_1 = f"Processing...\nJob ID: {expected_job_id}\nComplete."
    stdout_format_1 = stdout_text_1.encode('utf-8')
    # 2. ID as the last line (typical YC ID format)
    stdout_text_2 = f"Some other output\nBlah blah\n{expected_job_id}"
    stdout_format_2 = stdout_text_2.encode('utf-8')
    # 3. No clear ID found (should raise error)
    stdout_text_3 = "Processing...\nFinished."
    stdout_format_3 = stdout_text_3.encode('utf-8')

    # Test Case 1: Explicit Job ID
    mock_result_1 = MagicMock()
    mock_result_1.returncode = 0
    mock_result_1.stdout = stdout_format_1
    mock_result_1.stderr = b""
    mock_subprocess_run.return_value = mock_result_1

    job_id_1 = ds_client.submit_job_cli(config_path)
    assert job_id_1 == expected_job_id

    expected_cmd_base = [
        'datasphere', 'project', 'job', 'execute',
        '-p', ds_client.project_id,
        '-c', config_path,
    ]
    mock_subprocess_run.assert_called_with(
        expected_cmd_base,
        capture_output=True,
        check=False,
        text=False
    )

    # Test Case 2: ID as last line
    mock_result_2 = MagicMock()
    mock_result_2.returncode = 0
    mock_result_2.stdout = stdout_format_2
    mock_result_2.stderr = b""
    mock_subprocess_run.return_value = mock_result_2

    job_id_2 = ds_client.submit_job_cli(config_path)
    assert job_id_2 == expected_job_id
    # Check call again (redundant check of args, but ensures it was called)
    mock_subprocess_run.assert_called_with(expected_cmd_base, capture_output=True, check=False, text=False)

    # Test Case 3: No ID found (expect error)
    mock_result_3 = MagicMock()
    mock_result_3.returncode = 0
    mock_result_3.stdout = stdout_format_3
    mock_result_3.stderr = b""
    mock_subprocess_run.return_value = mock_result_3

    with pytest.raises(DataSphereClientError, match="Failed to parse Job ID"):
        ds_client.submit_job_cli(config_path)

@patch('deployment.app.datasphere_integration.client.subprocess.run')
def test_submit_job_cli_failure(mock_subprocess_run, ds_client):
    """Test failed job submission using the 'datasphere' CLI."""
    config_path = "path/to/config.yaml"
    error_message = "Something went wrong"

    # Mock subprocess.run to simulate failed CLI execution
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = b""
    mock_result.stderr = error_message.encode('utf-8')
    mock_subprocess_run.return_value = mock_result

    with pytest.raises(DataSphereClientError, match=f"DataSphere CLI command failed.*Error: {error_message}"):
        ds_client.submit_job_cli(config_path)

@patch('deployment.app.datasphere_integration.client.subprocess.run')
def test_submit_job_cli_file_not_found(mock_subprocess_run, ds_client):
    """Test CLI command not found error."""
    config_path = "path/to/config.yaml"

    # Mock subprocess.run to raise FileNotFoundError
    mock_subprocess_run.side_effect = FileNotFoundError("[Errno 2] No such file or directory: 'datasphere'")

    with pytest.raises(DataSphereClientError, match="'datasphere' command not found"):
        ds_client.submit_job_cli(config_path)

# --- Tests for SDK-based methods --- #

@patch(f'{mock_sdk_module_path}.SDK')
def test_get_job_status_sdk_success(MockSDKClass, ds_client_no_sdk_init):
    """Test successfully getting job status using the SDK."""
    job_id = "c1qabcdefg1234567890"
    expected_status_obj = MockJobStatus.SUCCESS
    expected_status_str = str(expected_status_obj)

    # Re-initialize the client *within the test* so it uses the MockSDKClass
    client = DataSphereClient(project_id="test_project", folder_id="test_folder")

    # Configure the mock SDK *instance* that the client WILL create
    mock_sdk_instance = MockSDKClass.return_value
    mock_job_wrapper = MockJobWrapper(job_id, expected_status_obj)
    mock_sdk_instance.get_job.return_value = mock_job_wrapper

    # Test the actual method call and expected outcome
    status = client.get_job_status(job_id)
    assert status == expected_status_str

    # Ensure the SDK was initialized (the mock was called)
    MockSDKClass.assert_called_once_with(oauth_token=None, profile=None)
    # Ensure the mocked SDK method was called as expected
    mock_sdk_instance.get_job.assert_called_once_with(job_id)

@patch(f'{mock_sdk_module_path}.SDK')
def test_get_job_status_sdk_failure(MockSDKClass, ds_client_no_sdk_init):
    """Test handling of SDK errors when getting job status."""
    job_id = "c1qabcdefg1234567890"
    sdk_error_message = "Job not found in Datasphere"

    # Re-initialize the client *within the test* so it uses the MockSDKClass
    client = DataSphereClient(project_id="test_project", folder_id="test_folder")

    # Configure the mock SDK instance to raise the *specific* error we want
    mock_sdk_instance = MockSDKClass.return_value
    # Simulate the SDK's get_job raising an error
    mock_sdk_instance.get_job.side_effect = Exception(sdk_error_message)

    with pytest.raises(DataSphereClientError, match=f"SDK error getting status.*{sdk_error_message}"):
        client.get_job_status(job_id)

    MockSDKClass.assert_called_once_with(oauth_token=None, profile=None)
    mock_sdk_instance.get_job.assert_called_once_with(job_id)

# Use the non-initializing fixture here too
def test_get_job_status_sdk_unavailable(ds_client_no_sdk_init):
    """Test error handling if SDK was not imported/initialized."""
    job_id = "c1qabcdefg1234567890"

    # The fixture ensures _sdk is None initially
    client = ds_client_no_sdk_init
    assert client._sdk is None # Verify precondition

    with pytest.raises(DataSphereClientError, match="DataSphere SDK is not available"):
        client.get_job_status(job_id)

# Removed old test test_submit_job_cli_success as it's replaced by test_submit_job_cli_success_id_parsing

# Add more tests for failure cases, different param formats etc. later 

# --- Download Tests --- #

@patch(f'{mock_sdk_module_path}.SDK')
def test_download_job_results_sdk_success(MockSDKClass, ds_client_no_sdk_init):
    """Test successfully downloading job results using the SDK."""
    job_id = "c1qabcdefg1234567890"
    output_dir = "/fake/output/dir"

    # Re-initialize client
    client = DataSphereClient(project_id="test_project", folder_id="test_folder")

    # Configure mock SDK instance
    mock_sdk_instance = MockSDKClass.return_value

    # Call the method - This will fail until implemented
    client.download_job_results(job_id, output_dir)

    # Assertions (verify SDK init and download call)
    MockSDKClass.assert_called_once_with(oauth_token=None, profile=None)
    mock_sdk_instance.download_job_files.assert_called_once_with(
        id=job_id,
        with_logs=False, # Assuming defaults
        with_diagnostics=False, # Assuming defaults
        output_dir=output_dir
    )

    # Remove the check for NotImplementedError
    # with pytest.raises(NotImplementedError):
    #     client.download_job_results(job_id, output_dir)

@patch(f'{mock_sdk_module_path}.SDK')
def test_download_job_results_sdk_failure(MockSDKClass, ds_client_no_sdk_init):
    """Test handling SDK errors during result download."""
    job_id = "c1qabcdefg1234567890"
    output_dir = "/fake/output/dir"
    sdk_error_message = "Download failed: Access denied"

    # Re-initialize client
    client = DataSphereClient(project_id="test_project", folder_id="test_folder")

    # Configure mock SDK to raise error
    mock_sdk_instance = MockSDKClass.return_value
    mock_sdk_instance.download_job_files.side_effect = Exception(sdk_error_message)

    # Call the method and assert the expected error
    with pytest.raises(DataSphereClientError, match=f"SDK error downloading results.*{sdk_error_message}"):
        client.download_job_results(job_id, output_dir)

    # Assertions (verify SDK init and download call attempt)
    MockSDKClass.assert_called_once_with(oauth_token=None, profile=None)
    mock_sdk_instance.download_job_files.assert_called_once_with(
        id=job_id, with_logs=False, with_diagnostics=False, output_dir=output_dir
    )

def test_download_job_results_sdk_unavailable(ds_client_no_sdk_init):
    """Test downloading results when SDK is unavailable."""
    job_id = "c1qabcdefg1234567890"
    output_dir = "/fake/output/dir"
    client = ds_client_no_sdk_init
    assert client._sdk is None

    with pytest.raises(DataSphereClientError, match="DataSphere SDK is not available"):
        client.download_job_results(job_id, output_dir)

# --- Cancellation Tests --- #

@patch(f'{mock_sdk_module_path}.SDK')
def test_cancel_job_sdk_success(MockSDKClass, ds_client_no_sdk_init):
    """Test successfully canceling a job using the SDK's underlying client."""
    job_id = "c1qabcdefg1234567890"

    # Re-initialize client to use the mock SDK
    client = DataSphereClient(project_id="test_project", folder_id="test_folder")

    # Configure the mock SDK instance and its *internal* client's cancel method
    mock_sdk_instance = MockSDKClass.return_value
    mock_internal_client = MagicMock()
    mock_sdk_instance.client = mock_internal_client

    # Call the method - This will fail until implemented
    client.cancel_job(job_id)

    # Assertions
    MockSDKClass.assert_called_once_with(oauth_token=None, profile=None)
    # Assert cancel was called on the internal client mock with keyword args
    mock_internal_client.cancel.assert_called_once_with(job_id=job_id, graceful=False)

    # Remove the check for NotImplementedError
    # with pytest.raises(NotImplementedError):
    #     client.cancel_job(job_id)

# Add test for cancel failure (e.g., SDK error)
@patch(f'{mock_sdk_module_path}.SDK')
def test_cancel_job_sdk_failure(MockSDKClass, ds_client_no_sdk_init):
    """Test handling SDK errors during job cancellation."""
    job_id = "c1qabcdefg1234567890"
    sdk_error_message = "Cannot cancel job in current state"

    # Re-initialize client
    client = DataSphereClient(project_id="test_project", folder_id="test_folder")

    # Configure mock SDK and internal client to raise error on cancel
    mock_sdk_instance = MockSDKClass.return_value
    mock_internal_client = MagicMock()
    mock_sdk_instance.client = mock_internal_client
    mock_internal_client.cancel.side_effect = Exception(sdk_error_message)

    with pytest.raises(DataSphereClientError, match=f"SDK error canceling Job ID.*{sdk_error_message}"):
        client.cancel_job(job_id)

    MockSDKClass.assert_called_once_with(oauth_token=None, profile=None)
    # Assert cancel was called on the internal client mock with keyword args
    mock_internal_client.cancel.assert_called_once_with(job_id=job_id, graceful=False)

def test_cancel_job_sdk_unavailable(ds_client_no_sdk_init):
    """Test canceling job when SDK is unavailable."""
    job_id = "c1qabcdefg1234567890"
    client = ds_client_no_sdk_init
    assert client._sdk is None

    with pytest.raises(DataSphereClientError, match="DataSphere SDK is not available"):
        client.cancel_job(job_id)

# Add test for cancel failure (e.g., SDK error) 