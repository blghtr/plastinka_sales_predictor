import pytest
from unittest.mock import patch, MagicMock
import subprocess
import re # Import re
import json # Added for list command mocking

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

# Dummy project/folder IDs for testing
TEST_PROJECT_ID = "c1qtestprojectid12345"
TEST_FOLDER_ID = "b1gtestfolderid123456"
TEST_JOB_ID = "c1qjobid123abcdefghij"
NEW_TEST_JOB_ID = "c1qnewjobid123456789"
EXISTING_JOB_IDS = {"c1qoldjob1", "c1qoldjob2"}

@pytest.fixture
def mock_sdk():
    """Fixture to mock the datasphere SDK."""
    sdk_mock = MagicMock()
    # Mock nested attributes if needed for other tests
    sdk_mock.get_job.return_value = MagicMock(job=MagicMock(status="RUNNING"))
    sdk_mock.download_job_files.return_value = None
    sdk_mock.client.cancel.return_value = None
    return sdk_mock

@pytest.fixture
def client(mock_sdk):
    """Fixture to create a DataSphereClient instance with mocked SDK."""
    # Patch the SDK import within the client module's scope during test execution
    with patch('deployment.app.datasphere_integration.client.datasphere.sdk.SDK', return_value=mock_sdk):
         # Initialize client - SDK mock will be used
        client_instance = DataSphereClient(project_id=TEST_PROJECT_ID, folder_id=TEST_FOLDER_ID)
        # Manually assign the mocked SDK instance if needed by tests directly
        client_instance._sdk = mock_sdk
        return client_instance

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
    # 1. Explicit "Job ID: ..." format - This case is no longer relevant for list command
    # stdout_text_1 = f"Processing...\nJob ID: {expected_job_id}\nComplete."
    # stdout_format_1 = stdout_text_1.encode('utf-8')
    
    # Valid JSON output for the list command (before execute)
    stdout_list_before = json.dumps([{"id": job_id} for job_id in EXISTING_JOB_IDS])
    stdout_format_list_before = stdout_list_before.encode('utf-8')
    
    # Valid JSON output for the list command (after execute)
    stdout_list_after = json.dumps([{"id": job_id} for job_id in (EXISTING_JOB_IDS | {expected_job_id})])
    stdout_format_list_after = stdout_list_after.encode('utf-8')

    # Output for the execute command (return code 0, stdout content doesn't matter)
    stdout_execute = b"Job submitted."
    
    # Configure mock_subprocess_run side effect
    def mock_run_side_effect(*args, **kwargs):
        cmd_list = args[0]
        if "list" in cmd_list:
            # Return list before or after based on call count
            if mock_subprocess_run.call_count == 1:
                mock_result = MagicMock(returncode=0, stdout=stdout_format_list_before, stderr=b"")
            else:
                mock_result = MagicMock(returncode=0, stdout=stdout_format_list_after, stderr=b"")
        elif "execute" in cmd_list:
            mock_result = MagicMock(returncode=0, stdout=stdout_execute, stderr=b"")
        else:
            mock_result = MagicMock(returncode=1, stdout=b"", stderr=b"Unknown command")
        return mock_result
        
    mock_subprocess_run.side_effect = mock_run_side_effect

    # Patch time.sleep
    with patch('deployment.app.datasphere_integration.client.time.sleep'):
        job_id_1 = ds_client.submit_job_cli(config_path)
        
    assert job_id_1 == expected_job_id
    assert mock_subprocess_run.call_count == 3 # list, execute, list

    # --- Old cases removed or integrated --- 
    # Test Case 2: ID as last line - Integrated into the new logic
    
    # Test Case 3: No ID found (should raise error) - Covered by a separate test

# Renamed old test to reflect its purpose (now covered by test_submit_job_cli_id_determination_failure)
# @patch('deployment.app.datasphere_integration.client.subprocess.run')
# def test_submit_job_cli_old_parsing_failure(mock_subprocess_run, ds_client):
#    ...

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

    # Expect error from the initial list call failing
    expected_error_match = f"Could not list jobs before submission: Failed to list DataSphere jobs. Exit code 1. Error: {error_message}"
    with pytest.raises(DataSphereClientError, match=re.escape(expected_error_match)):
        ds_client.submit_job_cli(config_path)
        
    # Check that only the list command was attempted
    mock_subprocess_run.assert_called_once()
    # Verify it was the list command
    called_cmd = mock_subprocess_run.call_args[0][0]
    assert 'list' in called_cmd
    assert 'execute' not in called_cmd

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

# --- Tests for submit_job_cli (Refactored) ---

# Mock _run_cli_command directly as it's now the central CLI interaction point
@patch('deployment.app.datasphere_integration.client.DataSphereClient._run_cli_command')
def test_submit_job_cli_success_list_diff(mock_run_cli, client):
    """Test successful job submission using list comparison."""
    config_file = "path/to/config.yaml"
    
    # Define expected command calls and their outputs
    list_cmd_args = ['datasphere', 'project', 'job', 'list', '-p', TEST_PROJECT_ID, '--format', 'json']
    execute_cmd_args = ['datasphere', 'project', 'job', 'execute', '-p', TEST_PROJECT_ID, '-c', config_file]
    
    # Output for the first list call (before execute)
    list_output_before = json.dumps([{"id": job_id} for job_id in EXISTING_JOB_IDS])
    
    # Output for the execute call (stdout doesn't matter much now, but return code must be 0)
    execute_output_stdout = "Job submitted."
    
    # Output for the second list call (after execute)
    list_output_after = json.dumps([{"id": job_id} for job_id in EXISTING_JOB_IDS | {NEW_TEST_JOB_ID}])

    # Configure the mock side effect based on the command called
    def run_cli_side_effect(cmd):
        cmd_str = ' '.join(cmd)
        if 'list' in cmd_str:
            # Determine if it's the first or second list call
            if mock_run_cli.call_count == 1:
                return (0, list_output_before, "")  # Success, stdout, stderr
            else:
                return (0, list_output_after, "")
        elif 'execute' in cmd_str:
            return (0, execute_output_stdout, "")
        else:
            return (1, "", "Unknown command") # Should not happen
            
    mock_run_cli.side_effect = run_cli_side_effect
    
    # Patch time.sleep to avoid actual delay during test
    with patch('deployment.app.datasphere_integration.client.time.sleep'):
        job_id = client.submit_job_cli(config_path=config_file)

    # Assertions
    assert job_id == NEW_TEST_JOB_ID
    assert mock_run_cli.call_count == 3 # list, execute, list
    # Check the calls were made with correct arguments
    mock_run_cli.assert_any_call(list_cmd_args)
    mock_run_cli.assert_any_call(execute_cmd_args)
    # The third call is also list_cmd_args

@patch('deployment.app.datasphere_integration.client.DataSphereClient._run_cli_command')
def test_submit_job_cli_id_determination_failure(mock_run_cli, client):
    """Test failure cases for determining Job ID via list comparison."""
    config_file = "path/to/config.yaml"
    
    list_output = json.dumps([{"id": job_id} for job_id in EXISTING_JOB_IDS])
    execute_output_stdout = "Job submitted."

    # Case 1: No new job ID found
    def run_cli_no_new_id(cmd):
        if 'list' in ' '.join(cmd): return (0, list_output, "")
        if 'execute' in ' '.join(cmd): return (0, execute_output_stdout, "")
        return (1, "", "Unknown")
    mock_run_cli.side_effect = run_cli_no_new_id
    mock_run_cli.reset_mock() # Reset call count for this case
    
    with patch('deployment.app.datasphere_integration.client.time.sleep'):
        with pytest.raises(DataSphereClientError, match="No new job found"):
             client.submit_job_cli(config_path=config_file)
    assert mock_run_cli.call_count == 3

    # Case 2: Multiple new job IDs found
    list_output_multiple_new = json.dumps([{"id": job_id} for job_id in EXISTING_JOB_IDS | {"new_id_1", "new_id_2"}])
    def run_cli_multiple_new_ids(cmd):
        cmd_str = ' '.join(cmd)
        if 'list' in cmd_str:
             if mock_run_cli.call_count == 1: return (0, list_output, "")
             else: return (0, list_output_multiple_new, "")
        if 'execute' in cmd_str: return (0, execute_output_stdout, "")
        return (1, "", "Unknown")
    mock_run_cli.side_effect = run_cli_multiple_new_ids
    mock_run_cli.reset_mock()
    
    with patch('deployment.app.datasphere_integration.client.time.sleep'):
        with pytest.raises(DataSphereClientError, match="Found multiple new jobs"):
             client.submit_job_cli(config_path=config_file)
    assert mock_run_cli.call_count == 3

@patch('deployment.app.datasphere_integration.client.DataSphereClient._run_cli_command')
def test_submit_job_cli_execute_failure(mock_run_cli, client):
    """Test failure when the 'execute' command itself fails."""
    config_file = "path/to/bad_config.yaml"
    list_cmd_args = ['datasphere', 'project', 'job', 'list', '-p', TEST_PROJECT_ID, '--format', 'json']
    execute_cmd_args = ['datasphere', 'project', 'job', 'execute', '-p', TEST_PROJECT_ID, '-c', config_file]
    list_output = json.dumps([{"id": job_id} for job_id in EXISTING_JOB_IDS])
    execute_error_stderr = "Error: Invalid config file."

    def run_cli_execute_fails(cmd):
        cmd_str = ' '.join(cmd)
        if 'list' in cmd_str: return (0, list_output, "") # List before works
        if 'execute' in cmd_str: return (1, "", execute_error_stderr) # Execute fails
        return (1, "", "Unknown")
    mock_run_cli.side_effect = run_cli_execute_fails

    with pytest.raises(DataSphereClientError, match=f"job execution command failed.*{execute_error_stderr}"):
        client.submit_job_cli(config_path=config_file)
    
    assert mock_run_cli.call_count == 2 # list, execute (no second list)
    mock_run_cli.assert_any_call(list_cmd_args)
    mock_run_cli.assert_any_call(execute_cmd_args)

@patch('deployment.app.datasphere_integration.client.DataSphereClient._run_cli_command')
def test_submit_job_cli_list_failure(mock_run_cli, client):
    """Test failure when the 'list' command fails (before or after execute)."""
    config_file = "path/to/config.yaml"
    list_cmd_args = ['datasphere', 'project', 'job', 'list', '-p', TEST_PROJECT_ID, '--format', 'json']
    execute_cmd_args = ['datasphere', 'project', 'job', 'execute', '-p', TEST_PROJECT_ID, '-c', config_file]
    list_error_stderr = "Error: Cannot connect to API."
    execute_output_stdout = "Job submitted."

    # Case 1: First list fails
    def run_cli_first_list_fails(cmd):
        if 'list' in ' '.join(cmd): return (1, "", list_error_stderr)
        return (1, "", "Unknown") # Should not be called
    mock_run_cli.side_effect = run_cli_first_list_fails
    mock_run_cli.reset_mock()
    
    with pytest.raises(DataSphereClientError, match=f"Could not list jobs before submission.*{list_error_stderr}"):
         client.submit_job_cli(config_path=config_file)
    assert mock_run_cli.call_count == 1
    mock_run_cli.assert_called_with(list_cmd_args)
    
    # Case 2: Second list fails
    list_output_before = json.dumps([{"id": job_id} for job_id in EXISTING_JOB_IDS])
    def run_cli_second_list_fails(cmd):
        cmd_str = ' '.join(cmd)
        if 'list' in cmd_str:
             if mock_run_cli.call_count == 1: return (0, list_output_before, "")
             else: return (1, "", list_error_stderr) # Second list fails
        if 'execute' in cmd_str: return (0, execute_output_stdout, "")
        return (1, "", "Unknown")
    mock_run_cli.side_effect = run_cli_second_list_fails
    mock_run_cli.reset_mock()
    
    with patch('deployment.app.datasphere_integration.client.time.sleep'):
        with pytest.raises(DataSphereClientError, match=f"Could not list jobs after submission.*{list_error_stderr}"):
             client.submit_job_cli(config_path=config_file)
    assert mock_run_cli.call_count == 3 # list, execute, list (failed)
    mock_run_cli.assert_any_call(list_cmd_args)
    mock_run_cli.assert_any_call(execute_cmd_args)

# Keep other tests (get_job_status, download_job_results, cancel_job)
# test_submit_job_cli_params_arg_removed is still relevant for signature check
# --- Tests for submit_job_cli --- # <--- Remove duplicate section marker

def test_submit_job_cli_params_arg_removed(client):
    """
    Test that submit_job_cli raises TypeError if 'params' is passed
    (once the argument is removed). This test should fail initially.
    """
    with pytest.raises(TypeError):
        # This call simulates passing the 'params' argument after it has been removed
        # from the function definition, which should raise a TypeError.
        client.submit_job_cli(config_path="dummy/config.yaml", params={"unused": "value"})

# Remove old submit tests that parsed stdout
# @patch('subprocess.run')
# def test_submit_job_cli_success(mock_run, client):
#    ...

# @patch('subprocess.run')
# def test_submit_job_cli_parsing_failure(mock_run, client):
#    ...

# @patch('subprocess.run')
# def test_submit_job_cli_command_failure(mock_run, client):
#    ...

# --- Tests for get_job_status ---

def test_get_job_status_success(client, mock_sdk):
    """Test getting job status successfully via SDK."""
    # Configure the mock SDK behavior for this test
    expected_status = "COMPLETED"
    mock_job_wrapper = MagicMock()
    mock_job_wrapper.job.status = expected_status # Simulate the nested structure
    mock_sdk.get_job.return_value = mock_job_wrapper

    status = client.get_job_status(job_id=TEST_JOB_ID)

    assert status == expected_status
    mock_sdk.get_job.assert_called_once_with(TEST_JOB_ID)

def test_get_job_status_sdk_error(client, mock_sdk):
    """Test error handling when SDK fails to get job status."""
    # Configure the mock SDK to raise an exception
    sdk_error_message = "Job not found"
    mock_sdk.get_job.side_effect = Exception(sdk_error_message)

    with pytest.raises(DataSphereClientError, match=f"SDK error getting status.*{sdk_error_message}"):
        client.get_job_status(job_id="nonexistent_job")

    mock_sdk.get_job.assert_called_once_with("nonexistent_job")

# --- Tests for download_job_results ---

def test_download_job_results_success(client, mock_sdk):
    """Test successful job results download via SDK."""
    output_dir = "/tmp/results"
    client.download_job_results(job_id=TEST_JOB_ID, output_dir=output_dir, with_logs=True)

    mock_sdk.download_job_files.assert_called_once_with(
        id=TEST_JOB_ID,
        with_logs=True,
        with_diagnostics=False,
        output_dir=output_dir
    )

def test_download_job_results_sdk_error(client, mock_sdk):
    """Test error handling when SDK fails to download results."""
    sdk_error_message = "Download failed"
    mock_sdk.download_job_files.side_effect = Exception(sdk_error_message)
    output_dir = "/tmp/results_fail"

    with pytest.raises(DataSphereClientError, match=f"SDK error downloading results.*{sdk_error_message}"):
        client.download_job_results(job_id=TEST_JOB_ID, output_dir=output_dir)

    mock_sdk.download_job_files.assert_called_once_with(
        id=TEST_JOB_ID,
        with_logs=False,
        with_diagnostics=False,
        output_dir=output_dir
    )

# --- Tests for cancel_job ---

def test_cancel_job_success(client, mock_sdk):
    """Test successful job cancellation via SDK."""
    client.cancel_job(job_id=TEST_JOB_ID, graceful=True)

    # Check that the internal SDK client's cancel method was called
    mock_sdk.client.cancel.assert_called_once_with(job_id=TEST_JOB_ID, graceful=True)

def test_cancel_job_sdk_error(client, mock_sdk):
    """Test error handling when SDK fails to cancel job."""
    sdk_error_message = "Cancellation forbidden"
    mock_sdk.client.cancel.side_effect = Exception(sdk_error_message)

    with pytest.raises(DataSphereClientError, match=f"SDK error canceling Job ID.*{sdk_error_message}"):
        client.cancel_job(job_id=TEST_JOB_ID)

    mock_sdk.client.cancel.assert_called_once_with(job_id=TEST_JOB_ID, graceful=False)

# Add more tests below
# ... existing code ... 