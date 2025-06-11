import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import logging
import json
from pathlib import Path
import os
import uuid

from deployment.app.services.datasphere_service import _process_job_results, DATASPHERE_PREDICTIONS_FILE, DATASPHERE_MODEL_FILE
from deployment.app.models.api_models import TrainingConfig, ModelConfig, OptimizerConfig, LRSchedulerConfig, TrainingDatasetConfig, JobType, JobStatus
from deployment.app.db.database import create_job, create_or_get_config, update_job_status

def create_test_training_config(model_id_base: str = "test-model-base") -> TrainingConfig:
    """Create a minimal TrainingConfig object for resilience testing."""
    model_config = ModelConfig(
        num_encoder_layers=2, num_decoder_layers=2, decoder_output_dim=8,
        temporal_width_past=12, temporal_width_future=6, temporal_hidden_size_past=64,
        temporal_hidden_size_future=64, temporal_decoder_hidden=128, batch_size=32,
        dropout=0.2, use_reversible_instance_norm=True, use_layer_norm=True
    )
    optimizer_config = OptimizerConfig(lr=0.001, weight_decay=1e-4)
    lr_shed_config = LRSchedulerConfig(T_0=10, T_mult=2)
    train_ds_config = TrainingDatasetConfig(alpha=0.05, span=100)
    
    return TrainingConfig(
        nn_model_config=model_config, optimizer_config=optimizer_config,
        lr_shed_config=lr_shed_config, train_ds_config=train_ds_config,
        lags=12, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
        model_id=model_id_base
    )

@pytest.mark.asyncio
async def test_partial_results_predictions_only_no_metrics(in_memory_db, fs, caplog):
    """
    Tests handling of a completed DataSphere job where metrics.json is missing.
    Expected: Model and predictions are saved, but training result record is not.
    """
    caplog.set_level(logging.INFO)
    db_conn = in_memory_db['conn']
    ds_job_id = "ds-job-no-metrics"
    test_config = create_test_training_config()

    # --- DB Setup: Create parent records to satisfy FOREIGN KEY constraints ---
    job_id = create_job(job_type=JobType.TRAINING, connection=db_conn)
    config_id = create_or_get_config(config_dict=test_config.model_dump(), connection=db_conn)

    update_job_status(job_id, "running", connection=db_conn)
    db_conn.commit()

    # --- File System Setup ---
    results_download_dir = Path("/tmp/ds_output/results")
    fs.create_dir(results_download_dir)

    model_path = results_download_dir / DATASPHERE_MODEL_FILE
    fs.create_file(model_path, contents="dummy model data")

    predictions_path = results_download_dir / DATASPHERE_PREDICTIONS_FILE
    predictions_header = "barcode,artist,album,cover_type,price_category,release_type,recording_decade,release_decade,style,record_year,0.05,0.25,0.5,0.75,0.95"
    predictions_row = "1,Artist,Album,Gatefold,A,LP,2010s,2020s,Rock,2023,1.0,2.0,3.0,4.0,5.0"

    fs.create_file(predictions_path, contents=f"{predictions_header}\n{predictions_row}")

    # --- Mock save_predictions_to_db and save_model_file_and_db to avoid FK constraint issues ---

    model_id = f"test-model-base_{uuid.uuid4().hex[:8]}"
    mock_save_model = AsyncMock(return_value=model_id)
    mock_save_predictions = MagicMock(return_value={"result_id": "test-pred-result", "predictions_count": 1})
    
    # Mock functions to avoid database errors
    mock_get_job = MagicMock(return_value={"job_id": job_id, "status": "running"})
    mock_update_job_status = MagicMock()

    # --- Call Function ---
    with patch('deployment.app.services.datasphere_service.get_db_connection', return_value=db_conn), \
         patch('deployment.app.services.datasphere_service.save_predictions_to_db', mock_save_predictions), \
         patch('deployment.app.services.datasphere_service.save_model_file_and_db', mock_save_model), \
         patch('deployment.app.services.datasphere_service.get_job', mock_get_job), \
         patch('deployment.app.services.datasphere_service.update_job_status', mock_update_job_status):
        await _process_job_results(
            job_id=job_id, ds_job_id=ds_job_id,
            results_dir=str(results_download_dir),
            config=test_config,
            metrics_data=None,
            model_path=str(model_path),
            predictions_path=str(predictions_path),
            polls=1, poll_interval=0.01, config_id=config_id
        )

    # --- Assertions ---
    assert "No metrics data available. Training result record not created." in caplog.text
    mock_save_model.assert_called_once()
    
    # Verify save_predictions_to_db was called with the same model_id returned by mock_save_model
    call_args = mock_save_predictions.call_args
    assert call_args is not None, "save_predictions_to_db was not called"
    assert call_args.kwargs['job_id'] == job_id
    assert call_args.kwargs['predictions_path'] == str(predictions_path)
    assert call_args.kwargs['model_id'] == model_id
    
    # Verify the job status was updated to completed
    mock_update_job_status.assert_called_once()
    update_args = mock_update_job_status.call_args
    assert update_args is not None
    assert update_args.args[0] == job_id
    assert update_args.args[1] == JobStatus.COMPLETED.value
    assert update_args.kwargs["progress"] == 100
    
    training_record = db_conn.execute("SELECT * FROM training_results WHERE job_id = ?", (job_id,)).fetchone()
    assert training_record is None

@pytest.mark.asyncio
async def test_partial_results_no_model(in_memory_db, fs, caplog):
    """
    Tests handling of a completed DataSphere job where model.onnx is missing.
    Expected: Training result is created if metrics exist, but model/preds are not.
    """
    caplog.set_level(logging.INFO)
    db_conn = in_memory_db['conn']
    ds_job_id = "ds-job-no-model"
    test_config = create_test_training_config()

    # --- DB Setup ---
    job_id = create_job(job_type=JobType.TRAINING, connection=db_conn)
    config_id = create_or_get_config(config_dict=test_config.model_dump(), connection=db_conn)
    update_job_status(job_id, "running", connection=db_conn)
    db_conn.commit()

    # --- File System Setup ---
    results_download_dir = Path("/tmp/ds_output/results")
    fs.create_dir(results_download_dir)

    metrics_path = results_download_dir / 'metrics.json'
    metrics_data = {"val_loss": 0.15, "train_loss": 0.2}
    fs.create_file(metrics_path, contents=json.dumps(metrics_data))

    predictions_path = results_download_dir / DATASPHERE_PREDICTIONS_FILE
    predictions_header = "barcode,artist,album,cover_type,price_category,release_type,recording_decade,release_decade,style,record_year,0.05,0.25,0.5,0.75,0.95"
    predictions_row = "1,Artist,Album,Gatefold,A,LP,2010s,2020s,Rock,2023,1.0,2.0,3.0,4.0,5.0"
    fs.create_file(predictions_path, contents=f"{predictions_header}\n{predictions_row}")

    # --- Mock create_training_result ---
    mock_create_training_result = MagicMock(return_value="test-training-result")
    
    # Mock functions to avoid database errors
    mock_get_job = MagicMock(return_value={"job_id": job_id, "status": "running"})
    mock_update_job_status = MagicMock()
    
    # --- Call Function ---
    with patch('deployment.app.services.datasphere_service.get_db_connection', return_value=db_conn), \
         patch('deployment.app.services.datasphere_service.create_training_result', mock_create_training_result), \
         patch('deployment.app.services.datasphere_service.get_job', mock_get_job), \
         patch('deployment.app.services.datasphere_service.update_job_status', mock_update_job_status):
        await _process_job_results(
            job_id=job_id, ds_job_id=ds_job_id,
            results_dir=str(results_download_dir),
            config=test_config,
            metrics_data=metrics_data,
            model_path=None,
            predictions_path=str(predictions_path),
            polls=1, poll_interval=0.01, config_id=config_id
        )

    # --- Assertions ---
    assert "Model file not found in results." in caplog.text
    mock_create_training_result.assert_called_once_with(
        job_id=job_id,
        config_id=config_id,
        metrics=metrics_data,
        model_id=None,
        duration=0,  # polls * poll_interval = 1 * 0.01 = 0.01, which gets converted to int 0
        config=test_config.model_dump()
    )
    
    # Verify the job status was updated to completed with the training result ID
    mock_update_job_status.assert_called_once()
    update_args = mock_update_job_status.call_args
    assert update_args is not None
    assert update_args.args[0] == job_id
    assert update_args.args[1] == JobStatus.COMPLETED.value
    assert update_args.kwargs["progress"] == 100
    assert update_args.kwargs["result_id"] == "test-training-result"
    
    model_record = db_conn.execute("SELECT * FROM models WHERE job_id = ?", (job_id,)).fetchone()
    assert model_record is None

@pytest.mark.asyncio
async def test_partial_results_no_predictions(in_memory_db, fs, caplog):
    """
    Tests handling of a completed DataSphere job where predictions.csv is missing.
    Expected: Model is saved, training result is created, but predictions are not.
    """
    caplog.set_level(logging.INFO)
    db_conn = in_memory_db['conn']
    ds_job_id = "ds-job-no-predictions"
    test_config = create_test_training_config()

    # --- DB Setup ---
    job_id = create_job(job_type=JobType.TRAINING, connection=db_conn)
    config_id = create_or_get_config(config_dict=test_config.model_dump(), connection=db_conn)
    update_job_status(job_id, "running", connection=db_conn)
    db_conn.commit()
    
    # --- File System Setup ---
    results_download_dir = Path("/tmp/ds_output/results")
    fs.create_dir(results_download_dir)
    
    model_path = results_download_dir / DATASPHERE_MODEL_FILE
    fs.create_file(model_path, contents="dummy model data")
    
    metrics_path = results_download_dir / 'metrics.json'
    metrics_data = {"val_loss": 0.12, "train_loss": 0.18}
    fs.create_file(metrics_path, contents=json.dumps(metrics_data))

    # --- Mock save_model_file_and_db and create_training_result with correct async implementation ---
    model_id = f"test-model-base_{uuid.uuid4().hex[:8]}"
    mock_save_model = AsyncMock(return_value=model_id)
    mock_create_training_result = MagicMock(return_value="test-training-result")
    
    # Mock functions to avoid database errors
    mock_get_job = MagicMock(return_value={"job_id": job_id, "status": "running"})
    mock_update_job_status = MagicMock()
    
    # --- Call Function ---
    with patch('deployment.app.services.datasphere_service.get_db_connection', return_value=db_conn), \
         patch('deployment.app.services.datasphere_service.save_model_file_and_db', mock_save_model), \
         patch('deployment.app.services.datasphere_service.create_training_result', mock_create_training_result), \
         patch('deployment.app.services.datasphere_service.get_job', mock_get_job), \
         patch('deployment.app.services.datasphere_service.update_job_status', mock_update_job_status):
        await _process_job_results(
            job_id=job_id, ds_job_id=ds_job_id,
            results_dir=str(results_download_dir),
            config=test_config,
            metrics_data=metrics_data,
            model_path=str(model_path),
            predictions_path=None,
            polls=1, poll_interval=0.01, config_id=config_id
        )

    # --- Assertions ---
    assert "No predictions file found, cannot save predictions to DB." in caplog.text
    mock_save_model.assert_called_once_with(
        job_id=job_id,
        model_path=str(model_path),
        ds_job_id=ds_job_id,
        config=test_config,
        metrics_data=metrics_data
    )
    mock_create_training_result.assert_called_once_with(
        job_id=job_id,
        config_id=config_id,
        metrics=metrics_data,
        model_id=model_id,
        duration=0,  # polls * poll_interval = 1 * 0.01 = 0.01, which gets converted to int 0
        config=test_config.model_dump()
    )
    
    # Verify the job status was updated to completed with the training result ID
    mock_update_job_status.assert_called_once()
    update_args = mock_update_job_status.call_args
    assert update_args is not None
    assert update_args.args[0] == job_id
    assert update_args.args[1] == JobStatus.COMPLETED.value
    assert update_args.kwargs["progress"] == 100
    assert update_args.kwargs["result_id"] == "test-training-result"
    
    pred_record = db_conn.execute("SELECT * FROM prediction_results WHERE job_id = ?", (job_id,)).fetchone()
    assert pred_record is None

@pytest.mark.asyncio
async def test_datasphere_job_fails_with_logs(mock_service_env, caplog):
    """
    Tests that if a DataSphere job fails, the system correctly captures the
    failure status, logs the error, and updates the job status accordingly.
    """
    # Set up log capture
    caplog.set_level(logging.ERROR)
    
    # Extract mocks from the fixture
    mock_client = mock_service_env["client"]
    mock_update_job_status = mock_service_env["update_job_status"]
    fs = mock_service_env["fs"]
    
    # Configure DataSphere client mocks for a failing job
    ds_job_id = "ds-job-with-failure"
    job_id = "test-failure-job-id"
    
    # Configure client to return FAILED job status
    mock_client.submit_job.return_value = ds_job_id
    mock_client.get_job_status.return_value = "FAILED"
    
    # Create log files to simulate error logs being downloaded
    logs_dir = Path(mock_service_env["settings"].datasphere_output_dir) / f"ds_job_{job_id}_logs"
    fs.create_dir(logs_dir)
    fs.create_file(logs_dir / "job_error.log", contents="Critical error: Out of memory during training")
    
    # Configure download_job_results to simulate downloading error logs
    def mock_download_logs(*args, **kwargs):
        # Explicitly create the logs directory if it doesn't exist
        output_dir = kwargs.get("output_dir", "")
        fs.create_dir(output_dir, exist_ok=True)
        fs.create_file(Path(output_dir) / "job_error.log", contents="Critical error: Out of memory during training")
    
    mock_client.download_job_results.side_effect = mock_download_logs
    
    # Create test training config
    training_config = create_test_training_config()
    
    # Import the service module from mock_service_env
    ds_module = mock_service_env.get("service_module", None)
    if ds_module is None:
        # If service_module is not provided by the fixture, import it directly
        from deployment.app.services import datasphere_service as ds_module
    
    # Run the job with our mocked client and expect it to fail
    with pytest.raises(RuntimeError, match=f"DS Job {ds_job_id} ended with status: FAILED"):
        await ds_module.run_job(
            job_id=job_id,
            training_config=training_config.model_dump(),
            config_id="test-config-id"
        )
    
    # Verify that the error was logged
    assert f"DS Job {ds_job_id} ended with status: FAILED" in caplog.text
    
    # Verify that update_job_status was called with the FAILED status
    # Find any call that has the job_id and FAILED/failed status
    failure_call_found = False
    for call_args in mock_update_job_status.call_args_list:
        args, kwargs = call_args
        if len(args) >= 2 and args[0] == job_id and (
            args[1] == JobStatus.FAILED.value or 
            args[1] == "failed" or 
            "status" in kwargs and (kwargs["status"] == JobStatus.FAILED.value or kwargs["status"] == "failed")
        ):
            failure_call_found = True
            break
    
    assert failure_call_found, "update_job_status was not called with a FAILED status"
    
    # Verify that download_job_results was called for downloading logs
    mock_client.download_job_results.assert_called()

@pytest.mark.asyncio
async def test_datasphere_job_polling_timeout(mock_service_env, caplog):
    """
    Tests that the system correctly handles a job that exceeds the maximum number
    of polling attempts, updating its status to TIMED_OUT and logging appropriate errors.
    """
    # Set up log capture
    caplog.set_level(logging.ERROR)
    
    # Extract mocks from the fixture
    mock_client = mock_service_env["client"]
    mock_update_job_status = mock_service_env["update_job_status"]
    settings = mock_service_env["settings"]
    
    # Configure client for a job that never completes
    ds_job_id = "ds-job-timeout"
    job_id = "test-timeout-job-id"
    
    # Override poll settings to make the test run quickly
    # Store original values to restore later
    original_max_polls = settings.datasphere.max_polls
    original_poll_interval = settings.datasphere.poll_interval
    
    # Set to small values for the test
    settings.datasphere.max_polls = 3
    settings.datasphere.poll_interval = 0.01
    
    # Always return "RUNNING" status to trigger the timeout
    mock_client.submit_job.return_value = ds_job_id
    mock_client.get_job_status.return_value = "RUNNING"
    
    # Create test training config
    training_config = create_test_training_config()
    
    # Import the service module from mock_service_env
    ds_module = mock_service_env.get("service_module", None)
    if ds_module is None:
        # If service_module is not provided by the fixture, import it directly
        from deployment.app.services import datasphere_service as ds_module
    
    # Run the job with our mocked client and expect it to timeout
    with pytest.raises(TimeoutError, match=f"DS Job {ds_job_id} execution timed out"):
        await ds_module.run_job(
            job_id=job_id,
            training_config=training_config.model_dump(),
            config_id="test-config-id"
        )
    
    # Verify that the error was logged
    assert "DS Job polling exceeded maximum attempts" in caplog.text or "timed out after" in caplog.text
    
    # Verify that get_job_status was called the expected number of times
    # Should be called max_polls times before timing out
    assert mock_client.get_job_status.call_count == settings.datasphere.max_polls
    
    # Verify that update_job_status was called with a TIMED_OUT or similar status
    timeout_call_found = False
    for call_args in mock_update_job_status.call_args_list:
        args, kwargs = call_args
        if len(args) >= 2 and args[0] == job_id and (
            "timeout" in str(args[1]).lower() or
            "timed_out" in str(args[1]).lower() or
            (len(args) > 2 and "timeout" in str(args[2]).lower()) or
            (len(args) > 2 and "timed_out" in str(args[2]).lower()) or
            ("error_message" in kwargs and "timeout" in str(kwargs["error_message"]).lower()) or
            ("error_message" in kwargs and "timed_out" in str(kwargs["error_message"]).lower()) or
            ("status_message" in kwargs and "timeout" in str(kwargs["status_message"]).lower()) or
            ("status_message" in kwargs and "timed_out" in str(kwargs["status_message"]).lower())
        ):
            timeout_call_found = True
            break
    
    assert timeout_call_found, "update_job_status was not called with a TIMED_OUT status or timeout error message"
    
    # Restore original settings to avoid affecting other tests
    settings.datasphere.max_polls = original_max_polls
    settings.datasphere.poll_interval = original_poll_interval 

@pytest.mark.asyncio
async def test_datasphere_client_connectivity_error(mock_service_env, caplog):
    """
    Tests that the system correctly handles connectivity errors with DataSphere,
    updating the job status and logging appropriate error messages.
    """
    # Set up log capture
    caplog.set_level(logging.ERROR)
    
    # Extract mocks from the fixture
    mock_client = mock_service_env["client"]
    mock_update_job_status = mock_service_env["update_job_status"]
    
    # Configure client for connectivity error
    job_id = "test-connectivity-error-job-id"
    
    # Configure submit_job to raise an exception simulating connectivity failure
    connectivity_error = ConnectionError("Failed to connect to DataSphere API: Network timeout")
    mock_client.submit_job.side_effect = connectivity_error
    
    # Create test training config
    training_config = create_test_training_config()
    
    # Import the service module from mock_service_env
    ds_module = mock_service_env.get("service_module", None)
    if ds_module is None:
        # If service_module is not provided by the fixture, import it directly
        from deployment.app.services import datasphere_service as ds_module
    
    # Run the job with our mocked client and expect it to fail with a connectivity error
    with pytest.raises(RuntimeError) as excinfo:
        await ds_module.run_job(
            job_id=job_id,
            training_config=training_config.model_dump(),
            config_id="test-config-id"
        )
    
    # Verify the error message contains connectivity information
    assert "Failed to connect" in str(excinfo.value) or "Network" in str(excinfo.value)
    
    # Verify the error was logged
    assert "Failed to connect" in caplog.text or "Network" in caplog.text
    
    # Verify that update_job_status was called with a FAILED status
    failure_call_found = False
    error_message_found = False
    
    for call_args in mock_update_job_status.call_args_list:
        args, kwargs = call_args
        # Check for FAILED status
        if len(args) >= 2 and args[0] == job_id and (
            args[1] == JobStatus.FAILED.value or 
            args[1] == "failed" or 
            "status" in kwargs and (kwargs["status"] == JobStatus.FAILED.value or kwargs["status"] == "failed")
        ):
            failure_call_found = True
        
        # Check for error message related to connectivity
        if "error_message" in kwargs and (
            "connect" in str(kwargs["error_message"]).lower() or
            "network" in str(kwargs["error_message"]).lower() or
            "timeout" in str(kwargs["error_message"]).lower()
        ):
            error_message_found = True
    
    assert failure_call_found, "update_job_status was not called with a FAILED status"
    assert error_message_found, "update_job_status was not called with a connectivity-related error message" 