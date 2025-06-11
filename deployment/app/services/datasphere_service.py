from datetime import datetime
import time
import os
import json
import uuid
import asyncio
import logging
import pandas as pd
import shutil
from pathlib import Path
import yaml
import subprocess
import sys # Added to use sys.executable
import tempfile
from typing import Optional, Tuple, Dict, Any, List
from pydantic import TypeAdapter
from deployment.app.db.database import (
    update_job_status, 
    create_training_result,
    create_or_get_config,
    create_model_record,
    get_recent_models,
    delete_model_record_and_file,
    get_active_config,
    get_best_config_by_metric,
    set_config_active,
    get_active_model,
    get_best_model_by_metric,
    set_model_active,
    delete_models_by_ids,
    get_all_models,
    get_job,
    create_prediction_result,
    get_or_create_multiindex_id,
    execute_many,
    get_db_connection,
    get_effective_config
)
from deployment.app.models.api_models import JobStatus, TrainingConfig
from deployment.app.config import AppSettings
from deployment.datasphere.client import DataSphereClient
from deployment.datasphere.prepare_datasets import get_datasets

from deployment.app.utils.file_utilities import _get_directory_hash
import zipfile

# Initialize settings
settings = AppSettings()

# Constants from settings
DATASPHERE_JOB_DIRECTORY = settings.datasphere_output_dir
DATASPHERE_MODEL_FILE = "model.onnx"
DATASPHERE_PREDICTIONS_FILE = "predictions.csv"
DATASPHERE_PREDICTIONS_FORMAT = "csv"
JOB_POLL_MAX = settings.datasphere.max_polls
JOB_POLL_INTERVAL = settings.datasphere.poll_interval
MAX_RETRIES = 3
RETRY_DELAY = 2
DATASPHERE_IMAGE_VARIABLES = ["PYTHONPATH", "PYTHONUNBUFFERED"]

# Timeouts for DataSphere Client operations
CLIENT_INIT_TIMEOUT_SECONDS = settings.datasphere.client_init_timeout_seconds # Default 1 minute
CLIENT_SUBMIT_TIMEOUT_SECONDS = settings.datasphere.client_submit_timeout_seconds # Default 2 minutes
CLIENT_STATUS_TIMEOUT_SECONDS = settings.datasphere.client_status_timeout_seconds # Default 30 seconds
CLIENT_DOWNLOAD_TIMEOUT_SECONDS = settings.datasphere.client_download_timeout_seconds # Default 5 minutes
CLIENT_CANCEL_TIMEOUT_SECONDS = settings.datasphere.client_cancel_timeout_seconds # Default 1 minute

# Set up logger
logger = logging.getLogger(__name__)

# Helper Functions for run_job stages




async def _prepare_job_datasets(job_id: str, config: TrainingConfig, start_date_for_dataset: str = None, end_date_for_dataset: str = None, output_dir: str = None) -> None:
    """Prepares datasets required for the job."""
    logger.info(f"[{job_id}] Stage 2: Preparing datasets...")

    logger.info(f"[{job_id}] Calling get_datasets with start_date: {start_date_for_dataset}, end_date: {end_date_for_dataset}, outputting to: {output_dir}")
    try:
        get_datasets(
            start_date=start_date_for_dataset,
            end_date=end_date_for_dataset,
            config=config,
            output_dir=output_dir
        )
        logger.info(f"[{job_id}] Datasets prepared in {output_dir}.")
        update_job_status(job_id, JobStatus.RUNNING.value, progress=10, status_message="Datasets prepared.")
    except Exception as e:
        logger.error(f"[{job_id}] Failed to prepare datasets: {e}", exc_info=True)
        update_job_status(job_id, JobStatus.FAILED.value, error_message=f"Failed to prepare datasets: {e}")
        raise RuntimeError(f"Failed to prepare datasets during job {job_id}. Original error: {e}") from e


async def _initialize_datasphere_client(job_id: str) -> DataSphereClient:
    """Initializes and returns the DataSphere client."""
    logger.info(f"[{job_id}] Stage 3: Initializing DataSphere client...")
    if not settings.datasphere or not settings.datasphere.client:
        update_job_status(job_id, JobStatus.FAILED.value, error_message="DataSphere client configuration is missing.")
        raise ValueError("DataSphere client configuration is missing in settings.")
    
    client_config = settings.datasphere.client # Get config dict

    try:
        logger.debug(f"[{job_id}] Attempting to initialize DataSphereClient with timeout {CLIENT_INIT_TIMEOUT_SECONDS}s.")
        # DataSphereClient constructor is synchronous, run in a thread
        client = await asyncio.wait_for(
            asyncio.to_thread(DataSphereClient, **client_config),
            timeout=CLIENT_INIT_TIMEOUT_SECONDS
        )
        logger.info(f"[{job_id}] DataSphere client initialized successfully.")
        update_job_status(job_id, JobStatus.RUNNING.value, progress=15, status_message="DataSphere client initialized.")
        return client
    except asyncio.TimeoutError:
        error_msg = f"DataSphere client initialization timed out after {CLIENT_INIT_TIMEOUT_SECONDS} seconds."
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg) from asyncio.TimeoutError # Keep original exception type for context
    except ImportError as e: # This might still be relevant if datasphere client has conditional imports
        logger.error(f"Error importing DataSphereClient or its dependencies: {e}. Ensure all necessary packages are installed.")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=f"Client Initialization Failed due to import error: {e}")
        raise RuntimeError(f"Failed to initialize DataSphere client due to import error: {e}") from e
    except Exception as e:
        error_msg = f"Failed to initialize DataSphere client: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        # Keep original exception type if it's a custom DataSphereClientError or similar
        if "DataSphereClientError" in str(type(e)): # Basic check
             raise
        raise RuntimeError(error_msg) from e



async def _verify_datasphere_job_inputs(job_id: str, input_dir_path: Path) -> None:
    """
    Verifies the integrity of all necessary DataSphere job input files before archiving and submission.
    Ensures the input directory exists, is a directory, and contains the config and dataset files.
    
    Args:
        job_id: The ID of the current job.
        input_dir_path: The path to the input directory to be verified.
        
    Raises:
        FileNotFoundError: If a critical file (config.json or dataset) is missing.
        RuntimeError: If the input directory is invalid or other general integrity issues.
    """
    logger.info(f"[{job_id}] Stage 4a.1: Verifying DataSphere job inputs in {input_dir_path}...")
    
    # 1. Verify Input Directory Existence and Type
    if not input_dir_path.exists():
        error_msg = f"Input directory '{input_dir_path}' does not exist for job {job_id}."
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg)
    
    if not input_dir_path.is_dir():
        error_msg = f"Input path '{input_dir_path}' is not a directory for job {job_id}."
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg)

    # 2. Validate config.json
    config_json_file = input_dir_path / "config.json"
    if not config_json_file.is_file():
        error_msg = f"Required config.json not found in input directory '{input_dir_path}' for job {job_id}."
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise FileNotFoundError(error_msg)
    
    # 3. Validate Dataset Files (e.g., train.dill, val.dill, full.dill) 
    # Based on prepare_datasets.py, it should produce 'train.dill' and 'val.dill'
    train_data_file = input_dir_path / "train.dill"
    val_data_file = input_dir_path / "val.dill"
    full_data_file = input_dir_path / "full.dill"
    if not train_data_file.is_file() or not val_data_file.is_file():
        missing_datasets = []
        if not train_data_file.is_file():
            missing_datasets.append("train.dill")
        if not val_data_file.is_file():
            missing_datasets.append("val.dill")
        if not full_data_file.is_file():
            missing_datasets.append("full.dill")
        error_msg = f"Missing required dataset files in input directory '{input_dir_path}' for job {job_id}: {', '.join(missing_datasets)}."
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"[{job_id}] DataSphere job inputs verified successfully.")
    update_job_status(job_id, JobStatus.RUNNING.value, progress=20, status_message="DataSphere job inputs verified.")


async def _prepare_datasphere_job_submission(job_id: str, config: TrainingConfig, target_input_dir: Path) -> Tuple[str, str]:
    """Prepares the parameters file and DataSphere config for job submission."""
    logger.info(f"[{job_id}] Stage 4a: Preparing DataSphere job submission inputs...")
    # Generate a unique ID component for this DS job run (used for output/results dir later)
    ds_job_run_suffix = f"ds_job_{job_id}_{uuid.uuid4().hex[:8]}"

    # Save training config as JSON
    config_json_path = str(target_input_dir / "config.json")
    logger.info(f"[{job_id}] Saving training config to {config_json_path}")
    with open(config_json_path, 'w') as f: # open() accepts string path
        # Use model_dump_json for Pydantic v2+
        json.dump(config.model_dump(), f, indent=2) # Save training config

    # Create ready DataSphere YAML config from template
    template_config_path = settings.datasphere_job_config_path
    ready_config_path = str(target_input_dir / "datasphere_config.yaml")
    
    logger.info(f"[{job_id}] Creating ready DataSphere config from template {template_config_path}")
    
    # Load template YAML config
    with open(template_config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # Add inputs section with input.zip in correct format (list of dict-like entries)
    config_data['inputs'] = ['input.zip: INPUT']
    
    # Save ready config
    with open(ready_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"[{job_id}] Ready DataSphere config saved to {ready_config_path}")

    # Return the suffix for identifying the run and the path to the ready config file
    return ds_job_run_suffix, ready_config_path


async def _archive_input_directory(job_id: str, input_dir: str) -> str:
    """Archives the contents of the input directory into input.zip."""
    logger.info(f"[{job_id}] Stage 4b: Archiving input directory {input_dir}...")
    update_job_status(job_id, JobStatus.RUNNING.value, status_message="Archiving input files...")

    archive_name = "input.zip"
    archive_path = os.path.join(input_dir, archive_name)

    # Basic check for directory existence before archiving
    input_dir_path = Path(input_dir)
    if not input_dir_path.exists() or not input_dir_path.is_dir():
        error_msg = f"Cannot archive: Input directory '{input_dir}' does not exist or is not a directory for job {job_id}."
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg)

    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in input_dir_path.rglob('*'):
                if not file_path.is_file():
                    continue
                
                relative_path = file_path.relative_to(input_dir_path)
                zipf.write(file_path, arcname=relative_path)

        logger.info(f"[{job_id}] Successfully created input archive at {archive_path}")
        update_job_status(job_id, JobStatus.RUNNING.value, progress=22, status_message="Input archive created.")
        return archive_path
    except Exception as e:
        logger.error(f"[{job_id}] Failed to create input archive from {input_dir}: {e}", exc_info=True)
        update_job_status(job_id, JobStatus.FAILED.value, error_message=f"Failed to archive inputs: {e}")
        raise RuntimeError(f"Failed to create input archive: {e}") from e


async def _submit_datasphere_job(job_id: str, client: DataSphereClient, ready_config_path: str) -> str:
    """
    Submits a job to DataSphere and returns the DS job ID.
    
    Args:
        job_id: The job ID in our system
        client: Initialized DataSphere client
        ready_config_path: Path to the ready configuration file
        
    Returns:
        The DataSphere job ID
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        RuntimeError: If job submission fails or times out
    """
    if not os.path.exists(ready_config_path):
        error_msg = f"DataSphere job config YAML not found at: {ready_config_path}"
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"[{job_id}] Submitting job to DataSphere using ready config: {ready_config_path}. Timeout: {CLIENT_SUBMIT_TIMEOUT_SECONDS}s")

    try:
        # client.submit_job is synchronous, run in a thread with timeout
        ds_job_id = await asyncio.wait_for(
            asyncio.to_thread(client.submit_job, config_path=ready_config_path),
            timeout=CLIENT_SUBMIT_TIMEOUT_SECONDS
        )
        logger.info(f"[{job_id}] DataSphere Job submitted, DS Job ID: {ds_job_id}")
        update_job_status(job_id, JobStatus.RUNNING.value, progress=25, 
                        status_message=f"DS Job {ds_job_id} submitted.")
        return ds_job_id
    except asyncio.TimeoutError:
        error_msg = f"DataSphere job submission timed out after {CLIENT_SUBMIT_TIMEOUT_SECONDS} seconds."
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg) from asyncio.TimeoutError
    except Exception as e:
        # This will catch DataSphereClientError from client.submit_job or other unexpected errors
        error_msg = f"Failed to submit DataSphere job: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        if "DataSphereClientError" in str(type(e)):
            raise # Re-raise original DataSphereClientError
        raise RuntimeError(error_msg) from e


async def _check_datasphere_job_status(job_id: str, ds_job_id: str, client: DataSphereClient) -> str:
    """
    Checks the status of a DataSphere job and returns the current status.
    
    Args:
        job_id: The job ID in our system
        ds_job_id: The DataSphere job ID
        client: Initialized DataSphere client
        
    Returns:
        The current status string from DataSphere
        
    Raises:
        RuntimeError: If getting status fails or times out
    """
    logger.debug(f"[{job_id}] Checking DataSphere job status for DS Job ID: {ds_job_id}. Timeout: {CLIENT_STATUS_TIMEOUT_SECONDS}s")
    try:
        # client.get_job_status is synchronous, run in a thread with timeout
        current_ds_status_str = await asyncio.wait_for(
            asyncio.to_thread(client.get_job_status, ds_job_id),
            timeout=CLIENT_STATUS_TIMEOUT_SECONDS
        )
        logger.debug(f"[{job_id}] Got status: {current_ds_status_str} for DS Job ID: {ds_job_id}")
        return current_ds_status_str
    except asyncio.TimeoutError:
        error_msg = f"DataSphere job status check timed out after {CLIENT_STATUS_TIMEOUT_SECONDS} seconds for DS Job ID: {ds_job_id}."
        logger.warning(f"[{job_id}] {error_msg}") # Log as warning, as polling might retry
        # Do not update job status here, let the polling loop handle overall job timeout/failure
        raise RuntimeError(error_msg) from asyncio.TimeoutError # Propagate to polling loop
    except Exception as status_exc:
        # This will catch DataSphereClientError or other unexpected errors
        error_msg = f"Error getting status for DS Job {ds_job_id}: {status_exc}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        # Do not update job status here, let the polling loop handle overall job timeout/failure
        if "DataSphereClientError" in str(type(status_exc)):
            raise # Re-raise original DataSphereClientError
        raise RuntimeError(error_msg) from status_exc


async def _download_datasphere_job_results(
    job_id: str, 
    ds_job_id: str, 
    client: DataSphereClient, 
    results_dir: str,
    ds_job_run_suffix: str
) -> Tuple[Dict[str, Any] | None, str | None, str | None]:
    """Downloads and processes the results from a DataSphere job."""
    logger.info(f"[{job_id}] Stage 5a: Downloading results for ds_job '{ds_job_id}' to '{results_dir}'...")

    metrics_data = None
    model_path = None
    predictions_path = None

    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"[{job_id}] Downloading results for DS Job {ds_job_id} to {results_dir}. Timeout: {CLIENT_DOWNLOAD_TIMEOUT_SECONDS}s")

    try:
        # client.download_job_results is synchronous, run in a thread with timeout
        await asyncio.wait_for(
            asyncio.to_thread(client.download_job_results, ds_job_id, results_dir),
            timeout=CLIENT_DOWNLOAD_TIMEOUT_SECONDS
        )
        logger.info(f"[{job_id}] Results for DS Job {ds_job_id} downloaded to {results_dir}")
        
        # Check if there's an output.zip archive that needs to be extracted
        output_zip_path = os.path.join(results_dir, "output.zip")
        if os.path.exists(output_zip_path):
            logger.info(f"[{job_id}] Found output.zip archive, extracting to {results_dir}")
            try:
                with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(results_dir)
                logger.info(f"[{job_id}] Successfully extracted output.zip archive")
                
                # Optional: Remove the archive file after extraction
                os.remove(output_zip_path)
                logger.debug(f"[{job_id}] Removed output.zip archive after extraction")
            except zipfile.BadZipFile as e:
                logger.error(f"[{job_id}] Invalid zip file format for output.zip: {e}")
                # Continue processing in case individual files exist
            except Exception as e:
                logger.error(f"[{job_id}] Error extracting output.zip: {e}", exc_info=True)
                # Continue processing in case individual files exist
        
        # Check for model, predictions, and metrics files
        model_path = os.path.join(results_dir, DATASPHERE_MODEL_FILE)
        predictions_path = os.path.join(results_dir, DATASPHERE_PREDICTIONS_FILE)
        metrics_path = os.path.join(results_dir, "metrics.json")

        if not os.path.exists(model_path):
            logger.warning(f"[{job_id}] Model file '{DATASPHERE_MODEL_FILE}' not found at {model_path}. Model will not be saved.")
            model_path = None
        
        if not os.path.exists(predictions_path):
            logger.warning(f"[{job_id}] Predictions file '{DATASPHERE_PREDICTIONS_FILE}' not found at {predictions_path}. Predictions will not be saved.")
            predictions_path = None
        
        if not os.path.exists(metrics_path):
            logger.warning(f"[{job_id}] Metrics file 'metrics.json' not found at {metrics_path}. Metrics will not be saved.")
            metrics_data = None
        else:
            try:
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                logger.info(f"[{job_id}] Loaded metrics from {metrics_path}")
            except json.JSONDecodeError as e:
                logger.error(f"[{job_id}] Failed to parse metrics.json from {metrics_path}: {e}", exc_info=True)
                metrics_data = None # Ensure it's None if parsing fails
                # Optionally, this could be a critical error causing the job to fail

        return metrics_data, predictions_path, model_path
    except asyncio.TimeoutError:
        error_msg = f"DataSphere job results download timed out after {CLIENT_DOWNLOAD_TIMEOUT_SECONDS} seconds for DS Job ID: {ds_job_id}."
        logger.error(f"[{job_id}] {error_msg}")
        # Update job status directly here if download failure is critical for the job outcome
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg) from asyncio.TimeoutError       
    except Exception as e:
        logger.error(f"[{job_id}] DataSphere job '{ds_job_id}' failed with an exception during result download: {e}", exc_info=True)
        # It's not a timeout, but it's a failure in the download process itself
        raise RuntimeError(f"DataSphere job result download failed for job {job_id}. Details: {e}") from e
    finally:
        # This block is for logging the downloaded files for debug purposes
        try:
            if os.path.exists(results_dir):
                logger.debug(f"[{job_id}] Contents of results directory '{results_dir}': {os.listdir(results_dir)}")
        except Exception as e:
            logger.warning(f"[{job_id}] Could not list contents of results directory '{results_dir}': {e}")


async def _download_logs_diagnostics(
    job_id: str, 
    ds_job_id: str, 
    client: DataSphereClient, 
    logs_dir: str, 
    is_success: bool = True
) -> None:
    """
    Downloads logs and diagnostics for a DataSphere job.
    
    Args:
        job_id: The job ID in our system
        ds_job_id: The DataSphere job ID
        client: Initialized DataSphere client
        logs_dir: Directory to download logs to
        is_success: Whether the job was successful (used for logging message)
        
    Returns:
        None
    """
    os.makedirs(logs_dir, exist_ok=True)
    status_str = "successful" if is_success else "failed"
    logger.info(f"[{job_id}] Downloading logs/diagnostics for {status_str} DS Job {ds_job_id} to {logs_dir}. Timeout: {CLIENT_DOWNLOAD_TIMEOUT_SECONDS}s")

    try:
        # client.download_job_results is synchronous, run in a thread with timeout
        await asyncio.wait_for(
            asyncio.to_thread(
                client.download_job_results,
                ds_job_id,
                logs_dir,
                with_logs=True,
                with_diagnostics=True
            ),
            timeout=CLIENT_DOWNLOAD_TIMEOUT_SECONDS
        )
        logger.info(f"[{job_id}] Logs/diagnostics for {status_str} DS Job {ds_job_id} downloaded to {logs_dir}")
    except asyncio.TimeoutError:
        # This is generally a non-critical error for the job's main outcome, so log and continue.
        error_msg = f"DataSphere logs/diagnostics download timed out after {CLIENT_DOWNLOAD_TIMEOUT_SECONDS} seconds for DS Job ID: {ds_job_id}."
        logger.warning(f"[{job_id}] {error_msg}")
    except Exception as dl_exc:
        # This is generally a non-critical error for the job's main outcome, so log and continue.
        logger.warning(f"[{job_id}] Failed to download logs/diagnostics for DS Job {ds_job_id}: {dl_exc}", exc_info=True)


async def _process_datasphere_job(
    job_id: str,
    client: DataSphereClient,
    ds_job_run_suffix: str,
    ds_job_specific_output_base_dir: str,
    ready_config_path: str
) -> Tuple[str, str, Dict[str, Any] | None, str | None, str | None, int]:
    """
    Coordinates the DataSphere job lifecycle including submission, monitoring, and result fetching.
    
    Args:
        job_id: ID of the job
        client: DataSphere client
        ds_job_run_suffix: Unique identifier for this run
        ds_job_specific_output_base_dir: Directory for this run's outputs/results
        ready_config_path: Path to the ready DataSphere config file
        
    Returns:
        Tuple of (ds_job_id, results_dir, metrics_data, model_path, predictions_path, polls)
    """
    logger.info(f"[{job_id}] Stage 5: Processing DataSphere job {ds_job_run_suffix}...")
    
    # Submit the job
    ds_job_id = await _submit_datasphere_job(job_id, client, ready_config_path)

    completed = False
    max_polls = settings.datasphere.max_polls
    poll_interval = settings.datasphere.poll_interval
    polls = 0
    metrics_data = None
    predictions_path = None
    model_path = None

    logger.info(f"[{job_id}] Polling DS Job {ds_job_id} status (max_polls={max_polls}, poll_interval={poll_interval}s).")
    logger.debug(f"DEBUG MONITOR: DS Job ID: {ds_job_id}, Max polls from settings: {settings.datasphere.max_polls}")

    # Monitor job status
    while not completed and polls < max_polls:
        logger.debug(f"DEBUG MONITOR: Entering poll loop, poll {polls + 1} for DS Job ID: {ds_job_id}")
        await asyncio.sleep(poll_interval)
        polls += 1

        try:
            current_ds_status_str = await _check_datasphere_job_status(job_id, ds_job_id, client)
            current_ds_status = current_ds_status_str.lower()
            logger.info(f"[{job_id}] DS Job {ds_job_id} status (poll {polls}/{max_polls}): {current_ds_status_str}")
        except Exception as status_exc:
            logger.error(f"[{job_id}] Error getting status for DS Job {ds_job_id} (poll {polls}/{max_polls}): {status_exc}. Will retry.")
            update_job_status(
                job_id,
                JobStatus.RUNNING.value,
                status_message=f"DS Job {ds_job_id}: Error polling status ({status_exc}) - Retrying..."
            )
            continue

        # Estimate progress
        current_progress = 25 + int((polls / max_polls) * 65)  # Progress from 25% to 90% during polling
        update_job_status(
            job_id,
            JobStatus.RUNNING.value,
            progress=current_progress,
            status_message=f"DS Job {ds_job_id}: {current_ds_status_str}"
        )

        if current_ds_status in ["completed", "success"]:
            completed = True
            logger.info(f"[{job_id}] DS Job {ds_job_id} completed. Downloading results...")

            # Create a dedicated directory for results
            results_download_dir = os.path.join(
                ds_job_specific_output_base_dir,
                "results"
            )
            
            # Download and process results
            metrics_data, predictions_path, model_path = await _download_datasphere_job_results(
                job_id, ds_job_id, client, results_download_dir, ds_job_run_suffix
            )

            # Download logs and diagnostics if configured
            download_diag_on_success = getattr(settings.datasphere, 'download_diagnostics_on_success', False)
            if download_diag_on_success:
                logs_dir = os.path.join(results_download_dir, "logs_diagnostics_success")
                await _download_logs_diagnostics(job_id, ds_job_id, client, logs_dir, is_success=True)
            else: 
                logger.info(f"[{job_id}] Skipping optional download of logs/diagnostics for successful job based on settings.")

        elif current_ds_status in ["failed", "error", "cancelled"]:
            # Define results directory for consistency
            results_download_dir = os.path.join(
                ds_job_specific_output_base_dir,
                "results"
            )
            
            error_detail = f"DS Job {ds_job_id} ended with status: {current_ds_status_str}."
            logger.error(f"[{job_id}] {error_detail}")

            # Download logs for failed job
            logs_dir = os.path.join(results_download_dir, "logs_diagnostics")
            await _download_logs_diagnostics(job_id, ds_job_id, client, logs_dir, is_success=False)
            error_detail += f" Logs/diagnostics may be available in {logs_dir}."

            update_job_status(job_id, JobStatus.FAILED.value, error_message=error_detail)
            raise RuntimeError(error_detail)

    if not completed:
        timeout_message = f"DS Job {ds_job_id} execution timed out after {polls} polls ({max_polls * poll_interval}s)."
        logger.error(f"[{job_id}] {timeout_message}")
        update_job_status(job_id, status=JobStatus.FAILED.value, error_message=str(timeout_message))
        raise TimeoutError(timeout_message)

    return ds_job_id, results_download_dir, metrics_data, model_path, predictions_path, polls


async def _perform_model_cleanup(job_id: str, current_model_id: str) -> None:
    """Prunes old, non-active models based on settings."""
    try:
        num_models_to_keep = getattr(settings, "max_models_to_keep", 5)
        if num_models_to_keep > 0:
            logger.info(f"[{job_id}] Checking for old models to prune (keeping last {num_models_to_keep})...")

            # Get IDs of models to keep (most recent ones, including potentially the current one if it becomes active later)
            # Note: The current model (current_model_id) is initially inactive.
            recent_kept_models_info = get_recent_models(limit=num_models_to_keep)
            kept_model_ids = {m['model_id'] for m in recent_kept_models_info} # Use a set for faster lookups
            logger.debug(f"[{job_id}] Initially identified models to keep (by recent creation): {kept_model_ids}")

            # Fetch all models (or a reasonable limit) to find candidates for deletion
            all_models_info = get_all_models(limit=1000)
            models_to_delete_ids = []

            for model_info in all_models_info:
                m_id = model_info['model_id']
                is_active = model_info.get('is_active', False)

                # Candidate for deletion if:
                # 1. It's NOT the model just created in *this* job run.
                # 2. It's NOT marked as active.
                # 3. It's NOT in the set of recently created models to keep.
                if m_id != current_model_id and not is_active and m_id not in kept_model_ids:
                    models_to_delete_ids.append(m_id)

            if models_to_delete_ids:
                logger.info(f"[{job_id}] Found {len(models_to_delete_ids)} older, non-active models to prune: {models_to_delete_ids}")
                delete_result = delete_models_by_ids(models_to_delete_ids) # Assuming this handles file deletion too
                deleted_count = delete_result.get("deleted_count", "N/A")
                failed_deletions = delete_result.get("failed_ids", [])
                logger.info(f"[{job_id}] Model cleanup result: Deleted {deleted_count} models. Failed: {failed_deletions}")
            else:
                logger.info(f"[{job_id}] No non-active models found eligible for pruning beyond the kept {num_models_to_keep}.")
        else:
            logger.info(f"[{job_id}] Model cleanup disabled (max_models_to_keep <= 0).")
    except Exception as cleanup_exc:
        logger.error(f"[{job_id}] Error during model cleanup: {cleanup_exc}", exc_info=True)
        # Non-fatal, log and continue


async def _process_job_results(
    job_id: str,
    ds_job_id: str,
    results_dir: str, # Base directory where results were downloaded.
    config: TrainingConfig,
    metrics_data: Dict[str, Any] | None,
    model_path: str | None,
    predictions_path: str | None,
    polls: int,
    poll_interval: float,
    config_id: str
):
    """Processes the results of a completed DataSphere job, saving artifacts and updating the database."""
    final_message = f"Job completed. DS Job ID: {ds_job_id}."
    warnings_list = []
    model_id_for_status = None

    try:
        # Save model and predictions if they exist
        if model_path:
            model_id_for_status = await save_model_file_and_db(
                job_id=job_id, 
                model_path=model_path, 
                ds_job_id=ds_job_id, 
                config=config, 
                metrics_data=metrics_data
            )
            logger.info(f"[{job_id}] Model saved with ID: {model_id_for_status}")

            if predictions_path:
                logger.info(f"[{job_id}] DEBUG_SERVICE: predictions_path exists. Proceeding to save predictions to DB from '{predictions_path}'...")
                prediction_result_info = save_predictions_to_db(
                    predictions_path=predictions_path,
                    job_id=job_id,
                    model_id=model_id_for_status # Pass current_model_id, which is confirmed not None here
                )
                logger.info(f"[{job_id}] Saved {prediction_result_info.get('predictions_count', 'N/A')} predictions to database with result_id: {prediction_result_info.get('result_id', 'N/A')}")
            else:
                logger.warning(f"[{job_id}] No predictions file found, cannot save predictions to DB.")
                warnings_list.append("Predictions file not found in results.")
        else:
            logger.warning(f"[{job_id}] DEBUG_SERVICE: model_path is None or empty. Skipping model and prediction saving.")
            warnings_list.append("Model file not found in results.")

        # Create a record of the training result with metrics
        training_result_id = None
        if metrics_data:
            logger.info(f"[{job_id}] DEBUG_SERVICE: metrics_data exists. Proceeding to create training result record...")
            training_result_id = create_training_result(
                job_id=job_id,
                config_id=config_id,
                metrics=metrics_data,  # Pass the dict directly
                model_id=model_id_for_status, # Can be None if model saving failed or no model
                duration=int(polls * poll_interval),
                config=config.model_dump() # Pass the dict directly
            )
            logger.info(f"[{job_id}] Training result record created: {training_result_id}")
            final_message += f" Training Result ID: {training_result_id}."
        else:
            logger.warning(f"[{job_id}] No metrics data available. Training result record not created.")
            warnings_list.append("Metrics data not found in results.")

        # Append warnings to the status message if any
        if warnings_list:
            final_message += " Processing warnings: " + "; ".join(warnings_list)

        # Update job status to completed
        update_job_status(job_id, JobStatus.COMPLETED.value, progress=100, status_message=final_message, result_id=training_result_id)
        logger.info(f"[{job_id}] Job processing completed. Final status message: {final_message}")

        # Perform model cleanup only if a model was successfully created in this run
        if model_id_for_status:
            await _perform_model_cleanup(job_id, model_id_for_status)
        else:
            logger.info(f"[{job_id}] Skipping model cleanup as no new model was created in this job run.")

    except Exception as e:
        error_msg = f"Error processing job results: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        # Ensure final status is FAILED if not already set by a more specific handler
        # Check if job status was already updated to FAILED by a helper, to avoid overwriting detailed message
        job_details = get_job(job_id) # This might fail if DB is unavailable, handle it.
        if job_details and job_details.get('status') != JobStatus.FAILED.value:
            update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg, status_message=error_msg)
        raise


async def _cleanup_directories(job_id: str, dir_paths: List[str]) -> List[str]:
    """Attempts to recursively delete the specified directories."""
    logger.info(f"[{job_id}] Stage 7: Cleaning up specified directories: {dir_paths}")
    cleanup_errors = []

    for dir_path in dir_paths:
        if dir_path and os.path.exists(dir_path) and os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                logger.info(f"[{job_id}] Successfully deleted directory: {dir_path}")
            except Exception as e_clean:
                error_msg = f"Error deleting directory {dir_path}: {e_clean}"
                logger.error(f"[{job_id}] {error_msg}")
                cleanup_errors.append(error_msg)
        elif not dir_path:
             logger.info(f"[{job_id}] Skipping cleanup for empty directory path.")
        else:
            logger.info(f"[{job_id}] Directory not found or not a directory, skipping cleanup: {dir_path}")
    
    # Return any cleanup errors to be handled by caller
    return cleanup_errors


def save_predictions_to_db(
    predictions_path: str, 
    job_id: str, 
    model_id: str,
    direct_db_connection=None
) -> dict:
    """
    Reads prediction results from a CSV file and saves them to the database.
    
    Args:
        predictions_path: Path to the CSV file with predictions
        job_id: The job ID that produced these predictions
        model_id: The model ID used for these predictions
        direct_db_connection: Optional direct database connection to use
        
    Returns:
        Dictionary with result_id and predictions_count
        
    Raises:
        FileNotFoundError: If the predictions file doesn't exist
        ValueError: If the predictions file has invalid format or missing required columns
    """
    # Verify the file exists
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    try:
        # Load predictions from CSV
        df = pd.read_csv(predictions_path)
                
        # Verify required columns exist
        required_columns = ['barcode', 'artist', 'album', 'cover_type', 
                           'price_category', 'release_type', 'recording_decade', 
                           'release_decade', 'style', 'record_year',
                           '0.05', '0.25', '0.5', '0.75', '0.95']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Predictions file missing required columns: {missing_cols}")
        
        # Generate a UUID for the prediction result
        result_id = str(uuid.uuid4())
        
        # Create a connection to the database
        conn = direct_db_connection if direct_db_connection else get_db_connection()
        
        try:
            conn.commit()
            
            # Begin transaction
            conn.execute("BEGIN TRANSACTION")
            
            # Create a record in prediction_results table
            timestamp = datetime.now().isoformat()
            conn.execute(
                """
                INSERT INTO prediction_results 
                (result_id, job_id, model_id, prediction_date, output_path, summary_metrics) 
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (result_id, job_id, model_id, timestamp, predictions_path, "{}")
            )
            
            # Prepare data for batch insert into fact_predictions
            predictions_data = []
            
            for _, row in df.iterrows():
                # Get or create multiindex_id
                multiindex_id = get_or_create_multiindex_id(
                    barcode=row['barcode'],
                    artist=row['artist'],
                    album=row['album'],
                    cover_type=row['cover_type'],
                    price_category=row['price_category'],
                    release_type=row['release_type'],
                    recording_decade=row['recording_decade'],
                    release_decade=row['release_decade'],
                    style=row['style'],
                    record_year=int(row['record_year']),
                    connection=conn
                )
                
                # Prepare prediction data
                prediction_row = (
                    result_id,
                    multiindex_id,
                    timestamp,  # prediction_date
                    model_id,   # model_id
                    row['0.05'],
                    row['0.25'],
                    row['0.5'],
                    row['0.75'],
                    row['0.95'],
                    timestamp   # created_at
                )
                predictions_data.append(prediction_row)
            
            # Batch insert all predictions
            try:
                # Проверка соединения и таблицы перед вставкой
                check_cursor = conn.cursor()
                check_cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fact_predictions'")
                table_exists = check_cursor.fetchone()
                if not table_exists:
                    raise ValueError("Table fact_predictions does not exist in the database")
                
                # Попробуем прямой вызов вместо execute_many
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT INTO fact_predictions
                    (result_id, multiindex_id, prediction_date, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    predictions_data
                )
            except Exception:
                # Если прямой вызов не сработал, используем execute_many
                execute_many(
                    """
                    INSERT INTO fact_predictions
                    (result_id, multiindex_id, prediction_date, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    predictions_data,
                    conn
                )
            
            # Commit the transaction instead of execute("COMMIT")
            conn.commit()
            
            return {
                "result_id": result_id,
                "predictions_count": len(df)
            }
            
        except Exception as e:
            # Rollback on error
            try:
                conn.rollback()
            except:
                pass
            raise e
        finally:
            # Не закрываем соединение, если это внешнее соединение
            if not direct_db_connection and conn:
                conn.close()
            
    except pd.errors.EmptyDataError:
        raise ValueError("Predictions file is empty")
    except pd.errors.ParserError:
        raise ValueError("Predictions file has invalid format")
    except Exception as e:
        # Do not wrap database connection errors
        import sqlite3
        if isinstance(e, FileNotFoundError):
            raise
        if isinstance(e, sqlite3.OperationalError):
            raise
        raise ValueError(f"Error processing predictions: {str(e)}")


# Main Orchestrator Function
async def run_job(job_id: str, training_config: dict, config_id: str, dataset_start_date: str = None, dataset_end_date: str = None) -> Dict[str, Any] | None:
    """
    Runs a DataSphere training job pipeline: setup, execution, monitoring, result processing, cleanup.
    """
    logger.debug(f"[{job_id}] run_job: Initial settings.project_root_dir = {settings.project_root_dir}")
    ds_job_id: Optional[str] = None
    ds_job_run_suffix: Optional[str] = None 
    client: Optional[DataSphereClient] = None
    job_completed_successfully = False

    try:
        with tempfile.TemporaryDirectory(dir=str(settings.datasphere_input_dir)) as temp_input_dir_str, \
             tempfile.TemporaryDirectory(dir=str(settings.datasphere_output_dir)) as temp_output_dir_str:
            temp_input_dir = Path(temp_input_dir_str)
            temp_output_dir = Path(temp_output_dir_str)

            try:
                # Initialize job status
                update_job_status(job_id, JobStatus.PENDING.value, progress=0, status_message="Initializing job.")

                # Stage 1: Get Parameters (already provided)
                if training_config is None:
                    raise ValueError("No training configuration was provided or found.")
                config = TrainingConfig(**training_config)

                # Stage 2: Prepare Datasets
                await _prepare_job_datasets(job_id, config, dataset_start_date, dataset_end_date, output_dir=str(temp_input_dir))

                # Stage 3: Initialize Client
                client = await _initialize_datasphere_client(job_id)

                # Stage 4a: Prepare DS Job Submission Inputs
                ds_job_run_suffix, ready_config_path = await _prepare_datasphere_job_submission(job_id, config, temp_input_dir)

                # Stage 4a.1: Verify DataSphere job inputs
                await _verify_datasphere_job_inputs(job_id, temp_input_dir)

                # Stage 4b: Archive Input Directory
                _ = await _archive_input_directory(job_id, temp_input_dir_str)

                # Stage 5: Submit and Monitor DS Job
                ds_job_id, results_dir_from_process, metrics_data, model_path, predictions_path, polls = await _process_datasphere_job(
                    job_id,
                    client,
                    ds_job_run_suffix,
                    temp_output_dir,
                    ready_config_path
                )

                # Stage 6: Process Results
                await _process_job_results(
                    job_id, ds_job_id, results_dir_from_process, config, metrics_data, model_path, predictions_path, polls, settings.datasphere.poll_interval, config_id
                )
                job_completed_successfully = True
                logger.info(f"[{job_id}] Job pipeline completed successfully.")
                
                return {
                    "job_id": job_id,
                    "status": JobStatus.COMPLETED.value,
                    "datasphere_job_id": ds_job_id,
                    "model_id": model_path.split(os.sep)[-2] if model_path else None, 
                    "message": "Job completed successfully"
                }
            except asyncio.CancelledError:
                logger.warning(f"[{job_id}] Job run was cancelled.")
                active_client = client if 'client' in locals() and client is not None else None
                current_ds_job_id = ds_job_id if 'ds_job_id' in locals() and ds_job_id is not None else None

                cancel_message = "Job cancelled by application."
                if active_client and current_ds_job_id:
                    logger.info(f"[{job_id}] Attempting to cancel DataSphere job {current_ds_job_id} due to task cancellation.")
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(active_client.cancel_job, current_ds_job_id, graceful=True),
                            timeout=CLIENT_CANCEL_TIMEOUT_SECONDS
                        )
                        logger.info(f"[{job_id}] DataSphere job {current_ds_job_id} cancellation request submitted.")
                        cancel_message += f" DataSphere job {current_ds_job_id} cancellation attempted."
                    except asyncio.TimeoutError:
                        logger.error(f"[{job_id}] Timeout trying to cancel DataSphere job {current_ds_job_id}.")
                        cancel_message += f" Timeout attempting to cancel DataSphere job {current_ds_job_id}."
                    except Exception as cancel_e:
                        logger.error(f"[{job_id}] Error trying to cancel DataSphere job {current_ds_job_id}: {cancel_e}", exc_info=True)
                        cancel_message += f" Error attempting to cancel DataSphere job {current_ds_job_id}: {cancel_e}."
                
                update_job_status(job_id, JobStatus.FAILED.value, error_message=cancel_message, status_message=cancel_message)
                
            except (ValueError, RuntimeError, TimeoutError, ImportError) as e:
                error_msg = f"Job pipeline failed: {str(e)}"
                logger.error(f"[{job_id}] {error_msg}", exc_info=isinstance(e, (RuntimeError, ImportError)))
                job_details = get_job(job_id)
                if job_details and job_details.get('status') != JobStatus.FAILED.value:
                    update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg, status_message=error_msg)
                raise
            except Exception as e:
                error_msg = f"Unexpected error in job pipeline: {str(e)}"
                logger.error(f"[{job_id}] {error_msg}", exc_info=True)
                job_details = get_job(job_id)
                if job_details and job_details.get('status') != JobStatus.FAILED.value:
                    update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg, status_message=error_msg)
                raise 
            finally:
                # The temporary directories are cleaned up automatically by the context manager.
                # Final log message for job run completion.
                current_ds_job_id_log = ds_job_id if 'ds_job_id' in locals() and ds_job_id else "N/A"
                current_ds_job_suffix_log = ds_job_run_suffix if 'ds_job_run_suffix' in locals() and ds_job_run_suffix else "N/A"
                ds_job_info_str = f"DS Job ID: {current_ds_job_id_log}, DS Job Suffix: {current_ds_job_suffix_log}"
                logger.info(f"[{job_id}] Job run processing finished. ({ds_job_info_str}). Temporary directories will be cleaned up automatically.")

    except Exception as e:
        # This will catch errors in creating the temporary directories themselves.
        error_msg = f"Failed to set up temporary directories for job: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg, status_message=error_msg)
        raise
        
async def save_model_file_and_db(
    job_id: str, 
    model_path: str, 
    ds_job_id: str,
    config: TrainingConfig,
    metrics_data: Optional[Dict[str, Any]]
) -> str:
    """Saves the model file reference to the database."""
    try:
        # Generate a unique model ID based on the config's model_id (base name)
        model_id = f"{config.model_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"[{job_id}] Generated new model ID: {model_id}")

        # Construct metadata
        file_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        metadata = {
            "file_size_bytes": file_size,
            "downloaded_from_ds_job": ds_job_id,
            "original_path": model_path,
            "config_model_id_base": config.model_id, # Store the base name for tracking
            "metrics": metrics_data or {} # Store metrics with the model
        }
        
        # Create a record in the database
        create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path=str(model_path),
            created_at=datetime.now(),
            metadata=metadata
        )
        logger.info(f"[{job_id}] Successfully created model record in DB for model_id: {model_id}")
        return model_id

    except Exception as e:
        error_msg = f"Failed to save model file record to DB for path {model_path}: {e}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        # We don't update job status here, the caller should handle it
        raise RuntimeError(error_msg) from e
