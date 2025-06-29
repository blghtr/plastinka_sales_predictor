import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from deployment.app.config import get_settings
from deployment.app.db.database import (
    create_model_record,
    create_training_result,
    delete_models_by_ids,
    execute_many,
    get_all_models,
    get_db_connection,
    get_job,
    get_or_create_multiindex_id,
    get_recent_models,
    update_job_status,
)
from deployment.app.models.api_models import JobStatus, TrainingConfig
from deployment.datasphere.client import DataSphereClient
from deployment.datasphere.prepare_datasets import get_datasets

# Initialize settings
# settings = AppSettings()

# Constants from settings
# The constants below are now fetched inside the functions where they are needed
# to ensure they use the latest configuration, especially in test environments.

# Set up logger
logger = logging.getLogger(__name__)

# Helper Functions for Project Input Linking

def create_project_input_link(archive_path: str, project_root: str) -> str:
    """
    Creates cross-platform link to archive in project root.
    
    Args:
        archive_path: Path to the temporary archive file
        project_root: Path to the project root directory
        
    Returns:
        Path to the created link/file in project root
        
    Raises:
        RuntimeError: If linking/copying fails
    """
    project_input_path = os.path.join(project_root, "input.zip")

    # Remove existing file/link if present
    if os.path.exists(project_input_path):
        try:
            os.remove(project_input_path)
        except OSError as e:
            logger.warning(f"Failed to remove existing input.zip: {e}")

    try:
        if os.name == 'posix':  # Linux/macOS - symlink
            os.symlink(archive_path, project_input_path)
            logger.info(f"Created symlink: {project_input_path} -> {archive_path}")
        else:  # Windows - hard link
            os.link(archive_path, project_input_path)
            logger.info(f"Created hard link: {project_input_path} -> {archive_path}")
    except OSError as e:
        # Fallback to copying if linking fails
        try:
            shutil.copy2(archive_path, project_input_path)
            logger.warning(f"Failed to create link ({e}), copied file instead: {project_input_path}")
        except Exception as copy_e:
            error_msg = f"Failed to create link and copy fallback failed: {copy_e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from copy_e

    return project_input_path


def cleanup_project_input_link(project_root: str) -> None:
    """
    Removes input.zip link/file from project root.
    
    Args:
        project_root: Path to the project root directory
    """
    project_input_path = os.path.join(project_root, "input.zip")
    try:
        if os.path.exists(project_input_path):
            os.remove(project_input_path)
            logger.info(f"Cleaned up project input link: {project_input_path}")
    except OSError as e:
        logger.warning(f"Failed to cleanup project input link: {e}")


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
    settings = get_settings()
    if not settings.datasphere or not settings.datasphere.client:
        update_job_status(job_id, JobStatus.FAILED.value, error_message="DataSphere client configuration is missing.")
        raise ValueError("DataSphere client configuration is missing in settings.")

    client_config = settings.datasphere.client # Get config dict

    try:
        # DataSphereClient constructor is synchronous, run in a thread
        client = await asyncio.wait_for(
            asyncio.to_thread(DataSphereClient, **client_config),
            timeout=settings.datasphere.client_init_timeout_seconds
        )
        logger.info(f"[{job_id}] DataSphere client initialized successfully.")
        update_job_status(job_id, JobStatus.RUNNING.value, progress=15, status_message="DataSphere client initialized.")
        return client
    except asyncio.TimeoutError:
        error_msg = f"DataSphere client initialization timed out after {settings.datasphere.client_init_timeout_seconds} seconds."
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

    if not train_data_file.is_file() or not val_data_file.is_file():
        missing_datasets = []
        if not train_data_file.is_file():
            missing_datasets.append("train.dill")
        if not val_data_file.is_file():
            missing_datasets.append("val.dill")

        error_msg = f"Missing required dataset files in input directory '{input_dir_path}' for job {job_id}: {', '.join(missing_datasets)}."
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"[{job_id}] DataSphere job inputs verified successfully.")
    update_job_status(job_id, JobStatus.RUNNING.value, progress=20, status_message="DataSphere job inputs verified.")


async def _prepare_datasphere_job_submission(job_id: str, config: TrainingConfig, target_input_dir: Path) -> str:
    """Prepares the parameters file for job submission."""
    logger.info(f"[{job_id}] Stage 4a: Preparing DataSphere job submission inputs...")

    # Save training config as JSON
    config_json_path = str(target_input_dir / "config.json")
    logger.info(f"[{job_id}] Saving training config to {config_json_path}")
    with open(config_json_path, 'w') as f: # open() accepts string path
        # Use model_dump_json for Pydantic v2+
        json.dump(config.model_dump(), f, indent=2) # Save training config

    logger.info(f"[{job_id}] DataSphere job submission inputs prepared.")
    update_job_status(job_id, JobStatus.RUNNING.value, progress=18, status_message="Job submission inputs prepared.")

    # Return the path to the static template config file
    return str(get_settings().datasphere_job_config_path)


async def _archive_input_directory(job_id: str, input_dir: str, archive_dir: str = None) -> str:
    """
    Archives the input directory into a zip file for DataSphere submission.
    
    Args:
        job_id: The job ID for logging
        input_dir: Path to the input directory to archive
        archive_dir: Optional directory where to create the archive. 
                    If None, creates in parent of input_dir (current behavior)
        
    Returns:
        Path to the created archive file
        
    Raises:
        RuntimeError: If archiving fails
    """
    logger.info(f"[{job_id}] Stage 4b: Archiving input directory '{input_dir}'...")

    # Choose where to create the archive
    input_dir_path = Path(input_dir)
    archive_name = f"{input_dir_path.name}.zip"

    if archive_dir:
        archive_path = Path(archive_dir) / archive_name
        logger.info(f"[{job_id}] Creating archive in specified directory: {archive_path}")
    else:
        # Current behavior for backward compatibility
        archive_path = input_dir_path.parent / archive_name
        logger.info(f"[{job_id}] Creating archive in parent directory: {archive_path}")

    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(input_dir):
                # Skip the build artifacts directory
                if '_build_artifacts' in dirs:
                    dirs.remove('_build_artifacts')
                    logger.info(f"[{job_id}] Excluding _build_artifacts directory from archive")
                
                for file in files:
                    file_path = os.path.join(root, file)
                    # Skip the archive file itself to avoid recursion
                    if os.path.abspath(file_path) == os.path.abspath(archive_path):
                        continue
                    # Calculate relative path from input_dir
                    relative_path = os.path.relpath(file_path, input_dir)
                    zipf.write(file_path, relative_path)

        logger.info(f"[{job_id}] Input directory archived to: {archive_path}")
        update_job_status(job_id, JobStatus.RUNNING.value, progress=23, status_message="Input directory archived.")
        return str(archive_path)
    except Exception as e:
        error_msg = f"Failed to archive input directory: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg) from e








async def _create_new_datasphere_job(
    job_id: str,
    client: DataSphereClient,
    ready_config_path: str,
    work_dir: str
) -> str:
    """
    Creates a new DataSphere job (current logic).
    
    Args:
        job_id: Our internal job ID
        client: DataSphere client instance
        ready_config_path: Path to job configuration
        work_dir: Working directory for temporary files
        
    Returns:
        DataSphere job ID of the new job
        
    Raises:
        RuntimeError: If job creation fails or times out
    """
    logger.info(f"[{job_id}] Creating new DataSphere job")

    try:
        # Use the existing submit_job method with timeout and work_dir
        ds_job_id = await asyncio.wait_for(
            asyncio.to_thread(client.submit_job, config_path=ready_config_path, work_dir=work_dir),
            timeout=get_settings().datasphere.client_submit_timeout_seconds
        )

        logger.info(f"[{job_id}] Successfully created new job: {ds_job_id}")
        return ds_job_id

    except asyncio.TimeoutError:
        error_msg = f"DataSphere job creation timed out after {get_settings().datasphere.client_submit_timeout_seconds} seconds"
        logger.error(f"[{job_id}] {error_msg}")
        raise RuntimeError(error_msg) from asyncio.TimeoutError
    except Exception as e:
        error_msg = f"Failed to create new DataSphere job: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        raise RuntimeError(error_msg) from e


async def _submit_datasphere_job(job_id: str, client: DataSphereClient, ready_config_path: str, work_dir: str) -> str:
    """
    Submits a new DataSphere job and returns the DS job ID.
    
    Args:
        job_id: The job ID in our system
        client: Initialized DataSphere client
        ready_config_path: Path to the ready configuration file
        work_dir: Working directory for temporary files
        
    Returns:
        The DataSphere job ID
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        RuntimeError: If job creation fails
    """
    if not os.path.exists(ready_config_path):
        error_msg = f"DataSphere job config YAML not found at: {ready_config_path}"
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Create new job
        logger.info(f"[{job_id}] Creating new DataSphere job")
        ds_job_id = await _create_new_datasphere_job(job_id, client, ready_config_path, work_dir)

        # Update status
        update_job_status(job_id, JobStatus.RUNNING.value, progress=25,
                        status_message=f"DS Job {ds_job_id} submitted.")
        return ds_job_id

    except Exception as e:
        if not isinstance(e, (FileNotFoundError, RuntimeError)):
            error_msg = f"DataSphere job creation failed: {str(e)}"
            logger.error(f"[{job_id}] {error_msg}", exc_info=True)
            update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
            raise RuntimeError(error_msg) from e
        raise


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
    try:
        # client.get_job_status is synchronous, run in a thread with timeout
        current_ds_status_str = await asyncio.wait_for(
            asyncio.to_thread(client.get_job_status, ds_job_id),
            timeout=get_settings().datasphere.client_status_timeout_seconds
        )
        return current_ds_status_str
    except asyncio.TimeoutError:
        error_msg = f"DataSphere job status check timed out after {get_settings().datasphere.client_status_timeout_seconds} seconds for DS Job ID: {ds_job_id}."
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
    results_dir: str
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Downloads and processes the results from a DataSphere job."""
    logger.info(f"[{job_id}] Stage 5a: Downloading results for ds_job '{ds_job_id}' to '{results_dir}'...")

    metrics_data = None
    model_path = None
    predictions_path = None

    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"[{job_id}] Downloading results for DS Job {ds_job_id} to {results_dir}. Timeout: {get_settings().datasphere.client_download_timeout_seconds}s")

    try:
        # client.download_job_results is synchronous, run in a thread with timeout
        await asyncio.wait_for(
            asyncio.to_thread(client.download_job_results, ds_job_id, results_dir),
            timeout=get_settings().datasphere.client_download_timeout_seconds
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
            except zipfile.BadZipFile as e:
                logger.error(f"[{job_id}] Invalid zip file format for output.zip: {e}")
                # Continue processing in case individual files exist
            except Exception as e:
                logger.error(f"[{job_id}] Error extracting output.zip: {e}", exc_info=True)
                # Continue processing in case individual files exist

        # Check for model, predictions, and metrics files
        model_path = os.path.join(results_dir, "model.onnx")
        predictions_path = os.path.join(results_dir, "predictions.csv")
        metrics_path = os.path.join(results_dir, "metrics.json")

        if not os.path.exists(model_path):
            logger.warning(f"[{job_id}] Model file 'model.onnx' not found at {model_path}. Model will not be saved.")
            model_path = None

        if not os.path.exists(predictions_path):
            logger.warning(f"[{job_id}] Predictions file 'predictions.csv' not found at {predictions_path}. Predictions will not be saved.")
            predictions_path = None

        if not os.path.exists(metrics_path):
            logger.warning(f"[{job_id}] Metrics file 'metrics.json' not found at {metrics_path}. Metrics will not be saved.")
            metrics_data = None
        else:
            try:
                with open(metrics_path) as f:
                    metrics_data = json.load(f)
                logger.info(f"[{job_id}] Loaded metrics from {metrics_path}")
            except json.JSONDecodeError as e:
                logger.error(f"[{job_id}] Failed to parse metrics.json from {metrics_path}: {e}", exc_info=True)
                metrics_data = None # Ensure it's None if parsing fails
                # Optionally, this could be a critical error causing the job to fail

        return metrics_data, predictions_path, model_path
    except asyncio.TimeoutError:
        error_msg = f"DataSphere job results download timed out after {get_settings().datasphere.client_download_timeout_seconds} seconds for DS Job ID: {ds_job_id}."
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
                pass  # Debug logging removed
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
    logger.info(f"[{job_id}] Downloading logs/diagnostics for {status_str} DS Job {ds_job_id} to {logs_dir}. Timeout: {get_settings().datasphere.client_download_timeout_seconds}s")

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
            timeout=get_settings().datasphere.client_download_timeout_seconds
        )
        logger.info(f"[{job_id}] Logs/diagnostics for {status_str} DS Job {ds_job_id} downloaded to {logs_dir}")
    except asyncio.TimeoutError:
        # This is generally a non-critical error for the job's main outcome, so log and continue.
        error_msg = f"DataSphere logs/diagnostics download timed out after {get_settings().datasphere.client_download_timeout_seconds} seconds for DS Job ID: {ds_job_id}."
        logger.warning(f"[{job_id}] {error_msg}")
    except Exception as dl_exc:
        # This is generally a non-critical error for the job's main outcome, so log and continue.
        logger.warning(f"[{job_id}] Failed to download logs/diagnostics for DS Job {ds_job_id}: {dl_exc}", exc_info=True)


async def _process_datasphere_job(
    job_id: str,
    client: DataSphereClient,
    ds_job_specific_output_base_dir: str,
    ready_config_path: str,
    work_dir: str
) -> tuple[str, str, dict[str, Any] | None, str | None, str | None, int]:
    """
    Coordinates the DataSphere job lifecycle including submission, monitoring, and result fetching.
    
    Args:
        job_id: ID of the job
        client: DataSphere client
        ds_job_specific_output_base_dir: Directory for this run's outputs/results
        ready_config_path: Path to the ready DataSphere config file
        work_dir: Working directory for temporary files
        
    Returns:
        Tuple of (ds_job_id, results_dir, metrics_data, model_path, predictions_path, polls)
    """
    logger.info(f"[{job_id}] Stage 5: Processing DataSphere job...")

    # Submit the job
    ds_job_id = await _submit_datasphere_job(job_id, client, ready_config_path, work_dir)

    completed = False
    settings = get_settings()
    max_polls = settings.datasphere.max_polls
    poll_interval = settings.datasphere.poll_interval
    polls = 0
    metrics_data = None
    predictions_path = None
    model_path = None

    logger.info(f"[{job_id}] Polling DS Job {ds_job_id} status (max_polls={max_polls}, poll_interval={poll_interval}s).")

    # Monitor job status
    while not completed and polls < max_polls:
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

        if current_ds_status in ["completed", "success"]:  # "success" kept for backward-compat
            completed = True
            logger.info(f"[{job_id}] DS Job {ds_job_id} completed. Downloading results...")

            # Download and process results
            metrics_data, predictions_path, model_path = await _download_datasphere_job_results(
                job_id, ds_job_id, client, ds_job_specific_output_base_dir
            )

            # Download logs and diagnostics if configured
            download_diag_on_success = getattr(get_settings().datasphere, 'download_diagnostics_on_success', False)
            if download_diag_on_success:
                logs_dir = os.path.join(ds_job_specific_output_base_dir, "logs_diagnostics_success")
                await _download_logs_diagnostics(job_id, ds_job_id, client, logs_dir, is_success=True)
            else:
                logger.info(f"[{job_id}] Skipping optional download of logs/diagnostics for successful job based on settings.")

        elif current_ds_status in ["failed", "cancelled", "cancelling"]:
            error_detail = f"DS Job {ds_job_id} ended with status: {current_ds_status_str}."
            logger.error(f"[{job_id}] {error_detail}")

            # Download logs for failed job
            logs_dir = os.path.join(ds_job_specific_output_base_dir, "logs_diagnostics")
            await _download_logs_diagnostics(job_id, ds_job_id, client, logs_dir, is_success=False)
            error_detail += f" Logs/diagnostics may be available in {logs_dir}."

            update_job_status(job_id, JobStatus.FAILED.value, error_message=error_detail)
            raise RuntimeError(error_detail)

    if not completed:
        timeout_message = f"DS Job {ds_job_id} execution timed out after {polls} polls ({max_polls * poll_interval}s)."
        logger.error(f"[{job_id}] {timeout_message}")
        update_job_status(job_id, status=JobStatus.FAILED.value, error_message=str(timeout_message))
        raise TimeoutError(timeout_message)

    return ds_job_id, ds_job_specific_output_base_dir, metrics_data, model_path, predictions_path, polls


async def _perform_model_cleanup(job_id: str, current_model_id: str) -> None:
    """Prunes old, non-active models based on settings."""
    try:
        num_models_to_keep = getattr(get_settings(), "max_models_to_keep", 5)
        if num_models_to_keep > 0:
            logger.info(f"[{job_id}] Checking for old models to prune (keeping last {num_models_to_keep})...")

            # Get IDs of models to keep (most recent ones, including potentially the current one if it becomes active later)
            # Note: The current model (current_model_id) is initially inactive.
            recent_kept_models_info = get_recent_models(limit=num_models_to_keep)
            kept_model_ids = {m['model_id'] for m in recent_kept_models_info} # Use a set for faster lookups

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
    metrics_data: dict[str, Any] | None,
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
            logger.warning(f"[{job_id}] Model path is None or empty. Skipping model and prediction saving.")
            warnings_list.append("Model file not found in results.")

        # Create a record of the training result with metrics
        training_result_id = None
        if metrics_data:
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





async def _cleanup_directories(job_id: str, dir_paths: list[str]) -> list[str]:
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

        # Calculate prediction month from job parameters
        prediction_month = None
        try:
            job_details = get_job(job_id, connection=direct_db_connection)
            if job_details and job_details.get('parameters'):
                params = json.loads(job_details['parameters'])
                dataset_end_date_str = params.get('dataset_end_date')

                if dataset_end_date_str:
                    # Parse dataset_end_date and add 1 month
                    dataset_end_date = datetime.fromisoformat(dataset_end_date_str.replace('Z', '+00:00'))
                    if dataset_end_date.month == 12:
                        prediction_month = dataset_end_date.replace(year=dataset_end_date.year + 1, month=1, day=1)
                    else:
                        prediction_month = dataset_end_date.replace(month=dataset_end_date.month + 1, day=1)

                    logger.info(f"Calculated prediction month: {prediction_month.strftime('%Y-%m')} for job {job_id}")
        except Exception as e:
            logger.warning(f"Could not calculate prediction month for job {job_id}: {e}")

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
                (result_id, job_id, model_id, prediction_date, prediction_month, output_path, summary_metrics) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (result_id, job_id, model_id, timestamp,
                 prediction_month.date().isoformat() if prediction_month else None,
                 predictions_path, "{}")
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
                    INSERT OR REPLACE INTO fact_predictions
                    (result_id, multiindex_id, prediction_date, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    predictions_data
                )
            except Exception:
                # Если прямой вызов не сработал, используем execute_many
                execute_many(
                    """
                    INSERT OR REPLACE INTO fact_predictions
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
async def run_job(job_id: str, training_config: dict, config_id: str, dataset_start_date: str = None, dataset_end_date: str = None) -> dict[str, Any] | None:
    """
    Runs a DataSphere training job pipeline: setup, execution, monitoring, result processing, cleanup.
    """
    ds_job_id: str | None = None
    client: DataSphereClient | None = None
    project_input_link_path: str | None = None
    job_completed_successfully = False

    try:
        with tempfile.TemporaryDirectory(dir=str(get_settings().datasphere_input_dir)) as temp_input_dir_str, \
             tempfile.TemporaryDirectory(dir=str(get_settings().datasphere_output_dir)) as temp_output_dir_str:
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
                static_config_path = await _prepare_datasphere_job_submission(job_id, config, temp_input_dir)

                # Stage 4a.1: Verify DataSphere job inputs
                await _verify_datasphere_job_inputs(job_id, temp_input_dir)

                # Stage 4b: Archive Input Directory
                archive_path = await _archive_input_directory(job_id, temp_input_dir_str, temp_input_dir_str)

                # Stage 4c: Create Project Link
                project_input_link_path = create_project_input_link(
                    archive_path,
                    str(get_settings().datasphere_job_dir)
                )
                logger.info(f"[{job_id}] Created project input link: {project_input_link_path}")
                update_job_status(job_id, JobStatus.RUNNING.value, progress=24, status_message="Project input link created.")

                # Stage 5: Submit and Monitor DS Job
                ds_job_id, results_dir_from_process, metrics_data, model_path, predictions_path, polls = await _process_datasphere_job(
                    job_id,
                    client,
                    temp_output_dir,
                    static_config_path,
                    temp_input_dir_str  # Pass temp_input_dir as work_dir for local modules
                )

                # Stage 6: Process Results
                await _process_job_results(
                    job_id, ds_job_id, results_dir_from_process, config, metrics_data, model_path, predictions_path, polls, get_settings().datasphere.poll_interval, config_id
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
                            timeout=get_settings().datasphere.client_cancel_timeout_seconds
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
                # Cleanup project input link regardless of success/failure
                if project_input_link_path:
                    cleanup_project_input_link(str(get_settings().datasphere_job_dir))

                # The temporary directories are cleaned up automatically by the context manager.
                # Final log message for job run completion.
                current_ds_job_id_log = ds_job_id if 'ds_job_id' in locals() and ds_job_id is not None else "N/A"
                logger.info(f"[{job_id}] Job run processing finished. DS Job ID: {current_ds_job_id_log}. Temporary directories and project links cleaned up automatically.")

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
    metrics_data: dict[str, Any] | None
) -> str:
    """
    Saves the model file to permanent storage and creates a database record.
    
    Args:
        job_id: The job ID that produced this model
        model_path: Path to the temporary model file (downloaded from DataSphere)
        ds_job_id: DataSphere job ID that produced this model
        config: Training configuration used
        metrics_data: Optional metrics data from training
        
    Returns:
        The generated model ID
        
    Raises:
        RuntimeError: If model file cannot be copied or database record cannot be created
    """
    try:
        # Generate a unique model ID based on the config's model_id (base name)
        model_id = f"{config.model_id}_{uuid.uuid4().hex[:8]}"
        logger.info(f"[{job_id}] Generated new model ID: {model_id}")

        # Verify source model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Source model file not found: {model_path}")

        # Create permanent storage path
        permanent_model_filename = f"{model_id}.onnx"
        permanent_model_path = os.path.join(get_settings().models_dir, permanent_model_filename)

        logger.info(f"[{job_id}] Copying model from temporary location '{model_path}' to permanent storage '{permanent_model_path}'")

        # Copy model file to permanent storage
        try:
            shutil.copy2(model_path, permanent_model_path)
            logger.info(f"[{job_id}] Model file successfully copied to permanent storage")
        except Exception as copy_error:
            raise RuntimeError(f"Failed to copy model file to permanent storage: {copy_error}") from copy_error

        # Verify the copy was successful
        if not os.path.exists(permanent_model_path):
            raise RuntimeError(f"Model file copy verification failed - file not found at destination: {permanent_model_path}")

        # Get file size from the permanent location
        file_size = os.path.getsize(permanent_model_path)

        # Construct metadata
        metadata = {
            "file_size_bytes": file_size,
            "downloaded_from_ds_job": ds_job_id,
            "original_temp_path": model_path,  # Keep reference to original temp path for debugging
            "permanent_storage_path": permanent_model_path,
            "config_model_id_base": config.model_id, # Store the base name for tracking
            "metrics": metrics_data or {} # Store metrics with the model
        }

        # Create a record in the database with the permanent path
        create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path=str(permanent_model_path),  # Save the permanent path
            created_at=datetime.now(),
            metadata=metadata
        )
        logger.info(f"[{job_id}] Successfully created model record in DB for model_id: {model_id} with permanent path: {permanent_model_path}")
        return model_id

    except Exception as e:
        error_msg = f"Failed to save model file and create DB record: {e}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        # We don't update job status here, the caller should handle it
        raise RuntimeError(error_msg) from e
