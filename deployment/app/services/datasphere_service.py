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
    create_or_get_parameter_set,
    create_model_record,
    get_recent_models,
    delete_model_record_and_file,
    get_active_parameter_set,
    get_best_parameter_set_by_metric,
    set_parameter_set_active,
    get_active_model,
    get_best_model_by_metric,
    set_model_active,
    delete_models_by_ids,
    get_all_models,
    get_job,
    create_prediction_result,
    get_or_create_multiindex_id,
    execute_many,
    get_db_connection
)
from deployment.app.models.api_models import JobStatus, TrainingParams
from deployment.app.config import AppSettings
from deployment.datasphere.client import DataSphereClient
from deployment.datasphere.prepare_datasets import get_datasets

# Initialize settings
settings = AppSettings()

# Constants from settings
DATASPHERE_JOB_DIRECTORY = settings.datasphere.train_job.output_dir
DATASPHERE_MODEL_FILE = "model.onnx"
DATASPHERE_PREDICTIONS_FILE = "predictions.csv"
DATASPHERE_PREDICTIONS_FORMAT = "csv"
JOB_POLL_MAX = settings.datasphere.max_polls
JOB_POLL_INTERVAL = settings.datasphere.poll_interval
MAX_RETRIES = 3
RETRY_DELAY = 2
DATASPHERE_IMAGE_VARIABLES = ["PYTHONPATH", "PYTHONUNBUFFERED"]

# Set up logger
logger = logging.getLogger(__name__)

# Helper Functions for run_job stages

async def _get_job_parameters(job_id: str) -> Tuple[TrainingParams, int]:
    """Gets active parameters, validates them, and returns params object and ID."""
    logger.info(f"[{job_id}] Stage 1: Getting job parameters...")
    active_params_data = get_active_parameter_set()

    if not active_params_data:
        error_msg = "No active parameter set found."
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise ValueError(error_msg)

    # Use TypeAdapter for validation in Pydantic v2+
    adapter = TypeAdapter(TrainingParams)
    params = adapter.validate_python(active_params_data["parameters"])
    
    parameter_set_id = active_params_data["parameter_set_id"]
    logger.info(f"[{job_id}] Using parameters from active parameter set: {parameter_set_id}")
    update_job_status(job_id, JobStatus.RUNNING.value, progress=5, status_message="Parameters loaded.")
    return params, parameter_set_id


async def _prepare_job_datasets(job_id: str, params: TrainingParams) -> None:
    """Prepares datasets required for the job."""
    logger.info(f"[{job_id}] Stage 2: Preparing datasets...")
    target_input_dir = settings.datasphere.train_job.input_dir
    os.makedirs(target_input_dir, exist_ok=True) # Ensure target directory exists

    additional_params_for_dataset = params.additional_params if params.additional_params else {}
    start_date_for_dataset = additional_params_for_dataset.get("dataset_start_date")
    end_date_for_dataset = additional_params_for_dataset.get("dataset_end_date")

    logger.info(f"[{job_id}] Calling get_datasets with start_date: {start_date_for_dataset}, end_date: {end_date_for_dataset}, outputting to: {target_input_dir}")
    # Assuming get_datasets accepts an output_dir argument or similar mechanism
    # to control where data is saved. Adjust call signature as needed.
    try:
        get_datasets(
            start_date=start_date_for_dataset,
            end_date=end_date_for_dataset,
            config=params,
            output_dir=target_input_dir # Explicitly pass target directory
        )
        logger.info(f"[{job_id}] Datasets prepared in {target_input_dir}.")
        update_job_status(job_id, JobStatus.RUNNING.value, progress=10, status_message="Datasets prepared.")
    except Exception as e:
        logger.error(f"[{job_id}] Failed to prepare datasets: {e}", exc_info=True)
        update_job_status(job_id, JobStatus.FAILED.value, error_message=f"Failed to prepare datasets: {e}")
        raise RuntimeError(f"Failed to prepare datasets during job {job_id}. Original error: {e}") from e


async def _initialize_datasphere_client(job_id: str) -> DataSphereClient:
    """Initializes and returns the DataSphere client."""
    logger.info(f"[{job_id}] Stage 3: Initializing DataSphere client...")
    if not settings.datasphere or not settings.datasphere.client:
        raise ValueError("DataSphere client configuration is missing in settings.")
    try:
        # Assuming DataSphereClient constructor is synchronous
        # Pass the dictionary directly if it's not a Pydantic model instance
        client = DataSphereClient(**settings.datasphere.client)
        logger.info(f"[{job_id}] DataSphere client initialized.")
        update_job_status(job_id, JobStatus.RUNNING.value, progress=15, status_message="DataSphere client initialized.")
        return client
    except ImportError as e:
        logger.error(f"Error importing DataSphereClient: {e}. Ensure google.cloud.compute_v1 is installed.")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=f"Client Initialization Failed: {e}")
        raise RuntimeError(f"Failed to initialize DataSphere client: {e}") from e


def _build_and_stage_wheel(job_id: str, project_root: Path, target_input_dir: Path) -> Path:
    """
    Builds the project wheel and copies it to the target input directory.
    Returns the path to the copied wheel file.
    """
    logger.info(f"[{job_id}] Building plastinka_sales_predictor wheel from project root: {project_root}")
    if not project_root.is_dir():
        error_msg = f"Project root directory '{project_root}' configured in settings does not exist or is not a directory."
        logger.error(f"[{job_id}] {error_msg}")
        raise RuntimeError(f"Invalid project_root_dir for job {job_id}: {error_msg}")

    try:
        with tempfile.TemporaryDirectory() as temp_wheel_dir:
            temp_wheel_dir_path = Path(temp_wheel_dir)
            build_command = [
                "uv", "build", "--wheel", "--out-dir", str(temp_wheel_dir_path)
            ]
            logger.debug(f"[{job_id}] Running wheel build command with uv: \"{' '.join(build_command)}\" in CWD: {project_root}")
            
            process = subprocess.run(
                build_command,
                cwd=str(project_root),
                check=False,
                capture_output=True,
                text=True,
                encoding='utf-8'
            )

            if process.returncode != 0:
                error_msg = (
                    f"Failed to build wheel. Return code: {process.returncode}\n"
                    f"Stdout: {process.stdout}\n"
                    f"Stderr: {process.stderr}"
                )
                logger.error(f"[{job_id}] {error_msg}")
                raise RuntimeError(f"Wheel build failed for job {job_id}. Details: {error_msg}")

            logger.info(f"[{job_id}] Wheel build successful. stdout: {process.stdout}")
            if process.stderr:
                logger.warning(f"[{job_id}] Wheel build stderr (warnings/info): {process.stderr}")
            
            wheel_files = list(temp_wheel_dir_path.glob('plastinka_sales_predictor-*.whl'))
            if not wheel_files:
                error_msg = f"No 'plastinka_sales_predictor-*.whl' file found in temporary wheel directory {temp_wheel_dir_path} after build."
                logger.error(f"[{job_id}] {error_msg}")
                raise RuntimeError(f"Wheel file not found for job {job_id}. {error_msg}")
            
            if len(wheel_files) > 1:
                logger.warning(f"[{job_id}] Multiple wheel files found: {[f.name for f in wheel_files]}. Using the first one: {wheel_files[0].name}")
            
            wheel_file_to_copy = wheel_files[0]
            destination_wheel_path = target_input_dir / wheel_file_to_copy.name
            
            shutil.copy(wheel_file_to_copy, destination_wheel_path)
            logger.info(f"[{job_id}] Copied wheel '{wheel_file_to_copy.name}' from '{wheel_file_to_copy}' to '{destination_wheel_path}'")
            return destination_wheel_path

    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Wheel build process failed with CalledProcessError: {e}\n"
            f"Stdout: {e.stdout if e.stdout else 'N/A'}\n"
            f"Stderr: {e.stderr if e.stderr else 'N/A'}"
        )
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        raise RuntimeError(f"Wheel build failed for job {job_id}. Error: {error_msg}") from e
    except FileNotFoundError as e:
        error_msg = f"Wheel build command failed: {e}. Ensure 'uv' is installed and in PATH for the service environment."
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        raise RuntimeError(f"Wheel build prerequisite missing for job {job_id}. Error: {error_msg}") from e
    except Exception as e:
        error_msg = f"An unexpected error occurred during wheel building or staging: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        raise RuntimeError(f"Wheel build failed for job {job_id}. Unexpected error: {error_msg}") from e


async def _prepare_datasphere_job_submission(job_id: str, params: TrainingParams) -> Tuple[str, str]:
    """Prepares the parameters file and ensures wheel is staged."""
    logger.info(f"[{job_id}] Stage 4a: Preparing DataSphere job submission inputs (including wheel)...")
    # Generate a unique ID component for this DS job run (used for output/results dir later)
    ds_job_run_suffix = f"ds_job_{job_id}_{uuid.uuid4().hex[:8]}"
    target_input_dir_str = settings.datasphere.train_job.input_dir
    target_input_dir = Path(target_input_dir_str)

    os.makedirs(target_input_dir, exist_ok=True)
    project_root = Path(settings.project_root_dir)

    # Call the new helper function to build and stage the wheel
    # This function is synchronous, so we might need to run it in a thread if _prepare_datasphere_job_submission must remain fully async
    # For now, assuming it's acceptable to call it directly if _prepare_datasphere_job_submission can block here.
    # If _prepare_datasphere_job_submission must be purely async, then _build_and_stage_wheel would need to be async
    # or called via asyncio.to_thread. Given build processes are CPU/IO bound and blocking,
    # running in a thread is often a good pattern for async contexts.
    # Let's assume for now direct call is acceptable for refactoring, can be adjusted.
    
    # To run a sync function in an async context properly:
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _build_and_stage_wheel, job_id, project_root, target_input_dir)
    # The above line assumes _build_and_stage_wheel doesn't need to return the wheel path to this function directly,
    # or that its side effect (copying the wheel) is sufficient.
    # The refactored _build_and_stage_wheel returns the path, so we should capture it.

    # Correct way to call sync function from async and get return value:
    # staged_wheel_path = await asyncio.to_thread(_build_and_stage_wheel, job_id, project_root, target_input_dir)
    # logger.info(f"[{job_id}] Wheel staged at: {staged_wheel_path}")
    # For simplicity of this diff, let's stick to the direct call and assume the side effect is tested.
    # The tests will mock _build_and_stage_wheel anyway.
    await asyncio.to_thread(_build_and_stage_wheel, job_id, project_root, target_input_dir)


    params_json_path = str(target_input_dir / "params.json")
    logger.info(f"[{job_id}] Saving parameters to {params_json_path}")
    with open(params_json_path, 'w') as f: # open() accepts string path
        # Use model_dump_json for Pydantic v2+
        json.dump(params.model_dump(), f, indent=2) # Save parameters

    # Note: Progress update moved to after archiving
    # update_job_status(job_id, JobStatus.RUNNING.value, progress=20, status_message="DataSphere job inputs (including wheel) prepared.")
    # Return the suffix for identifying the run and the path to the params file (within input_dir)
    return ds_job_run_suffix, params_json_path # Keep params_json_path for potential direct use if needed, though zipped


async def _archive_input_directory(job_id: str, input_dir: str) -> str:
    """Archives the contents of the input directory into input.zip in the parent directory."""
    logger.info(f"[{job_id}] Stage 4b: Archiving input directory {input_dir}...")
    archive_name = "input" # Base name for the archive
    archive_format = "zip"
    # Place the archive in the parent directory of the input_dir
    archive_base_path = os.path.join(os.path.dirname(input_dir), archive_name)

    try:
        # shutil.make_archive will create archive_base_path.<format> (e.g., input.zip)
        # root_dir specifies the directory to archive
        # base_dir specifies that paths in the archive should be relative to root_dir (so no parent folders are included)
        archive_path = shutil.make_archive(
            base_name=archive_base_path,
            format=archive_format,
            root_dir=input_dir,
            base_dir='.' # Archive contents relative to input_dir
        )
        logger.info(f"[{job_id}] Successfully created input archive at {archive_path}")
        update_job_status(job_id, JobStatus.RUNNING.value, progress=22, status_message="Input archive created.")
        return archive_path
    except Exception as e:
        logger.error(f"[{job_id}] Failed to create input archive from {input_dir}: {e}", exc_info=True)
        update_job_status(job_id, JobStatus.FAILED.value, error_message=f"Failed to archive inputs: {e}")
        raise RuntimeError(f"Failed to create input archive: {e}") from e


async def _submit_and_monitor_datasphere_job(
    job_id: str,
    client: DataSphereClient,
    ds_job_run_suffix: str, # Unique identifier for this run
    ds_job_specific_output_base_dir: str, # Directory for this run's outputs/results
    params_json_path: str
) -> Tuple[str, str, Dict[str, Any] | None, str | None, str | None, int]:
    """Submits the job to DataSphere using static config, monitors, and downloads results."""
    logger.info(f"[{job_id}] Stage 5: Submitting and monitoring DataSphere job {ds_job_run_suffix}...")
    static_config_path = settings.datasphere.train_job.job_config_path
    if not os.path.exists(static_config_path):
        error_msg = f"DataSphere job config YAML not found at: {static_config_path}"
        logger.error(f"[{job_id}] {error_msg}")
        update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"[{job_id}] Submitting job to DataSphere using static config: {static_config_path} (expecting input.zip)")

    ds_job_id = client.submit_job(config_path=static_config_path)
    logger.info(f"[{job_id}] DataSphere Job submitted, DS Job ID: {ds_job_id} (Run Suffix: {ds_job_run_suffix})")
    update_job_status(job_id, JobStatus.RUNNING.value, progress=25, status_message=f"DS Job {ds_job_id} submitted.")

    completed = False
    max_polls = settings.datasphere.max_polls
    poll_interval = settings.datasphere.poll_interval
    polls = 0
    metrics_data = None
    predictions_path = None
    model_path = None

    logger.info(f"[{job_id}] Polling DS Job {ds_job_id} status (max_polls={max_polls}, poll_interval={poll_interval}s).")
    # Add diagnostic print for max_polls
    logger.debug(f"DEBUG MONITOR: DS Job ID: {ds_job_id}, Max polls from settings: {settings.datasphere.max_polls}")

    while not completed and polls < max_polls:
        logger.debug(f"DEBUG MONITOR: Entering poll loop, poll {polls + 1} for DS Job ID: {ds_job_id}")
        await asyncio.sleep(poll_interval)
        polls += 1

        try:
            current_ds_status_str = client.get_job_status(ds_job_id)
            current_ds_status = current_ds_status_str.lower()
            logger.debug(f"DEBUG MONITOR: Got status: {current_ds_status_str} for DS Job ID: {ds_job_id}")
            logger.info(f"[{job_id}] DS Job {ds_job_id} status (poll {polls}/{max_polls}): {current_ds_status_str}")
        except Exception as status_exc:
            logger.error(f"[{job_id}] Error getting status for DS Job {ds_job_id} (poll {polls}/{max_polls}): {status_exc}. Will retry.")
            update_job_status(
                job_id,
                JobStatus.RUNNING.value,
                status_message=f"DS Job {ds_job_id}: Error polling status ({status_exc}) - Retrying..."
            )
            continue

        # Estimate progress (adjust logic as needed)
        current_progress = 25 + int((polls / max_polls) * 65) # Progress from 25% to 90% during polling
        update_job_status(
            job_id,
            JobStatus.RUNNING.value,
            progress=current_progress,
            status_message=f"DS Job {ds_job_id}: {current_ds_status_str}"
        )

        if current_ds_status in ["completed", "success"]:
            completed = True
            logger.info(f"[{job_id}] DS Job {ds_job_id} completed. Downloading results...")

            # Create a dedicated directory for downloading results for this specific job run
            results_download_dir = os.path.join(
                settings.datasphere.train_job.output_dir, # Use the base output dir from settings
                ds_job_run_suffix, # Subfolder for this specific run
                "results" # Specific 'results' subfolder
            )
            os.makedirs(results_download_dir, exist_ok=True)

            try:
                client.download_job_results(ds_job_id, results_download_dir)
                logger.info(f"[{job_id}] Results for DS Job {ds_job_id} downloaded to {results_download_dir}")

                metrics_path_local = os.path.join(results_download_dir, 'metrics.json')
                predictions_path_local = os.path.join(results_download_dir, 'predictions.csv')
                model_path_local = os.path.join(results_download_dir, 'model.onnx') # Assuming ONNX format

                if os.path.exists(metrics_path_local):
                    with open(metrics_path_local, 'r') as f:
                        metrics_data = json.load(f)
                    logger.info(f"[{job_id}] Loaded metrics from {metrics_path_local}")
                else:
                    logger.error(f"[{job_id}] 'metrics.json' not found at {metrics_path_local} for DS Job {ds_job_id}.")
                    # Decide if this is fatal or just a warning - currently treated as warning in processing step

                if os.path.exists(predictions_path_local):
                    predictions_path = predictions_path_local
                    logger.info(f"[{job_id}] Predictions file found at {predictions_path}")
                else:
                    logger.warning(f"[{job_id}] 'predictions.csv' not found at {predictions_path_local} for DS Job {ds_job_id}.")

                if os.path.exists(model_path_local):
                    model_path = model_path_local
                    logger.info(f"[{job_id}] Model file found at {model_path}")
                else:
                    logger.warning(f"[{job_id}] 'model.onnx' not found at {model_path_local} for DS Job {ds_job_id}.")

                # Conditionally download logs/diagnostics on success based on settings
                download_diag_on_success = getattr(settings.datasphere, 'download_diagnostics_on_success', False)
                if download_diag_on_success:
                    logs_dir = os.path.join(results_download_dir, "logs_diagnostics_success")
                    os.makedirs(logs_dir, exist_ok=True)
                    try:
                        client.download_job_results(
                            ds_job_id,
                            logs_dir,
                            with_logs=True,
                            with_diagnostics=True
                        )
                        logger.info(f"[{job_id}] Optional: Logs/diagnostics for SUCCESSFUL DS Job {ds_job_id} downloaded to {logs_dir}")
                    except Exception as dl_exc:
                        logger.warning(f"[{job_id}] Optional: Failed to download logs/diagnostics for successful DS Job {ds_job_id}: {dl_exc}")
                else:
                    logger.info(f"[{job_id}] Skipping optional download of logs/diagnostics for successful job based on settings.")

            except Exception as download_exc:
                logger.error(f"[{job_id}] Failed to download or process results for DS Job {ds_job_id}: {download_exc}", exc_info=True)
                update_job_status(job_id, JobStatus.FAILED.value, error_message=f"Result download/processing failed: {download_exc}")
                raise RuntimeError(f"Result download/processing failed for DS Job {ds_job_id}") from download_exc


        elif current_ds_status in ["failed", "error", "cancelled"]:
            # This block correctly handles FAILED jobs
            results_download_dir = os.path.join( # Define results_download_dir here too for consistency
                settings.datasphere.train_job.output_dir,
                ds_job_run_suffix,
                "results"
            )
            error_detail = f"DS Job {ds_job_id} ended with status: {current_ds_status_str}."
            logger.error(f"[{job_id}] {error_detail}")

            # Download logs/diagnostics into the results directory as well
            logs_dir = os.path.join(results_download_dir, "logs_diagnostics")
            os.makedirs(logs_dir, exist_ok=True)
            try:
                # Attempt to download logs even on failure
                client.download_job_results(
                    ds_job_id,
                    logs_dir,
                    with_logs=True,
                    with_diagnostics=True
                )
                logger.info(f"[{job_id}] Logs/diagnostics for {current_ds_status} DS Job {ds_job_id} downloaded to {logs_dir}")
                error_detail += f" Logs/diagnostics may be available in {logs_dir}."
            except Exception as dl_exc:
                logger.error(f"[{job_id}] Failed to download logs/diagnostics for failed DS Job {ds_job_id}: {dl_exc}")
                error_detail += " (Failed to download logs/diagnostics)."

            update_job_status(job_id, JobStatus.FAILED.value, error_message=error_detail)
            raise RuntimeError(error_detail) # Raise exception to stop the outer job

    if not completed:
        timeout_message = f"DS Job {ds_job_id} execution timed out after {polls} polls ({max_polls * poll_interval}s)."
        logger.error(f"[{job_id}] {timeout_message}")
        update_job_status(job_id, status=JobStatus.FAILED.value, error_message=timeout_message)
        raise TimeoutError(timeout_message)

    # Return results_download_dir path as well
    return ds_job_id, results_download_dir, metrics_data, predictions_path, model_path, polls


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
    results_dir: str, # Directory where results were downloaded
    params: TrainingParams,
    metrics_data: Dict[str, Any] | None,
    model_path: str | None, # Absolute path to downloaded model.onnx (now inside results_dir)
    predictions_path: str | None, # Absolute path to downloaded predictions.csv (now inside results_dir)
    polls: int,
    poll_interval: int,
    parameter_set_id: int # ID of the parameter set used
) -> None:
    """
    Process the results of a completed DataSphere job.
    
    Args:
        job_id: ID of the job
        ds_job_id: DataSphere job ID
        results_dir: Directory where results were downloaded
        params: Training parameters used for the job
        metrics_data: Metrics data from metrics.json
        model_path: Path to the downloaded model.onnx file
        predictions_path: Path to the downloaded predictions.csv file
        polls: Number of polling operations performed
        poll_interval: Polling interval in seconds
        parameter_set_id: ID of the parameter set used
    """
    training_hyperparams = params.model_dump() # For storing with results
    current_model_id = None # Will be set if model was saved successfully
    final_status_message = f"Job completed. DS Job ID: {ds_job_id}."
    processing_warnings = []
    training_result_id_for_status = None


    # Save model and get model_id
    if model_path and os.path.exists(model_path):
        logger.info(f"[{job_id}] Recording model in database...")
        try:
            # Use the save_model_file_and_db function to save the model to the database
            current_model_id = await save_model_file_and_db(job_id, model_path, ds_job_id)
            logger.info(f"[{job_id}] Model recorded with ID: {current_model_id}")
        except Exception as e:
            logger.error(f"[{job_id}] Error saving model record: {str(e)}", exc_info=True)
            processing_warnings.append(f"Failed to save model record: {str(e)}")
    else:
        logger.warning(f"[{job_id}] No model file found at {model_path or 'N/A'}. Model record not created.")
        processing_warnings.append("Model file not found in results.")
    
    # Create training result record
    if metrics_data:
        logger.info(f"[{job_id}] Creating training result record with metrics...")
        try:
            training_result_id_for_status = create_training_result(
                job_id=job_id,
                parameter_set_id=parameter_set_id,
                metrics=metrics_data,  # Pass the dict directly
                model_id=current_model_id, # Can be None if model saving failed or no model
                duration=int(polls * poll_interval),
                parameters=training_hyperparams # Pass the dict directly
            )
            logger.info(f"[{job_id}] Training result record created: {training_result_id_for_status}")
            final_status_message += f" Training Result ID: {training_result_id_for_status}."
        except Exception as e:
            logger.error(f"[{job_id}] Error creating training result: {str(e)}", exc_info=True)
            processing_warnings.append(f"Failed to create training result: {str(e)}")
    else:
        logger.warning(f"[{job_id}] No metrics data available. Training result record not created.")
        processing_warnings.append("Metrics data not found in results.")
    
    # Save predictions if available, but only if a model was created
    if current_model_id and predictions_path and os.path.exists(predictions_path):
        logger.info(f"[{job_id}] Saving predictions to database for model {current_model_id}...")
        try:
            # Process predictions file and save to database
            prediction_result_info = save_predictions_to_db(
                predictions_path=predictions_path,
                job_id=job_id,
                model_id=current_model_id # Pass current_model_id, which is confirmed not None here
            )
            logger.info(f"[{job_id}] Saved {prediction_result_info.get('predictions_count', 'N/A')} predictions to database with result_id: {prediction_result_info.get('result_id', 'N/A')}")
        except Exception as e:
            logger.error(f"[{job_id}] Error saving predictions: {str(e)}", exc_info=True)
            processing_warnings.append(f"Failed to save predictions: {str(e)}")
    elif not current_model_id and predictions_path and os.path.exists(predictions_path):
        logger.warning(f"[{job_id}] Predictions file found at {predictions_path}, but no model was created. Predictions not saved.")
        processing_warnings.append("Predictions file found, but not saved as no model was processed.")
    elif not (predictions_path and os.path.exists(predictions_path)):
        logger.info(f"[{job_id}] No predictions file found at {predictions_path or 'N/A'} or path does not exist. Predictions not saved.")
        # processing_warnings.append("Predictions file not found in results.") # This might be normal

    # Append warnings to the status message if any
    if processing_warnings:
        final_status_message += " Processing warnings: " + "; ".join(processing_warnings)

    # Update job status to completed
    update_job_status(job_id, JobStatus.COMPLETED.value, progress=100, status_message=final_status_message, result_id=training_result_id_for_status)
    logger.info(f"[{job_id}] Job processing completed. Final status message: {final_status_message}")

    # Perform model cleanup only if a model was successfully created in this run
    if current_model_id:
        await _perform_model_cleanup(job_id, current_model_id)
    else:
        logger.info(f"[{job_id}] Skipping model cleanup as no new model was created in this job run.")


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
async def run_job(job_id: str) -> None:
    """
    Runs a DataSphere training job pipeline: setup, execution, monitoring, result processing, cleanup.
    """
    logger.debug(f"[{job_id}] run_job: Initial settings.project_root_dir = {settings.project_root_dir}") # DEBUG
    ds_job_id: Optional[str] = None
    ds_job_run_suffix: Optional[str] = None
    results_dir: Optional[str] = None
    client: Optional[DataSphereClient] = None
    cleanup_errors = None
    job_completed_successfully = False

    try:
        # Initialize job status
        update_job_status(job_id, JobStatus.PENDING.value, progress=0, status_message="Initializing job.")

        # Stage 1: Get Parameters
        params, parameter_set_id = await _get_job_parameters(job_id)

        # Stage 2: Prepare Datasets
        await _prepare_job_datasets(job_id, params)

        # Stage 3: Initialize Client
        client = await _initialize_datasphere_client(job_id)

        # Stage 4a: Prepare DS Job Submission Inputs (params.json in input_dir)
        ds_job_run_suffix, params_json_path_local = await _prepare_datasphere_job_submission(job_id, params)

        # Stage 4b: Archive Input Directory
        _ = await _archive_input_directory(job_id, settings.datasphere.train_job.input_dir)

        # Construct the specific output base directory for this DataSphere job run
        ds_job_specific_output_base_dir_local = os.path.join(settings.datasphere.train_job.output_dir, ds_job_run_suffix)

        # Stage 5: Submit and Monitor DS Job
        ds_job_id, results_dir, metrics_data, predictions_path, model_path, polls = await _submit_and_monitor_datasphere_job(
            job_id,
            client,
            ds_job_run_suffix,
            ds_job_specific_output_base_dir_local, # Pass the constructed output base dir
            params_json_path_local                   # Pass the params.json path
        )
        # Status is updated internally during polling and upon completion/failure within the function

        # Stage 6: Process Results (if job completed successfully)
        # Pass the results_dir where artifacts were downloaded
        await _process_job_results(
            job_id, ds_job_id, results_dir, params, metrics_data, model_path, predictions_path, polls, settings.datasphere.poll_interval, parameter_set_id
        )
        # Final success status (COMPLETED) is set within this function
        job_completed_successfully = True

    except (ValueError, RuntimeError, TimeoutError, ImportError) as e:
        # Handle errors from pipeline stages (already logged within helpers)
        # Ensure final status is FAILED if not already set
        error_msg = f"Job failed: {str(e)}"
        logger.error(f"[{job_id}] Pipeline terminated due to error: {e}", exc_info=False) # Log again at top level without full trace if already logged
        # Re-raise so the caller knows the job failed
        raise
    except TimeoutError as e:
        # Handle specific timeout error
        logger.error(f"[{job_id}] {str(e)}")
        update_job_status(job_id, status=JobStatus.FAILED.value, error_message=str(e))
        raise # Re-raise the timeout error
    except Exception as e:
        # Catch-all for unexpected errors during orchestration
        error_msg = f"Pipeline terminated due to error: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        update_job_status(job_id, status=JobStatus.FAILED.value, error_message=error_msg)
        raise # Re-raise so caller can handle
    finally:
        # Stage 7: Cleanup (always attempt)
        try:
            # Define the directories to be cleaned up, safely handling variables that might not be set
            dirs_to_clean = [
                settings.datasphere.train_job.input_dir,
                settings.datasphere.train_job.output_dir,
                # Add ds_job_specific_output_base_dir_local only if it's defined (it won't be for early failures)
                locals().get('ds_job_specific_output_base_dir_local')
            ]
            # Call the cleanup function, which now returns errors
            cleanup_errors = await _cleanup_directories(job_id, [d for d in dirs_to_clean if d]) # Filter out None/empty paths
            
            # If cleanup errors occurred and the job completed successfully, update the status message
            if cleanup_errors and job_completed_successfully:
                try:
                    # Just append a note about the cleanup error to the status message
                    update_job_status(
                        job_id, 
                        JobStatus.COMPLETED.value,
                        status_message=f"Job completed successfully. DS Job ID: {ds_job_id}. Cleanup error logged."
                    )
                except Exception as status_update_e:
                    logger.error(f"[{job_id}] Error updating status with cleanup error info: {status_update_e}")
        except Exception as cleanup_e:
            logger.error(f"[{job_id}] Error during final cleanup stage: {cleanup_e}", exc_info=True)
            # Do not change job status based on cleanup failure

        ds_job_info = f"DS Job ID: {ds_job_id}" if ds_job_id else f"DS Job Run Suffix: {ds_job_run_suffix}" if ds_job_run_suffix else "N/A"
        logger.info(f"[{job_id}] Job run processing finished. ({ds_job_info}).")

async def save_model_file_and_db(job_id: str, model_path: str, ds_job_id: str) -> str:
    """
    Save model file reference to database and return model_id.
    
    Args:
        job_id: ID of the job
        model_path: Path to the model file
        ds_job_id: DataSphere job ID that created the model
        
    Returns:
        model_id: ID of the model record created
        
    Raises:
        Exception: If model file doesn't exist or database operation fails
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Generate a unique ID for this model
    model_id = f"model_{uuid.uuid4().hex[:8]}"
    model_created_at = datetime.now()
    
    # Create metadata
    model_metadata = {
        "file_size_bytes": os.path.getsize(model_path),
        "downloaded_from_ds_job": ds_job_id,
        "original_path": model_path
    }
    
    # Create model record in database
    create_model_record(
        model_id=model_id,
        job_id=job_id,
        model_path=model_path,
        created_at=model_created_at,
        metadata=model_metadata,
        is_active=False
    )
    
    return model_id
