from datetime import datetime
import time
import os
import json
import uuid
import asyncio
import logging
from typing import Optional, Tuple, Dict, Any, List
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
    get_job
)
from deployment.app.models.api_models import JobStatus, TrainingParams
from deployment.app.config import settings
from deployment.datasphere.client import DataSphereClient
from deployment.datasphere.prepare_datasets import get_datasets
import shutil
from pathlib import Path
import yaml
from pydantic import TypeAdapter

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


async def _prepare_datasphere_job_submission(job_id: str, params: TrainingParams) -> Tuple[str, str]:
    """Prepares the parameters file within the main input directory."""
    logger.info(f"[{job_id}] Stage 4a: Preparing DataSphere job submission inputs...")
    # Generate a unique ID component for this DS job run (used for output/results dir later)
    ds_job_run_suffix = f"ds_job_{job_id}_{uuid.uuid4().hex[:8]}"
    target_input_dir = settings.datasphere.train_job.input_dir # Use the main input dir

    # Ensure the main input directory exists (datasets should already be there)
    os.makedirs(target_input_dir, exist_ok=True)

    params_json_path = os.path.join(target_input_dir, "params.json")
    logger.info(f"[{job_id}] Saving parameters to {params_json_path}")
    with open(params_json_path, 'w') as f:
        # Use model_dump_json for Pydantic v2+
        json.dump(params.model_dump(), f, indent=2) # Save parameters

    # Note: Progress update moved to after archiving
    # update_job_status(job_id, JobStatus.RUNNING.value, progress=20, status_message="DataSphere job inputs prepared.")
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
    while not completed and polls < max_polls:
        await asyncio.sleep(poll_interval)
        polls += 1

        try:
            current_ds_status_str = client.get_job_status(ds_job_id)
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
    """Processes the results of a completed DataSphere job."""
    logger.info(f"[{job_id}] Stage 6: Processing job results for DS Job {ds_job_id}...")

    training_hyperparams = params.model_dump() # For storing with results
    current_model_id = None # Will be set if model is successfully recorded

    # --- Calculate Duration ---
    training_duration_seconds = metrics_data.get("training_duration_seconds") if metrics_data else None
    if training_duration_seconds is None:
        # Estimate duration if not provided in metrics
        training_duration_seconds = polls * poll_interval
        logger.warning(f"[{job_id}] training_duration_seconds not in metrics, estimated as {training_duration_seconds}s.")
    else:
        logger.info(f"[{job_id}] Training duration from metrics: {training_duration_seconds}s.")


    # --- Handle Model Record Creation and Cleanup ---
    if model_path and os.path.exists(model_path):
        # Generate a unique ID for this specific model artifact
        current_model_id = f'{params.model_id}_{uuid.uuid4().hex[:8]}' # More unique ID
        model_created_at = datetime.utcnow()
        model_metadata = {
            "file_size_bytes": os.path.getsize(model_path),
            "downloaded_from_ds_job": ds_job_id,
            "original_path": model_path, # Keep track of where it was downloaded
            "results_dir": results_dir, # Add results dir for context
            # Add other relevant metadata if available
        }

        # Persist the model file to a more permanent location if desired
        # For now, we'll record the downloaded path. Consider moving/copying later.
        # model_storage_path = _persist_model_artifact(job_id, current_model_id, model_path)

        create_model_record(
            model_id=current_model_id,
            job_id=job_id,
            model_path=model_path, # Record the path where it currently is
            created_at=model_created_at,
            metadata=model_metadata,
            is_active=False # Activation is a separate step, potentially manual or based on evaluation
        )
        logger.info(f"[{job_id}] Model record created (initially inactive): {current_model_id} at path {model_path}")

        # Perform cleanup of older models *after* successfully creating the new record
        await _perform_model_cleanup(job_id, current_model_id)

    elif model_path:
        logger.warning(f"[{job_id}] Model artifact path specified ({model_path}) but file not found. Cannot create model record.")
    else:
        logger.info(f"[{job_id}] No model artifact path available from DS job. Cannot create model record.")


    # --- Create Training Result ---
    if metrics_data:
        training_result_id = create_training_result(
            job_id=job_id,
            model_id=current_model_id, # Link to model if created, else None
            parameter_set_id=parameter_set_id, # Link to the parameters used
            metrics=metrics_data,
            parameters=training_hyperparams, # Store params used for this result
            duration=int(training_duration_seconds)
        )
        logger.info(f"[{job_id}] Training result {training_result_id} stored (Params: {parameter_set_id}, Model: {current_model_id or 'N/A'}).")
        # Update final status to COMPLETED with result ID
        update_job_status(
            job_id=job_id,
            status=JobStatus.COMPLETED.value,
            progress=100,
            result_id=training_result_id,
            status_message=f"Job completed. DS Job ID: {ds_job_id}. Training Result ID: {training_result_id}"
        )
    else:
        # Job technically completed, but without metrics, we can't create a full result
        logger.warning(f"[{job_id}] Metrics not found for DS Job {ds_job_id}. Skipping creation of training_result entry.")
        update_job_status(
            job_id,
            JobStatus.COMPLETED.value, # Mark as completed, but indicate missing metrics
            progress=100,
            status_message=f"Job completed. DS Job ID: {ds_job_id}. Metrics missing."
        )

    # Log final artifact locations for reference
    if current_model_id:
        logger.info(f"[{job_id}] Final Model ID: {current_model_id} (Results Dir: {results_dir})")
    if predictions_path and os.path.exists(predictions_path):
        logger.info(f"[{job_id}] Predictions available at: {predictions_path} (in Results Dir: {results_dir})")
    elif predictions_path:
         logger.warning(f"[{job_id}] Predictions artifact path specified ({predictions_path}) but file not found.")


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


# Main Orchestrator Function
async def run_job(job_id: str) -> None:
    """
    Runs a DataSphere training job pipeline: setup, execution, monitoring, result processing, cleanup.
    """
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
