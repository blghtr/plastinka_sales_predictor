import asyncio
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from deployment.app.config import get_settings
from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.models.api_models import JobStatus, TrainingConfig
from deployment.app.services.job_registries.input_preparator_registry import (
    get_input_preparator,
)
from deployment.app.services.job_registries.job_type_registry import (
    JobTypeConfig,
    get_job_type_config,
)
from deployment.app.services.job_registries.result_processor_registry import (
    process_job_results_unified,
)
from deployment.app.utils.retry import retry_async_with_backoff
from deployment.app.utils.validation import ValidationError
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
        if os.name == "posix":  # Linux/macOS - symlink
            os.symlink(archive_path, project_input_path)
            logger.info(f"Created symlink: {project_input_path} -> {archive_path}")
        else:  # Windows - hard link
            os.link(archive_path, project_input_path)
            logger.info(f"Created hard link: {project_input_path} -> {archive_path}")
    except OSError as e:
        # Fallback to copying if linking fails
        try:
            shutil.copy2(archive_path, project_input_path)
            logger.warning(
                f"Failed to create link ({e}), copied file instead: {project_input_path}"
            )
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


async def _prepare_job_datasets(
    job_id: str,
    dal: DataAccessLayer,
    start_date_for_dataset: str = None,
    end_date_for_dataset: str = None,
    output_dir: str = None,
    job_config: JobTypeConfig | None = None,
) -> None:
    """Prepares datasets required for the job."""
    logger.info(f"[{job_id}] Stage 2: Preparing datasets...")

    logger.info(
        f"[{job_id}] Calling get_datasets with start_date: "
        f"{start_date_for_dataset}, end_date: {end_date_for_dataset}, "
        f"outputting to: {output_dir}"
    )

    try:
        get_datasets(
            start_date=start_date_for_dataset,
            end_date=end_date_for_dataset,
            output_dir=output_dir,
            datasets_to_generate=job_config.datasets_to_generate if job_config else [],
            dal=dal,
        )
        logger.info(f"[{job_id}] Datasets prepared in {output_dir}.")
        dal.update_job_status(
            job_id,
            JobStatus.RUNNING.value,
            progress=10,
            status_message="Datasets prepared.",
        )
    except Exception as e:
        logger.error(f"[{job_id}] Failed to prepare datasets: {e}", exc_info=True)
        dal.update_job_status(
            job_id,
            JobStatus.FAILED.value,
            error_message=f"Failed to prepare datasets: {e}",
        )
        raise RuntimeError(
            f"Failed to prepare datasets during job {job_id}. Original error: {e}"
        ) from e


async def _initialize_datasphere_client(job_id: str, dal: DataAccessLayer) -> DataSphereClient:
    """Initializes and returns the DataSphere client."""
    logger.info(f"[{job_id}] Stage 3: Initializing DataSphere client...")
    settings = get_settings()
    if not settings.datasphere or not settings.datasphere.client:
        dal.update_job_status(
            job_id,
            JobStatus.FAILED.value,
            error_message="DataSphere client configuration is missing.",
        )
        raise ValueError("DataSphere client configuration is missing in settings.")

    client_config = settings.datasphere.client  # Get config dict

    try:
        # DataSphereClient constructor is synchronous, run in a thread
        client = await asyncio.wait_for(
            asyncio.to_thread(DataSphereClient, **client_config),
            timeout=settings.datasphere.client_init_timeout_seconds,
        )
        logger.info(f"[{job_id}] DataSphere client initialized successfully.")
        dal.update_job_status(
            job_id,
            JobStatus.RUNNING.value,
            progress=15,
            status_message="DataSphere client initialized.",
        )

        return client
    except asyncio.TimeoutError:
        error_msg = f"DataSphere client initialization timed out after {settings.datasphere.client_init_timeout_seconds} seconds."
        logger.error(f"[{job_id}] {error_msg}")
        dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(
            error_msg
        ) from asyncio.TimeoutError  # Keep original exception type for context
    except (
        ImportError
    ) as e:  # This might still be relevant if datasphere client has conditional imports
        logger.error(
            f"Error importing DataSphereClient or its dependencies: {e}. Ensure all necessary packages are installed."
        )
        dal.update_job_status(
            job_id,
            JobStatus.FAILED.value,
            error_message=f"Client Initialization Failed due to import error: {e}",
        )
        raise RuntimeError(
            f"Failed to initialize DataSphere client due to import error: {e}"
        ) from e
    except Exception as e:
        error_msg = f"Failed to initialize DataSphere client: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        # Keep original exception type if it's a custom DataSphereClientError or similar
        if "DataSphereClientError" in str(type(e)):  # Basic check
            raise
        raise RuntimeError(error_msg) from e


async def _verify_datasphere_job_inputs(job_id: str, input_dir_path: Path, job_config: JobTypeConfig, dal: DataAccessLayer) -> None:
    """
    Verify that all required input files for the given job type exist.

    Args:
        job_id: Internal job identifier.
        input_dir_path: Directory that should contain the input files.
        job_config: JobTypeConfig describing required/optional files.
    """
    logger.info(
        f"[{job_id}] Stage 4a.1: Verifying DataSphere job inputs (job_type={job_config.name}) in {input_dir_path}..."
    )
    # Directory existence
    if not input_dir_path.exists() or not input_dir_path.is_dir():
        error_msg = f"Input directory '{input_dir_path}' is invalid for job {job_id}."
        logger.error(f"[{job_id}] {error_msg}")
        dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg)

    # Check required files
    missing_required: list[str] = []
    for req_file in job_config.required_input_files:
        if not (input_dir_path / req_file).is_file():
            missing_required.append(req_file)

    if missing_required:
        error_msg = (
            f"Missing required input files for job {job_id}: {', '.join(missing_required)}"
        )
        logger.error(f"[{job_id}] {error_msg}")
        dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"[{job_id}] All required input files verified successfully.")
    dal.update_job_status(
        job_id,
        JobStatus.RUNNING.value,
        progress=20,
        status_message="DataSphere job inputs verified.",
    )


async def _prepare_job_inputs_unified(
    job_id: str,
    config: TrainingConfig | None,
    target_input_dir: Path,
    job_config: JobTypeConfig,
    dal: DataAccessLayer,
) -> str:
    """
    Prepares the input files for a DataSphere job submission using a registered preparator.
    Returns the path to the main job config file (e.g., config.yaml).
    """
    logger.info(
        f"[{job_id}] Stage 4a: Preparing DataSphere job submission inputs via '{job_config.input_preparator_name}'..."
    )

    # Get and run the specific preparator for the job type
    try:
        preparator = get_input_preparator(job_config.input_preparator_name)
        await preparator(job_id, config, target_input_dir, job_config, dal)
    except Exception as e:
        error_msg = f"Failed during input preparation with '{job_config.input_preparator_name}': {e}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg) from e

    logger.info(f"[{job_id}] DataSphere job submission inputs prepared.")
    dal.update_job_status(
        job_id,
        JobStatus.RUNNING.value,
        progress=18,
        status_message="Job submission inputs prepared.",
    )

    # Return the path to the job-specific config file for DataSphere client
    settings = get_settings()
    script_dir = job_config.get_script_dir(settings)
    return os.path.join(script_dir, job_config.config_filename)


async def _archive_input_directory(
    job_id: str, input_dir: str, dal: DataAccessLayer, archive_dir: str = None
) -> str:
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
        logger.info(
            f"[{job_id}] Creating archive in specified directory: {archive_path}"
        )
    else:
        # Current behavior for backward compatibility
        archive_path = input_dir_path.parent / archive_name
        logger.info(f"[{job_id}] Creating archive in parent directory: {archive_path}")

    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(input_dir):
                # Skip the build artifacts directory
                if "_build_artifacts" in dirs:
                    dirs.remove("_build_artifacts")
                    logger.info(
                        f"[{job_id}] Excluding _build_artifacts directory from archive"
                    )

                for file in files:
                    file_path = os.path.join(root, file)
                    # Skip the archive file itself to avoid recursion
                    if os.path.abspath(file_path) == os.path.abspath(archive_path):
                        continue
                    # Calculate relative path from input_dir
                    relative_path = os.path.relpath(file_path, input_dir)
                    zipf.write(file_path, relative_path)

        logger.info(f"[{job_id}] Input directory archived to: {archive_path}")
        dal.update_job_status(
            job_id,
            JobStatus.RUNNING.value,
            progress=23,
            status_message="Input directory archived.",
        )
        return str(archive_path)
    except Exception as e:
        error_msg = f"Failed to archive input directory: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg) from e


@retry_async_with_backoff(max_tries=3, base_delay=2.0, max_delay=30.0, component="datasphere_submit")
async def _create_new_datasphere_job(
    job_id: str, client: DataSphereClient, ready_config_path: str, work_dir: str
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
            asyncio.to_thread(
                client.submit_job, config_path=ready_config_path, work_dir=work_dir
            ),
            timeout=get_settings().datasphere.client_submit_timeout_seconds,
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


async def _submit_datasphere_job(
    job_id: str, client: DataSphereClient, ready_config_path: str, work_dir: str, dal: DataAccessLayer
) -> str:
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
        dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise FileNotFoundError(error_msg)

    try:
        # Create new job
        logger.info(f"[{job_id}] Creating new DataSphere job")
        ds_job_id = await _create_new_datasphere_job(
            job_id, client, ready_config_path, work_dir
        )

        # Update status
        dal.update_job_status(
            job_id,
            JobStatus.RUNNING.value,
            progress=25,
            status_message=f"DS Job {ds_job_id} submitted.",
        )
        return ds_job_id

    except Exception as e:
        if not isinstance(e, FileNotFoundError | RuntimeError):
            error_msg = f"DataSphere job creation failed: {str(e)}"
            logger.error(f"[{job_id}] {error_msg}", exc_info=True)
            dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
            raise RuntimeError(error_msg) from e
        raise


@retry_async_with_backoff(max_tries=2, base_delay=1.0, max_delay=10.0, component="datasphere_status")
async def _check_datasphere_job_status(
    job_id: str, ds_job_id: str, client: DataSphereClient
) -> str:
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
            timeout=get_settings().datasphere.client_status_timeout_seconds,
        )
        return current_ds_status_str
    except asyncio.TimeoutError:
        error_msg = f"DataSphere job status check timed out after {get_settings().datasphere.client_status_timeout_seconds} seconds for DS Job ID: {ds_job_id}."
        logger.warning(
            f"[{job_id}] {error_msg}"
        )  # Log as warning, as polling might retry
        # Do not update job status here, let the polling loop handle overall job timeout/failure
        raise RuntimeError(
            error_msg
        ) from asyncio.TimeoutError  # Propagate to polling loop
    except Exception as status_exc:
        # This will catch DataSphereClientError or other unexpected errors
        error_msg = f"Error getting status for DS Job {ds_job_id}: {status_exc}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        # Do not update job status here, let the polling loop handle overall job timeout/failure
        if "DataSphereClientError" in str(type(status_exc)):
            raise  # Re-raise original DataSphereClientError
        raise RuntimeError(error_msg) from status_exc


@retry_async_with_backoff(max_tries=3, base_delay=5.0, max_delay=60.0, component="datasphere_download")
async def _download_datasphere_job_results(
    job_id: str,
    ds_job_id: str,
    client: DataSphereClient,
    results_dir: str,
    job_config: JobTypeConfig,
    dal: DataAccessLayer,
) -> dict[str, Any]:
    """
    Downloads and processes the results from a DataSphere job in a unified way using job_config.

    Args:
        job_id: Internal job identifier.
        ds_job_id: DataSphere job identifier.
        client: DataSphere client instance.
        results_dir: Directory to download results to.
        job_config: JobTypeConfig describing expected output files and their roles.

    Returns:
        Dictionary {role: value}, where value is a path to file or parsed data (for 'metrics'), or None if not found.
    """
    logger.info(
        f"[{job_id}] Stage 5a: Downloading results for ds_job '{ds_job_id}' to '{results_dir}'..."
    )

    os.makedirs(results_dir, exist_ok=True)
    logger.info(
        f"[{job_id}] Downloading results for DS Job {ds_job_id} to {results_dir}. Timeout: {get_settings().datasphere.client_download_timeout_seconds}s"
    )

    try:
        # Download results (sync, so run in thread)
        await asyncio.wait_for(
            asyncio.to_thread(client.download_job_results, ds_job_id, results_dir),
            timeout=get_settings().datasphere.client_download_timeout_seconds,
        )
        logger.info(
            f"[{job_id}] Results for DS Job {ds_job_id} downloaded to {results_dir}"
        )

        # Extract output.zip if present
        output_zip_path = os.path.join(results_dir, "output.zip")
        if os.path.exists(output_zip_path):
            logger.info(f"[{job_id}] Found output.zip archive, extracting to {results_dir}")
            try:
                with zipfile.ZipFile(output_zip_path, "r") as zip_ref:
                    zip_ref.extractall(results_dir)
                logger.info(f"[{job_id}] Successfully extracted output.zip archive")
                os.remove(output_zip_path)
            except zipfile.BadZipFile as e:
                logger.error(f"[{job_id}] Invalid zip file format for output.zip: {e}")
            except Exception as e:
                logger.error(f"[{job_id}] Error extracting output.zip: {e}", exc_info=True)

        # Унифицированная обработка по ролям
        results_by_role = {}
        for expected_file in job_config.expected_output_files:
            file_path = os.path.join(results_dir, expected_file)
            role = job_config.output_file_roles.get(expected_file, expected_file)
            if os.path.exists(file_path):
                if role == "metrics":
                    try:
                        with open(file_path) as f:
                            results_by_role[role] = json.load(f)
                        logger.info(f"[{job_id}] Loaded metrics from {file_path}")
                    except json.JSONDecodeError as e:
                        logger.error(f"[{job_id}] Failed to parse metrics.json from {file_path}: {e}", exc_info=True)
                        results_by_role[role] = None
                else:
                    results_by_role[role] = file_path
                    logger.info(f"[{job_id}] Found output file for role '{role}': {file_path}")
            else:
                logger.warning(f"[{job_id}] Expected output file '{expected_file}' (role '{role}') not found in {results_dir}")
                results_by_role[role] = None

        return results_by_role
    except asyncio.TimeoutError:
        error_msg = f"DataSphere job results download timed out after {get_settings().datasphere.client_download_timeout_seconds} seconds for DS Job ID: {ds_job_id}."
        logger.error(f"[{job_id}] {error_msg}")
        dal.update_job_status(job_id, JobStatus.FAILED.value, error_message=error_msg)
        raise RuntimeError(error_msg) from asyncio.TimeoutError
    except Exception as e:
        logger.error(
            f"[{job_id}] DataSphere job '{ds_job_id}' failed with an exception during result download: {e}",
            exc_info=True,
        )
        raise RuntimeError(
            f"DataSphere job result download failed for job {job_id}. Details: {e}"
        ) from e
    finally:
        try:
            if os.path.exists(results_dir):
                pass  # Debug logging removed
        except Exception as e:
            logger.warning(
                f"[{job_id}] Could not list contents of results directory '{results_dir}': {e}"
            )


async def _download_logs_diagnostics(
    job_id: str,
    ds_job_id: str,
    client: DataSphereClient,
    logs_dir: str,
    is_success: bool = True,
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
    logger.info(
        f"[{job_id}] Downloading logs/diagnostics for {status_str} DS Job {ds_job_id} to {logs_dir}. Timeout: {get_settings().datasphere.client_download_timeout_seconds}s"
    )

    try:
        # client.download_job_results is synchronous, run in a thread with timeout
        await asyncio.wait_for(
            asyncio.to_thread(
                client.download_job_results,
                ds_job_id,
                logs_dir,
                with_logs=True,
                with_diagnostics=True,
            ),
            timeout=get_settings().datasphere.client_download_timeout_seconds,
        )
        logger.info(
            f"[{job_id}] Logs/diagnostics for {status_str} DS Job {ds_job_id} downloaded to {logs_dir}"
        )
    except asyncio.TimeoutError:
        # This is generally a non-critical error for the job's main outcome, so log and continue.
        error_msg = f"DataSphere logs/diagnostics download timed out after {get_settings().datasphere.client_download_timeout_seconds} seconds for DS Job ID: {ds_job_id}."
        logger.warning(f"[{job_id}] {error_msg}")
    except Exception as dl_exc:
        # This is generally a non-critical error for the job's main outcome, so log and continue.
        logger.warning(
            f"[{job_id}] Failed to download logs/diagnostics for DS Job {ds_job_id}: {dl_exc}",
            exc_info=True,
        )


async def _process_datasphere_job(
    job_id: str,
    client: DataSphereClient,
    ds_job_specific_output_base_dir: str,
    ready_config_path: str,
    work_dir: str,
    job_config: JobTypeConfig,
    dal: DataAccessLayer,
) -> tuple[str, str, dict[str, Any] | None, dict[str, Any], int]:
    """
    Coordinates the DataSphere job lifecycle including submission, monitoring, and result fetching.
    Returns: Tuple of (ds_job_id, results_dir, metrics_data, output_files_by_role, polls)
    """
    logger.info(f"[{job_id}] Stage 5: Processing DataSphere job...")

    # Submit the job
    ds_job_id = await _submit_datasphere_job(
        job_id, client, ready_config_path, work_dir, dal
    )

    completed = False
    settings = get_settings()
    max_polls = settings.datasphere.max_polls
    poll_interval = settings.datasphere.poll_interval
    polls = 0
    metrics_data = None
    output_files_by_role = {}

    logger.info(
        f"[{job_id}] Polling DS Job {ds_job_id} status (max_polls={max_polls}, poll_interval={poll_interval}s)."
    )

    # Monitor job status
    while not completed and polls < max_polls:
        await asyncio.sleep(poll_interval)
        polls += 1

        try:
            current_ds_status_str = await _check_datasphere_job_status(
                job_id, ds_job_id, client
            )
            current_ds_status = current_ds_status_str.lower()
            logger.info(
                f"[{job_id}] DS Job {ds_job_id} status (poll {polls}/{max_polls}): {current_ds_status_str}"
            )
        except Exception as status_exc:
            logger.error(
                f"[{job_id}] Error getting status for DS Job {ds_job_id} (poll {polls}/{max_polls}): {status_exc}. Will retry."
            )
            dal.update_job_status(
                job_id,
                JobStatus.RUNNING.value,
                status_message=f"DS Job {ds_job_id}: Error polling status ({status_exc}) - Retrying...",
            )
            continue

        # Estimate progress
        current_progress = 25 + int(
            (polls / max_polls) * 65
        )  # Progress from 25% to 90% during polling
        dal.update_job_status(
            job_id,
            JobStatus.RUNNING.value,
            progress=current_progress,
            status_message=f"DS Job {ds_job_id}: {current_ds_status_str}",
        )

        if current_ds_status in [
            "completed",
            "success",
        ]:  # "success" kept for backward-compat
            completed = True
            logger.info(
                f"[{job_id}] DS Job {ds_job_id} completed. Downloading results..."
            )

            # Download and process results (унифицировано)
            output_files_by_role = await _download_datasphere_job_results(
                job_id, ds_job_id, client, ds_job_specific_output_base_dir, job_config, dal
            )
            metrics_data = output_files_by_role.get("metrics")

            # Download logs and diagnostics if configured
            download_diag_on_success = getattr(
                get_settings().datasphere, "download_diagnostics_on_success", False
            )
            if download_diag_on_success:
                logs_dir = os.path.join(
                    ds_job_specific_output_base_dir, "logs_diagnostics_success"
                )
                await _download_logs_diagnostics(
                    job_id, ds_job_id, client, logs_dir, is_success=True
                )
            else:
                logger.info(
                    f"[{job_id}] Skipping optional download of logs/diagnostics for successful job based on settings."
                )

        elif current_ds_status in ["failed", "cancelled", "cancelling"]:
            error_detail = (
                f"DS Job {ds_job_id} ended with status: {current_ds_status_str}."
            )
            logger.error(f"[{job_id}] {error_detail}")

            # Download logs for failed job
            logs_dir = os.path.join(ds_job_specific_output_base_dir, "logs_diagnostics")
            await _download_logs_diagnostics(
                job_id, ds_job_id, client, logs_dir, is_success=False
            )
            error_detail += f" Logs/diagnostics may be available in {logs_dir}."

            dal.update_job_status(
                job_id, JobStatus.FAILED.value, error_message=error_detail
            )
            raise RuntimeError(error_detail)

    if not completed:
        timeout_message = f"DS Job {ds_job_id} execution timed out after {polls} polls ({max_polls * poll_interval}s)."
        logger.error(f"[{job_id}] {timeout_message}")
        dal.update_job_status(
            job_id, status=JobStatus.FAILED.value, error_message=str(timeout_message)
        )
        raise TimeoutError(timeout_message)

    return (
        ds_job_id,
        ds_job_specific_output_base_dir,
        metrics_data,
        output_files_by_role,
        polls,
    )


async def _perform_model_cleanup(job_id: str, current_model_id: str, dal: DataAccessLayer) -> None:
    """Prunes old, non-active models based on settings."""
    try:
        num_models_to_keep = getattr(get_settings(), "max_models_to_keep", 5)
        if num_models_to_keep > 0:
            logger.info(
                f"[{job_id}] Checking for old models to prune (keeping last {num_models_to_keep})..."
            )

            # Get IDs of models to keep (most recent ones, including potentially the current one if it becomes active later)
            # Note: The current model (current_model_id) is initially inactive.
            recent_kept_models_info = dal.get_recent_models(limit=num_models_to_keep)
            kept_model_ids = {
                m["model_id"] for m in recent_kept_models_info
            }  # Use a set for faster lookups

            # Fetch all models (or a reasonable limit) to find candidates for deletion
            all_models_info = dal.get_all_models(limit=1000)
            models_to_delete_ids = []

            for model_info in all_models_info:
                m_id = model_info["model_id"]
                is_active = model_info.get("is_active", False)

                # Candidate for deletion if:
                # 1. It's NOT the model just created in *this* job run.
                # 2. It's NOT marked as active.
                # 3. It's NOT in the set of recently created models to keep.
                if (
                    m_id != current_model_id
                    and not is_active
                    and m_id not in kept_model_ids
                ):
                    models_to_delete_ids.append(m_id)

            if models_to_delete_ids:
                logger.info(
                    f"[{job_id}] Found {len(models_to_delete_ids)} older, non-active models to prune: {models_to_delete_ids}"
                )
                delete_result = dal.delete_models_by_ids(models_to_delete_ids)
                deleted_count = delete_result.get("deleted_count", "N/A")
                failed_deletions = delete_result.get("failed_ids", [])
                logger.info(
                    f"[{job_id}] Model cleanup result: Deleted {deleted_count} models. Failed: {failed_deletions}"
                )
            else:
                logger.info(
                    f"[{job_id}] No non-active models found eligible for pruning beyond the kept {num_models_to_keep}."
                )
        else:
            logger.info(f"[{job_id}] Model cleanup disabled (max_models_to_keep <= 0).")
    except Exception as cleanup_exc:
        logger.error(
            f"[{job_id}] Error during model cleanup: {cleanup_exc}", exc_info=True
        )
        # Non-fatal, log and continue


# NOTE: _process_job_results has been moved to result_processors.py
# as process_training_results in the unified result processing architecture


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
            logger.info(
                f"[{job_id}] Directory not found or not a directory, skipping cleanup: {dir_path}"
            )
    # Return any cleanup errors to be handled by caller
    return cleanup_errors


def save_predictions_to_db(
    predictions_path: str, job_id: str, model_id: str, dal: DataAccessLayer
) -> dict:
    """
    Reads prediction results from a CSV file and saves them to the database.

    Args:
        predictions_path: Path to the CSV file with predictions
        job_id: The job ID that produced these predictions
        model_id: The model ID used for these predictions

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
        try:
            df = pd.read_csv(
                predictions_path, 
                dtype={'barcode': object}             
            )
            # Identify integer columns (like recording_year) vs float columns (like quantiles)
            int_cols = ['recording_year']  # Only recording_year should be treated as integer
            float_cols = [col for col in df.select_dtypes(include="number").columns if col not in int_cols]

            if int_cols:
                df.loc[:, int_cols] = (
                    df.loc[:, int_cols]
                    .fillna(0)
                    .round(0)
                    .astype("int64")
                )

            if float_cols:
                df.loc[:, float_cols] = (
                    df
                    .loc[:, float_cols]
                    .fillna(0)
                    .round(2)
                    .astype("float64")
                )
            
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid predictions file format: {e}")

        # NEW: Check if DataFrame is empty after reading
        if df.empty:
            raise ValueError(f"Predictions file is empty or invalid format: {predictions_path}")

        # Verify required columns exist
        # TODO: move to config, add type validation
        required_columns = [
            "barcode",
            "artist",
            "album",
            "cover_type",
            "price_category",
            "release_type",
            "recording_decade",
            "release_decade",
            "style",
            "recording_year",
            "0.05",
            "0.25",
            "0.5",
            "0.75",
            "0.95",
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Predictions file missing required columns: {missing_cols}"
            )

        # Get prediction month from job parameters using DAL method
        prediction_month: date = dal.get_job_prediction_month(job_id)

        prediction_result_id = None
        try:
            prediction_result_id = dal.create_prediction_result(
                job_id=job_id,
                prediction_month=prediction_month,
                model_id=model_id,
                output_path=predictions_path,
                summary_metrics="{}",
            )

        except Exception as e:
            logger.error(
                f"[{job_id}] Error creating prediction result: {e}"
            )

        if prediction_result_id is not None:
            try:
                insert_result = dal.insert_predictions(
                        result_id=prediction_result_id,
                        model_id=model_id,
                        prediction_month=prediction_month,
                        df=df,
                    )
                return insert_result
            except Exception as e:
                raise e
        else:
            error_msg = f"No prediction result created for job {job_id}"
            logger.error(f"[{job_id}] {error_msg}")
            raise ValueError(error_msg)

    except pd.errors.EmptyDataError as e:
        raise ValueError("Predictions file is empty") from e
    except pd.errors.ParserError as e:
        raise ValueError("Predictions file has invalid format") from e
    except Exception as e:
        # Do not wrap database connection errors
        if isinstance(e, FileNotFoundError):
            raise

        raise ValueError(f"Error processing predictions: {str(e)}") from e
    

# Allowed job types
JOB_TYPE_TRAIN = "train"
JOB_TYPE_TUNE = "tune"


# Main Orchestrator Function
async def run_job(
    job_id: str,
    config: dict | None = None,
    config_id: str | None = None,
    job_type: str = JOB_TYPE_TRAIN,
    dataset_start_date: str = None,
    dataset_end_date: str = None,
    additional_job_params: dict | None = None,
    dal: DataAccessLayer = None,
) -> dict[str, Any] | None:
    """
    Runs a DataSphere training job pipeline: setup, execution, monitoring, result processing, cleanup.
    """
    ds_job_id: str | None = None
    client: DataSphereClient | None = None
    project_input_link_path: str | None = None
    job_completed_successfully = False
    try:
        config = TrainingConfig(**config)
    except (ValidationError, ValueError) as e:
        logger.error(f"Configuration is invalid: {e}", exc_info=True)
        raise ValueError(f"Invalid configuration provided: {e}") from e

    try:
        with (
            tempfile.TemporaryDirectory(
                dir=str(get_settings().datasphere_input_dir)
            ) as temp_input_dir_str,
            tempfile.TemporaryDirectory(
                dir=str(get_settings().datasphere_output_dir)
            ) as temp_output_dir_str,
        ):
            temp_input_dir = Path(temp_input_dir_str)
            temp_output_dir = Path(temp_output_dir_str)

            try:
                # Get job configuration from registry (returns a COPY)
                job_config = get_job_type_config(job_type)

                # Merge dynamic parameters passed from API (if any)
                if additional_job_params:
                    job_config.additional_params.update(additional_job_params)

                logger.info(
                    f"[{job_id}] Using job configuration for type: "
                    f"{job_config.name} with additional_params={job_config.additional_params}"
                )

                # Initialize job status
                dal.update_job_status(
                    job_id,
                    JobStatus.PENDING.value,
                    progress=0,
                    status_message="Initializing job.",
                )

                # Stage 2: Prepare Datasets
                await _prepare_job_datasets(
                    job_id,
                    dal,
                    dataset_start_date,
                    dataset_end_date,
                    output_dir=str(temp_input_dir),
                    job_config=job_config,
                )

                # Stage 3: Initialize Client
                client = await _initialize_datasphere_client(job_id, dal)

                # Stage 4a: Prepare DS Job Submission Inputs
                config_path = await _prepare_job_inputs_unified(
                    job_id,
                    config,
                    temp_input_dir,
                    job_config,
                    dal,
                )

                # Stage 4a.1: Verify DataSphere job inputs
                await _verify_datasphere_job_inputs(
                    job_id, temp_input_dir, job_config, dal
                )

                # Stage 4b: Archive Input Directory
                archive_path = await _archive_input_directory(
                    job_id, temp_input_dir_str, dal, temp_input_dir_str
                )

                # Stage 4c: Create Project Link
                # Put input.zip inside the specific job scripts directory so that
                # DataSphere job picks it up correctly (train vs tune)
                job_scripts_dir = job_config.get_script_dir(get_settings())
                project_input_link_path = create_project_input_link(
                    archive_path, job_scripts_dir
                )
                logger.info(
                    f"[{job_id}] Created project input link: {project_input_link_path}"
                )
                dal.update_job_status(
                    job_id,
                    JobStatus.RUNNING.value,
                    progress=24,
                    status_message="Project input link created.",
                )

                # Stage 5: Submit and Monitor DS Job
                (
                    ds_job_id,
                    results_dir_from_process,
                    metrics_data,
                    output_files_by_role,
                    polls,
                ) = await _process_datasphere_job(
                    job_id,
                    client,
                    temp_output_dir,
                    config_path,
                    temp_input_dir_str,  # Pass temp_input_dir as work_dir for local modules
                    job_config,
                    dal,
                )

                # Stage 6: Process Results using unified registry
                await process_job_results_unified(
                    job_id=job_id,
                    processor_name=job_config.result_processor_name,
                    ds_job_id=ds_job_id,
                    results_dir=results_dir_from_process,
                    config=config,
                    metrics_data=metrics_data,
                    output_files=output_files_by_role,
                    polls=polls,
                    poll_interval=get_settings().datasphere.poll_interval,
                    config_id=config_id,
                    dal=dal, # Pass the dal
                )
                job_completed_successfully = True
                logger.info(f"[{job_id}] Job pipeline completed successfully.")

                return {
                    "job_id": job_id,
                    "status": JobStatus.COMPLETED.value,
                    "datasphere_job_id": ds_job_id,
                    "message": "Job completed successfully",
                }
            except asyncio.CancelledError:
                logger.warning(f"[{job_id}] Job run was cancelled.")
                active_client = (
                    client if "client" in locals() and client is not None else None
                )
                current_ds_job_id = (
                    ds_job_id
                    if "ds_job_id" in locals() and ds_job_id is not None
                    else None
                )

                cancel_message = "Job cancelled by application."
                if active_client and current_ds_job_id:
                    logger.info(
                        f"[{job_id}] Attempting to cancel DataSphere job {current_ds_job_id} due to task cancellation."
                    )
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(
                                active_client.cancel_job,
                                current_ds_job_id,
                                graceful=True,
                            ),
                            timeout=get_settings().datasphere.client_cancel_timeout_seconds,
                        )
                        logger.info(
                            f"[{job_id}] DataSphere job {current_ds_job_id} cancellation request submitted."
                        )
                        cancel_message += f" DataSphere job {current_ds_job_id} cancellation attempted."
                    except asyncio.TimeoutError:
                        logger.error(
                            f"[{job_id}] Timeout trying to cancel DataSphere job {current_ds_job_id}."
                        )
                        cancel_message += f" Timeout attempting to cancel DataSphere job {current_ds_job_id}."
                    except Exception as cancel_e:
                        logger.error(
                            f"[{job_id}] Error trying to cancel DataSphere job {current_ds_job_id}: {cancel_e}",
                            exc_info=True,
                        )
                        cancel_message += f" Error attempting to cancel DataSphere job {current_ds_job_id}: {cancel_e}."

                dal.update_job_status(
                    job_id,
                    JobStatus.FAILED.value,
                    error_message=cancel_message,
                    status_message=cancel_message
                )

            except (ValueError, RuntimeError, TimeoutError, ImportError) as e:
                error_msg = f"Job pipeline failed: {str(e)}"
                logger.error(
                    f"[{job_id}] {error_msg}",
                    exc_info=isinstance(e, RuntimeError | ImportError),
                )
                job_details = dal.get_job(job_id)
                if job_details and job_details.get("status") != JobStatus.FAILED.value:
                    dal.update_job_status(
                        job_id,
                        JobStatus.FAILED.value,
                        error_message=error_msg,
                        status_message=error_msg,
                    )
                raise
            except Exception as e:
                error_msg = f"Unexpected error in job pipeline: {str(e)}"
                logger.error(f"[{job_id}] {error_msg}", exc_info=True)
                job_details = dal.get_job(job_id)
                if job_details and job_details.get("status") != JobStatus.FAILED.value:
                    dal.update_job_status(
                        job_id,
                        JobStatus.FAILED.value,
                        error_message=error_msg,
                        status_message=error_msg,
                    )
                raise
            finally:
                # Cleanup project input link regardless of success/failure
                if project_input_link_path:
                    cleanup_project_input_link(job_scripts_dir)

                # The temporary directories are cleaned up automatically by the context manager.
                # Final log message for job run completion.
                current_ds_job_id_log = (
                    ds_job_id
                    if "ds_job_id" in locals() and ds_job_id is not None
                    else "N/A"
                )
                logger.info(
                    f"[{job_id}] Job run processing finished. DS Job ID: {current_ds_job_id_log}. Temporary directories and project links cleaned up automatically."
                )

    except Exception as e:
        # This will catch errors in creating the temporary directories themselves.
        error_msg = f"Failed to set up temporary directories for job: {str(e)}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        dal.update_job_status(
            job_id,
            JobStatus.FAILED.value,
            error_message=error_msg,
            status_message=error_msg,
        )
        raise


async def save_model_file_and_db(
    job_id: str,
    model_path: str,
    ds_job_id: str,
    config: TrainingConfig,
    metrics_data: dict[str, Any] | None,
    dal: DataAccessLayer,
) -> str:
    """
    Saves the model file to permanent storage and creates a database record.

    Args:
        job_id: The job ID that produced this model
        model_path: Path to the temporary model file (downloaded from DataSphere)
        ds_job_id: DataSphere job ID that produced this model
        config: Training configuration used
        metrics_data: Optional metrics data from training
        connection: Optional existing database connection to use

    Returns:
        The generated model ID

    Raises:
        RuntimeError: If model file cannot be copied or database record cannot be created
    """
    try:
        # Generate a unique model ID based on the config's model_id (base name)
        model_id = config.model_id
        logger.info(f"[{job_id}] Using model ID: {model_id}")

        # Verify source model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Source model file not found: {model_path}")

        # Create permanent storage path
        permanent_model_filename = f"{model_id}.onnx"
        permanent_model_path = os.path.join(
            get_settings().models_dir, permanent_model_filename
        )

        logger.info(
            f"[{job_id}] Copying model from temporary location '{model_path}' to permanent storage '{permanent_model_path}'"
        )

        # Copy model file to permanent storage
        try:
            shutil.copy2(model_path, permanent_model_path)
            logger.info(
                f"[{job_id}] Model file successfully copied to permanent storage"
            )
        except Exception as copy_error:
            raise RuntimeError(
                f"Failed to copy model file to permanent storage: {copy_error}"
            ) from copy_error

        # Verify the copy was successful
        if not os.path.exists(permanent_model_path):
            raise RuntimeError(
                f"Model file copy verification failed - file not found at destination: {permanent_model_path}"
            )

        # Get file size from the permanent location
        file_size = os.path.getsize(permanent_model_path)

        # Construct metadata
        metadata = {
            "file_size_bytes": file_size,
            "downloaded_from_ds_job": ds_job_id,
            "original_temp_path": model_path,  # Keep reference to original temp path for debugging
            "permanent_storage_path": permanent_model_path,
            "config_model_id_base": config.model_id,  # Store the base name for tracking
            "metrics": metrics_data or {},  # Store metrics with the model
        }

        # Create a record in the database with the permanent path
        dal.create_model_record(
            model_id=model_id,
            job_id=job_id,
            model_path=str(permanent_model_path),  # Save the permanent path
            created_at=datetime.now(),
            metadata=metadata,
        )
        logger.info(
            f"[{job_id}] Successfully created model record in DB for model_id: {model_id} with permanent path: {permanent_model_path}"
        )
        return model_id

    except Exception as e:
        error_msg = f"Failed to save model file and create DB record: {e}"
        logger.error(f"[{job_id}] {error_msg}", exc_info=True)
        # We don't update job status here, the caller should handle it
        raise RuntimeError(error_msg) from e


