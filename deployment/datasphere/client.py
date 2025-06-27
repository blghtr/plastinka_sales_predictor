import logging
import os
import yaml
import tempfile
from pathlib import Path
from typing import Optional, List, Dict
from datasphere.client import Client as DatasphereClient
from datasphere.api import jobs_pb2 as jobs
from datasphere.config import Config, parse_config, check_limits
from datasphere.pyenv import define_py_env
from datasphere.files import prepare_local_modules, prepare_inputs
from io import StringIO

logger = logging.getLogger(__name__)

# Mapping of numeric JobStatus codes (as defined in DataSphere gRPC API) to canonical
# lowercase status strings that the rest of the service expects.
# Based on Yandex DataSphere JobStatus enum documentation:
# 0=JOB_STATUS_UNSPECIFIED, 1=CREATING, 2=EXECUTING, 3=UPLOADING_OUTPUT, 
# 4=SUCCESS, 5=ERROR, 6=CANCELLED, 7=CANCELLING, 8=PREPARING
# Confirmed by actual observations: 2=running, 5=failed, 6=cancelled
# If DataSphere extends the enum, add the new values here – keeping mapping logic
# isolated in this module avoids scattering magic numbers across the codebase.
JOB_STATUS_MAP: Dict[int, str] = {
    0: "unspecified",     # JOB_STATUS_UNSPECIFIED
    1: "pending",         # CREATING
    2: "running",         # EXECUTING ✓ confirmed
    3: "running",         # UPLOADING_OUTPUT
    4: "completed",       # SUCCESS
    5: "failed",          # ERROR ✓ confirmed
    6: "cancelled",       # CANCELLED ✓ confirmed
    7: "cancelling",      # CANCELLING
    8: "pending",         # PREPARING
}


def _normalize_status(raw_status) -> str:
    """Convert DataSphere SDK job.status into a canonical lowercase string.

    The SDK may return either a string (e.g. "SUCCESS") or an int (enum value).
    This helper makes downstream polling logic agnostic of that detail.
    Unknown values are passed through in lowercase string form so that the caller
    can decide what to do.
    """
    # Numeric enum – try to map directly
    try:
        int_code = int(raw_status)
        # If int(raw_status) succeeded but raw_status was actually a string that
        # contains digits, we'll still get here – acceptable.
        if int_code in JOB_STATUS_MAP:
            return JOB_STATUS_MAP[int_code]
    except (ValueError, TypeError):
        # Not a numeric code – fall back to string path
        pass

    # Fallback for string enums like "SUCCESS", "ERROR", etc.
    return str(raw_status).lower()


class DataSphereClientError(Exception):
    """Custom exception for DataSphere client errors."""
    pass


class DataSphereClient:
    """Client for interacting with Yandex DataSphere Jobs using official DataSphere client."""

    def __init__(self, project_id: str, folder_id: str, oauth_token: str = None, yc_profile: str = None):
        """
        Initializes the DataSphere client.

        Args:
            project_id: Yandex DataSphere project ID.
            folder_id: Yandex Cloud folder ID (may be used for future functionality).
            oauth_token: Yandex Cloud OAuth token (optional, uses profile/env if None).
            yc_profile: Yandex Cloud CLI profile name (optional).
        """
        
        if not project_id:
            raise ValueError("DataSphere project_id is required")
        if not folder_id:
            raise ValueError("Yandex Cloud folder_id is required")

        self.project_id = project_id
        self.folder_id = folder_id
        
        try:
            self._client = DatasphereClient(oauth_token=oauth_token, yc_profile=yc_profile)
            logger.info("DataSphere client initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize DataSphere client: {e}")
            raise DataSphereClientError(f"Failed to initialize official DataSphere client: {e}") from e

        logger.info(
            f"DataSphereClient initialized. Project ID: '{self.project_id}', Folder ID: '{self.folder_id}'"
        )

    def _prepare_job_with_local_modules(self, config_path: str, work_dir: str):
        """
        Prepares job parameters with local modules support.
        
        Args:
            config_path: Path to the job configuration YAML file.
            work_dir: Working directory for temporary files.
            
        Returns:
            Tuple of (job_params, cfg, sha256_to_display_path)
            
        Raises:
            DataSphereClientError: If preparation fails.
        """
        logger.info(f"Preparing job with local modules support from config: {config_path}")
        logger.info(f"Using work directory: {work_dir}")
        
        try:
            # Parse the config using datasphere's parser directly
            with open(config_path, 'r') as f:
                cfg = parse_config(f)
        except Exception as e:
            logger.error(f"Failed to parse configuration using DataSphere parser: {e}")
            raise DataSphereClientError(f"Invalid job configuration format: {e}") from e
        
        # Determine python environment if specified in config
        py_env = None
        if cfg.env.python:
            try:
                py_env = define_py_env(cfg.python_root_modules, cfg.env.python)
            except Exception as e:
                logger.error(f"Failed to define python environment: {e}")
                raise DataSphereClientError(f"Failed to define python environment: {e}") from e
        
        try:
            # Prepare inputs (archives directories if needed)
            cfg.inputs = prepare_inputs(cfg.inputs, work_dir)
            
            # Prepare local modules if python environment is defined
            local_modules = []
            sha256_to_display_path = {}
            
            if py_env:
                local_module_paths = prepare_local_modules(py_env, work_dir)
                local_modules = [f.get_file() for f in local_module_paths]
                
                # Create mapping for display paths
                sha256_to_display_path = {
                    f.sha256: p for f, p in zip(local_modules, py_env.local_modules_paths)
                }
            
            # Check limits
            check_limits(cfg, local_modules)
            
            # Generate job parameters
            job_params = cfg.get_job_params(py_env, local_modules)
            
            logger.info("Job parameters prepared successfully with local modules support")
            return job_params, cfg, sha256_to_display_path
            
        except Exception as e:
            logger.error(f"Failed to prepare job parameters: {e}")
            raise DataSphereClientError(f"Failed to prepare job parameters: {e}") from e

    def submit_job(self, config_path: str, work_dir: str) -> str:
        """
        Submits a DataSphere Job using the official client API.
        Parses the job configuration file, creates the job, and executes it.
        Supports local modules and dependencies specified in the config.

        Args:
            config_path: Path to the job configuration YAML file.
            work_dir: Working directory for temporary files.

        Returns:
            The submitted job ID.

        Raises:
            DataSphereClientError: If parsing fails or if the job submission fails.
        """
        logger.info(f"Submitting job with configuration file: {config_path}")
        
        try:
            # Prepare job parameters with local modules support
            job_params, cfg, sha256_to_display_path = self._prepare_job_with_local_modules(
                config_path, work_dir
            )
            
            # Create the job
            try:
                job_id = self._client.create(
                    job_params, 
                    cfg, 
                    self.project_id, 
                    sha256_to_display_path
                )
                logger.info(f"Successfully created job with ID: {job_id}")
                
                # Execute the job
                op, _ = self._client.execute(job_id)
                logger.info(f"Successfully started job execution. Operation ID: {op.id}")
                
                return job_id
            except Exception as e:
                logger.error(f"Failed to create or execute job: {e}")
                raise DataSphereClientError(f"Failed to submit job: {e}") from e
                
        except Exception as e:
            if not isinstance(e, DataSphereClientError):
                logger.error(f"Unexpected error during job submission: {e}")
                raise DataSphereClientError(f"Failed to submit job: {e}") from e
            raise


    def clone_job(self, source_job_id: str, config_path: str, work_dir: str, config_overrides: dict = None) -> str:
        """
        Clones an existing DataSphere job with new input files and optional config overrides.
        
        This method creates a new job based on an existing successful job, allowing for
        faster execution by reusing the cached Python environment from the source job.
        
        Args:
            source_job_id: ID of the source job to clone from
            config_path: Path to the job configuration YAML file (for new inputs)
            work_dir: Working directory for temporary files.
            config_overrides: Optional dictionary of configuration overrides
            
        Returns:
            The ID of the cloned job
            
        Raises:
            DataSphereClientError: If cloning fails or source job is invalid
        """
        logger.info(f"Cloning DataSphere job {source_job_id} with config: {config_path}")
        
        try:
            # Verify source job exists and is in a valid state
            source_job = self._client.get(source_job_id)
            if not source_job:
                raise DataSphereClientError(f"Source job {source_job_id} not found")
            
            # Check if source job is in a state that allows cloning
            source_status = str(source_job.status)
            if source_status not in ["COMPLETED", "SUCCESS"]:
                logger.warning(f"Source job {source_job_id} status is {source_status}, cloning may fail")
            
        except Exception as e:
            logger.error(f"Failed to verify source job {source_job_id}: {e}")
            raise DataSphereClientError(f"Failed to verify source job: {e}") from e
        
        # Change working directory to config file directory for relative path resolution
        config_path_obj = Path(config_path)
        config_dir = config_path_obj.parent
        original_cwd = Path.cwd()
        
        try:
            os.chdir(config_dir)
            
            # Prepare job parameters with local modules support (for new inputs)
            job_params, cfg, sha256_to_display_path = self._prepare_job_with_local_modules(config_path, work_dir)
            
            # Apply configuration overrides if provided
            if config_overrides:
                # Update job parameters with overrides
                for key, value in config_overrides.items():
                    if hasattr(job_params, key):
                        setattr(job_params, key, value)
            
            # Clone the job using the DataSphere client's clone functionality
            try:
                cloned_job_id = self._client.clone(
                    source_job_id=source_job_id,
                    job_params=job_params,
                    config=cfg,
                    project_id=self.project_id,
                    sha256_to_display_path=sha256_to_display_path
                )
                logger.info(f"Successfully cloned job {source_job_id} -> {cloned_job_id}")
                
                # Execute the cloned job
                op, _ = self._client.execute(cloned_job_id)
                logger.info(f"Successfully started cloned job execution. Operation ID: {op.id}")
                
                return cloned_job_id
                
            except AttributeError:
                # Fallback: If clone method doesn't exist, use create with clone hint
                logger.warning("DataSphere client doesn't support clone method, using create with clone hint")
                
                # Add clone hint to job parameters if supported
                if hasattr(job_params, 'clone_from'):
                    job_params.clone_from = source_job_id
                elif hasattr(job_params, 'parent_job_id'):
                    job_params.parent_job_id = source_job_id
                
                cloned_job_id = self._client.create(job_params, cfg, self.project_id, sha256_to_display_path)
                logger.info(f"Successfully created job with clone hint: {cloned_job_id}")
                
                # Execute the job
                op, _ = self._client.execute(cloned_job_id)
                logger.info(f"Successfully started job execution. Operation ID: {op.id}")
                
                return cloned_job_id
                
            except Exception as e:
                logger.error(f"Failed to clone job {source_job_id}: {e}")
                raise DataSphereClientError(f"Failed to clone job: {e}") from e
                
        except Exception as e:
            if not isinstance(e, DataSphereClientError):
                logger.error(f"Unexpected error during job cloning: {e}")
                raise DataSphereClientError(f"Failed to clone job: {e}") from e
            raise
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)

    def _read_yaml_config(self, config_path: str) -> dict:
        """Read and parse a YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise DataSphereClientError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise DataSphereClientError(f"Invalid YAML configuration: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error reading configuration file: {e}")
            raise DataSphereClientError(f"Failed to read configuration file: {e}") from e

    def get_job_status(self, job_id: str) -> str:
        """Gets and normalises the status of a DataSphere Job.

        Args:
            job_id: The ID of the job to check.

        Returns:
            Canonical lowercase status string (e.g. "running", "completed", "failed").

        Raises:
            DataSphereClientError: If the job status cannot be retrieved.
        """
        logger.info(f"Getting status for DataSphere Job ID: {job_id}")

        try:
            job = self._client.get(job_id)
            status_raw = job.status  # could be int(enum) or str
            status_norm = _normalize_status(status_raw)
            logger.info(f"Job ID {job_id} status: {status_norm} (raw: {status_raw})")
            return status_norm
        except Exception as e:
            logger.error(f"Failed to get status for Job ID {job_id}: {e}")
            raise DataSphereClientError(f"Failed to get job status: {e}") from e

    def download_job_results(self, job_id: str, output_dir: str, with_logs: bool = False, with_diagnostics: bool = False):
        """Downloads the results (and optionally logs/diagnostics) of a DataSphere Job.

        Args:
            job_id: The ID of the job whose files should be downloaded.
            output_dir: The local directory to download files into.
            with_logs: Whether to include job logs (default: False).
            with_diagnostics: Whether to include diagnostic files (default: False).

        Raises:
            DataSphereClientError: If the download operation fails.
        """
        logger.info(f"Downloading files for Job ID: {job_id} to {output_dir}")
        logger.info(f"Download options: logs={with_logs}, diagnostics={with_diagnostics}")
        
        try:
            # Get job to determine file lists
            job = self._client.get(job_id)
            
            # Collect appropriate files based on options
            files_to_download = list(job.output_files)
            if with_logs:
                files_to_download.extend(job.log_files)
            if with_diagnostics:
                files_to_download.extend(job.diagnostic_files)
            
            # Download the files
            logger.info(f"Downloading {len(files_to_download)} files")
            self._client.download_files(job_id, files_to_download, output_dir)
            logger.info(f"Successfully downloaded files for Job ID: {job_id}")
        except Exception as e:
            logger.error(f"Failed to download files for Job ID {job_id}: {e}")
            raise DataSphereClientError(f"Failed to download job files: {e}") from e

    def cancel_job(self, job_id: str, graceful: bool = False):
        """Cancels a running DataSphere Job.

        Args:
            job_id: The ID of the job to cancel.
            graceful: Whether to attempt graceful shutdown (default: False).

        Raises:
            DataSphereClientError: If the cancel operation fails.
        """
        logger.info(f"Canceling DataSphere Job ID: {job_id} (graceful={graceful})")
        
        try:
            self._client.cancel(job_id, graceful=graceful)
            logger.info(f"Successfully canceled Job ID: {job_id}")
        except Exception as e:
            logger.error(f"Failed to cancel Job ID {job_id}: {e}")
            raise DataSphereClientError(f"Failed to cancel job: {e}") from e
            
    def list_jobs(self) -> List[jobs.Job]:
        """Lists all jobs in the project.
        
        Returns:
            A list of Job objects.
            
        Raises:
            DataSphereClientError: If listing jobs fails.
        """
        logger.info(f"Listing jobs for project: {self.project_id}")
        
        try:
            jobs_list = self._client.list(self.project_id)
            logger.info(f"Found {len(jobs_list)} jobs")
            return jobs_list
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise DataSphereClientError(f"Failed to list jobs: {e}") from e
            
    def get_job(self, job_id: str) -> jobs.Job:
        """Gets detailed information about a job.
        
        Args:
            job_id: The ID of the job to get.
            
        Returns:
            A Job object with detailed information.
            
        Raises:
            DataSphereClientError: If getting job details fails.
        """
        logger.info(f"Getting details for job: {job_id}")
        
        try:
            job = self._client.get(job_id)
            return job
        except Exception as e:
            logger.error(f"Failed to get job details: {e}")
            raise DataSphereClientError(f"Failed to get job details: {e}") from e 