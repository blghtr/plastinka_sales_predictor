import logging
import yaml
from pathlib import Path
from typing import Optional, List
from datasphere.client import Client as DatasphereClient
from datasphere.api import jobs_pb2 as jobs
from datasphere.config import Config

logger = logging.getLogger(__name__)


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

    def submit_job(self, config_path: str) -> str:
        """
        Submits a DataSphere Job using the official client API.
        Parses the job configuration file, creates the job, and executes it.

        Args:
            config_path: Path to the job configuration YAML file.

        Returns:
            The submitted job ID.

        Raises:
            DataSphereClientError: If parsing fails or if the job submission fails.
        """
        logger.info(f"Submitting job with configuration file: {config_path}")
        
        try:
            # Read and parse the YAML configuration file
            config_data = self._read_yaml_config(config_path)
            
            # Convert config to Config object as expected by the official client
            from datasphere.config import parse_config
            from io import StringIO
            
            # Convert dict back to YAML string for parse_config
            config_yaml = yaml.dump(config_data)
            config_io = StringIO(config_yaml)
            
            # Parse the config using datasphere's parser
            try:
                cfg = parse_config(config_io)
            except Exception as e:
                logger.error(f"Failed to parse configuration using DataSphere parser: {e}")
                raise DataSphereClientError(f"Invalid job configuration format: {e}") from e
            
            # Create the job
            try:
                # Prepare job parameters and empty file maps (simplified)
                job_params = cfg.get_job_params(py_env=None, local_modules=[])
                sha256_to_display_path = {}
                
                # Create the job and get the ID
                job_id = self._client.create(job_params, cfg, self.project_id, sha256_to_display_path)
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
        """Gets the status of a DataSphere Job.

        Args:
            job_id: The ID of the job to check.

        Returns:
            The status of the job as a string.

        Raises:
            DataSphereClientError: If the job status cannot be retrieved.
        """
        logger.info(f"Getting status for DataSphere Job ID: {job_id}")
        
        try:
            job = self._client.get(job_id)
            status = str(job.status)
            logger.info(f"Job ID {job_id} status: {status}")
            return status
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