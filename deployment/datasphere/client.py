import subprocess
import logging
import re
import json
import time # Added for potential delay
import datasphere

logger = logging.getLogger(__name__)


class DataSphereClientError(Exception):
    """Custom exception for DataSphere client errors."""
    pass

class DataSphereClient:  # TODO: Проверить, действительно ли нам нужен свой клиент или можно использовать datasphere.client.Client 
    """Client for interacting with Yandex DataSphere Jobs (SDK and CLI fallback)."""

    def __init__(self, project_id: str, folder_id: str, oauth_token: str = None, yc_profile: str = None):
        """
        Initializes the client and the DataSphere SDK instance.

        Args:
            project_id: Yandex DataSphere project ID.
            folder_id: Yandex Cloud folder ID.
            oauth_token: Yandex Cloud OAuth token (optional, uses profile/env if None).
            yc_profile: Yandex Cloud CLI profile name (optional).
        """
        if not project_id:
            raise ValueError("DataSphere project_id is required")
        # folder_id might not be used by SDK directly, but good to have
        if not folder_id:
            raise ValueError("Yandex Cloud folder_id is required")

        self.project_id = project_id
        self.folder_id = folder_id
        self._sdk = None

        if datasphere and datasphere.sdk: # Check if SDK was imported successfully
            try:
                # Initialize SDK - uses OAuth token or falls back to yc profile/env vars
                self._sdk = datasphere.sdk.SDK(oauth_token=oauth_token, profile=yc_profile)
                logger.info("DataSphere SDK initialized successfully.")
            except Exception as e:
                # Catch potential errors during SDK initialization (e.g., auth issues)
                logger.exception(f"Failed to initialize DataSphere SDK: {e}")
                # Client can still function for CLI operations if SDK init fails
        else:
            logger.warning("DataSphere SDK not available, only CLI operations supported.")

        logger.info(
            f"DataSphereClient initialized. Project ID: '{self.project_id}', Folder ID: '{self.folder_id}'"
        )

    def _ensure_sdk_available(self):
        """Checks if the SDK was initialized successfully."""
        if not self._sdk:
            raise DataSphereClientError("DataSphere SDK is not available or failed to initialize.")

    def _run_cli_command(self, cmd: list) -> tuple[int, str, str]:
        """Runs a CLI command and returns return code, stdout, stderr."""
        logger.info(f"Running DataSphere CLI command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=False, # Don't raise CalledProcessError automatically
                text=False
            )
            stdout = result.stdout.decode('utf-8', errors='replace').strip()
            stderr = result.stderr.decode('utf-8', errors='replace').strip()
            return result.returncode, stdout, stderr
        except FileNotFoundError:
            logger.error(f"'{cmd[0]}' command not found. Is it installed and in PATH?")
            raise DataSphereClientError(f"'{cmd[0]}' command not found. Is the datasphere library installed?")
        except Exception as e:
            logger.exception("An unexpected error occurred during CLI execution.")
            raise DataSphereClientError(f"An unexpected error occurred during CLI execution: {e}") from e

    def _list_job_ids_cli(self) -> set[str]:
        """Lists DataSphere Job IDs for the project using the CLI."""
        cmd = [
            'datasphere', 'project', 'job', 'list',
            '-p', self.project_id,
            '--format', 'json' # Request JSON output for easier parsing
        ]
        returncode, stdout, stderr = self._run_cli_command(cmd)

        if returncode != 0:
            error_message = f"Failed to list DataSphere jobs. Exit code {returncode}."
            error_details = stderr or stdout
            logger.error(f"{error_message}\nDetails:\n{error_details}")
            raise DataSphereClientError(f"{error_message} Error: {error_details}")

        try:
            # Parse JSON output
            jobs_data = json.loads(stdout)
            if not isinstance(jobs_data, list):
                raise ValueError("Expected a list of jobs from CLI output")
            
            job_ids = set(job['id'] for job in jobs_data if isinstance(job, dict) and 'id' in job)
            logger.debug(f"Found {len(job_ids)} job IDs: {job_ids}")
            return job_ids
        except json.JSONDecodeError as e:
            error_message = "Failed to parse JSON output from 'datasphere project job list' command."
            logger.error(f"{error_message}\nSTDOUT:\n{stdout}\nError: {e}")
            raise DataSphereClientError(f"{error_message} Error: {e}") from e
        except (KeyError, TypeError, ValueError) as e:
            error_message = "Unexpected format in JSON output from 'datasphere project job list' command."
            logger.error(f"{error_message}\nSTDOUT:\n{stdout}\nError: {e}")
            raise DataSphereClientError(f"{error_message} Error: {e}") from e

    def submit_job_cli(self, config_path: str) -> str:
        """
        Submits a DataSphere Job using the 'datasphere' CLI tool.
        Determines the new job ID by comparing job lists before and after execution.

        Args:
            config_path: Path to the job configuration YAML file.

        Returns:
            The submitted job ID.

        Raises:
            DataSphereClientError: If the CLI command fails, job ID cannot be determined,
                                   or multiple new jobs appear.
        """
        logger.info("Attempting to submit job and determine ID via list comparison.")
        
        # 1. Get job IDs before execution
        try:
            ids_before = self._list_job_ids_cli()
        except DataSphereClientError as e:
            logger.error(f"Failed to list jobs before execution: {e}")
            raise DataSphereClientError(f"Could not list jobs before submission: {e}") from e

        # 2. Execute the job
        execute_cmd = [
            'datasphere', 'project', 'job', 'execute',
            '-p', self.project_id,
            '-c', config_path,
        ]
        returncode, stdout, stderr = self._run_cli_command(execute_cmd)

        if returncode != 0:
            error_message = f"DataSphere job execution command failed. Exit code {returncode}."
            error_details = stderr or stdout
            logger.error(f"{error_message}\nDetails:\n{error_details}")
            raise DataSphereClientError(f"{error_message} Error: {error_details}")

        logger.info(f"Job execution command succeeded. Output:\n{stdout}")
        
        # 3. Get job IDs after execution (with potential delay)
        # Sometimes the list might not update instantly
        time.sleep(2) # Add a small delay
        try:
            ids_after = self._list_job_ids_cli()
        except DataSphereClientError as e:
            logger.error(f"Failed to list jobs after execution: {e}")
            raise DataSphereClientError(f"Could not list jobs after submission: {e}") from e
            
        # 4. Find the difference
        new_ids = ids_after - ids_before

        if len(new_ids) == 1:
            new_job_id = new_ids.pop()
            logger.info(f"Successfully submitted DataSphere Job via CLI. Determined ID: {new_job_id}")
            return new_job_id
        elif len(new_ids) == 0:
            error_message = "Failed to determine new Job ID: No new job found after execution."
            logger.error(f"{error_message} Before: {ids_before}, After: {ids_after}")
            raise DataSphereClientError(error_message)
        else:
            error_message = f"Failed to determine unique Job ID: Found multiple new jobs ({len(new_ids)}): {new_ids}"
            logger.error(f"{error_message} Before: {ids_before}, After: {ids_after}")
            raise DataSphereClientError(error_message)

    def get_job_status(self, job_id: str) -> str:
        """Gets the status of a DataSphere Job using the SDK."""
        self._ensure_sdk_available()
        try:
            logger.info(f"Getting status for DataSphere Job ID: {job_id} via SDK")
            job_wrapper = self._sdk.get_job(job_id) # type: ignore

            # Assuming status is an attribute on the nested job object
            # The actual structure might be different (e.g., enum)
            if not job_wrapper or not hasattr(job_wrapper, 'job') or not hasattr(job_wrapper.job, 'status'):
                 logger.error(f"Could not retrieve valid job status object for Job ID: {job_id}")
                 raise DataSphereClientError(f"Invalid job data received from SDK for Job ID: {job_id}")

            status = job_wrapper.job.status
            # Convert status enum/object to string if necessary - for now, assume it's usable directly
            logger.info(f"Job ID {job_id} status: {status}")
            # Convert status to string representation if it's an enum
            return str(status)

        except Exception as e:
            # Catch potential SDK errors (e.g., job not found, network issues)
            logger.exception(f"Failed to get status for Job ID {job_id} using SDK: {e}")
            raise DataSphereClientError(f"SDK error getting status for Job ID {job_id}: {e}") from e

    # --- Placeholder methods for other functionalities (to be implemented) ---

    def download_job_results(self, job_id: str, output_dir: str, with_logs: bool = False, with_diagnostics: bool = False):
        """Downloads the results (and optionally logs/diagnostics) of a DataSphere Job using the SDK.

        Args:
            job_id: The ID of the job whose files should be downloaded.
            output_dir: The local directory to download files into.
            with_logs: Whether to include job logs (default: False).
            with_diagnostics: Whether to include diagnostic files (default: False).

        Raises:
            DataSphereClientError: If the SDK is unavailable or the download operation fails.
            # Note: The underlying SDK method might raise other exceptions (e.g., FileNotFoundError if output_dir invalid).
        """
        self._ensure_sdk_available()
        try:
            logger.info(f"Attempting to download files for Job ID: {job_id} to {output_dir} via SDK")
            logger.info(f"Download options: logs={with_logs}, diagnostics={with_diagnostics}")

            # Ensure the SDK instance has the download method
            if not hasattr(self._sdk, 'download_job_files'):
                 raise DataSphereClientError("SDK structure mismatch: cannot find download_job_files method.")

            self._sdk.download_job_files(
                id=job_id,
                with_logs=with_logs,
                with_diagnostics=with_diagnostics,
                output_dir=output_dir
            ) # type: ignore
            logger.info(f"Successfully requested download for Job ID: {job_id}")
            # Note: Actual download happens within the SDK method.

        except Exception as e:
            logger.exception(f"Failed to download results for Job ID {job_id} using SDK: {e}")
            raise DataSphereClientError(f"SDK error downloading results for Job ID {job_id}: {e}") from e

    def cancel_job(self, job_id: str, graceful: bool = False):
        """Cancels a running DataSphere Job using the SDK.

        Args:
            job_id: The ID of the job to cancel.
            graceful: Whether to attempt graceful shutdown (default: False).
                      Note: Behavior depends on SDK/backend implementation.

        Raises:
            DataSphereClientError: If the SDK is unavailable or the cancel operation fails.
        """
        self._ensure_sdk_available()
        try:
            logger.info(f"Attempting to cancel DataSphere Job ID: {job_id} (graceful={graceful}) via SDK")
            # The SDK class has a _cancel method which calls self.client.cancel
            # Let's call the internal client directly as patching _cancel is harder.
            # Assuming the _sdk instance has a 'client' attribute based on SDK source.
            if not hasattr(self._sdk, 'client') or not hasattr(self._sdk.client, 'cancel'):
                raise DataSphereClientError("SDK structure mismatch: cannot find internal client or cancel method.")

            self._sdk.client.cancel(job_id=job_id, graceful=graceful) # type: ignore
            logger.info(f"Cancel request sent for Job ID: {job_id}")
            # Note: Cancellation is likely asynchronous, this method doesn't wait for confirmation.

        except Exception as e:
            logger.exception(f"Failed to cancel Job ID {job_id} using SDK: {e}")
            raise DataSphereClientError(f"SDK error canceling Job ID {job_id}: {e}") from e 