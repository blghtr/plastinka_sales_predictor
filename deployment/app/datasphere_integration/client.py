import subprocess
import logging
import re
import json

logger = logging.getLogger(__name__)

# Import the datasphere SDK
try:
    import datasphere.sdk
    # Also import specific types if needed for hints, though mocking might make this less critical
    # from datasphere.api import jobs_pb2 as ds_jobs
    # from datasphere.sdk import JobWrapper
except ImportError:
    # Handle cases where the SDK might not be installed, although it should be
    # Log a warning or potentially disable SDK features
    datasphere = None # type: ignore
    logger.warning("DataSphere SDK (datasphere.sdk) not found. SDK features will be unavailable.")

class DataSphereClientError(Exception):
    """Custom exception for DataSphere client errors."""
    pass

class DataSphereClient:
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

    def submit_job_cli(self, config_path: str, params: dict = None) -> str:
        """
        Submits a DataSphere Job using the 'datasphere' CLI tool.

        Args:
            config_path: Path to the job configuration YAML file.
            params: Optional dictionary of parameters. NOTE: The 'datasphere' CLI doesn't directly accept
                    key-value params via flags. These would typically be handled by modifying the
                    config.yaml or passing input files defined within it. This argument is currently ignored.

        Returns:
            The submitted job ID.

        Raises:
            DataSphereClientError: If the CLI command fails or the job ID cannot be parsed.
            FileNotFoundError: If the config_path does not exist.
            ValueError: If params dict is provided but currently unsupported by this method.
        """
        # TODO: Revisit parameter handling. Options:
        # 1. Modify config.yaml dynamically before execution (complex, error-prone).
        # 2. Enforce parameters are handled via input files defined in config.yaml.
        # 3. Investigate if 'datasphere job execute' has hidden param options.
        if params:
            # For now, raise error if params are provided, as they aren't used.
            logger.warning("'params' argument provided to submit_job_cli but is currently ignored by the 'datasphere' CLI wrapper.")
            # Or raise ValueError("Parameter passing via dict is not supported by the datasphere CLI wrapper.")

        # Command uses 'datasphere' CLI tool, not 'yc datasphere'
        cmd = [
            'datasphere', # Use the specific tool
            'project', 'job', 'execute',
            '-p', self.project_id, # Use -p for project ID
            '-c', config_path,
            # No apparent flags for folder, format, async, or direct params
        ]

        logger.info(f"Running DataSphere CLI command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                check=False,
                text=False
            )

            stdout = result.stdout.decode('utf-8', errors='replace').strip()
            stderr = result.stderr.decode('utf-8', errors='replace').strip()

            if result.returncode != 0:
                # datasphere CLI errors might not be JSON
                error_message = f"DataSphere CLI command failed with exit code {result.returncode}."
                error_details = stderr or stdout # Prioritize stderr
                logger.error(f"{error_message}\nDetails:\n{error_details}")
                raise DataSphereClientError(f"{error_message} Error: {error_details}")

            # Parse Job ID from stdout - Reverting to brittle parsing
            # Assumption: The CLI outputs the Job ID on a line, possibly the last one.
            # We need to run the actual command or find better docs to confirm output.
            job_id = None
            lines = stdout.splitlines()
            # Try finding a line that looks like an ID (common YC format: c1q*********)
            id_pattern = r"^[a-zA-Z0-9]{20}$" # Typical YC ID format
            job_id_match = re.search(r"Job ID:\s*(\S+)", stdout, re.IGNORECASE) # Check specific text first

            if job_id_match:
                 job_id = job_id_match.group(1)
            else:
                # Look for likely ID on the last lines
                for line in reversed(lines):
                    potential_id = line.strip()
                    if re.match(id_pattern, potential_id):
                        job_id = potential_id
                        logger.info(f"Parsed potential Job ID from output line: {job_id}")
                        break

            if not job_id:
                error_message = "Failed to parse Job ID from DataSphere CLI output."
                logger.error(f"{error_message}\nSTDOUT:\n{stdout}")
                raise DataSphereClientError(f"{error_message} Output: {stdout}")

            logger.info(f"Successfully submitted DataSphere Job via CLI. ID: {job_id}")
            return job_id

        except FileNotFoundError:
            logger.error("'datasphere' command not found. Ensure datasphere CLI (pip install datasphere) is installed and in PATH.")
            raise DataSphereClientError("'datasphere' command not found. Is the datasphere library installed?")
        except Exception as e:
            # logger.exception handles including exception info automatically
            logger.exception("An unexpected error occurred during DataSphere CLI execution.")
            # Re-raise specific client error or the original exception depending on desired handling
            if isinstance(e, DataSphereClientError):
                raise
            # Wrap the original exception for context
            raise DataSphereClientError(f"An unexpected error occurred: {e}") from e

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