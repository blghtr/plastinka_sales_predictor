#!/usr/bin/env python
"""
Script for deploying cloud functions to Yandex Cloud.
"""
import os
import sys
import json
import argparse
import logging
import subprocess
import zipfile
import tempfile
import shutil
import time
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('deploy_cloud_functions')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy cloud functions to Yandex Cloud')
    
    parser.add_argument(
        '--function-type', '-t',
        choices=['training', 'prediction', 'all'],
        default='all',
        help='Type of function to deploy'
    )
    
    parser.add_argument(
        '--folder-id', '-f',
        help='Yandex Cloud folder ID'
    )
    
    parser.add_argument(
        '--service-account-id', '-s',
        help='Yandex Cloud service account ID'
    )
    
    parser.add_argument(
        '--bucket-name', '-b',
        default=os.environ.get('YANDEX_CLOUD_BUCKET', 'plastinka-ml-data'),
        help='Cloud storage bucket name'
    )
    
    parser.add_argument(
        '--memory', '-m',
        type=int,
        default=int(os.environ.get('CLOUD_FUNCTION_MEMORY', '512')),
        help='Memory limit in MB'
    )
    
    parser.add_argument(
        '--timeout', '-to',
        type=int,
        default=int(os.environ.get('CLOUD_FUNCTION_TIMEOUT', '300')),
        help='Function timeout in seconds'
    )
    
    parser.add_argument(
        '--env-file', '-e',
        default=os.environ.get('CLOUD_ENV_FILE', '.env'),
        help='Path to environment variables file'
    )
    
    parser.add_argument(
        '--requirements-file', '-r',
        default=os.environ.get('CLOUD_REQUIREMENTS_FILE', 'deployment/scripts/cloud_requirements.txt'),
        help='Path to requirements file for cloud functions'
    )
    
    parser.add_argument(
        '--skip-zipgen', '-z',
        action='store_true',
        help='Skip generating ZIP file (use existing)'
    )
    
    parser.add_argument(
        '--rollback-on-failure', '-R',
        action='store_true',
        default=bool(os.environ.get('ENABLE_ROLLBACK', 'false').lower() == 'true'),
        help='Enable automatic rollback on deployment failure'
    )
    
    # New arguments
    parser.add_argument(
        '--concurrent', '-C',
        action='store_true',
        default=bool(os.environ.get('ENABLE_CONCURRENT_DEPLOY', 'false').lower() == 'true'),
        help='Deploy functions concurrently'
    )
    
    parser.add_argument(
        '--dry-run', '-D',
        action='store_true',
        help='Perform dry run without actual deployment'
    )
    
    parser.add_argument(
        '--verify', '-V',
        action='store_true',
        default=bool(os.environ.get('VERIFY_DEPLOYMENT', 'false').lower() == 'true'),
        help='Verify function after deployment'
    )
    
    return parser.parse_args()


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        # Check if yc CLI is installed
        subprocess.run(['yc', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Yandex Cloud CLI is installed")
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Error: Yandex Cloud CLI (yc) is not installed or not in PATH")
        logger.error("Please install it from https://cloud.yandex.com/en/docs/cli/quickstart")
        sys.exit(1)


def get_env_vars(env_file: Optional[str]) -> Dict[str, str]:
    """
    Get environment variables from file or environment.
    
    Args:
        env_file: Path to environment variables file
        
    Returns:
        Dict of environment variables
    """
    env_vars = {
        'API_ENDPOINT': os.environ.get('FASTAPI_CALLBACK_BASE_URL', 'http://localhost:8000'),
        'API_KEY': os.environ.get('CLOUD_CALLBACK_AUTH_TOKEN', ''),
        'STORAGE_BUCKET': os.environ.get('YANDEX_CLOUD_BUCKET', 'plastinka-ml-data'),
        'STORAGE_ACCESS_KEY': os.environ.get('YANDEX_CLOUD_ACCESS_KEY', ''),
        'STORAGE_SECRET_KEY': os.environ.get('YANDEX_CLOUD_SECRET_KEY', ''),
        'STORAGE_ENDPOINT': os.environ.get('YANDEX_CLOUD_STORAGE_ENDPOINT', 'https://storage.yandexcloud.net'),
        'STORAGE_REGION': os.environ.get('YANDEX_CLOUD_REGION', 'ru-central1'),
        'LOG_LEVEL': os.environ.get('LOG_LEVEL', 'INFO')
    }
    
    if env_file and os.path.exists(env_file):
        logger.info(f"Loading environment variables from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key] = value
    
    # Check for required variables
    missing_vars = []
    for var in ['STORAGE_ACCESS_KEY', 'STORAGE_SECRET_KEY']:
        if not env_vars.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.warning("Function deployment may fail without these variables")
    
    return env_vars


def create_function_zip(function_type: str, requirements_file: str, skip_zipgen: bool) -> str:
    """
    Create ZIP file for cloud function.
    
    Args:
        function_type: Type of function ('training' or 'prediction')
        requirements_file: Path to requirements file
        skip_zipgen: Skip ZIP generation if True
        
    Returns:
        Path to ZIP file
    """
    # Define paths
    root_dir = Path(__file__).parent.parent.parent
    function_dir = root_dir / 'plastinka_sales_predictor' / 'cloud_functions' / function_type
    zip_file = root_dir / 'deployment' / 'dist' / f"{function_type}_function.zip"
    
    # Create dist directory if it doesn't exist
    os.makedirs(os.path.dirname(zip_file), exist_ok=True)
    
    # Skip ZIP generation if requested and file exists
    if skip_zipgen and os.path.exists(zip_file):
        logger.info(f"Using existing ZIP file: {zip_file}")
        return str(zip_file)
    
    logger.info(f"Creating ZIP file for {function_type} function")
    
    # Create temporary directory for building the ZIP
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy function code to temp directory
        temp_function_dir = Path(temp_dir) / function_type
        shutil.copytree(function_dir, temp_function_dir)
        
        # Copy requirements file
        shutil.copy(requirements_file, temp_function_dir / 'requirements.txt')
        
        # Create ZIP file
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(temp_function_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_function_dir)
                    zipf.write(file_path, arcname)
    
    logger.info(f"Created ZIP file: {zip_file}")
    return str(zip_file)


def get_current_function_version(function_name: str, folder_id: str) -> Optional[str]:
    """
    Get the current version ID of a function.
    
    Args:
        function_name: Name of the function
        folder_id: Yandex Cloud folder ID
        
    Returns:
        Version ID or None if function doesn't exist or has no versions
    """
    try:
        # Get the latest version of the function
        cmd = f"yc serverless function version list --function-name={function_name} --folder-id={folder_id} --format=json"
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        versions = json.loads(result.stdout)
        if versions:
            # Return the ID of the latest version
            return versions[0].get('id')
        
        return None
    except subprocess.CalledProcessError:
        # Function doesn't exist or has no versions
        return None
    except Exception as e:
        logger.error(f"Error getting function version: {str(e)}")
        return None


def rollback_function(function_name: str, folder_id: str, version_id: Optional[str]) -> bool:
    """
    Rollback a function to a previous version.
    
    Args:
        function_name: Name of the function
        folder_id: Yandex Cloud folder ID
        version_id: Version ID to rollback to
        
    Returns:
        True if rollback was successful
    """
    if not version_id:
        logger.warning(f"No previous version found for {function_name}, cannot rollback")
        return False
    
    try:
        logger.info(f"Rolling back {function_name} to version {version_id}")
        cmd = f"yc serverless function version set-tag default --function-name={function_name} --folder-id={folder_id} --id={version_id}"
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"Successfully rolled back {function_name} to version {version_id}")
        return True
    except Exception as e:
        logger.error(f"Error rolling back {function_name}: {str(e)}")
        return False


def deploy_function(
    function_type: str,
    zip_file: str,
    folder_id: str,
    service_account_id: str,
    memory: int,
    timeout: int,
    env_vars: Dict[str, str],
    enable_rollback: bool,
    dry_run: bool = False,
    verify: bool = False
) -> bool:
    """
    Deploy a cloud function to Yandex Cloud.
    
    Args:
        function_type: Type of function ('training' or 'prediction')
        zip_file: Path to ZIP file
        folder_id: Yandex Cloud folder ID
        service_account_id: Service account ID
        memory: Memory limit in MB
        timeout: Function timeout in seconds
        env_vars: Environment variables
        enable_rollback: Enable rollback on failure
        dry_run: Perform dry run without actual deployment
        verify: Verify function after deployment
        
    Returns:
        True if deployment was successful
    """
    # Define function name
    function_name = f"plastinka-{function_type}-function"
    
    # In dry-run mode, just log what would happen
    if dry_run:
        logger.info(f"DRY RUN: Would deploy {function_type} function: {function_name}")
        logger.info(f"DRY RUN: ZIP file: {zip_file}")
        logger.info(f"DRY RUN: Memory: {memory}MB, Timeout: {timeout}s")
        logger.info(f"DRY RUN: Environment variables: {list(env_vars.keys())}")
        return True
    
    # Display progress indicator
    logger.info(f"Deploying {function_type} function: {function_name}")
    logger.info("[1/5] Preparing deployment...")
    
    # Store current version for potential rollback
    previous_version = None
    if enable_rollback:
        previous_version = get_current_function_version(function_name, folder_id)
        if previous_version:
            logger.info(f"[2/5] Stored previous version {previous_version} for potential rollback")
        else:
            logger.info("[2/5] No previous version found for rollback")
    else:
        logger.info("[2/5] Rollback disabled, skipping version check")
    
    # Format environment variables for yc command
    env_string = " ".join([f"--environment {k}={v}" for k, v in env_vars.items()])
    
    # Create yc command
    cmd = (
        f"yc serverless function create "
        f"--name {function_name} "
        f"--folder-id {folder_id} "
        f"--description \"Plastinka {function_type} function\" "
        f"--memory {memory}m "
        f"--execution-timeout {timeout}s "
        f"--service-account-id {service_account_id}"
    )
    
    # Check if function already exists
    logger.info("[3/5] Checking if function exists...")
    check_cmd = f"yc serverless function get --name {function_name} --folder-id {folder_id}"
    try:
        subprocess.run(
            check_cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        logger.info(f"Function {function_name} already exists, updating...")
    except subprocess.CalledProcessError:
        # Function doesn't exist, create it
        try:
            logger.info(f"Creating function: {function_name}")
            subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create function {function_name}: {e}")
            return False
    
    # Deploy new version
    try:
        # Create version command
        version_cmd = (
            f"yc serverless function version create "
            f"--function-name {function_name} "
            f"--folder-id {folder_id} "
            f"--runtime python311 "
            f"--entrypoint handler "
            f"--memory {memory}m "
            f"--execution-timeout {timeout}s "
            f"--service-account-id {service_account_id} "
            f"--source-path {zip_file} "
            f"{env_string}"
        )
        
        # Run command
        logger.info(f"[4/5] Creating version for {function_name}")
        process = subprocess.run(
            version_cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Parse version ID from output
        version_id = None
        if '"id": "' in process.stdout:
            version_id = process.stdout.split('"id": "')[1].split('"')[0]
            logger.info(f"Created version {version_id} for {function_name}")
        else:
            logger.info("Version created successfully")
        
        # Retrieve and display function URL
        if version_id:
            function_url = get_function_url(function_name, folder_id)
            if function_url:
                logger.info(f"Function URL: {function_url}")
        
        logger.info(f"[5/5] Function {function_name} deployed successfully")
        
        # Verify function if requested
        if verify and version_id:
            if verify_function_deployment(function_name, folder_id, version_id):
                logger.info(f"Verification of {function_name} successful")
            else:
                logger.warning(f"Verification of {function_name} failed")
                # We don't fail the deployment since it technically succeeded
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to deploy function {function_name}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        
        # Perform rollback if enabled
        if enable_rollback and previous_version:
            logger.warning(f"Deployment failed, attempting rollback for {function_name}")
            rollback_function(function_name, folder_id, previous_version)
        
        return False


def get_function_url(function_name: str, folder_id: str) -> Optional[str]:
    """
    Get the HTTP URL for a function.
    
    Args:
        function_name: Name of the function
        folder_id: Yandex Cloud folder ID
        
    Returns:
        Function URL or None if not available
    """
    try:
        cmd = f"yc serverless function get --name {function_name} --folder-id {folder_id} --format=json"
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        function_data = json.loads(result.stdout)
        return function_data.get("http_invoke_url")
    except Exception as e:
        logger.error(f"Error getting function URL: {str(e)}")
        return None


def verify_function_deployment(function_name: str, folder_id: str, version_id: str) -> bool:
    """
    Verify that a function was deployed correctly.
    
    Args:
        function_name: Name of the function
        folder_id: Yandex Cloud folder ID
        version_id: Version ID
        
    Returns:
        True if verification was successful
    """
    logger.info(f"Verifying deployment of {function_name}...")
    
    # Verify function exists and is active
    try:
        # Wait for function to be ready
        for attempt in range(3):
            cmd = f"yc serverless function version get --function-name={function_name} --folder-id={folder_id} --id={version_id} --format=json"
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            version_data = json.loads(result.stdout)
            status = version_data.get("status")
            
            if status == "ACTIVE":
                logger.info(f"Function {function_name} version {version_id} is active")
                return True
            
            logger.info(f"Function status: {status}, waiting for ACTIVE... (attempt {attempt+1}/3)")
            time.sleep(5)
        
        logger.warning(f"Function {function_name} version {version_id} did not become active within timeout")
        return False
    except Exception as e:
        logger.error(f"Error verifying function deployment: {str(e)}")
        return False


async def deploy_functions_concurrently(functions_to_deploy, deployment_args):
    """
    Deploy multiple functions concurrently.
    
    Args:
        functions_to_deploy: List of function types to deploy
        deployment_args: Dictionary of deployment arguments
    
    Returns:
        Dictionary with deployment results
    """
    loop = asyncio.get_event_loop()
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = []
        for function_type in functions_to_deploy:
            logger.info(f"Scheduling deployment of {function_type} function")
            
            # Create ZIP file
            zip_file = create_function_zip(
                function_type=function_type,
                requirements_file=deployment_args['requirements_file'],
                skip_zipgen=deployment_args['skip_zipgen']
            )
            
            # Create task for concurrent deployment
            task = loop.run_in_executor(
                executor,
                deploy_function,
                function_type,
                zip_file,
                deployment_args['folder_id'],
                deployment_args['service_account_id'],
                deployment_args['memory'],
                deployment_args['timeout'],
                deployment_args['env_vars'],
                deployment_args['enable_rollback'],
                deployment_args['dry_run'],
                deployment_args['verify']
            )
            tasks.append((function_type, task))
        
        # Wait for all deployments to complete
        for function_type, task in tasks:
            success = await task
            results[function_type] = "success" if success else "failed"
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Folder ID and service account ID can come from arguments or environment variables
    folder_id = args.folder_id or os.environ.get('YANDEX_CLOUD_FOLDER_ID')
    service_account_id = args.service_account_id or os.environ.get('YANDEX_CLOUD_SERVICE_ACCOUNT_ID')
    
    # Validate required parameters
    if not folder_id:
        logger.error("Error: folder_id is required")
        logger.error("Provide it with --folder-id argument or YANDEX_CLOUD_FOLDER_ID environment variable")
        sys.exit(1)
    
    if not service_account_id:
        logger.error("Error: service_account_id is required")
        logger.error("Provide it with --service-account-id argument or YANDEX_CLOUD_SERVICE_ACCOUNT_ID environment variable")
        sys.exit(1)
    
    # Get environment variables
    env_vars = get_env_vars(args.env_file)
    
    # Function types to deploy
    function_types = ['training', 'prediction'] if args.function_type == 'all' else [args.function_type]
    
    # Results tracking
    results = {}
    
    if args.dry_run:
        logger.info("=== DRY RUN MODE ENABLED ===")
        logger.info("No actual deployment will occur")
    
    # Create deployment arguments dictionary
    deployment_args = {
        'folder_id': folder_id,
        'service_account_id': service_account_id,
        'memory': args.memory,
        'timeout': args.timeout,
        'env_vars': env_vars,
        'enable_rollback': args.rollback_on_failure,
        'requirements_file': args.requirements_file,
        'skip_zipgen': args.skip_zipgen,
        'dry_run': args.dry_run,
        'verify': args.verify
    }
    
    # Deploy functions (either concurrently or sequentially)
    if args.concurrent and len(function_types) > 1:
        logger.info("=== CONCURRENT DEPLOYMENT ENABLED ===")
        try:
            # Get event loop or create one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run concurrent deployment
            results = loop.run_until_complete(
                deploy_functions_concurrently(function_types, deployment_args)
            )
        except Exception as e:
            logger.error(f"Error during concurrent deployment: {str(e)}")
            sys.exit(1)
    else:
        # Deploy functions sequentially
        for function_type in function_types:
            logger.info(f"\n=== Processing {function_type} function ===")
            
            # Create ZIP file
            zip_file = create_function_zip(function_type, args.requirements_file, args.skip_zipgen)
            
            # Deploy function
            success = deploy_function(
                function_type=function_type,
                zip_file=zip_file,
                folder_id=folder_id,
                service_account_id=service_account_id,
                memory=args.memory,
                timeout=args.timeout,
                env_vars=env_vars,
                enable_rollback=args.rollback_on_failure,
                dry_run=args.dry_run,
                verify=args.verify
            )
            
            results[function_type] = "success" if success else "failed"
    
    # Print summary
    logger.info("\n=== Deployment summary ===")
    for function_type, status in results.items():
        logger.info(f"{function_type}: {status}")
    
    # Check if any deployment failed
    if "failed" in results.values():
        logger.error("Some deployments failed, check logs for details")
        sys.exit(1)
    else:
        logger.info("All deployments completed successfully")


if __name__ == '__main__':
    main() 