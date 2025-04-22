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
from pathlib import Path
from typing import Dict, Any, List, Optional

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
        default='plastinka-ml-data',
        help='Cloud storage bucket name'
    )
    
    parser.add_argument(
        '--memory', '-m',
        type=int,
        default=512,
        help='Memory limit in MB'
    )
    
    parser.add_argument(
        '--timeout', '-to',
        type=int,
        default=300,
        help='Function timeout in seconds'
    )
    
    parser.add_argument(
        '--env-file', '-e',
        help='Path to environment variables file'
    )
    
    parser.add_argument(
        '--requirements-file', '-r',
        default='deployment/scripts/cloud_requirements.txt',
        help='Path to requirements file for cloud functions'
    )
    
    parser.add_argument(
        '--skip-zipgen', '-z',
        action='store_true',
        help='Skip generating ZIP file (use existing)'
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
        'LOG_LEVEL': 'INFO'
    }
    
    if env_file and os.path.exists(env_file):
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


def deploy_function(
    function_type: str,
    zip_file: str,
    folder_id: str,
    service_account_id: str,
    memory: int,
    timeout: int,
    env_vars: Dict[str, str]
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
        
    Returns:
        True if deployment was successful
    """
    # Define function name
    function_name = f"plastinka-{function_type}-function"
    
    logger.info(f"Deploying {function_type} function: {function_name}")
    
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
            logger.error(f"Error creating function: {e}")
            logger.error(f"stderr: {e.stderr.decode()}")
            return False
    
    # Deploy new version
    deploy_cmd = (
        f"yc serverless function version create "
        f"--function-name {function_name} "
        f"--folder-id {folder_id} "
        f"--runtime python311 "
        f"--entrypoint handler "
        f"--source-path {zip_file} "
        f"{env_string}"
    )
    
    try:
        logger.info("Deploying function version...")
        subprocess.run(
            deploy_cmd, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        logger.info(f"Successfully deployed {function_type} function")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error deploying function: {e}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr.decode()}")
        return False


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Validate folder ID and service account ID
    folder_id = args.folder_id or os.environ.get('YANDEX_CLOUD_FOLDER_ID')
    service_account_id = args.service_account_id or os.environ.get('YANDEX_CLOUD_SERVICE_ACCOUNT_ID')
    
    if not folder_id:
        logger.error("Error: Folder ID is required")
        logger.error("Provide it with --folder-id or set YANDEX_CLOUD_FOLDER_ID environment variable")
        sys.exit(1)
    
    if not service_account_id:
        logger.error("Error: Service account ID is required")
        logger.error("Provide it with --service-account-id or set YANDEX_CLOUD_SERVICE_ACCOUNT_ID environment variable")
        sys.exit(1)
    
    # Get environment variables
    env_vars = get_env_vars(args.env_file)
    
    # Override bucket name if specified
    if args.bucket_name:
        env_vars['STORAGE_BUCKET'] = args.bucket_name
    
    # Deploy functions
    functions_to_deploy = []
    if args.function_type == 'all':
        functions_to_deploy = ['training', 'prediction']
    else:
        functions_to_deploy = [args.function_type]
    
    successful = 0
    for function_type in functions_to_deploy:
        # Create ZIP file
        zip_file = create_function_zip(function_type, args.requirements_file, args.skip_zipgen)
        
        # Deploy function
        if deploy_function(
            function_type, 
            zip_file, 
            folder_id, 
            service_account_id, 
            args.memory, 
            args.timeout, 
            env_vars
        ):
            successful += 1
    
    # Print summary
    logger.info(f"Deployment summary: {successful}/{len(functions_to_deploy)} functions deployed successfully")
    
    if successful < len(functions_to_deploy):
        sys.exit(1)
    
    # Get function information
    logger.info("Function URLs:")
    for function_type in functions_to_deploy:
        function_name = f"plastinka-{function_type}-function"
        cmd = f"yc serverless function get --name {function_name} --folder-id {folder_id} --format json"
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            function_info = json.loads(result.stdout)
            logger.info(f"  {function_type}: {function_info.get('id')}")
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            logger.error(f"Error getting function info: {e}")
    
    logger.info("Deployment completed")


if __name__ == "__main__":
    main() 