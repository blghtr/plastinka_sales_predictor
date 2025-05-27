#!/usr/bin/env python
"""
Script to check environment variables and generate a template .env file if needed.
"""
import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("env_checker")

# Define the project root relative to this script's location
# This script is in deployment/scripts/, so project_root is ../../
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define required environment variables
REQUIRED_VARS = {
    # Cloud Storage
    "YANDEX_CLOUD_ACCESS_KEY": "Access key for Yandex Cloud storage",
    "YANDEX_CLOUD_SECRET_KEY": "Secret key for Yandex Cloud storage",
    
    # Cloud Functions
    "YANDEX_CLOUD_FOLDER_ID": "Yandex Cloud folder ID",
    "YANDEX_CLOUD_SERVICE_ACCOUNT_ID": "Yandex Cloud service account ID",
    "YANDEX_CLOUD_API_KEY": "API key for Yandex Cloud",
    
    # Callbacks
    "CLOUD_CALLBACK_AUTH_TOKEN": "Authentication token for cloud function callbacks"
}

# Optional environment variables with defaults
OPTIONAL_VARS = {
    "APP_ENV": ("Application environment", "development"),
    "ALLOWED_ORIGINS": ("Comma-separated list of allowed CORS origins", "http://localhost:3000"),
    "DATABASE_PATH": ("Path to SQLite database file", "deployment/data/plastinka.db"),
    "LOG_LEVEL": ("Application logging level", "INFO"),
    "YANDEX_CLOUD_STORAGE_ENDPOINT": ("Endpoint URL for cloud storage", "https://storage.yandexcloud.net"),
    "YANDEX_CLOUD_REGION": ("Region for cloud storage", "ru-central1"),
    "YANDEX_CLOUD_BUCKET": ("Bucket name for ML data", "plastinka-ml-data"),
    "FASTAPI_CALLBACK_BASE_URL": ("Base URL for cloud function callbacks", "http://localhost:8000"),
    "MAX_UPLOAD_SIZE": ("Maximum size for file uploads in bytes", "52428800"),
    "CLOUD_FUNCTION_MEMORY": ("Memory limit for cloud functions in MB", "512"),
    "CLOUD_FUNCTION_TIMEOUT": ("Timeout for cloud functions in seconds", "300"),
    "ENABLE_ROLLBACK": ("Enable automatic rollback on deployment failure", "false")
}


def check_environment():
    """Check environment variables and return missing required ones."""
    missing = []
    for var in REQUIRED_VARS:
        if not os.environ.get(var):
            missing.append(var)
    
    return missing


def generate_env_template(output_path=None): # Changed default to None
    """Generate a template .env file with all variables."""
    if output_path is None: # Check against None
        # Default to .env.template in the project root
        output_path = PROJECT_ROOT / ".env.template"
    else:
        output_path = Path(output_path) # Convert provided path string to Path object
        
    if output_path.exists():
        logger.warning(f"File {output_path} already exists. Will not overwrite.")
        return False
    
    logger.info(f"Generating environment template at {output_path}")
    
    with open(output_path, "w") as f:
        f.write("# Plastinka Sales Predictor API - Environment Variables\n")
        f.write("# Generated template - fill with your values\n\n")
        
        # Required variables
        f.write("# Required Variables\n")
        for var, desc in REQUIRED_VARS.items():
            f.write(f"# {desc}\n{var}=\n\n")
        
        # Optional variables
        f.write("# Optional Variables (with defaults)\n")
        for var, (desc, default) in OPTIONAL_VARS.items():
            f.write(f"# {desc}\n{var}={default}\n\n")
    
    logger.info("Template generated. Please fill it with appropriate values.")
    return True


def main():
    """Main function."""
    logger.info("Checking environment variables...")
    
    # Check for required environment variables
    missing = check_environment()
    
    if missing:
        logger.warning(f"Missing {len(missing)} required environment variables:")
        for var in missing:
            logger.warning(f"  - {var}: {REQUIRED_VARS[var]}")
        
        # Create .env.template file in project root if .env or .env.template doesn't exist there
        env_file_path = PROJECT_ROOT / ".env"
        env_template_file_path = PROJECT_ROOT / ".env.template"

        if not env_file_path.exists() and not env_template_file_path.exists():
            logger.info(f"No {env_file_path.name} or {env_template_file_path.name} found in project root ({PROJECT_ROOT}). Generating template...")
            generate_env_template() # This will default to PROJECT_ROOT / ".env.template"
            logger.info(f"Please fill {env_template_file_path} with your values and rename it to {env_file_path.name} in the project root.")
            
        return 1
    else:
        logger.info("All required environment variables are set.")
        return 0


if __name__ == "__main__":
    # If an argument is provided, use it as the output path
    if len(sys.argv) > 1:
        generate_env_template(sys.argv[1])
    else:
        sys.exit(main()) 