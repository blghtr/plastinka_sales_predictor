#!/usr/bin/env python
"""
Script to check environment variables and generate a template .env file if needed.
"""

import logging
import os
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("env_checker")

# Define the project root relative to this script's location
# This script is in deployment/scripts/, so project_root is ../../
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Define required environment variables
REQUIRED_VARS = {
    # DataSphere Integration
    "DATASPHERE_PROJECT_ID": "DataSphere project ID for ML model training and inference",
    "DATASPHERE_FOLDER_ID": "DataSphere folder ID (may differ from YANDEX_CLOUD_FOLDER_ID)",

    # Service Account Authentication (Recommended for Production)
    "DATASPHERE_YC_PROFILE": "YC CLI profile name for service account authentication (e.g., 'datasphere-prod')",
        
    # API Configuration
    "API_X_API_KEY": "API key for DataSphere API access",
}


def check_environment():
    """Check environment variables and return missing required ones."""
    missing = []
    for var in REQUIRED_VARS:
        if not os.environ.get(var):
            missing.append(var)

    return missing


def generate_env_template(output_path=None):
    """Generate a template .env file with all variables."""
    if output_path is None:
        # Default to .env.template in the project root
        output_path = PROJECT_ROOT / ".env.template"
    else:
        output_path = Path(output_path)

    if output_path.exists():
        logger.warning(f"File {output_path} already exists. Will not overwrite.")
        return False

    logger.info(f"Generating environment template at {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Plastinka Sales Predictor API - Environment Variables\n")
        f.write("# Generated template - fill with your values\n\n")

        # Required variables
        f.write("# Required Variables\n")
        for var, desc in REQUIRED_VARS.items():
            f.write(f"# {desc}\n{var}=\n\n")

    logger.info("Template generated. Please fill it with appropriate values.")
    return True


def main():
    """Main function."""
    logger.info(f"Project root directory: {PROJECT_ROOT}")
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
            logger.info(
                f"No {env_file_path.name} or {env_template_file_path.name} found in project root ({PROJECT_ROOT}). Generating template..."
            )
            generate_env_template()
            logger.info(
                f"Please fill {env_template_file_path} with your values and rename it to {env_file_path.name} in the project root."
            )

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
