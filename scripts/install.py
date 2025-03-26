#!/usr/bin/env python3
"""Installation script for Plastinka Sales Predictor using UV."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and handle errors."""
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed: {' '.join(cmd)}")
        print(f"Error output: {e.stderr}")
        sys.exit(1)


def main():
    print("Installing Plastinka Sales Predictor...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Create virtual environment
    print("Creating virtual environment...")
    run_command(["uv", "venv"])
    
    print("Change working dir to project root...")
    os.chdir(project_root)

    # Install project in development mode
    print("Installing project dependencies...")
    run_command(["uv", "pip", "install", "-e", "."])
    
    # Create main directories
    print("Creating main directories...")
    run_command(["mkdir", "-p", "datasets", "models", "results", "logs", "configs"])

    print("\nInstallation completed successfully!")

if __name__ == "__main__":
    main()