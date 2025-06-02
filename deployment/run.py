#!/usr/bin/env python
import uvicorn
import os
import argparse
from pathlib import Path
import sys

# Find the project root directory (one level up from deployment)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Add the project root to sys.path
# This allows importing modules like 'deployment.app.main' correctly
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Create necessary directories
def create_directories():
    # Define paths relative to this script's location
    script_dir = Path(__file__).resolve().parent
    
    # Directories to be created relative to script_dir (which is '.../deployment/')
    relative_dirs = [
        "data",                 # -> deployment/data
        "data/uploads",         # -> deployment/data/uploads
        "data/predictions",     # -> deployment/data/predictions
        "data/reports",         # -> deployment/data/reports
        "logs"                  # -> deployment/logs
    ]
    
    for rel_dir in relative_dirs:
        # Construct the absolute path relative to the script directory
        abs_dir_path = script_dir / rel_dir
        abs_dir_path.mkdir(parents=True, exist_ok=True)

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run the Plastinka Sales Predictor API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    return parser.parse_args()

# Main function
def main():
    args = parse_args()
    
    # Create directories
    create_directories()
    
    # Run the application
    uvicorn.run(
        "deployment.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        app_dir="."
    )

if __name__ == "__main__":
    main() 