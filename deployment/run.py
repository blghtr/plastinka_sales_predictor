#!/usr/bin/env python
import uvicorn
import os
import argparse
from pathlib import Path

# Create necessary directories
def create_directories():
    directories = [
        "deployment/data",
        "deployment/data/uploads",
        "deployment/data/predictions",
        "deployment/data/reports",
        "deployment/logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

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
        workers=args.workers
    )

if __name__ == "__main__":
    main() 