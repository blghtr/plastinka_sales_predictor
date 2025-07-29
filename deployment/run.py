#!/usr/bin/env python
"""
Development server runner for Plastinka Sales Predictor API.
For production deployment, use gunicorn with systemd service.
"""
import argparse
from pathlib import Path

import uvicorn

from deployment.app.config import get_settings

# Find the project root directory (one level up from deployment)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Plastinka Sales Predictor API (Development)"
    )
    parser.add_argument(
        "--host", default=None, help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to bind to (overrides config)"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    # Remove workers argument as it's for production use with gunicorn

    return parser.parse_args()


# Main function
def main():
    args = parse_args()
    settings = get_settings()

    # Use configuration values with command line overrides
    host = args.host if args.host is not None else settings.api.host
    port = args.port if args.port is not None else settings.api.port

    # Run the application (single worker for development)
    uvicorn.run(
        "deployment.app.main:app",
        host=host,
        port=port,
        reload=args.reload,
        # Remove workers parameter - gunicorn handles workers in production
        app_dir=".",
    )


if __name__ == "__main__":
    main()
