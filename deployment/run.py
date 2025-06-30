#!/usr/bin/env python
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
        description="Run the Plastinka Sales Predictor API"
    )
    parser.add_argument(
        "--host", default=None, help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to bind to (overrides config)"
    )
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )

    return parser.parse_args()


# Main function
def main():
    args = parse_args()
    settings = get_settings()

    # Use configuration values with command line overrides
    host = args.host if args.host is not None else settings.api.host
    port = args.port if args.port is not None else settings.api.port

    # Run the application
    uvicorn.run(
        "deployment.app.main:app",
        host=host,
        port=port,
        reload=args.reload,
        workers=args.workers,
        app_dir=".",
    )


if __name__ == "__main__":
    main()
