import debugpy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

# Import configuration
from deployment.app.config import settings

# Import database initializer
from deployment.app.db.schema import init_db

# Import API routers
from deployment.app.api.jobs import router as jobs_router
from deployment.app.api.health import router as health_router
from deployment.app.api.models_configs import router as models_params_router
from deployment.app.api.admin import router as admin_router

# Import utils
from deployment.app.utils.error_handling import configure_error_handlers

# Create directories
Path("deployment/data").mkdir(parents=True, exist_ok=True)
Path("deployment/logs").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.api.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("deployment/logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("plastinka")

# Check required environment variables
def check_environment():
    """Check for required environment variables."""
    required_vars = {
        "YANDEX_CLOUD_ACCESS_KEY": "Cloud Storage Access Key",
        "YANDEX_CLOUD_SECRET_KEY": "Cloud Storage Secret Key",
        "YANDEX_CLOUD_FOLDER_ID": "Cloud Folder ID",
        "YANDEX_CLOUD_API_KEY": "Cloud API Key",
        "CLOUD_CALLBACK_AUTH_TOKEN": "Cloud Callback Authentication Token"
    }
    
    missing = []
    for var, desc in required_vars.items():
        if not os.environ.get(var):
            missing.append(f"{var} ({desc})")
    
    if missing:
        logger.warning(f"Missing {len(missing)} required environment variables:")
        for var in missing:
            logger.warning(f"  - {var}")
        logger.warning("The application may not function correctly. See docs/environment_variables.md for details.")
        
        if settings.is_production:
            logger.error("Production environment detected but missing required variables. Exiting.")
            sys.exit(1)
    
    return not missing

# Define lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Initializing application")
    # Check environment variables
    check_environment()
    # Get database path from environment variable
    db_path = os.getenv("DATABASE_URL")
    if not db_path:
        logger.error("DATABASE_URL environment variable not set")
        raise ValueError("DATABASE_URL environment variable not set")

    # Initialize database
    try:
        if init_db(db_path=db_path):
            logger.info("Database initialized successfully")
        else:
            logger.warning("Database initialization completed with warnings")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        raise

    logger.info("Application started successfully")
    
    yield  # Application running
    
    # Shutdown logic (if any)
    logger.info("Application shutting down")

# Create FastAPI application with lifespan
app = FastAPI(
    title="Plastinka Sales Predictor API",
    description="API for predicting vinyl record sales",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Configure error handlers
app = configure_error_handlers(app)

# Include API routers
app.include_router(jobs_router)
app.include_router(health_router)
app.include_router(models_params_router)
app.include_router(admin_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Plastinka Sales Predictor API",
        "docs": "/docs",
        "version": "0.1.0"
    }

# Run the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host=settings.api.host, 
        port=settings.api.port, 
        reload=settings.api.debug
    ) 