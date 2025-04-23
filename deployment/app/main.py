from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Import configuration
from app.config import settings

# Import database initializer
from app.db.schema import init_db

# Import API routers
from app.api.jobs import router as jobs_router

# Import utils
from app.utils.error_handling import configure_error_handlers

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

# Create FastAPI application
app = FastAPI(
    title="Plastinka Sales Predictor API",
    description="API for predicting vinyl record sales",
    version="0.1.0"
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

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Plastinka Sales Predictor API",
        "docs": "/docs",
        "version": "0.1.0"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "components": {
            "api": "healthy",
            "database": "unknown",
            "config": "degraded" if not check_environment() else "healthy"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Check database connection
    try:
        import sqlite3
        conn = sqlite3.connect(settings.db.path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}", exc_info=True)
        health_status["components"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    # Update overall status
    if any(status == "unhealthy" for status in health_status["components"].values()):
        health_status["status"] = "unhealthy"
    elif any(status == "degraded" for status in health_status["components"].values()):
        health_status["status"] = "degraded"
    
    return health_status

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing application")
    
    # Check environment variables
    check_environment()
    
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}", exc_info=True)
        raise

    logger.info("Application started successfully")


# Run the application with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host=settings.api.host, 
        port=settings.api.port, 
        reload=settings.api.debug
    ) 