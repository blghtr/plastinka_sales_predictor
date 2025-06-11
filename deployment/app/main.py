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
from deployment.app.api.health import ComponentHealth, check_environment

print("[DEBUG main.py] Top of main.py")
sys.stdout.flush()

# Import configuration
print("[DEBUG main.py] Importing settings...")
sys.stdout.flush()
from deployment.app.config import settings
print("[DEBUG main.py] Settings imported.")
sys.stdout.flush()

# Import database initializer
print("[DEBUG main.py] Importing init_db...")
sys.stdout.flush()
from deployment.app.db.schema import init_db
print("[DEBUG main.py] init_db imported.")
sys.stdout.flush()

# Import API routers
print("[DEBUG main.py] Importing routers...")
sys.stdout.flush()
from deployment.app.api.jobs import router as jobs_router
from deployment.app.api.health import router as health_router
from deployment.app.api.models_configs import router as models_params_router
from deployment.app.api.admin import router as admin_router
print("[DEBUG main.py] Routers imported.")
sys.stdout.flush()

# Import utils
print("[DEBUG main.py] Importing error_handling...")
sys.stdout.flush()
from deployment.app.utils.error_handling import configure_error_handlers
print("[DEBUG main.py] error_handling imported.")
sys.stdout.flush()

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
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

# Lifespan context manager for application startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("lifespan() called - Startup")
    
    # Perform environment checks
    logger.info("lifespan: Calling check_environment()...")
    env_health = check_environment() # This will now call the imported function
    
    if env_health.status != "healthy": 
        logger.warning(f"Missing environment variables: {env_health.details.get('missing_variables')}")
        logger.warning("The application may not function correctly. See docs/environment_variables.md for details.")
    logger.info("lifespan: check_environment() returned.") 

    # Initialize database
    logger.info("lifespan: Calling init_db()...")
    # Ensure init_db uses the computed database_path property from settings
    db_initialized_successfully = init_db(db_path=settings.database_path)
    if not db_initialized_successfully:
        logger.error("Database initialization failed during startup. Check logs.")
        # Depending on policy, might raise an exception here to stop startup
    else:
        logger.info("Database initialization completed.") # Changed from "completed with warnings"
    logger.info("lifespan: init_db() returned.")
    
    logger.info("lifespan: Yielding...")
    yield
    logger.info("lifespan() - Shutdown")

# Create FastAPI application with lifespan
print("[DEBUG main.py] About to create FastAPI app...")
sys.stdout.flush()
app = FastAPI(
    title="Plastinka Sales Predictor API",
    description="API for predicting vinyl record sales",
    version="0.1.0",
    lifespan=lifespan
)
print("[DEBUG main.py] FastAPI app created.")
sys.stdout.flush()

# Add CORS middleware with restricted origins
print("[DEBUG main.py] Adding CORS middleware...")
sys.stdout.flush()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
print("[DEBUG main.py] CORS middleware added.")
sys.stdout.flush()

# Configure error handlers
print("[DEBUG main.py] Configuring error handlers...")
sys.stdout.flush()
app = configure_error_handlers(app)
print("[DEBUG main.py] Error handlers configured.")
sys.stdout.flush()

# Include API routers
print("[DEBUG main.py] Including jobs_router...")
sys.stdout.flush()
app.include_router(jobs_router)
print("[DEBUG main.py] jobs_router included.")
sys.stdout.flush()

print("[DEBUG main.py] Including health_router...")
sys.stdout.flush()
app.include_router(health_router)
print("[DEBUG main.py] health_router included.")
sys.stdout.flush()

print("[DEBUG main.py] Including models_params_router...")
sys.stdout.flush()
app.include_router(models_params_router)
print("[DEBUG main.py] models_params_router included.")
sys.stdout.flush()

print("[DEBUG main.py] Including admin_router...")
sys.stdout.flush()
app.include_router(admin_router)
print("[DEBUG main.py] admin_router included.")
sys.stdout.flush()

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