import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from deployment.app.api.admin import router as admin_router
from deployment.app.api.health import router as health_router

# Import API routers
from deployment.app.api.jobs import router as jobs_router
from deployment.app.api.models_configs import router as models_params_router

# Import configuration
from deployment.app.config import get_settings

# Import database initializer
from deployment.app.db.schema import init_db

# Apply centralised logging configuration before the rest of the app starts.
from deployment.app.logger_config import configure_logging

# Import utils
from deployment.app.utils.error_handling import configure_error_handlers

# Получаем актуальную версию пакета
from plastinka_sales_predictor import __version__ as app_version

# This sets up a single console handler that integrates with Uvicorn / FastAPI
# and avoids spawning extra threads or writing to dataset-specific files.
configure_logging()

# Configure logging
logger = logging.getLogger(__name__)  # Use __name__ for module-specific logger


# Lifespan context manager for application startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Environment checks are performed via the health endpoint
    settings = get_settings()
    # Initialize database
    db_initialized_successfully = init_db(db_path=settings.database_path)
    if not db_initialized_successfully:
        logger.error("Database initialization failed during startup. Check logs.")
        # Depending on policy, might raise an exception here to stop startup
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()
    yield


# Create FastAPI application with lifespan
app = FastAPI(
    title="Plastinka Sales Predictor API",
    description="API for predicting vinyl record sales",
    version=app_version,
    lifespan=lifespan,
)

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().api.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
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
        "version": app_version,
    }


# Run the application with uvicorn
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
    )
