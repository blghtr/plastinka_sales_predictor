import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html

from deployment.app.api.admin import router as admin_router
from deployment.app.api.auth import router as auth_router
from deployment.app.api.health import router as health_router
from deployment.app.api.jobs import router as jobs_router
from deployment.app.api.models_configs import router as models_params_router
from deployment.app.api.results import router as results_router
from deployment.app.config import get_settings

settings = get_settings()
from deployment.app.db.connection import close_db_pool, get_db_pool, init_db_pool
from deployment.app.db.schema_postgresql import init_postgres_schema
from deployment.app.logger_config import configure_logging
from deployment.app.services.auth import get_docs_user
from deployment.app.utils.error_handling import configure_error_handlers
from plastinka_sales_predictor import __version__ as app_version

# Apply centralised logging configuration before the rest of the app starts.
configure_logging()
logger = logging.getLogger(__name__)  # Use __name__ for module-specific logger

# Lifespan context manager for application startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize PostgreSQL connection pool
    try:
        pool = await init_db_pool()
        logger.info("Database pool initialized successfully")
        
        # Initialize database schema
        schema_initialized = await init_postgres_schema(pool)
        if not schema_initialized:
            logger.error("Database schema initialization failed during startup. Check logs.")
            # Depending on policy, might raise an exception here to stop startup
            raise RuntimeError("Database schema initialization failed")
        logger.info("Database schema initialized successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed during startup: {e}", exc_info=True)
        # Depending on policy, might raise an exception here to stop startup
        raise

    yield

    # Close database pool on shutdown
    try:
        await close_db_pool()
        logger.info("Database pool closed successfully")
    except Exception as e:
        logger.error(f"Error closing database pool during shutdown: {e}", exc_info=True)

# Create FastAPI application with lifespan
app = FastAPI(
    title="Plastinka Sales Predictor API",
    description="API for predicting vinyl record sales",
    version=app_version,
    lifespan=lifespan,
    docs_url=None,  # Disable default /docs
    redoc_url=None, # Disable default /redoc
)


# Configure error handlers
app = configure_error_handlers(app)

# Add CORS middleware
if settings.api.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include API routers
app.include_router(jobs_router)
app.include_router(health_router)
app.include_router(models_params_router)
app.include_router(admin_router)
app.include_router(results_router)
app.include_router(auth_router)


# Custom protected docs and redoc URLs
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request, username: str = Depends(get_docs_user)):
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
    )

@app.get("/redoc", include_in_schema=False)
async def custom_redoc_html(request: Request, username: str = Depends(get_docs_user)):
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
    )
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

    uvicorn.run(
        "app.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug,
    )
