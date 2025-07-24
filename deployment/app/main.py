# Apply centralised logging configuration before the rest of the app starts.
from deployment.app.logger_config import configure_logging
import logging
# Configure logging
configure_logging()
logger = logging.getLogger(__name__)  # Use __name__ for module-specific logger

# Import context manager for lifespan
from contextlib import asynccontextmanager

# Import FastAPI dependencies
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles

# Import API routers
from deployment.app.api.jobs import router as jobs_router
from deployment.app.api.models_configs import router as models_params_router
from deployment.app.api.results import router as results_router
from deployment.app.api.admin import router as admin_router
from deployment.app.api.health import router as health_router
from deployment.app.api.auth import router as auth_router

# Import configuration
from deployment.app.config import get_settings

# Import database initializer
from deployment.app.db.schema import init_db

# Import utils
from deployment.app.utils.error_handling import configure_error_handlers

# Import security dependencies
from deployment.app.services.auth import get_docs_user

# Получаем актуальную версию пакета
from plastinka_sales_predictor import __version__ as app_version

settings = get_settings()

# Lifespan context manager for application startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    db_initialized_successfully = init_db(db_path=settings.database_path)
    if not db_initialized_successfully:
        logger.error("Database initialization failed during startup. Check logs.")
        # Depending on policy, might raise an exception here to stop startup

    yield

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