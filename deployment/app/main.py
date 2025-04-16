from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from pathlib import Path
from datetime import datetime

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
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("deployment/logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("plastinka")

# Create FastAPI application
app = FastAPI(
    title="Plastinka Sales Predictor API",
    description="API for predicting vinyl record sales",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        },
        "timestamp": datetime.now().isoformat()
    }
    
    # Check database connection
    try:
        import sqlite3
        conn = sqlite3.connect("deployment/data/plastinka.db")
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}", exc_info=True)
        health_status["components"]["database"] = "unhealthy"
        health_status["status"] = "degraded"
    
    return health_status

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing application")
    
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
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 