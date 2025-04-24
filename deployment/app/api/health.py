"""
Health and monitoring endpoints.
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any, List
from pydantic import BaseModel
import logging
import os
import time
import psutil
import sqlite3
from datetime import datetime, timedelta

from app.utils.retry_monitor import get_retry_statistics, reset_retry_statistics
from app.config import settings

router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={
        200: {"description": "Success"},
        500: {"description": "Server Error"}
    }
)

logger = logging.getLogger(__name__)

class ComponentHealth(BaseModel):
    """Component health status."""
    status: str
    details: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    timestamp: str
    uptime_seconds: int
    components: Dict[str, ComponentHealth]

class SystemStatsResponse(BaseModel):
    """System statistics response model."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    process_memory_mb: float
    open_files: int
    active_threads: int
    timestamp: str

class RetryStatsResponse(BaseModel):
    """Retry statistics response model."""
    total_retries: int
    successful_retries: int
    exhausted_retries: int
    successful_after_retry: int
    high_failure_operations: List[str]
    operation_stats: Dict[str, Dict[str, Any]]
    exception_stats: Dict[str, Dict[str, Any]]
    timestamp: str

# Track application start time
start_time = time.time()

def check_environment() -> ComponentHealth:
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
        details = {
            "missing_variables": missing,
            "message": "Some required environment variables are missing"
        }
        return ComponentHealth(status="degraded", details=details)
    
    return ComponentHealth(status="healthy")

def check_database() -> ComponentHealth:
    """Check database connection."""
    try:
        conn = sqlite3.connect(settings.db.path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        
        # Check if required tables exist
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND ("
            "name='cloud_functions' OR "
            "name='cloud_function_executions' OR "
            "name='cloud_storage_objects'"
            ")"
        )
        tables = [row[0] for row in cursor.fetchall()]
        required_tables = ['cloud_functions', 'cloud_function_executions', 'cloud_storage_objects']
        missing_tables = [table for table in required_tables if table not in tables]
        
        conn.close()
        
        if missing_tables:
            return ComponentHealth(
                status="degraded", 
                details={"missing_tables": missing_tables}
            )
        
        return ComponentHealth(status="healthy")
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}", exc_info=True)
        return ComponentHealth(
            status="unhealthy", 
            details={"error": str(e)}
        )

@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns status of all system components.
    """
    version = os.environ.get("APP_VERSION", "1.0.0")
    uptime = int(time.time() - start_time)
    
    # Check individual components
    components = {
        "api": ComponentHealth(status="healthy"),
        "database": check_database(),
        "config": check_environment()
    }
    
    # Determine overall status
    overall_status = "healthy"
    if any(comp.status == "unhealthy" for comp in components.values()):
        overall_status = "unhealthy"
    elif any(comp.status == "degraded" for comp in components.values()):
        overall_status = "degraded"
    
    logger.debug("Health check requested")
    
    return {
        "status": overall_status,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime,
        "components": components
    }

@router.get("/system", response_model=SystemStatsResponse)
async def system_stats():
    """
    Get detailed system statistics.
    Returns CPU, memory, and disk usage.
    """
    process = psutil.Process(os.getpid())
    
    # Calculate system stats
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage('/').percent,
        "process_memory_mb": process.memory_info().rss / (1024 * 1024),
        "open_files": len(process.open_files()),
        "active_threads": process.num_threads(),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/retry-stats", response_model=RetryStatsResponse)
async def retry_statistics():
    """
    Get detailed statistics about operation retries.
    Returns information about retry patterns and failure rates.
    """
    # Get retry statistics from the monitor
    stats = get_retry_statistics()
    
    return {
        "total_retries": stats["total_retries"],
        "successful_retries": stats["successful_retries"],
        "exhausted_retries": stats["exhausted_retries"],
        "successful_after_retry": stats["successful_after_retry"],
        "high_failure_operations": stats["high_failure_operations"],
        "operation_stats": stats["operation_stats"],
        "exception_stats": stats["exception_stats"],
        "timestamp": stats["timestamp"]
    }

@router.post("/retry-stats/reset")
async def reset_retry_stats():
    """
    Reset all retry statistics.
    """
    reset_retry_statistics()
    return {"status": "ok", "message": "Retry statistics reset successfully"} 