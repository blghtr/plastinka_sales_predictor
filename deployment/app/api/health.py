"""
Health and monitoring endpoints.
"""

import logging
import os
import sqlite3
import time
from datetime import datetime
from typing import Any

import psutil
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from deployment.app.config import get_settings
from deployment.app.services.auth import get_current_api_key_validated
from deployment.app.utils.environment import get_environment_status, ComponentHealth
from deployment.app.utils.retry_monitor import (
    get_retry_statistics,
    reset_retry_statistics,
)

import calendar
from datetime import timedelta

router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={200: {"description": "Success"}, 500: {"description": "Server Error"}},
)

logger = logging.getLogger(__name__)





class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str
    timestamp: str
    uptime_seconds: int
    components: dict[str, ComponentHealth]

    model_config = ConfigDict(from_attributes=True)


class SystemStatsResponse(BaseModel):
    """System statistics response model."""

    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    process_memory_mb: float
    open_files: int
    active_threads: int
    timestamp: str

    model_config = ConfigDict(from_attributes=True)


class RetryStatsResponse(BaseModel):
    """Retry statistics response model."""

    total_retries: int
    successful_retries: int
    exhausted_retries: int
    successful_after_retry: int
    high_failure_operations: list[str]
    operation_stats: dict[str, dict[str, Any]]
    exception_stats: dict[str, dict[str, Any]]
    timestamp: str

    model_config = ConfigDict(from_attributes=True)


# Track application start time
start_time = time.time()


def check_monotonic_months(table_name: str, conn) -> list[str]:
    """
    Проверяет, что месяцы в data_date идут подряд без пропусков (по всей таблице).
    Возвращает список пропущенных месяцев в формате YYYY-MM-01.
    """
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT data_date FROM {table_name}")
    dates = sorted(row[0] for row in cursor.fetchall())
    if not dates:
        return []
    # Преобразуем к datetime.date
    date_objs = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates]
    # Строим полный диапазон месяцев
    min_date, max_date = date_objs[0], date_objs[-1]
    current = min_date
    missing = []
    while current <= max_date:
        if current not in date_objs:
            missing.append(current.strftime("%Y-%m-%d"))
        # Переход к следующему месяцу
        year, month = current.year, current.month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        current = current.replace(year=year, month=month, day=1)
    return missing


def check_database() -> ComponentHealth:
    """Check database connection and monotonicity of months in sales & changes."""
    conn = None  # Initialize conn to None
    try:
        conn = sqlite3.connect(get_settings().database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()

        # Check if required tables exist
        required_tables = [
            "jobs",
            "models",
            "configs",
            "training_results",
            "prediction_results",
            "job_status_history",
            "dim_multiindex_mapping",
            "fact_sales",
            "fact_stock",
            "fact_prices",
            "fact_stock_changes",
            "fact_predictions",
            "sqlite_sequence",
            "processing_runs",
            "data_upload_results",
            "report_results",
        ]

        # Construct the query to check for these tables efficiently
        placeholders = ", ".join(["?" for _ in required_tables])
        query_check_tables = f"SELECT name FROM sqlite_master WHERE type='table' AND name IN ({placeholders})"
        cursor.execute(query_check_tables, tuple(required_tables))
        tables_found = [row[0] for row in cursor.fetchall()]
        missing_tables = [table for table in required_tables if table not in tables_found]
        if missing_tables:
            logger.warning(f"Database degraded: missing tables {missing_tables}")
            return ComponentHealth(
                status="degraded", details={"missing_tables": missing_tables}
            )

        # --- Проверка монотонности месяцев ---
        monotonicity_issues = {}
        for table in ["fact_sales", "fact_stock_changes"]:
            missing_months = check_monotonic_months(table, conn)
            if missing_months:
                monotonicity_issues[table] = missing_months
        if monotonicity_issues:
            logger.error(f"Database unhealthy: missing months in tables: {monotonicity_issues}")
            return ComponentHealth(
                status="unhealthy", details={"missing_months": monotonicity_issues}
            )

        return ComponentHealth(status="healthy")
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}", exc_info=True)
        return ComponentHealth(status="unhealthy", details={"error": str(e)})
    finally:
        if conn:
            conn.close()


@router.get("", response_model=HealthResponse)
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
        "config": get_environment_status(),
    }

    # Determine overall status and HTTP status code
    overall_status = "healthy"
    http_status_code = 200  # Default to 200 OK

    if any(comp.status == "unhealthy" for comp in components.values()):
        overall_status = "unhealthy"
        http_status_code = 503  # Service Unavailable
    elif any(comp.status == "degraded" for comp in components.values()):
        overall_status = "degraded"
        # Degraded status still returns 200 OK

    return JSONResponse(
        status_code=http_status_code,
        content={
            "status": overall_status,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "components": {k: v.model_dump() for k, v in components.items()},
        },
    )


@router.get("/system", response_model=SystemStatsResponse)
async def system_stats(api_key: bool = Depends(get_current_api_key_validated)):
    """
    Get detailed system statistics.
    Returns CPU, memory, and disk usage.
    """
    process = psutil.Process(os.getpid())

    # Calculate system stats
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage("/").percent,
        "process_memory_mb": process.memory_info().rss / (1024 * 1024),
        "open_files": len(process.open_files()),
        "active_threads": process.num_threads(),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/retry-stats", response_model=RetryStatsResponse)
async def retry_statistics(api_key: bool = Depends(get_current_api_key_validated)):
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
        "timestamp": stats["timestamp"],
    }


@router.post("/retry-stats/reset")
async def reset_retry_stats(api_key: bool = Depends(get_current_api_key_validated)):
    """
    Reset all retry statistics.
    """
    reset_retry_statistics()
    return {"status": "ok", "message": "Retry statistics reset successfully"}
