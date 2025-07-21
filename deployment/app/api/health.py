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
from fastapi import APIRouter, Depends, status, HTTPException
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

from ..dependencies import get_dal_system
from ..db.data_access_layer import DataAccessLayer
from deployment.app.models.api_models import ErrorDetailResponse
from deployment.app.utils.error_handling import ErrorDetail

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


def check_monotonic_months(table_name: str, dal: DataAccessLayer) -> list[str]:
    """
    Проверяет, что месяцы в data_date идут подряд без пропусков (по всей таблице).
    Возвращает список пропущенных месяцев в формате YYYY-MM-01.
    """
    dates_rows = dal.execute_raw_query(
        f"SELECT DISTINCT data_date FROM {table_name}", fetchall=True
    )
    dates = sorted(row["data_date"] for row in dates_rows) if dates_rows else []

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


def check_database(dal: DataAccessLayer) -> ComponentHealth:
    """Perform database health checks."""
    logger.info("Performing database health check...")
    status = "healthy"
    details = {}

    try:
        # 1. Basic connection check: try to execute a simple query
        dal.execute_raw_query("SELECT 1", fetchall=False)

        # 2. Check for required tables
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
            "processing_runs",
            "data_upload_results",
            "report_results",
        ]

        # Construct the query to check for these tables efficiently using batching
        query_template = "SELECT name FROM sqlite_master WHERE type='table' AND name IN ({placeholders})"
        tables_found_rows = dal.execute_query_with_batching(
            query_template, required_tables, connection=None
        )
        tables_found = [row["name"] for row in tables_found_rows] if tables_found_rows else []

        missing_tables = [table for table in required_tables if table not in tables_found]
        if missing_tables:
            status = "degraded"
            details["missing_tables"] = missing_tables
            logger.warning(f"Database degraded: missing tables {missing_tables}")

        # 3. Check for monotonic growth of months in fact tables
        missing_months_sales = check_monotonic_months("fact_sales", dal)
        if missing_months_sales:
            status = "unhealthy"
            details["missing_months_sales"] = missing_months_sales
            logger.warning(
                f"Database unhealthy: missing months in fact_sales: {missing_months_sales}"
            )

        missing_months_stock_changes = check_monotonic_months(
            "fact_stock_changes", dal
        )
        if missing_months_stock_changes:
            status = "unhealthy"
            details["missing_months_stock_changes"] = missing_months_stock_changes
            logger.warning(
                f"Database unhealthy: missing months in fact_stock_changes: {missing_months_stock_changes}"
            )

    except Exception as e:
        status = "unhealthy"
        details["error"] = str(e)
        logger.error(f"Database health check failed: {e}", exc_info=True)

    return ComponentHealth(status=status, details=details)


router = APIRouter(
    prefix="/health",
    tags=["health"],
    responses={
        200: {"description": "Success", "model": HealthResponse},
        503: {"description": "Service Unavailable", "model": ErrorDetailResponse},
    },
)


@router.get("", response_model=HealthResponse)
async def health_check(
    dal: DataAccessLayer = Depends(get_dal_system)
):
    """
    Comprehensive health check endpoint.
    Returns status of all system components.
    """
    version = os.environ.get("APP_VERSION", "1.0.0")
    uptime = int(time.time() - start_time)

    # Check individual components
    components = {
        "api": ComponentHealth(status="healthy"),
        "database": check_database(dal),
        "config": get_environment_status(),
    }

    # Check active model metric
    active_model_metric = dal.get_active_model_primary_metric()
    settings = get_settings()

    if active_model_metric is None:
        components["active_model_metric"] = ComponentHealth(
            status="degraded",
            details={
                "reason": "No active model found or primary metric not available."
            },
        )
    elif active_model_metric < settings.metric_thesh_for_health_check:
        if settings.default_metric_higher_is_better:
            components["active_model_metric"] = ComponentHealth(
                status="degraded",
                details={
                    "reason": f"Active model primary metric ({active_model_metric:.4f}) is below threshold ({settings.metric_thesh_for_health_check:.4f})."
                },
            )
    else:
        if not settings.default_metric_higher_is_better:
            components["active_model_metric"] = ComponentHealth(
                status="degraded",
                details={
                    "reason": f"Active model primary metric ({active_model_metric:.4f}) is above threshold ({settings.metric_thesh_for_health_check:.4f})."
                },
            )
        else:
            components["active_model_metric"] = ComponentHealth(
                status="healthy",
                details={
                    "metric_value": active_model_metric,
                    "threshold": settings.metric_thesh_for_health_check,
                },
            )

    # Determine overall status and HTTP status code
    overall_status = "healthy"
    http_status_code = 200  # Default to 200 OK

    if any(comp.status == "unhealthy" for comp in components.values()):
        overall_status = "unhealthy"
        http_status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif any(comp.status == "degraded" for comp in components.values()):
        overall_status = "degraded"
        http_status_code = status.HTTP_200_OK

    # If the overall status is unhealthy or degraded, raise HTTPException
    if overall_status != "healthy":
        error_message = f"API is {overall_status}."
        if overall_status == "unhealthy":
            error_code = "service_unavailable"
        else: # degraded
            error_code = "service_degraded"

        details = {k: v.model_dump() for k, v in components.items()}

        error = ErrorDetail(
            message=error_message,
            code=error_code,
            status_code=http_status_code,
            details=details,
        )
        # Log the error, though for health checks it might be excessive if frequently polled
        # error.log_error(None) # No request context here

        raise HTTPException(
            status_code=error.status_code,
            detail=error.message
        )

    # If healthy, return the regular HealthResponse
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
