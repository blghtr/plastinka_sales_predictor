"""Database queries for prediction month management."""

import logging
from datetime import date, datetime, timedelta
from typing import Any

import asyncpg

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_query

logger = logging.getLogger("plastinka.database")


async def get_next_prediction_month(connection: asyncpg.Connection) -> date:
    """
    Finds the last month with complete data in fact_sales and returns the next month.
    
    Args:
        connection: Required database connection
        
    Returns:
        date: The next prediction month
    """
    query = """
        SELECT
            TO_CHAR(DATE_TRUNC('month', data_date), 'YYYY-MM') as month,
            COUNT(DISTINCT EXTRACT(DAY FROM data_date))::integer as distinct_days,
            EXTRACT(DAY FROM (DATE_TRUNC('month', data_date) + INTERVAL '1 month' - INTERVAL '1 day'))::integer as days_in_month
        FROM fact_sales
        GROUP BY DATE_TRUNC('month', data_date)
        HAVING COUNT(DISTINCT EXTRACT(DAY FROM data_date))::integer = EXTRACT(DAY FROM (DATE_TRUNC('month', data_date) + INTERVAL '1 month' - INTERVAL '1 day'))::integer
        ORDER BY month DESC
        LIMIT 1
    """
    try:
        result = await execute_query(query, connection=connection)
        if result and result.get("month"):
            last_full_month = datetime.strptime(result["month"], "%Y-%m").date()
            # Return the next month
            return (last_full_month.replace(day=1) + timedelta(days=32)).replace(day=1)
        else:
            # If no full month is found, default to the month after the latest data point
            latest_data_query = "SELECT MAX(data_date) as max_date FROM fact_sales"
            latest_data_result = await execute_query(latest_data_query, connection=connection)
            if latest_data_result and latest_data_result.get("max_date"):
                max_date = latest_data_result["max_date"]
                if isinstance(max_date, str):
                    max_date = date.fromisoformat(max_date)
                elif isinstance(max_date, datetime):
                    max_date = max_date.date()
                return (max_date.replace(day=1) + timedelta(days=32)).replace(day=1)
            # Fallback to the current month's next month if no data exists at all
            today = date.today()
            return (today.replace(day=1) + timedelta(days=32)).replace(day=1)

    except (DatabaseError, TypeError, ValueError) as e:
        logger.error(f"Failed to get next prediction month: {e}")
        # Fallback in case of any error
        today = date.today()
        return (today.replace(day=1) + timedelta(days=32)).replace(day=1)


async def get_latest_prediction_month(connection: asyncpg.Connection) -> date | None:
    """
    Get the most recent prediction_month from the prediction_results table.
    
    Args:
        connection: Required database connection
        
    Returns:
        date: The latest prediction month or None
    """
    query = "SELECT MAX(prediction_month) as latest_month FROM prediction_results"
    try:
        result = await execute_query(query, connection=connection)
        if result and result.get("latest_month"):
            latest_month = result["latest_month"]
            if isinstance(latest_month, str):
                return date.fromisoformat(latest_month)
            elif isinstance(latest_month, datetime):
                return latest_month.date()
            elif isinstance(latest_month, date):
                return latest_month
        return None
    except (DatabaseError, TypeError, ValueError) as e:
        logger.error(f"Failed to get latest prediction month: {e}")
        return None

