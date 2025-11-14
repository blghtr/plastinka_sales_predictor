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


async def get_predictions_history(
    multiindex_id: int | None = None,
    prediction_month_from: date | None = None,
    prediction_month_to: date | None = None,
    include_actuals: bool = True,
    include_features: bool = True,
    connection: asyncpg.Connection = None,
) -> dict[str, Any]:
    """
    Retrieve predictions history with optional actuals and features data.
    
    Aggregates predictions, actual sales, and report features by multiindex_id
    for dashboard visualization.
    
    Args:
        multiindex_id: Optional filter by specific multiindex_id
        prediction_month_from: Optional start date for prediction range (first day of month)
        prediction_month_to: Optional end date for prediction range (first day of month)
        include_actuals: Whether to include actual sales data from fact_sales
        include_features: Whether to include report_features data
        connection: Required database connection
        
    Returns:
        Dictionary with keys:
        - predictions: List of prediction records with metadata
        - actuals: List of actual sales records (if include_actuals=True)
        - features: List of feature records (if include_features=True)
    """
    results = {
        "predictions": [],
        "actuals": [],
        "features": []
    }
    
    # Build WHERE clause conditions
    where_clauses = []
    params: list[Any] = []
    param_num = 1
    
    if multiindex_id is not None:
        where_clauses.append(f"fp.multiindex_id = ${param_num}")
        params.append(multiindex_id)
        param_num += 1
    
    if prediction_month_from is not None:
        where_clauses.append(f"fp.prediction_month >= ${param_num}")
        params.append(prediction_month_from)
        param_num += 1
    
    if prediction_month_to is not None:
        where_clauses.append(f"fp.prediction_month <= ${param_num}")
        params.append(prediction_month_to)
        param_num += 1
    
    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    # Query predictions with metadata
    predictions_query = f"""
        SELECT 
            fp.multiindex_id,
            fp.prediction_month,
            fp.model_id,
            fp.quantile_05,
            fp.quantile_25,
            fp.quantile_50,
            fp.quantile_75,
            fp.quantile_95,
            dmm.barcode,
            dmm.artist,
            dmm.album,
            dmm.cover_type,
            dmm.price_category,
            dmm.release_type,
            dmm.recording_decade,
            dmm.release_decade,
            dmm.style,
            dmm.recording_year
        FROM fact_predictions fp
        JOIN dim_multiindex_mapping dmm ON fp.multiindex_id = dmm.multiindex_id
        WHERE {where_clause}
        ORDER BY fp.multiindex_id, fp.prediction_month, fp.model_id
    """
    
    try:
        predictions = await execute_query(
            predictions_query,
            params=tuple(params),
            fetchall=True,
            connection=connection
        ) or []
        results["predictions"] = predictions
        
        # Query actuals if requested
        if include_actuals:
            actuals_where_clauses = []
            actuals_params: list[Any] = []
            actuals_param_num = 1
            
            if multiindex_id is not None:
                actuals_where_clauses.append(f"multiindex_id = ${actuals_param_num}")
                actuals_params.append(multiindex_id)
                actuals_param_num += 1
            
            if prediction_month_from is not None:
                actuals_where_clauses.append(f"data_date >= ${actuals_param_num}")
                actuals_params.append(prediction_month_from)
                actuals_param_num += 1
            
            if prediction_month_to is not None:
                actuals_where_clauses.append(f"data_date <= ${actuals_param_num}")
                actuals_params.append(prediction_month_to)
                actuals_param_num += 1
            
            actuals_where_clause = " AND ".join(actuals_where_clauses) if actuals_where_clauses else "1=1"
            
            actuals_query = f"""
                SELECT 
                    multiindex_id,
                    data_date as month,
                    value as sales_count
                FROM fact_sales
                WHERE {actuals_where_clause}
                ORDER BY multiindex_id, data_date
            """
            
            actuals = await execute_query(
                actuals_query,
                params=tuple(actuals_params),
                fetchall=True,
                connection=connection
            ) or []
            results["actuals"] = actuals
        
        # Query features if requested
        if include_features:
            features_where_clauses = []
            features_params: list[Any] = []
            features_param_num = 1
            
            if multiindex_id is not None:
                features_where_clauses.append(f"rf.multiindex_id = ${features_param_num}")
                features_params.append(multiindex_id)
                features_param_num += 1
            
            if prediction_month_from is not None:
                features_where_clauses.append(f"rf.data_date >= ${features_param_num}")
                features_params.append(prediction_month_from)
                features_param_num += 1
            
            if prediction_month_to is not None:
                features_where_clauses.append(f"rf.data_date <= ${features_param_num}")
                features_params.append(prediction_month_to)
                features_param_num += 1
            
            features_where_clause = " AND ".join(features_where_clauses) if features_where_clauses else "1=1"
            
            features_query = f"""
                SELECT 
                    rf.multiindex_id,
                    rf.data_date as month,
                    rf.availability,
                    rf.confidence,
                    rf.masked_mean_sales_items,
                    rf.masked_mean_sales_rub,
                    rf.lost_sales as lost_sales_rub
                FROM report_features rf
                WHERE {features_where_clause}
                ORDER BY rf.multiindex_id, rf.data_date
            """
            
            features = await execute_query(
                features_query,
                params=tuple(features_params),
                fetchall=True,
                connection=connection
            ) or []
            results["features"] = features
        
        return results
        
    except DatabaseError as e:
        logger.error(f"Failed to get predictions history: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_predictions_history: {e}", exc_info=True)
        raise DatabaseError(f"Failed to retrieve predictions history: {str(e)}") from e
