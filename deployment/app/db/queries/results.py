"""Database queries for result management (training, tuning, prediction)."""

import json
import logging
import uuid
from datetime import date, datetime
from typing import Any

import asyncpg
import pandas as pd

from deployment.app.db.exceptions import DatabaseError
from deployment.app.db.queries.core import execute_query, execute_many
from deployment.app.db.utils import generate_id, json_default_serializer, split_ids_for_batching

logger = logging.getLogger("plastinka.database")


async def create_training_result(
    job_id: str,
    model_id: str,
    config_id: str,
    metrics: dict[str, Any],
    duration: int | None,
    connection: asyncpg.Connection = None,
) -> str:
    """Creates a record for a completed training job and handles auto-activation."""
    result_id = str(uuid.uuid4())
    metrics_json = json.dumps(metrics, default=json_default_serializer)

    # Insert the training result
    query = """
    INSERT INTO training_results
    (result_id, job_id, model_id, config_id, metrics, duration)
    VALUES ($1, $2, $3, $4, $5, $6)
    """
    params = (result_id, job_id, model_id, config_id, metrics_json, duration)
    await execute_query(query=query, connection=connection, params=params)
    logger.info(f"Created training result: {result_id} for job: {job_id}")

    # Update the job with the result_id
    update_query = "UPDATE jobs SET result_id = $1 WHERE job_id = $2"
    await execute_query(query=update_query, connection=connection, params=(result_id, job_id))
    logger.info(f"Updated job {job_id} with result_id: {result_id}")

    # Auto-activate best config
    import deployment.app.db.queries.configs as configs_queries
    await configs_queries.auto_activate_best_config_if_enabled(connection=connection)

    # Auto-activate best model
    if model_id:
        import deployment.app.db.queries.models as models_queries
        await models_queries.auto_activate_best_model_if_enabled(connection=connection)
    
    return result_id


async def create_tuning_result(
    job_id: str,
    config_id: str,
    metrics: dict[str, Any],
    duration: int | None,
    connection: asyncpg.Connection = None,
) -> str:
    """Creates a record for a completed tuning job."""
    result_id = str(uuid.uuid4())
    metrics_json = json.dumps(metrics, default=json_default_serializer)

    query = """
    INSERT INTO tuning_results
    (result_id, job_id, config_id, metrics, duration)
    VALUES ($1, $2, $3, $4, $5)
    """
    params = (result_id, job_id, config_id, metrics_json, duration)
    await execute_query(query, connection=connection, params=params)
    logger.info(f"Created tuning result: {result_id} for job: {job_id}")
    return result_id


async def create_prediction_result(
    job_id: str,
    model_id: str | None = None,
    output_path: str | None = None,
    summary_metrics: dict[str, Any] | None = None,
    prediction_month: date | str | None = None,
    connection: asyncpg.Connection = None,
) -> str:
    """
    Create or update a prediction result record.
    If a record for the job_id and prediction_month exists, it's updated.
    Otherwise, a new one is created.
    """
    result_id = generate_id()

    # Convert prediction_month to date object for asyncpg
    # Handle both date objects and string formats (YYYY-MM or YYYY-MM-DD)
    if isinstance(prediction_month, date):
        prediction_month_value = prediction_month
    elif isinstance(prediction_month, str):
        # Try parsing YYYY-MM format first, then YYYY-MM-DD
        try:
            if len(prediction_month) == 7:  # YYYY-MM format
                prediction_month_value = date.fromisoformat(prediction_month + "-01")
            else:
                prediction_month_value = date.fromisoformat(prediction_month)
        except ValueError:
            logger.error(f"Invalid prediction_month format: {prediction_month}")
            raise ValueError(f"Invalid prediction_month format: {prediction_month}. Expected YYYY-MM or YYYY-MM-DD")
    elif prediction_month is None:
        prediction_month_value = None
    else:
        raise ValueError(f"Invalid prediction_month type: {type(prediction_month)}")

    # Check if a result already exists for this job_id and month
    check_query = "SELECT result_id FROM prediction_results WHERE job_id = $1 AND prediction_month = $2"
    existing_result = await execute_query(
        check_query, params=(job_id, prediction_month_value), connection=connection
    )

    if existing_result:
        # Update existing record
        result_id = existing_result["result_id"]
        query = """
            UPDATE prediction_results
            SET model_id = COALESCE($1, model_id),
                output_path = COALESCE($2, output_path),
                summary_metrics = COALESCE($3, summary_metrics)
            WHERE result_id = $4
        """
        params = (
            model_id,
            output_path,
            json.dumps(summary_metrics, default=json_default_serializer) if summary_metrics else None,
            result_id,
        )
        logger.info(f"Updating prediction result for job {job_id} and month {prediction_month_value}")
    else:
        # Insert new record
        query = """
        INSERT INTO prediction_results (result_id, job_id, model_id, output_path, summary_metrics, prediction_month)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        params = (
            result_id,
            job_id,
            model_id,
            output_path,
            json.dumps(summary_metrics, default=json_default_serializer) if summary_metrics else None,
            prediction_month_value,
        )
        logger.info(f"Creating new prediction result for job {job_id} and month {prediction_month_value}")

    try:
        await execute_query(query, connection=connection, params=params)

        # Update the job with the result_id
        update_query = "UPDATE jobs SET result_id = $1 WHERE job_id = $2"
        await execute_query(update_query, connection=connection, params=(result_id, job_id))
        logger.info(f"Updated job {job_id} with result_id: {result_id}")

        return result_id
    except DatabaseError:
        logger.error(f"Failed to create/update prediction result for job {job_id}")
        raise


async def get_training_results(
    result_id: str | None = None,
    limit: int = 100,
    connection: asyncpg.Connection = None
) -> dict | list[dict]:
    """
    Get training result(s) by ID or a list of recent results.
    If result_id is provided, returns a single dict.
    If result_id is None, returns a list of dicts, ordered by creation date and limited.
    """
    if result_id:
        query = "SELECT * FROM training_results WHERE result_id = $1"
        result = await execute_query(query, connection=connection, params=(result_id,))
        # Parse JSONB metrics
        if result and result.get("metrics"):
            if isinstance(result["metrics"], dict):
                pass  # Already parsed
            elif isinstance(result["metrics"], str):
                try:
                    result["metrics"] = json.loads(result["metrics"])
                except json.JSONDecodeError:
                    result["metrics"] = {}
        return result
    else:
        query = "SELECT * FROM training_results ORDER BY created_at DESC LIMIT $1"
        results = await execute_query(query, connection=connection, params=(limit,), fetchall=True) or []
        # Parse JSONB metrics
        for r in results:
            if r.get("metrics"):
                if isinstance(r["metrics"], dict):
                    pass
                elif isinstance(r["metrics"], str):
                    try:
                        r["metrics"] = json.loads(r["metrics"])
                    except json.JSONDecodeError:
                        r["metrics"] = {}
        return results


async def get_tuning_results(
    connection: asyncpg.Connection,
    result_id: str | None = None,
    metric_name: str | None = None,
    higher_is_better: bool | None = None,
    limit: int = 100,
) -> dict | list[dict]:
    """
    Get tuning result(s).
    - If result_id is provided, fetches a single result by its ID.
    - If result_id is None, fetches a list of results, sorted by a specified metric
      or by creation date if no metric is provided.
    """
    from deployment.app.db.types import ALLOWED_METRICS
    
    # Fetch a single result by ID
    if result_id:
        query = "SELECT * FROM tuning_results WHERE result_id = $1"
        result = await execute_query(query, connection=connection, params=(result_id,), fetchall=False)
        # Parse JSONB metrics
        if result and result.get("metrics"):
            if isinstance(result["metrics"], dict):
                pass
            elif isinstance(result["metrics"], str):
                try:
                    result["metrics"] = json.loads(result["metrics"])
                except json.JSONDecodeError:
                    result["metrics"] = {}
        return result

    # Fetch a list of results, with optional sorting
    params = [limit]

    if metric_name:
        if higher_is_better is None:
            raise ValueError("`higher_is_better` must be specified when `metric_name` is provided.")
        if metric_name not in ALLOWED_METRICS:
            raise ValueError(f"Invalid metric name: {metric_name}")

        order_direction = "DESC" if higher_is_better else "ASC"
        json_path = f"'{metric_name}'"

        query = f"""
            SELECT *, (metrics->>{json_path})::float as metric_value
            FROM tuning_results
            WHERE metrics IS NOT NULL 
              AND jsonb_typeof(metrics) = 'object' 
              AND (metrics->>{json_path}) IS NOT NULL
            ORDER BY metric_value {order_direction}
            LIMIT $1
        """
    else:
        # Default sorting by creation date if no metric is specified
        query = "SELECT * FROM tuning_results ORDER BY created_at DESC LIMIT $1"

    try:
        results = await execute_query(query, connection=connection, params=tuple(params), fetchall=True) or []
        # Parse JSONB metrics
        for r in results:
            if r.get("metrics"):
                if isinstance(r["metrics"], dict):
                    pass
                elif isinstance(r["metrics"], str):
                    try:
                        r["metrics"] = json.loads(r["metrics"])
                    except json.JSONDecodeError:
                        r["metrics"] = {}
        return results
    except DatabaseError as e:
        logger.error(f"Failed to get tuning results: {e}")
        raise


async def get_prediction_result(
    result_id: str,
    connection: asyncpg.Connection = None
) -> dict:
    """Get prediction result by ID"""
    query = "SELECT * FROM prediction_results WHERE result_id = $1"
    result = await execute_query(query=query, connection=connection, params=(result_id,))
    # Parse JSONB summary_metrics
    if result and result.get("summary_metrics"):
        if isinstance(result["summary_metrics"], dict):
            pass
        elif isinstance(result["summary_metrics"], str):
            try:
                result["summary_metrics"] = json.loads(result["summary_metrics"])
            except json.JSONDecodeError:
                result["summary_metrics"] = {}
    return result


async def get_prediction_results_by_month(
    prediction_month: str,
    model_id: str | None = None,
    connection: asyncpg.Connection = None
) -> list[dict]:
    """Get all prediction results for a specific month, optionally filtered by model_id."""
    query = "SELECT * FROM prediction_results WHERE prediction_month = $1"
    params = [prediction_month]

    if model_id:
        query += " AND model_id = $2"
        params.append(model_id)

    query += " ORDER BY prediction_month DESC"
    results = await execute_query(query, connection=connection, params=tuple(params), fetchall=True) or []
    # Parse JSONB summary_metrics
    for r in results:
        if r.get("summary_metrics"):
            if isinstance(r["summary_metrics"], dict):
                pass
            elif isinstance(r["summary_metrics"], str):
                try:
                    r["summary_metrics"] = json.loads(r["summary_metrics"])
                except json.JSONDecodeError:
                    r["summary_metrics"] = {}
    return results


async def get_predictions(
    job_ids: list[str],
    model_id: str | None = None,
    prediction_month: date | None = None,
    connection: asyncpg.Connection = None
) -> list[dict]:
    """
    Extract prediction data for the given training jobs.

    Args:
        job_ids: List of training job IDs
        model_id: Optional model_id for filtering
        prediction_month: Optional prediction month for filtering
        connection: Required database connection

    Returns:
        List of dictionaries with prediction data
    """
    if not job_ids:
        return []

    # Use batching for large job_ids lists
    
    all_predictions = []
    
    for batch in split_ids_for_batching(job_ids, batch_size=1000):
        placeholders = ", ".join(f"${i+1}" for i in range(len(batch)))
        
        # Get prediction results for these jobs
        base_query = f"""
            SELECT DISTINCT pr.result_id, pr.job_id, pr.model_id
            FROM prediction_results pr
            WHERE pr.job_id IN ({placeholders})
        """
        
        params_list: list[Any] = list(batch)

        if model_id:
            base_query += " AND pr.model_id = $" + str(len(params_list) + 1)
            params_list.append(model_id)

        if prediction_month:
            base_query += " AND pr.prediction_month = $" + str(len(params_list) + 1)
            params_list.append(prediction_month.isoformat() if isinstance(prediction_month, date) else prediction_month)

        prediction_results = await execute_query(
            base_query, params=tuple(params_list), fetchall=True, connection=connection
        ) or []

        if not prediction_results:
            continue

        result_ids = [pr["result_id"] for pr in prediction_results]
        
        # Get actual prediction data
        for result_ids_batch in split_ids_for_batching(result_ids, batch_size=1000):
            result_placeholders = ", ".join(f"${i+1}" for i in range(len(result_ids_batch)))
            
            predictions_query = f"""
                SELECT
                    fp.result_id,
                    dmm.multiindex_id,
                    dmm.barcode,
                    dmm.artist,
                    dmm.album,
                    dmm.cover_type,
                    dmm.price_category,
                    dmm.release_type,
                    dmm.recording_decade,
                    dmm.release_decade,
                    dmm.style,
                    dmm.recording_year,
                    fp.model_id,
                    fp.prediction_month,
                    fp.quantile_05,
                    fp.quantile_25,
                    fp.quantile_50,
                    fp.quantile_75,
                    fp.quantile_95
                FROM fact_predictions fp
                JOIN dim_multiindex_mapping dmm ON fp.multiindex_id = dmm.multiindex_id
                WHERE fp.result_id IN ({result_placeholders})
            """

            query_params = list(result_ids_batch)
            
            if model_id:
                predictions_query += " AND fp.model_id = $" + str(len(query_params) + 1)
                query_params.append(model_id)

            if prediction_month:
                predictions_query += " AND fp.prediction_month = $" + str(len(query_params) + 1)
                query_params.append(prediction_month.isoformat() if isinstance(prediction_month, date) else prediction_month)

            predictions_query += " ORDER BY dmm.artist, dmm.album, fp.prediction_month"

            batch_predictions = await execute_query(
                predictions_query, params=tuple(query_params), fetchall=True, connection=connection
            ) or []
            
            all_predictions.extend(batch_predictions)

    return all_predictions


async def get_data_upload_result(
    result_id: str,
    connection: asyncpg.Connection = None
) -> dict:
    """Get data upload result by ID"""
    query = "SELECT * FROM data_upload_results WHERE result_id = $1"
    result = await execute_query(query=query, connection=connection, params=(result_id,))
    # Parse JSONB features_generated
    if result and result.get("features_generated"):
        if isinstance(result["features_generated"], list):
            pass  # Already parsed
        elif isinstance(result["features_generated"], str):
            try:
                result["features_generated"] = json.loads(result["features_generated"])
            except json.JSONDecodeError:
                result["features_generated"] = []
    return result


async def get_report_result(result_id: str, connection: asyncpg.Connection = None) -> dict:
    """
    Get report result by ID
    """
    query = "SELECT * FROM report_results WHERE result_id = $1"
    return await execute_query(query=query, connection=connection, params=(result_id,))


async def create_data_upload_result(
    job_id: str,
    records_processed: int,
    features_generated: list[str],
    processing_run_id: int,
    connection: asyncpg.Connection = None,
) -> str:
    """
    Create a data upload result record

    Args:
        job_id: Associated job ID
        records_processed: Number of records processed
        features_generated: List of feature types generated
        processing_run_id: ID of the processing run
        connection: Required database connection to use

    Returns:
        Generated result ID
    """
    result_id = generate_id()

    query = """
    INSERT INTO data_upload_results (result_id, job_id, records_processed,
                                    features_generated, processing_run_id)
    VALUES ($1, $2, $3, $4, $5)
    """

    params = (
        result_id,
        job_id,
        records_processed,
        json.dumps(features_generated, default=json_default_serializer),
        processing_run_id,
    )

    await execute_query(query, connection=connection, params=params)
    return result_id


async def insert_predictions(
    result_id: str,
    model_id: str,
    prediction_month: date,
    df: pd.DataFrame,
    connection: asyncpg.Connection = None,
):
    """
    Insert predictions into fact_predictions table.
    
    Args:
        result_id: Result ID for the predictions
        model_id: Model ID used for predictions
        prediction_month: Month for predictions
        df: DataFrame with predictions (must have columns: 0.05, 0.25, 0.5, 0.75, 0.95)
        connection: Required database connection
    """
    import deployment.app.db.queries.multiindex as multiindex_queries
    
    timestamp = datetime.now().isoformat()
    
    # Prepare tuples for batch multiindex creation
    tuples_to_process = (
        df[
            [
                'barcode', 
                'artist', 
                'album', 
                'cover_type', 
                'price_category', 
                'release_type', 
                'recording_decade', 
                'release_decade', 
                'style', 
                'recording_year']
        ]
        .fillna('None')
        .values
        .tolist()
    )
    
    # Get all multiindex_ids in batch
    multiindex_ids = await multiindex_queries.get_or_create_multiindex_ids_batch(tuples_to_process, connection)
    
    # Normalize prediction_month to ISO string
    if isinstance(prediction_month, date):
        pm_str = prediction_month.isoformat()
    else:
        pm_str = str(prediction_month) if prediction_month is not None else None
    if not pm_str:
        # Defensive fallback to first day of current month if missing/invalid
        pm_str = date.today().replace(day=1).isoformat()
    
    # Prepare predictions data
    predictions_data = pd.DataFrame({
        "result_id": [result_id] * len(df),
        "multiindex_id": multiindex_ids,
        "prediction_month": [pm_str] * len(df),
        "model_id": [model_id] * len(df),
        "quantile_05": df['0.05'].values,
        "quantile_25": df['0.25'].values,
        "quantile_50": df['0.5'].values,
        "quantile_75": df['0.75'].values,
        "quantile_95": df['0.95'].values,
        "created_at": [timestamp] * len(df)
    })

    # Use PostgreSQL ON CONFLICT instead of INSERT OR REPLACE
    # The unique constraint is on (multiindex_id, prediction_month, model_id)
    query = """
        INSERT INTO fact_predictions
        (result_id, multiindex_id, prediction_month, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (multiindex_id, prediction_month, model_id) 
        DO UPDATE SET
            result_id = EXCLUDED.result_id,
            quantile_05 = EXCLUDED.quantile_05,
            quantile_25 = EXCLUDED.quantile_25,
            quantile_50 = EXCLUDED.quantile_50,
            quantile_75 = EXCLUDED.quantile_75,
            quantile_95 = EXCLUDED.quantile_95,
            created_at = EXCLUDED.created_at
    """
    
    # Convert prediction_month and created_at strings back to date/datetime objects for asyncpg
    from datetime import datetime as dt
    params_list = [
        (
            row["result_id"],
            row["multiindex_id"],
            date.fromisoformat(row["prediction_month"]) if isinstance(row["prediction_month"], str) else row["prediction_month"],
            row["model_id"],
            row["quantile_05"],
            row["quantile_25"],
            row["quantile_50"],
            row["quantile_75"],
            row["quantile_95"],
            dt.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
        )
        for _, row in predictions_data.iterrows()
    ]
    
    await execute_many(query, params_list, connection=connection)
    
    return {"result_id": result_id, "predictions_count": len(df)}

