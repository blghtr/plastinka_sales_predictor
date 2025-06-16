"""
Data retention management for optimizing database storage.

This module provides functions for implementing data retention policies,
including cleanup of historical predictions and management of model files.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

from ..config import settings
from .database import get_db_connection, ALLOWED_METRICS # Import ALLOWED_METRICS

logger = logging.getLogger(__name__)

def cleanup_old_predictions(days_to_keep: Optional[int] = None, conn=None) -> int:
    """
    Remove prediction records older than the specified retention period.
    
    Args:
        days_to_keep: Number of days to keep predictions for.
                      If None, uses the value from settings.
        conn: Optional database connection. If None, a new connection is created.
    
    Returns:
        Number of records removed
    """
    if days_to_keep is None:
        days_to_keep = settings.data_retention.prediction_retention_days
    
    retention_date = datetime.now() - timedelta(days=days_to_keep)
    cutoff_date_str = retention_date.strftime('%Y-%m-%d')
    
    # Track if we created this connection or if it was passed in
    connection_created = False
    if conn is None:
        conn = get_db_connection()
        connection_created = True
    
    cursor = conn.cursor()
    
    try:
        # Count records to be deleted
        cursor.execute(
            "SELECT COUNT(*) FROM fact_predictions WHERE DATE(prediction_date) < DATE(?)", 
            (cutoff_date_str,)
        )
        result = cursor.fetchone()
        # Handle both tuple (normal SQLite) and dict (test with dict_factory) result formats
        count = result[0] if isinstance(result, (tuple, list)) else list(result.values())[0]
        
        if count > 0:
            # Delete old predictions
            cursor.execute(
                "DELETE FROM fact_predictions WHERE DATE(prediction_date) < DATE(?)", 
                (cutoff_date_str,)
            )
            conn.commit()
            logger.info(f"Deleted {count} predictions older than {cutoff_date_str}")
        else:
            logger.info(f"No predictions found older than {cutoff_date_str}")
        
        return count
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error cleaning up old predictions: {str(e)}")
        return 0
    
    finally:
        # Only close the connection if we created it
        if connection_created:
            conn.close()

def cleanup_old_historical_data(
    sales_days_to_keep: Optional[int] = None,
    stock_days_to_keep: Optional[int] = None,
    conn=None
) -> Dict[str, int]:
    """
    Remove historical sales and stock data older than the specified retention period.
    
    Args:
        sales_days_to_keep: Number of days to keep sales data.
                           If None, uses the value from settings.
        stock_days_to_keep: Number of days to keep stock data.
                           If None, uses the value from settings.
        conn: Optional database connection. If None, a new connection is created.
    
    Returns:
        Dictionary with counts of removed records by type
    """
    if sales_days_to_keep is None:
        sales_days_to_keep = settings.data_retention.sales_retention_days
    if stock_days_to_keep is None:
        stock_days_to_keep = settings.data_retention.stock_retention_days
    
    # Calculate cutoff dates
    sales_cutoff = datetime.now() - timedelta(days=sales_days_to_keep)
    stock_cutoff = datetime.now() - timedelta(days=stock_days_to_keep)
    
    sales_cutoff_str = sales_cutoff.strftime('%Y-%m-%d')
    stock_cutoff_str = stock_cutoff.strftime('%Y-%m-%d')
    
    # Track if we created this connection or if it was passed in
    connection_created = False
    if conn is None:
        conn = get_db_connection()
        connection_created = True
    
    cursor = conn.cursor()
    result = {"sales": 0, "stock": 0, "stock_changes": 0, "prices": 0}
    
    try:
        # Clean up sales data
        cursor.execute(
            "SELECT COUNT(*) FROM fact_sales WHERE data_date < ?",
            (sales_cutoff_str,)
        )
        sales_count = cursor.fetchone()[0]
        
        if sales_count > 0:
            cursor.execute(
                "DELETE FROM fact_sales WHERE data_date < ?",
                (sales_cutoff_str,)
            )
            result["sales"] = sales_count
            logger.info(f"Deleted {sales_count} sales records older than {sales_cutoff_str}")
        
        # Clean up stock data
        cursor.execute(
            "SELECT COUNT(*) FROM fact_stock WHERE data_date < ?",
            (stock_cutoff_str,)
        )
        stock_count = cursor.fetchone()[0]
        
        if stock_count > 0:
            cursor.execute(
                "DELETE FROM fact_stock WHERE data_date < ?",
                (stock_cutoff_str,)
            )
            result["stock"] = stock_count
            logger.info(f"Deleted {stock_count} stock records older than {stock_cutoff_str}")
        
        # Clean up stock change data
        cursor.execute(
            "SELECT COUNT(*) FROM fact_stock_changes WHERE data_date < ?",
            (stock_cutoff_str,)
        )
        changes_count = cursor.fetchone()[0]
        
        if changes_count > 0:
            cursor.execute(
                "DELETE FROM fact_stock_changes WHERE data_date < ?",
                (stock_cutoff_str,)
            )
            result["stock_changes"] = changes_count
            logger.info(f"Deleted {changes_count} stock change records older than {stock_cutoff_str}")
        
        # Clean up price data
        cursor.execute(
            "SELECT COUNT(*) FROM fact_prices WHERE data_date < ?",
            (sales_cutoff_str,)
        )
        prices_count = cursor.fetchone()[0]
        
        if prices_count > 0:
            cursor.execute(
                "DELETE FROM fact_prices WHERE data_date < ?",
                (sales_cutoff_str,)
            )
            result["prices"] = prices_count
            logger.info(f"Deleted {prices_count} price records older than {sales_cutoff_str}")
        
        conn.commit()
        return result
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error cleaning up historical data: {str(e)}")
        return result
    
    finally:
        # Only close the connection if we created it
        if connection_created:
            conn.close()

def cleanup_old_models(models_to_keep: Optional[int] = None, 
                      inactive_days_to_keep: Optional[int] = None,
                      conn=None) -> List[str]:
    """
    Clean up old model records and files based on retention policy.
    
    For each active parameter set, keeps the top N models ranked by validation metric.
    For inactive models, removes those older than the specified period.
    
    Args:
        models_to_keep: Number of models to keep per parameter set.
                        If None, uses the value from settings.
        inactive_days_to_keep: Number of days to keep inactive models.
                               If None, uses the value from settings.
        conn: Optional database connection. If None, a new connection is created.
    
    Returns:
        List of model IDs that were deleted
    """
    if models_to_keep is None:
        models_to_keep = settings.data_retention.models_to_keep
    if inactive_days_to_keep is None:
        inactive_days_to_keep = settings.data_retention.inactive_model_retention_days
    
    # Track if we created this connection or if it was passed in
    connection_created = False
    if conn is None:
        conn = get_db_connection()
        connection_created = True
    
    cursor = conn.cursor()
    
    deleted_model_ids = []
    
    try:
        # 1. Get active parameter sets
        cursor.execute("SELECT parameter_set_id FROM parameter_sets WHERE is_active = 1")
        active_param_sets = [row[0] for row in cursor.fetchall()]
        
        # 2. For each active parameter set, keep only top N models by metric
        default_metric = settings.default_metric
        higher_is_better = settings.default_metric_higher_is_better
        
        order_direction = "DESC" if higher_is_better else "ASC"
        
        if default_metric not in ALLOWED_METRICS:
            logger.error(f"Invalid default_metric '{default_metric}' provided to cleanup_models_for_active_parameter_sets.")
            # Fallback to a safe default or raise an error
            # For now, let's raise an error as this indicates a configuration issue.
            raise ValueError(f"Invalid default_metric: {default_metric}. Allowed metrics are: {ALLOWED_METRICS}")

        json_path = f"'$.{default_metric}'" # metric_name is validated

        for param_set_id in active_param_sets:
            # Get models for this parameter set, sorted by metric
            # The default_metric for JSON_EXTRACT and order_direction are now safely constructed.
            query = f"""
                SELECT m.model_id, m.model_path
                FROM models m
                JOIN training_results tr ON m.model_id = tr.model_id
                WHERE tr.parameter_set_id = ?
                ORDER BY json_extract(tr.metrics, {json_path}) {order_direction}
            """
            cursor.execute(query, (param_set_id,))
            
            models = cursor.fetchall()
            
            # If we have more models than we need to keep, delete the excess
            if len(models) > models_to_keep:
                models_to_delete = models[models_to_keep:]
                
                for model_id, model_path in models_to_delete:
                    # Check if model is currently used in any predictions
                    cursor.execute(
                        "SELECT COUNT(*) FROM fact_predictions WHERE model_id = ?", 
                        (model_id,)
                    )
                    prediction_count = cursor.fetchone()[0]
                    
                    if prediction_count == 0:
                        # Safe to delete model as it's not referenced by predictions
                        cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                        
                        # Delete associated training results
                        cursor.execute(
                            "DELETE FROM training_results WHERE model_id = ?", 
                            (model_id,)
                        )
                        
                        # Delete physical file if it exists
                        if model_path and os.path.exists(model_path):
                            try:
                                os.remove(model_path)
                                logger.info(f"Deleted model file: {model_path}")
                            except OSError as e:
                                logger.warning(f"Failed to delete model file {model_path}: {str(e)}")
                        
                        deleted_model_ids.append(model_id)
                        logger.info(f"Deleted model {model_id} (excess model for parameter set {param_set_id})")
        
        # 3. Clean up inactive models older than retention period
        retention_date = datetime.now() - timedelta(days=inactive_days_to_keep)
        cutoff_date_str = retention_date.strftime('%Y-%m-%d %H:%M:%S')
        
        cursor.execute("""
            SELECT model_id, model_path 
            FROM models 
            WHERE is_active = 0 AND created_at < ?
        """, (cutoff_date_str,))
        
        inactive_models = cursor.fetchall()
        for model_id, model_path in inactive_models:
            # Check if model is used in any predictions
            cursor.execute(
                "SELECT COUNT(*) FROM fact_predictions WHERE model_id = ?", 
                (model_id,)
            )
            prediction_count = cursor.fetchone()[0]
            
            if prediction_count == 0:
                # Safe to delete
                cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                
                # Delete associated training results
                cursor.execute(
                    "DELETE FROM training_results WHERE model_id = ?", 
                    (model_id,)
                )
                
                # Delete physical file
                if model_path and os.path.exists(model_path):
                    try:
                        os.remove(model_path)
                        logger.info(f"Deleted model file: {model_path}")
                    except OSError as e:
                        logger.warning(f"Failed to delete model file {model_path}: {str(e)}")
                
                deleted_model_ids.append(model_id)
                logger.info(f"Deleted inactive model {model_id} (older than {cutoff_date_str})")
        
        conn.commit()
        return deleted_model_ids
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error cleaning up old models: {str(e)}")
        return []
    
    finally:
        # Only close the connection if we created it
        if connection_created:
            conn.close()

def run_cleanup_job() -> None:
    """
    Run a complete cleanup job, applying all data retention policies.
    
    This function is designed to be called periodically (e.g., via cron job)
    to maintain optimal database size and remove unnecessary files.
    """
    if not settings.data_retention.cleanup_enabled:
        logger.info("Data retention cleanup is disabled in settings")
        return
    
    logger.info("Starting data retention cleanup job")
    
    # Create a single connection for all operations
    conn = get_db_connection()
    
    try:
        # Clean up old predictions
        try:
            predictions_removed = cleanup_old_predictions(conn=conn)
            logger.info(f"Removed {predictions_removed} old predictions")
        except Exception as e:
            logger.error(f"Error in prediction cleanup: {str(e)}")
        
        # Clean up old models
        try:
            models_removed = cleanup_old_models(conn=conn)
            logger.info(f"Removed {len(models_removed)} old models: {', '.join(models_removed) if models_removed else 'none'}")
        except Exception as e:
            logger.error(f"Error in model cleanup: {str(e)}")
        
        # Clean up historical data
        try:
            historical_data_results = cleanup_old_historical_data(conn=conn)
            logger.info(f"Historical data cleanup results: {historical_data_results}")
        except Exception as e:
            logger.error(f"Error in historical data cleanup: {str(e)}")
        
        logger.info("Data retention cleanup job completed")
    
    finally:
        # Close the connection when done
        conn.close() 