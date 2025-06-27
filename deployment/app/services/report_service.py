from datetime import datetime, timedelta
import time
import os
import json
import uuid
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import sqlite3
from collections import defaultdict

from deployment.app.db.database import (
    update_job_status, create_report_result, execute_query, get_db_connection
)
from deployment.app.db.feature_storage import load_features
from deployment.app.models.api_models import JobStatus, ReportParams, ReportType

logger = logging.getLogger(__name__)


async def generate_report(job_id: str, params: ReportParams) -> None:
    """
    Generate a report using the specified parameters.
    
    Args:
        job_id: ID of the job
        params: Report parameters
    """
    try:
        # Update job status to running
        update_job_status(job_id, JobStatus.RUNNING.value, progress=0)
        
        # Prepare report parameters
        report_type = params.report_type
        start_date = params.start_date
        end_date = params.end_date
        filters = params.filters or {}
        
        # Log the parameters
        update_job_status(
            job_id,
            JobStatus.RUNNING.value,
            progress=10,
            status_message=f"Generating {report_type} report"
        )
        
        # Create output directory
        output_dir = Path(f"deployment/data/reports/{report_type}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report based on type
        output_path = await _generate_report_by_type(
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            output_dir=output_dir,
            job_id=job_id
        )
        
        # Create result
        result_id = create_report_result(
            job_id=job_id,
            report_type=report_type.value,
            parameters={
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "filters": filters
            },
            output_path=str(output_path)
        )
        
        # Update job as completed
        update_job_status(
            job_id,
            JobStatus.COMPLETED.value,
            progress=100,
            result_id=result_id
        )
        
    except Exception as e:
        # Update job as failed with error message
        update_job_status(
            job_id,
            JobStatus.FAILED.value,
            error_message=str(e)
        )
        # Re-raise for logging
        raise


async def _generate_report_by_type(
    report_type: ReportType,
    start_date: datetime,
    end_date: datetime,
    filters: dict,
    output_dir: Path,
    job_id: str
) -> Path:
    """
    Generate a report based on its type.
    
    Args:
        report_type: Type of report to generate
        start_date: Start date for the report
        end_date: End date for the report
        filters: Additional filters for the report
        output_dir: Directory to save the report
        job_id: ID of the job
        
    Returns:
        Path to the generated report
    """
    
    if report_type == ReportType.PREDICTION_REPORT:
        return await _generate_prediction_report(
            prediction_month=start_date,
            filters=filters,
            output_dir=output_dir,
            job_id=job_id
        )
    elif report_type == ReportType.SALES_SUMMARY:
        return await _generate_sales_summary_report(
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            output_dir=output_dir,
            job_id=job_id
        )
    elif report_type == ReportType.MODEL_PERFORMANCE:
        return await _generate_model_performance_report(
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            output_dir=output_dir,
            job_id=job_id
        )
    elif report_type == ReportType.PREDICTION_ACCURACY:
        return await _generate_prediction_accuracy_report(
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            output_dir=output_dir,
            job_id=job_id
        )
    elif report_type == ReportType.INVENTORY_ANALYSIS:
        return await _generate_inventory_analysis_report(
            start_date=start_date,
            end_date=end_date,
            filters=filters,
            output_dir=output_dir,
            job_id=job_id
        )
    else:
        raise ValueError(f"Unsupported report type: {report_type}")


async def _generate_prediction_report(
    prediction_month: datetime,
    filters: dict,
    output_dir: Path,
    job_id: str
) -> Path:
    """
    Generate an enhanced prediction report for a specific month with enriched metrics.
    
    Args:
        prediction_month: Month for which predictions are needed
        filters: Additional filters for the report
        output_dir: Directory to save the report
        job_id: ID of the job
        
    Returns:
        Path to the generated report
    """
    logger.info(f"[{job_id}] Generating enhanced prediction report for month: {prediction_month.strftime('%Y-%m')}")
    
    # Update progress
    update_job_status(job_id, JobStatus.RUNNING.value, progress=20, 
                     status_message="Finding training jobs for prediction month")
    
    # Find training jobs that would have predictions for this month
    training_jobs = _find_training_jobs_for_prediction_month(prediction_month)
    
    if not training_jobs:
        logger.warning(f"[{job_id}] No training jobs found for prediction month {prediction_month.strftime('%Y-%m')}")
        # Create empty report
        output_path = output_dir / f"prediction_report_{prediction_month.strftime('%Y_%m')}_empty.csv"
        pd.DataFrame({
            'message': [f'No predictions found for month {prediction_month.strftime("%Y-%m")}']
        }).to_csv(output_path, index=False)
        return output_path
    
    logger.info(f"[{job_id}] Found {len(training_jobs)} training jobs for prediction month")
    
    # Update progress
    update_job_status(job_id, JobStatus.RUNNING.value, progress=40, 
                     status_message="Extracting predictions from database")
    
    # Extract predictions for these jobs
    predictions_data = _extract_predictions_for_jobs(training_jobs, filters)
    
    if predictions_data.empty:
        logger.warning(f"[{job_id}] No prediction data found for training jobs")
        # Create empty report
        output_path = output_dir / f"prediction_report_{prediction_month.strftime('%Y_%m')}_no_data.csv"
        pd.DataFrame({
            'message': [f'No prediction data found for month {prediction_month.strftime("%Y-%m")}']
        }).to_csv(output_path, index=False)
        return output_path
    
    # NEW: Load raw features and process them for enrichment
    update_job_status(job_id, JobStatus.RUNNING.value, progress=60, 
                     status_message="Loading raw features for enrichment")
    
    try:
        raw_features = _load_raw_features_for_report(prediction_month)
        
        if raw_features:
            # Update progress
            update_job_status(job_id, JobStatus.RUNNING.value, progress=70, 
                             status_message="Processing features for enriched metrics")
            
            # Process features to get enriched metrics
            processed_features = _process_features_for_report(raw_features, prediction_month)
            
            if processed_features:
                # Update progress
                update_job_status(job_id, JobStatus.RUNNING.value, progress=80, 
                                 status_message="Extracting enriched columns")
                
                # Extract enriched columns for the target month
                enriched_columns = _extract_features_for_month(processed_features, prediction_month)
                
                if not enriched_columns.empty:
                    # Join prediction data with enriched metrics
                    update_job_status(job_id, JobStatus.RUNNING.value, progress=85, 
                                     status_message="Joining predictions with enriched metrics")
                    
                    predictions_data = _join_predictions_with_enriched_metrics(
                        predictions_data, enriched_columns
                    )
                    logger.info(f"[{job_id}] Successfully enhanced predictions with enriched metrics")
                else:
                    logger.warning(f"[{job_id}] No enriched columns could be extracted for {prediction_month.strftime('%Y-%m')}")
            else:
                logger.warning(f"[{job_id}] No processed features available for enrichment")
        else:
            logger.warning(f"[{job_id}] No raw features available for enrichment")
            
    except Exception as e:
        logger.warning(f"[{job_id}] Could not load/process enriched features: {e}")
        # Continue with basic report if enrichment fails
    
    # Update progress
    update_job_status(job_id, JobStatus.RUNNING.value, progress=90, 
                     status_message="Generating final report")
    
    # Generate enhanced report file
    output_path = output_dir / f"prediction_report_{prediction_month.strftime('%Y_%m')}_enhanced.csv"
    predictions_data.to_csv(output_path, index=False)
    
    logger.info(f"[{job_id}] Enhanced prediction report generated: {output_path} with {len(predictions_data)} records")
    
    # Log column summary
    enriched_cols = ['Средние продажи (шт)', 'Средние продажи (руб)', 'Потерянные продажи (руб)']
    present_enriched_cols = [col for col in enriched_cols if col in predictions_data.columns]
    
    if present_enriched_cols:
        logger.info(f"[{job_id}] Report includes enriched columns: {present_enriched_cols}")
    else:
        logger.info(f"[{job_id}] Report generated with basic prediction data only")
    
    return output_path


def _find_training_jobs_for_prediction_month(prediction_month: datetime, connection: sqlite3.Connection = None) -> List[Dict[str, Any]]:
    """
    Find training jobs that have predictions for the specified month.
    
    Now simplified: just look for prediction_results with the target prediction_month.
    
    Args:
        prediction_month: Month for which we need predictions
        connection: Optional database connection to use
        
    Returns:
        List of training job records
    """
    # Format prediction month as YYYY-MM-01 for database comparison
    target_month_str = prediction_month.strftime('%Y-%m-01')
    
    logger.info(f"Looking for training jobs with predictions for month: {prediction_month.strftime('%Y-%m')}")
    
    query = """
        SELECT DISTINCT j.job_id, j.parameters, j.created_at, j.status
        FROM jobs j
        JOIN prediction_results pr ON j.job_id = pr.job_id
        WHERE j.job_type = 'training' 
        AND j.status = 'completed'
        AND pr.prediction_month = ?
    """
    
    try:
        jobs = execute_query(query, (target_month_str,), fetchall=True, connection=connection)
        if jobs:
            logger.info(f"Found {len(jobs)} training jobs with predictions for {prediction_month.strftime('%Y-%m')}")
            for job in jobs:
                logger.info(f"Found job: {job['job_id']}")
        else:
            logger.info(f"No training jobs found with predictions for {prediction_month.strftime('%Y-%m')}")
        
        return jobs or []
        
    except Exception as e:
        logger.error(f"Error finding training jobs with predictions: {e}")
        return []


def _extract_predictions_for_jobs(training_jobs: List[Dict[str, Any]], filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract prediction data for the given training jobs.
    
    Args:
        training_jobs: List of training job records
        filters: Additional filters to apply
        
    Returns:
        DataFrame with prediction data
    """
    if not training_jobs:
        return pd.DataFrame()
    
    job_ids = [job['job_id'] for job in training_jobs]
    job_ids_placeholder = ','.join(['?' for _ in job_ids])
    
    # Get prediction results for these jobs
    query = f"""
        SELECT DISTINCT pr.result_id, pr.job_id, pr.model_id
        FROM prediction_results pr
        WHERE pr.job_id IN ({job_ids_placeholder})
    """
    
    try:
        prediction_results = execute_query(query, tuple(job_ids), fetchall=True)
        
        if not prediction_results:
            logger.warning("No prediction results found for training jobs")
            return pd.DataFrame()
        
        result_ids = [pr['result_id'] for pr in prediction_results]
        result_ids_placeholder = ','.join(['?' for _ in result_ids])
        
        # Get actual prediction data
        predictions_query = f"""
            SELECT 
                fp.result_id,
                fp.model_id,
                fp.prediction_date,
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
                dmm.record_year
            FROM fact_predictions fp
            JOIN dim_multiindex_mapping dmm ON fp.multiindex_id = dmm.multiindex_id
            WHERE fp.result_id IN ({result_ids_placeholder})
            ORDER BY dmm.artist, dmm.album, fp.prediction_date
        """
        
        predictions_data = execute_query(predictions_query, tuple(result_ids), fetchall=True)
        
        if not predictions_data:
            logger.warning("No prediction data found in fact_predictions")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions_data)
        
        # Apply filters if any
        if filters:
            df = _apply_filters_to_dataframe(df, filters)
        
        logger.info(f"Extracted {len(df)} prediction records")
        return df
        
    except Exception as e:
        logger.error(f"Error extracting predictions: {e}")
        return pd.DataFrame()


def _apply_filters_to_dataframe(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to the predictions DataFrame.
    
    Args:
        df: DataFrame to filter
        filters: Dictionary of filters to apply
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for filter_key, filter_value in filters.items():
        if filter_key in filtered_df.columns:
            if isinstance(filter_value, list):
                filtered_df = filtered_df[filtered_df[filter_key].isin(filter_value)]
            else:
                filtered_df = filtered_df[filtered_df[filter_key] == filter_value]
    
    return filtered_df


# New enrichment functions for enhanced prediction reports
def _load_raw_features_for_report(prediction_month: datetime) -> Dict[str, pd.DataFrame]:
    """
    Load raw features needed for enriched report generation.
    
    Args:
        prediction_month: Month for which to load features
        
    Returns:
        Dictionary of raw features
    """
    try:
        logger.info(f"Loading raw features for enriched report generation for {prediction_month.strftime('%Y-%m')}")
        
        raw_features = load_features(
            store_type='sql',
            feature_types=['sales', 'change', 'stock', 'prices']
        )
        
        # Log what we loaded
        for feature_type, df in raw_features.items():
            if hasattr(df, 'shape'):
                logger.info(f"Loaded {feature_type}: {df.shape}")
            else:
                logger.info(f"Loaded {feature_type}: {type(df)}")
        
        return raw_features
        
    except Exception as e:
        logger.error(f"Error loading raw features for report: {e}", exc_info=True)
        return {}


def _adapt_features_schema(raw_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Adapt features loaded from database to match notebook expectations.
    
    The feature_storage.load_features() returns DataFrames with MultiIndex structure
    reconstructed from dim_multiindex_mapping. This function ensures the schema
    matches what the notebook's process_features function expects.
    
    Args:
        raw_features: Raw features from database
        
    Returns:
        Adapted features with proper schema
    """
    adapted_features = {}
    
    for feature_type, df in raw_features.items():
        if df.empty:
            adapted_features[feature_type] = df
            continue
            
        try:
            if feature_type in ['sales', 'change']:
                # These have dates in the columns (from pivot), restructure for notebook compatibility
                adapted_df = df.copy()
                
                # Ensure we have the right structure - MultiIndex rows, date columns
                if isinstance(adapted_df.index, pd.MultiIndex):
                    # The database system should already have the correct MultiIndex
                    # We need to reshape to have dates in index for notebook compatibility
                    
                    # Melt the DataFrame to get dates as index level
                    melted = adapted_df.reset_index().melt(
                        id_vars=[col for col in adapted_df.reset_index().columns if col not in adapted_df.columns],
                        var_name='_date',
                        value_name=feature_type
                    )
                    
                    # Set MultiIndex with _date as the last level
                    index_cols = [col for col in melted.columns if col not in ['_date', feature_type]]
                    melted = melted.set_index(index_cols + ['_date'])
                    
                    adapted_features[feature_type] = melted
                else:
                    adapted_features[feature_type] = adapted_df
                    
            else:
                # stock and prices can be used as-is
                adapted_features[feature_type] = df
                
        except Exception as e:
            logger.error(f"Error adapting {feature_type} schema: {e}", exc_info=True)
            adapted_features[feature_type] = df
    
    return adapted_features


def _process_features_for_report(raw_features: Dict[str, pd.DataFrame], prediction_month: datetime) -> Dict[str, pd.DataFrame]:
    """
    Simplified version of notebook's process_features adapted for database system.
    
    Key adaptations:
    1. Works with feature_storage.load_features() output format
    2. Uses dynamic prediction_month instead of hardcoded date
    3. Handles English column names from database schema
    4. Returns only the specific metrics needed for reports
    
    Args:
        raw_features: Raw features from database
        prediction_month: Month for which to generate features
        
    Returns:
        Dictionary of processed features
    """
    try:
        logger.info(f"Processing features for report generation for {prediction_month.strftime('%Y-%m')}")
        
        # Adapt schema to match notebook expectations
        adapted_features = _adapt_features_schema(raw_features)
        
        # Check if we have required features
        required_features = ['sales', 'change', 'stock', 'prices']
        missing_features = [f for f in required_features if f not in adapted_features or adapted_features[f].empty]
        
        if missing_features:
            logger.warning(f"Missing required features for processing: {missing_features}")
            return {}
        
        # Apply the core processing logic (adapted from notebook)
        new_features = defaultdict(list)
        
        sales = adapted_features['sales']
        daily_movements = adapted_features['change']
        stock = adapted_features['stock']
        prices = adapted_features['prices']
        
        # Get index structure (non-date columns)
        idx = [i for i in daily_movements.index.names if not i.startswith('_')]
        
        # Group by month periods
        change_groups = daily_movements.groupby(
            pd.to_datetime(daily_movements.index.get_level_values('_date')).to_period('M')
        )
        
        sales_groups = sales.abs().groupby(
            pd.to_datetime(sales.index.get_level_values('_date')).to_period('M')
        )
        
        # Process each month (core algorithm from notebook)
        for month, daily_change in change_groups:
            try:
                logger.debug(f"Processing month: {month}")
                
                daily_change = daily_change.reset_index()
                change_pivot = daily_change.pivot_table(
                    index=idx,
                    columns='_date',
                    values='change',
                    aggfunc='sum',
                    fill_value=0,
                )

                # Get corresponding sales data
                try:
                    daily_sales = sales_groups.get_group(month).reset_index()
                    sales_pivot = daily_sales.pivot_table(
                        index=idx,
                        columns='_date',
                        values='sales',
                        aggfunc='sum',
                        fill_value=0,
                    )
                except KeyError:
                    logger.warning(f"No sales data found for month {month}")
                    continue

                # Update stock with changes
                stock = stock.join(change_pivot, how='outer').fillna(0).cumsum(axis=1)
                
                try:
                    stock = stock.sort_index(axis=1)
                except TypeError:
                    # Mixed-type columns, keep current order
                    pass
                
                last_stock, stock = stock.iloc[:, -1:], stock.iloc[:, :-1]
                
                # Calculate confidence and availability
                conf = stock.clip(0, 5) / 5
                month_conf = conf.mean(1).rename(month)

                in_stock = stock.clip(0, 1)
                availability_mask = in_stock > 0
                in_stock_frac = in_stock.mean(1).rename(month)

                # Calculate masked sales metrics
                common_index = sales_pivot.index
                masked_mean_sales_items = (
                    sales_pivot.where(availability_mask).mean(axis=1, skipna=True)
                    .fillna(0).round(2).loc[common_index].rename(month)
                )

                # Get prices for this index
                try:
                    prices_for_index = prices.loc[masked_mean_sales_items.index, 'prices'].rename(month)
                except KeyError:
                    logger.warning(f"Price data not found for some items in month {month}")
                    prices_for_index = pd.Series(0, index=masked_mean_sales_items.index).rename(month)

                # Calculate sales in rubles
                masked_mean_sales_rub = (
                    masked_mean_sales_items.mul(prices_for_index)
                    .fillna(0).loc[common_index].astype(int)
                )

                # Calculate lost sales
                lost_sales = (
                    (~availability_mask).astype(int).sum(axis=1, skipna=True)
                    .rename(month).mul(masked_mean_sales_rub)
                    .fillna(0).loc[common_index]
                )

                # Update stock for next iteration
                stock = last_stock

                # Store calculated metrics
                new_features['availability'].append(in_stock_frac)
                new_features['confidence'].append(month_conf)
                new_features['masked_mean_sales_items'].append(masked_mean_sales_items)
                new_features['masked_mean_sales_rub'].append(masked_mean_sales_rub)
                new_features['lost_sales'].append(lost_sales)
                
            except Exception as e:
                logger.error(f'Error processing month {month}: {e}', exc_info=True)
                continue

        # Consolidate features by month
        processed_features = {}
        for k, v in new_features.items():
            if v:  # Only process if we have data
                try:
                    k_df = pd.concat(v, axis=1).fillna(0).T
                    k_df = k_df.set_index(k_df.index.to_timestamp('ms'))
                    processed_features[k] = k_df
                except Exception as e:
                    logger.error(f"Error consolidating feature {k}: {e}")
                    continue
        
        logger.info(f"Successfully processed {len(processed_features)} feature types")
        return processed_features
        
    except Exception as e:
        logger.error(f"Error in process_features_for_report: {e}", exc_info=True)
        return {}


def _extract_features_for_month(processed_features: Dict[str, pd.DataFrame], target_month: datetime) -> pd.DataFrame:
    """
    Extract features for specific prediction month with Russian business names.
    
    Args:
        processed_features: Processed features dictionary
        target_month: Target month for extraction
        
    Returns:
        DataFrame with enriched columns for the target month
    """
    try:
        target_date = pd.to_datetime(target_month.strftime('%Y-%m-01'))
        logger.info(f"Extracting features for target date: {target_date}")
        
        columns_data = []
        feature_mapping = [
            ('masked_mean_sales_items', 'Средние продажи (шт)'),
            ('masked_mean_sales_rub', 'Средние продажи (руб)'),
            ('lost_sales', 'Потерянные продажи (руб)')
        ]
        
        for feature_key, display_name in feature_mapping:
            if feature_key in processed_features:
                feature_df = processed_features[feature_key]
                logger.debug(f"Processing feature {feature_key}, shape: {feature_df.shape}")
                logger.debug(f"Available dates: {feature_df.columns.tolist()}")
                
                # Find the closest date to our target
                available_dates = pd.to_datetime(feature_df.columns)
                
                # Try exact match first
                if target_date in available_dates:
                    selected_date = target_date
                else:
                    # Find closest date in the same month
                    same_month_dates = available_dates[
                        (available_dates.year == target_date.year) & 
                        (available_dates.month == target_date.month)
                    ]
                    
                    if len(same_month_dates) > 0:
                        selected_date = same_month_dates[0]
                    else:
                        # Use the latest available date
                        selected_date = available_dates.max()
                        logger.warning(f"No data for target month {target_month.strftime('%Y-%m')}, using {selected_date}")
                
                if selected_date in feature_df.columns:
                    column_data = pd.DataFrame(
                        feature_df[selected_date]
                    ).rename(columns={selected_date: display_name})
                    columns_data.append(column_data)
                    logger.debug(f"Added column {display_name} with {len(column_data)} rows")
        
        if columns_data:
            result = pd.concat(columns_data, axis=1)
            logger.info(f"Successfully extracted features: {result.shape}")
            return result
        else:
            logger.warning("No feature columns could be extracted")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error extracting features for month: {e}", exc_info=True)
        return pd.DataFrame()


def _join_predictions_with_enriched_metrics(predictions_df: pd.DataFrame, enriched_columns: pd.DataFrame) -> pd.DataFrame:
    """
    Join prediction data with enriched metrics using product attribute matching.
    
    Args:
        predictions_df: DataFrame with prediction data
        enriched_columns: DataFrame with enriched metrics
        
    Returns:
        Enhanced predictions DataFrame
    """
    try:
        if enriched_columns.empty:
            logger.warning("No enriched columns to join")
            return predictions_df
        
        logger.info(f"Joining predictions ({predictions_df.shape}) with enriched metrics ({enriched_columns.shape})")
        
        # Create a mapping key for predictions using product attributes
        prediction_key_cols = ['barcode', 'artist', 'album', 'cover_type', 'price_category', 
                             'release_type', 'recording_decade', 'release_decade', 'style', 'record_year']
        
        # Ensure all key columns exist in predictions
        available_key_cols = [col for col in prediction_key_cols if col in predictions_df.columns]
        
        if not available_key_cols:
            logger.error("No matching key columns found for joining")
            return predictions_df
        
        # Create composite key for predictions
        predictions_with_key = predictions_df.copy()
        predictions_with_key['_join_key'] = predictions_with_key[available_key_cols].apply(
            lambda x: tuple(x.values), axis=1
        )
        
        # Create composite key for enriched data
        enriched_with_key = enriched_columns.copy()
        if isinstance(enriched_columns.index, pd.MultiIndex):
            # MultiIndex case - combine all levels
            enriched_with_key['_join_key'] = enriched_columns.index.to_series().apply(
                lambda x: tuple(x) if isinstance(x, tuple) else (x,)
            )
        else:
            # Single index case
            enriched_with_key['_join_key'] = enriched_columns.index.to_series().apply(
                lambda x: (x,)
            )
        
        # Reset index for joining
        enriched_with_key = enriched_with_key.reset_index(drop=True)
        
        # Perform the join
        result = predictions_with_key.merge(
            enriched_with_key[['_join_key'] + list(enriched_columns.columns)],
            on='_join_key',
            how='left'
        )
        
        # Clean up
        result = result.drop('_join_key', axis=1)
        
        logger.info(f"Successfully joined data, result shape: {result.shape}")
        
        # Log summary of enriched data
        for col in enriched_columns.columns:
            if col in result.columns:
                non_null_count = result[col].notna().sum()
                logger.info(f"Column '{col}': {non_null_count}/{len(result)} rows have data")
        
        return result
        
    except Exception as e:
        logger.error(f"Error joining predictions with enriched metrics: {e}", exc_info=True)
        return predictions_df


# Stub implementations for other report types
async def _generate_sales_summary_report(
    start_date: datetime,
    end_date: datetime,
    filters: dict,
    output_dir: Path,
    job_id: str
) -> Path:
    """Generate sales summary report (stub implementation)."""
    logger.info(f"[{job_id}] Generating sales summary report (stub)")
    output_path = output_dir / f"sales_summary_{uuid.uuid4().hex[:8]}.csv"
    pd.DataFrame({'message': ['Sales summary report not yet implemented']}).to_csv(output_path, index=False)
    return output_path


async def _generate_model_performance_report(
    start_date: datetime,
    end_date: datetime,
    filters: dict,
    output_dir: Path,
    job_id: str
) -> Path:
    """Generate model performance report (stub implementation)."""
    logger.info(f"[{job_id}] Generating model performance report (stub)")
    output_path = output_dir / f"model_performance_{uuid.uuid4().hex[:8]}.csv"
    pd.DataFrame({'message': ['Model performance report not yet implemented']}).to_csv(output_path, index=False)
    return output_path


async def _generate_prediction_accuracy_report(
    start_date: datetime,
    end_date: datetime,
    filters: dict,
    output_dir: Path,
    job_id: str
) -> Path:
    """Generate prediction accuracy report (stub implementation)."""
    logger.info(f"[{job_id}] Generating prediction accuracy report (stub)")
    output_path = output_dir / f"prediction_accuracy_{uuid.uuid4().hex[:8]}.csv"
    pd.DataFrame({'message': ['Prediction accuracy report not yet implemented']}).to_csv(output_path, index=False)
    return output_path


async def _generate_inventory_analysis_report(
    start_date: datetime,
    end_date: datetime,
    filters: dict,
    output_dir: Path,
    job_id: str
) -> Path:
    """Generate inventory analysis report (stub implementation)."""
    logger.info(f"[{job_id}] Generating inventory analysis report (stub)")
    output_path = output_dir / f"inventory_analysis_{uuid.uuid4().hex[:8]}.csv"
    pd.DataFrame({'message': ['Inventory analysis report not yet implemented']}).to_csv(output_path, index=False)
    return output_path
    