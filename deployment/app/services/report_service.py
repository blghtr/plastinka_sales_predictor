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
    execute_query, get_db_connection, get_active_model
)
from deployment.app.db.feature_storage import load_features
from deployment.app.models.api_models import ReportParams, ReportType

logger = logging.getLogger(__name__)


def generate_report(params: ReportParams) -> pd.DataFrame:
    """
    Generate a prediction report using the specified parameters.
    
    Args:
        params: Report parameters (only prediction_report supported)
        
    Returns:
        DataFrame with prediction report data
    """
    # Validate report type
    if params.report_type != ReportType.PREDICTION_REPORT:
        raise ValueError(f"Unsupported report type: {params.report_type}")
    
    # Determine model_id: explicit in params or fallback to active model
    model_id: Optional[str] = params.model_id
    if model_id is None:
        try:
            active_model_info = get_active_model()
            model_id = active_model_info.get("model_id") if active_model_info else None
            if model_id:
                logger.info(f"Using active model {model_id} for prediction report")
        except Exception as e:
            logger.warning(f"Could not fetch active model: {e}")
            model_id = None  # Proceed without filtering if not available

    # Prepare report parameters
    prediction_month = params.prediction_month
    filters = params.filters or {}
    
    logger.info(f"Generating prediction report for {prediction_month.strftime('%Y-%m')}")
    
    # Generate report
    return _generate_prediction_report(
        prediction_month=prediction_month,
        filters=filters,
        model_id=model_id
    )


def _generate_prediction_report(
    prediction_month: datetime,
    filters: dict,
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate an enhanced prediction report for a specific month with enriched metrics.
    
    Args:
        prediction_month: Month for which predictions are needed
        filters: Additional filters for the report
        model_id: Optional model_id for filtering
        
    Returns:
        DataFrame with prediction data
    """
    logger.info(f"Generating enhanced prediction report for month: {prediction_month.strftime('%Y-%m')}")
    
    # Find training jobs that would have predictions for this month and model (if specified)
    training_jobs = _find_training_jobs_for_prediction_month(prediction_month, model_id)
    
    if not training_jobs:
        logger.warning(f"No training jobs found for prediction month {prediction_month.strftime('%Y-%m')}")
        # Return empty DataFrame with message
        return pd.DataFrame({
            'message': [f'No predictions found for month {prediction_month.strftime("%Y-%m")}']
        })
    
    logger.info(f"Found {len(training_jobs)} training jobs for prediction month")
    
    # Extract predictions for these jobs
    predictions_data = _extract_predictions_for_jobs(training_jobs, filters, model_id)
    
    if predictions_data.empty:
        logger.warning("No prediction data found for training jobs")
        # Return empty DataFrame with message
        return pd.DataFrame({
            'message': [f'No prediction data found for month {prediction_month.strftime("%Y-%m")}']
        })
    
    # Try to load raw features and process them for enrichment
    try:
        raw_features = _load_raw_features_for_report(prediction_month)
        
        if raw_features:
            # Process features to get enriched metrics
            processed_features = _process_features_for_report(raw_features, prediction_month)
            
            if processed_features:
                # Extract enriched columns for the target month
                enriched_columns = _extract_features_for_month(processed_features, prediction_month)
                
                if not enriched_columns.empty:
                    # Join prediction data with enriched metrics
                    predictions_data = _join_predictions_with_enriched_metrics(
                        predictions_data, enriched_columns
                    )
                    logger.info("Successfully enhanced predictions with enriched metrics")
                else:
                    logger.warning(f"No enriched columns could be extracted for {prediction_month.strftime('%Y-%m')}")
            else:
                logger.warning("No processed features available for enrichment")
        else:
            logger.warning("No raw features available for enrichment")
            
    except Exception as e:
        logger.warning(f"Could not load/process enriched features: {e}")
        # Continue with basic report if enrichment fails
    
    logger.info(f"Enhanced prediction report generated with {len(predictions_data)} records")
    
    # Log column summary
    enriched_cols = ['Средние продажи (шт)', 'Средние продажи (руб)', 'Потерянные продажи (руб)']
    present_enriched_cols = [col for col in enriched_cols if col in predictions_data.columns]
    
    if present_enriched_cols:
        logger.info(f"Report includes enriched columns: {present_enriched_cols}")
    else:
        logger.info("Report generated with basic prediction data only")
    
    return predictions_data


def _find_training_jobs_for_prediction_month(
    prediction_month: datetime,
    model_id: Optional[str] = None,
    connection: sqlite3.Connection = None
) -> List[Dict[str, Any]]:
    """
    Find training jobs that have predictions for the specified month and model.
    
    Now simplified: just look for prediction_results with the target prediction_month.
    
    Args:
        prediction_month: Month for which we need predictions
        model_id: Optional model_id for filtering
        connection: Optional database connection to use
        
    Returns:
        List of training job records
    """
    # Format prediction month as YYYY-MM-01 for database comparison
    target_month_str = prediction_month.strftime('%Y-%m-01')
    
    logger.info(f"Looking for training jobs with predictions for month: {prediction_month.strftime('%Y-%m')}")
    
    base_query = """
        SELECT DISTINCT j.job_id, j.parameters, j.created_at, j.status
        FROM jobs j
        JOIN prediction_results pr ON j.job_id = pr.job_id
        WHERE j.job_type = 'training' 
        AND j.status = 'completed'
        AND pr.prediction_month = ?
    """

    params: List[Any] = [target_month_str]

    if model_id:
        base_query += " AND pr.model_id = ?"
        params.append(model_id)

    try:
        jobs = execute_query(base_query, tuple(params), fetchall=True, connection=connection)
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


def _extract_predictions_for_jobs(
    training_jobs: List[Dict[str, Any]],
    filters: Dict[str, Any],
    model_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract prediction data for the given training jobs.
    
    Args:
        training_jobs: List of training job records
        filters: Additional filters to apply
        model_id: Optional model_id for filtering
        
    Returns:
        DataFrame with prediction data
    """
    if not training_jobs:
        return pd.DataFrame()
    
    job_ids = [job['job_id'] for job in training_jobs]
    job_ids_placeholder = ','.join(['?' for _ in job_ids])
    
    # Get prediction results for these jobs
    base_query = f"""
        SELECT DISTINCT pr.result_id, pr.job_id, pr.model_id
        FROM prediction_results pr
        WHERE pr.job_id IN ({job_ids_placeholder})
    """

    params_list: List[Any] = job_ids.copy()

    if model_id:
        base_query += " AND pr.model_id = ?"
        params_list.append(model_id)

    try:
        prediction_results = execute_query(base_query, tuple(params_list), fetchall=True)
        
        if not prediction_results:
            logger.warning("No prediction results found for training jobs")
            return pd.DataFrame()
        
        result_ids = [pr['result_id'] for pr in prediction_results]
        result_ids_placeholder = ','.join(['?' for _ in result_ids])
        
        # Get actual prediction data
        predictions_query = f"""
            SELECT 
                fp.result_id,
                dmm.barcode,
                dmm.artist,
                dmm.album,
                dmm.cover_type,
                dmm.price_category,
                dmm.release_type,
                dmm.recording_decade,
                dmm.release_decade,
                dmm.style,
                dmm.record_year,
                fp.model_id,
                fp.prediction_date,
                fp.quantile_05,
                fp.quantile_25,
                fp.quantile_50,
                fp.quantile_75,
                fp.quantile_95
            FROM fact_predictions fp
            JOIN dim_multiindex_mapping dmm ON fp.multiindex_id = dmm.multiindex_id
            WHERE fp.result_id IN ({result_ids_placeholder})
            ORDER BY dmm.artist, dmm.album, fp.prediction_date
        """

        if model_id:
            predictions_query += " AND fp.model_id = ?"

        query_params = result_ids.copy()
        if model_id:
            query_params.append(model_id)

        predictions_data = execute_query(predictions_query, tuple(query_params), fetchall=True)
        
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


# Enrichment functions for enhanced prediction reports
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
                    # Get index column names (these become columns after reset_index)
                    index_column_names = list(adapted_df.index.names)
                    melted = adapted_df.reset_index().melt(
                        id_vars=index_column_names,
                        var_name='_date',
                        value_name=f"{feature_type}_value"
                    )
                    
                    # Rename the new value column back to its original intended name
                    melted.rename(columns={f"{feature_type}_value": feature_type}, inplace=True)
                    
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
        adapted_features = raw_features # _adapt_features_schema(raw_features)
        
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
                logger.debug(f"Available dates: {feature_df.index.tolist()}")
                
                # Find the closest date to our target
                available_dates = pd.to_datetime(feature_df.index)
                logger.debug(f"Available dates: {available_dates}")

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
                
                if selected_date in feature_df.index:
                    column_data = pd.DataFrame({display_name: feature_df.loc[selected_date]})
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
    