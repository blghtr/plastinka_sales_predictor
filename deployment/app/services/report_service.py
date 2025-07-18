import logging
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from deployment.app.config import get_settings
from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.db.database import (
    execute_query,
    get_active_model,
    update_job_status,
)
from deployment.app.db.feature_storage import load_features
from deployment.app.models.api_models import JobStatus, ReportParams, ReportType

logger = logging.getLogger(__name__)


def generate_report(params: ReportParams, dal: DataAccessLayer) -> pd.DataFrame:
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
    model_id: str | None = params.model_id
    if model_id is None:
        try:
            active_model_info = dal.get_active_model()
            model_id = active_model_info.get("model_id") if active_model_info else None
            if model_id:
                logger.info(f"Using active model {model_id} for prediction report")
        except Exception as e:
            logger.warning(f"Could not fetch active model: {e}")
            model_id = None  # Proceed without filtering if not available

    # Prepare report parameters
    prediction_month = params.prediction_month
    filters = params.filters or {}

    if prediction_month is None:
        raise ValueError("Prediction month must be provided to generate a report.")

    logger.info(
        f"Generating prediction report for {prediction_month.strftime('%Y-%m')}"
    )

    # Generate report
    return _generate_prediction_report(
        prediction_month=prediction_month,
        filters=filters,
        model_id=model_id,
        dal=dal,
    )


def _generate_prediction_report(
    prediction_month: datetime,
    filters: dict,
    dal: DataAccessLayer,
    model_id: str | None = None,
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
    logger.info(
        f"Generating enhanced prediction report for month: {prediction_month.strftime('%Y-%m')}"
    )

    # Find training jobs that would have predictions for this month and model (if specified)
    training_jobs = _find_training_jobs_for_prediction_month(
        prediction_month, dal, model_id
    )

    if not training_jobs:
        logger.warning(
            f"No training jobs found for prediction month {prediction_month.strftime('%Y-%m')}"
        )
        # Return empty DataFrame with message
        return pd.DataFrame(
            {
                "message": [
                    f"No predictions found for month {prediction_month.strftime('%Y-%m')}"
                ]
            }
        )

    logger.info(f"Found {len(training_jobs)} training jobs for prediction month")

    # Extract predictions for these jobs
    predictions_data = _extract_predictions_for_jobs(
        training_jobs, filters, dal, model_id
    )

    if predictions_data.empty:
        logger.warning("No prediction data found for training jobs")
        # Return empty DataFrame with message
        return pd.DataFrame(
            {
                "message": [
                    f"No prediction data found for month {prediction_month.strftime('%Y-%m')}"
                ]
            }
        )

    # Try to load raw features and process them for enrichment
    try:
        raw_features = _load_raw_features_for_report(prediction_month)

        if raw_features:
            # Process features to get enriched metrics
            processed_features = _process_features_for_report(
                raw_features, prediction_month
            )

            if processed_features:
                # Extract enriched columns for the target month
                enriched_columns = _extract_features_for_month(
                    processed_features, prediction_month
                )

                if not enriched_columns.empty:
                    # Join prediction data with enriched metrics
                    predictions_data = _join_predictions_with_enriched_metrics(
                        predictions_data, enriched_columns
                    )
                    logger.info(
                        "Successfully enhanced predictions with enriched metrics"
                    )
                else:
                    logger.warning(
                        f"No enriched columns could be extracted for {prediction_month.strftime('%Y-%m')}"
                    )
            else:
                logger.warning("No processed features available for enrichment")
        else:
            logger.warning("No raw features available for enrichment")

    except Exception as e:
        logger.warning(f"Could not load/process enriched features: {e}")
        # Continue with basic report if enrichment fails

    logger.info(
        f"Enhanced prediction report generated with {len(predictions_data)} records"
    )

    # Log column summary
    enriched_cols = [
        "Средние продажи (шт)",
        "Средние продажи (руб)",
        "Потерянные продажи (руб)",
    ]
    present_enriched_cols = [
        col for col in enriched_cols if col in predictions_data.columns
    ]

    if present_enriched_cols:
        logger.info(f"Report includes enriched columns: {present_enriched_cols}")
    else:
        logger.info("Report generated with basic prediction data only")

    return predictions_data


def _find_training_jobs_for_prediction_month(
    prediction_month: datetime,
    dal: DataAccessLayer,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Finds completed training jobs that have predictions for the specified month and model.
    This is now achieved by fetching prediction results and then retrieving the associated job details.
    """
    logger.info(
        f"Looking for prediction results for month: {prediction_month.strftime('%Y-%m')}"
    )

    try:
        # Use the DAL to get prediction results for the month and optional model
        prediction_results = dal.get_prediction_results_by_month(
            prediction_month=prediction_month, model_id=model_id
        )

        if not prediction_results:
            logger.info(
                f"No prediction results found for {prediction_month.strftime('%Y-%m')}"
            )
            return []

        # Get unique job IDs from the results
        job_ids = list(set(res["job_id"] for res in prediction_results))

        # Fetch the full job details for each unique job ID
        jobs = []
        for job_id in job_ids:
            job_details = dal.get_job(job_id)
            if job_details and job_details.get("status") == "completed":
                jobs.append(job_details)

        if jobs:
            logger.info(
                f"Found {len(jobs)} completed training jobs with predictions for {prediction_month.strftime('%Y-%m')}"
            )
        else:
            logger.warning(
                f"Found prediction results, but no associated completed training jobs for {prediction_month.strftime('%Y-%m')}"
            )

        return jobs

    except Exception as e:
        logger.error(
            f"Error finding training jobs with predictions via DAL: {e}", exc_info=True
        )
        return []


def _extract_predictions_for_jobs(
    training_jobs: list[dict[str, Any]],
    filters: dict[str, Any],
    dal: DataAccessLayer,  # Pass the DAL instance
    model_id: str | None = None,
) -> pd.DataFrame:
    """
    Extract prediction data for the given training jobs using the DataAccessLayer.

    Args:
        training_jobs: List of training job records
        filters: Additional filters to apply
        dal: The DataAccessLayer instance
        model_id: Optional model_id for filtering

    Returns:
        DataFrame with prediction data
    """
    if not training_jobs:
        return pd.DataFrame()

    job_ids = [job["job_id"] for job in training_jobs]

    try:
        # Use the new DAL method to fetch predictions
        predictions_data = dal.get_predictions_for_jobs(job_ids=job_ids, model_id=model_id)

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
        logger.error(f"Error extracting predictions using DAL: {e}", exc_info=True)
        return pd.DataFrame()


def _apply_filters_to_dataframe(
    df: pd.DataFrame, filters: dict[str, Any]
) -> pd.DataFrame:
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
def _load_raw_features_for_report(
    prediction_month: datetime,
) -> dict[str, pd.DataFrame]:
    """
    Load raw features needed for enriched report generation.

    Args:
        prediction_month: Month for which to load features

    Returns:
        Dictionary of raw features
    """
    try:
        logger.info(
            f"Loading raw features for enriched report generation for {prediction_month.strftime('%Y-%m')}"
        )

        raw_features = load_features(
            store_type="sql", feature_types=["sales", "change", "stock", "prices"]
        )

        # Log what we loaded
        for feature_type, df in raw_features.items():
            if hasattr(df, "shape"):
                logger.info(f"Loaded {feature_type}: {df.shape}")
            else:
                logger.info(f"Loaded {feature_type}: {type(df)}")

        return raw_features

    except Exception as e:
        logger.error(f"Error loading raw features for report: {e}", exc_info=True)
        return {}


def _process_features_for_report(
    raw_features: dict[str, pd.DataFrame], prediction_month: datetime
) -> dict[str, pd.DataFrame]:
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
        logger.info(
            f"Processing features for report generation for {prediction_month.strftime('%Y-%m')}"
        )

        # Check if we have required features
        required_features = ["sales", "change", "stock", "prices"]
        missing_features = [
            f for f in required_features
            if f not in raw_features or raw_features[f].empty
        ]

        if missing_features:
            logger.warning(
                f"Missing required features for processing: {missing_features}"
            )
            return {}

        # Apply the core processing logic (adapted from notebook)
        new_features = defaultdict(list)

        sales = raw_features["sales"]
        change = raw_features["change"]
        stock = raw_features["stock"].T
        prices = raw_features["prices"].T

        # Group by month periods
        change_groups = change.groupby(
            change.index.to_period("M")
        )

        sales_groups = sales.abs().groupby(
            sales.index.to_period("M")
        )

        # Process each month (core algorithm from notebook)
        for month, daily_change in change_groups:
            try:
                change_pivot = daily_change.T

                # Get corresponding sales data
                try:
                    sales_pivot = sales_groups.get_group(month).T
                except KeyError:
                    logger.warning(f"No sales data found for month {month}")
                    continue

                # Update stock with changes
                stock = stock.join(
                    change_pivot,
                    how="outer"
                ).fillna(0).cumsum(axis=1)

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
                    sales_pivot.where(availability_mask)
                    .mean(axis=1, skipna=True)
                    .fillna(0)
                    .round(2)
                    .loc[common_index]
                    .rename(month)
                )

                # Get prices for this index
                try:
                    prices_for_index = prices.loc[
                        masked_mean_sales_items.index,
                    ].rename(columns=lambda _: month).iloc[:, 0]
                except KeyError:
                    logger.warning(
                        f"Price data not found for some items in month {month}"
                    )
                    prices_for_index = pd.Series(
                        0, index=masked_mean_sales_items.index, name=month
                    )

                # Calculate sales in rubles
                masked_mean_sales_rub = (
                    masked_mean_sales_items.mul(prices_for_index)
                    .fillna(0)
                    .loc[common_index]
                    .astype(int)
                )

                # Calculate lost sales
                lost_sales = (
                    (~availability_mask)
                    .astype(int)
                    .sum(axis=1, skipna=True)
                    .rename(month)
                    .mul(masked_mean_sales_rub)
                    .fillna(0)
                    .loc[common_index]
                )

                # Update stock for next iteration
                stock = last_stock

                # Store calculated metrics
                new_features["availability"].append(in_stock_frac)
                new_features["confidence"].append(month_conf)
                new_features["masked_mean_sales_items"].append(masked_mean_sales_items)
                new_features["masked_mean_sales_rub"].append(masked_mean_sales_rub)
                new_features["lost_sales"].append(lost_sales)

            except Exception as e:
                logger.error(f"Error processing month {month}: {e}", exc_info=True)
                continue

        # Consolidate features by month
        processed_features = {}
        for k, v in new_features.items():
            if v:  # Only process if we have data
                try:
                    k_df = pd.concat(v, axis=1).fillna(0).T
                    k_df = k_df.set_index(k_df.index.to_timestamp("ms"))
                    processed_features[k] = k_df
                except Exception as e:
                    logger.error(f"Error consolidating feature {k}: {e}")
                    continue

        logger.info(f"Successfully processed {len(processed_features)} feature types")
        return processed_features

    except Exception as e:
        logger.error(f"Error in process_features_for_report: {e}", exc_info=True)
        return {}


def _extract_features_for_month(
    processed_features: dict[str, pd.DataFrame], target_month: datetime
) -> pd.DataFrame:
    """
    Extract features for specific prediction month with Russian business names.

    Args:
        processed_features: Processed features dictionary
        target_month: Target month for extraction

    Returns:
        DataFrame with enriched columns for the target month
    """
    try:
        target_date = pd.to_datetime(target_month.strftime("%Y-%m-01"))
        logger.info(f"Extracting features for target date: {target_date}")

        columns_data = []
        feature_mapping = [
            ("masked_mean_sales_items", "Средние продажи (шт)"),
            ("masked_mean_sales_rub", "Средние продажи (руб)"),
            ("lost_sales", "Потерянные продажи (руб)"),
        ]

        for feature_key, display_name in feature_mapping:
            if feature_key in processed_features:
                feature_df = processed_features[feature_key]
                # Find the closest date to our target
                available_dates = pd.to_datetime(feature_df.index)

                # Try exact match first
                if target_date in available_dates:
                    selected_date = target_date
                else:
                    # Find closest date in the same month
                    same_month_dates = available_dates[
                        (available_dates.year == target_date.year)
                        & (available_dates.month == target_date.month)
                    ]

                    if len(same_month_dates) > 0:
                        selected_date = same_month_dates[0]
                    else:
                        # Use the latest available date
                        selected_date = available_dates.max()
                        logger.warning(
                            f"No data for target month {target_month.strftime('%Y-%m')}, using {selected_date}"
                        )

                if selected_date in feature_df.index:
                    column_data = pd.DataFrame(
                        {display_name: feature_df.loc[selected_date]}
                    )
                    columns_data.append(column_data)


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


def _join_predictions_with_enriched_metrics(
    predictions_df: pd.DataFrame, enriched_columns: pd.DataFrame
) -> pd.DataFrame:
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

        logger.info(
            f"Joining predictions ({predictions_df.shape}) with enriched metrics ({enriched_columns.shape})"
        )

        # Create a mapping key for predictions using product attributes
        prediction_key_cols = list(predictions_df.columns[1:11])

        # Ensure all key columns exist in predictions
        available_key_cols = [
            col for col in prediction_key_cols if col in predictions_df.columns
        ]

        if not available_key_cols:
            logger.error("No matching key columns found for joining")
            return predictions_df

        # Create composite key for predictions
        predictions_with_multiindex = predictions_df.copy().fillna('None')
        predictions_with_multiindex = predictions_with_multiindex.set_index(
            prediction_key_cols
        )

        enriched_with_multiindex = enriched_columns.copy()
        merged_df = pd.merge(
            predictions_with_multiindex,
            enriched_with_multiindex,
            left_index=True,
            right_index=True,
            how="left",
        )
        logger.info(f"Successfully joined data, result shape: {merged_df.shape}")

        # Log summary of enriched data
        for col in enriched_columns.columns:
            if col in merged_df.columns:
                non_null_count = merged_df[col].notna().sum()
                logger.info(
                    f"Column '{col}': {non_null_count}/{len(merged_df)} rows have data"
                )

        return merged_df.fillna('None')

    except Exception as e:
        logger.error(
            f"Error joining predictions with enriched metrics: {e}", exc_info=True
        )
        return predictions_df



