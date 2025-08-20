import logging

import pandas as pd

from deployment.app.db import feature_storage
from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.models.api_models import ReportParams, ReportType

logger = logging.getLogger(__name__)

def generate_report(params: ReportParams, dal: DataAccessLayer) -> pd.DataFrame:
    """
    Generate a prediction report by fetching pre-calculated features.

    Args:
        params: Report parameters, including prediction_month and filters.
        dal: DataAccessLayer instance.

    Returns:
        DataFrame with prediction report data.
    """

    if params.report_type != ReportType.PREDICTION_REPORT:
        raise ValueError(f"Unsupported report type: {params.report_type}")

    prediction_month = params.prediction_month
    if prediction_month is None:
        raise ValueError("Prediction month must be provided to generate a report.")

    logger.info(
        f"Generating prediction report for {prediction_month.strftime('%Y-%m')} from pre-calculated features."
    )

    # Get active model
    active_model = dal.get_active_model()
    if not active_model:
        raise ValueError("No active model found. Cannot generate report.")
    model_id = active_model["model_id"]

    predictions = dal.get_predictions(
        prediction_month=prediction_month, 
        model_id=model_id,
    )
    if predictions.empty:
        logger.warning(f"No predictions found for active model '{model_id}' "
                       f"for month {prediction_month.strftime('%Y-%m')}")
        raise ValueError("No predictions found")
    
    multiidx_have_prediction = predictions["multiindex_id"].tolist()
    
    # Fetch pre-calculated features from the database using the DAL
    report_features = dal.get_report_features(
        prediction_month=prediction_month,
        multiidx_ids=multiidx_have_prediction,
        feature_subset=["avg_sales_items", "avg_sales_rub", "lost_sales_rub"],
    )

    if report_features.empty:
        logger.warning(f"No pre-calculated report features found for month {prediction_month.strftime('%Y-%m')}")
        return pd.DataFrame()

    if len(report_features) != len(multiidx_have_prediction):
        logger.warning(
            f"Number of report features ({len(report_features)}) "
            f"does not match number of multiidx_ids ({len(multiidx_have_prediction)}). "
            "Some metrics will not be available."
        )

    # Convert to DataFrame
    report_df = predictions.merge(
        report_features, 
        on="multiindex_id", 
        how="left"
    )

    # Check if predictions are missing
    if report_df['quantile_50'].isnull().all():
        logger.error(f"No predictions found for active model '{model_id}' for month {prediction_month.strftime('%Y-%m')}")
        raise ValueError(f"No predictions found for active model '{model_id}' for the specified month.")

    # Rename columns to human-readable names for the report
    feature_mapping = {
        'masked_mean_sales_items': 'Средние продажи (шт)',
        'masked_mean_sales_rub': 'Средние продажи (руб)',
        'lost_sales': 'Потерянные продажи (руб)',
    }
    report_df.rename(columns=feature_mapping, inplace=True)

    logger.info(f"Successfully generated report with {len(report_df)} records.")
    return report_df



