import logging
import pandas as pd
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

    # Fetch pre-calculated features from the database using the DAL
    report_data = dal.get_report_features(
        prediction_month=prediction_month, model_id=model_id, filters=params.filters
    )

    if not report_data:
        logger.warning(f"No pre-calculated report features found for month {prediction_month.strftime('%Y-%m')}")
        return pd.DataFrame({"message": [f"No report data found for month {prediction_month.strftime('%Y-%m')}"]})

    # Convert to DataFrame
    report_df = pd.DataFrame(report_data)

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



