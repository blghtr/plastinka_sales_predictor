import logging
import calendar

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

    # --- Refactored prediction fetching ---
    prediction_results = dal.get_prediction_results_by_month(
        prediction_month=prediction_month, model_id=model_id
    )
    if not prediction_results:
        logger.warning(
            f"No prediction results found for active model '{model_id}' for month {prediction_month.strftime('%Y-%m')}"
        )
        raise ValueError("No predictions found")

    job_ids = list(set(r["job_id"] for r in prediction_results))

    predictions_list = dal.get_predictions(
        job_ids=job_ids, model_id=model_id, prediction_month=prediction_month
    )
    if not predictions_list:
        logger.warning(
            f"No prediction data found for active model '{model_id}' for month {prediction_month.strftime('%Y-%m')}"
        )
        raise ValueError("No predictions found")

    predictions = pd.DataFrame(predictions_list)
    # --- End of refactoring ---

    if predictions.empty:
        logger.warning(
            f"No predictions found for active model '{model_id}' "
            f"for month {prediction_month.strftime('%Y-%m')}"
        )
        raise ValueError("No predictions found")

    multiidx_have_prediction = predictions["multiindex_id"].unique().tolist()

    # --- Refactored feature fetching ---
    start_date = prediction_month.replace(day=1)
    end_date = start_date.replace(
        day=calendar.monthrange(start_date.year, start_date.month)[1]
    )

    all_features = feature_storage.load_features(
        dal=dal,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )
    report_features_df = all_features.get("report_features")

    if report_features_df is not None and not report_features_df.empty:
        report_features_df = report_features_df[
            report_features_df["multiindex_id"].isin(multiidx_have_prediction)
        ]

        agg_spec = {
            "availability": "mean",
            "confidence": "mean",
            "masked_mean_sales_items": "mean",
            "masked_mean_sales_rub": "mean",
            "lost_sales": "sum",
        }

        existing_agg_spec = {
            k: v for k, v in agg_spec.items() if k in report_features_df.columns
        }

        if existing_agg_spec:
            report_features = (
                report_features_df.groupby("multiindex_id")
                .agg(existing_agg_spec)
                .reset_index()
            )
        else:
            report_features = pd.DataFrame(columns=["multiindex_id"])

    else:
        report_features = pd.DataFrame(
            columns=[
                "multiindex_id",
                "availability",
                "confidence",
                "masked_mean_sales_items",
                "masked_mean_sales_rub",
                "lost_sales",
            ]
        )
    # --- End of refactoring ---

    if report_features.empty and not multiidx_have_prediction:
        logger.warning(
            f"No pre-calculated report features found for month {prediction_month.strftime('%Y-%m')}"
        )
        # Create empty dataframe with expected columns to avoid merge errors and to have columns for report
        report_features = pd.DataFrame(
            columns=[
                "multiindex_id",
                "masked_mean_sales_items",
                "masked_mean_sales_rub",
                "lost_sales",
            ]
        )

    if len(report_features) != len(multiidx_have_prediction):
        logger.warning(
            f"Number of report features ({len(report_features)}) "
            f"does not match number of multiidx_ids with predictions ({len(multiidx_have_prediction)}). "
            "Some metrics will not be available."
        )

    # Convert to DataFrame
    report_df = predictions.merge(report_features, on="multiindex_id", how="left")

    # Check if predictions are missing
    if report_df["quantile_50"].isnull().all():
        logger.error(
            f"No predictions found for active model '{model_id}' for month {prediction_month.strftime('%Y-%m')}"
        )
        raise ValueError(
            f"No predictions found for active model '{model_id}' for the specified month."
        )

    # Rename columns to human-readable names for the report
    feature_mapping = {
        "masked_mean_sales_items": "Средние продажи (шт)",
        "masked_mean_sales_rub": "Средние продажи (руб)",
        "lost_sales": "Потерянные продажи (руб)",
    }
    report_df.rename(columns=feature_mapping, inplace=True)

    logger.info(f"Successfully generated report with {len(report_df)} records.")
    return report_df



