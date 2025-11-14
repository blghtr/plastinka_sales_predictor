import logging
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.dependencies import get_dal_for_general_user
from deployment.app.models.api_models import (
    ActualItem,
    DateRangeMetadata,
    FeatureItem,
    PredictionHistoryItem,
    PredictionHistoryMetadata,
    PredictionHistoryParams,
    PredictionHistoryResponse,
    PredictionItem,
    PredictionQuantiles,
    ProductMetadata,
)
from deployment.app.services.auth import get_unified_auth
from deployment.app.utils.error_handling import AppValidationError, ErrorDetail

logger = logging.getLogger("plastinka.api.predictions")

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


def get_prediction_history_params(
    multiindex_id: int | None = Query(None, description="Filter by specific multiindex_id"),
    prediction_month_from: str | None = Query(None, description="Start date for prediction range (YYYY-MM-01)"),
    prediction_month_to: str | None = Query(None, description="End date for prediction range (YYYY-MM-01)"),
    include_actuals: bool = Query(True, description="Include actual sales data"),
    include_features: bool = Query(True, description="Include report_features data"),
) -> PredictionHistoryParams:
    """
    Dependency function to parse and validate query parameters for predictions history.
    
    Converts string dates to date objects and creates PredictionHistoryParams model.
    Validation happens in model_validator which raises ValidationError (AppValidationError).
    """
    # Parse date strings to date objects
    parsed_from = None
    parsed_to = None
    
    if prediction_month_from:
        try:
            parsed_from = date.fromisoformat(prediction_month_from)
        except ValueError as e:
            raise AppValidationError(
                "Invalid date format. Expected YYYY-MM-01",
                details={
                    "parameter": "prediction_month_from",
                    "value": prediction_month_from,
                    "error": str(e)
                }
            ) from e
    
    if prediction_month_to:
        try:
            parsed_to = date.fromisoformat(prediction_month_to)
        except ValueError as e:
            raise AppValidationError(
                "Invalid date format. Expected YYYY-MM-01",
                details={
                    "parameter": "prediction_month_to",
                    "value": prediction_month_to,
                    "error": str(e)
                }
            ) from e
    
    # Create and validate model (validation happens in model_validator)
    # If ValidationError is raised, it will be caught by AppValidationError handler
    try:
        return PredictionHistoryParams(
            multiindex_id=multiindex_id,
            prediction_month_from=parsed_from,
            prediction_month_to=parsed_to,
            include_actuals=include_actuals,
            include_features=include_features,
        )
    except AppValidationError:
        # Re-raise AppValidationError as-is (it's already the right type)
        raise


def _transform_to_response(data: dict[str, Any]) -> PredictionHistoryResponse:
    """
    Transform database results into PredictionHistoryResponse.
    
    Groups predictions, actuals, and features by multiindex_id.
    
    Args:
        data: Dictionary with keys: predictions, actuals, features
        
    Returns:
        PredictionHistoryResponse with grouped data
    """
    predictions = data.get("predictions", [])
    actuals = data.get("actuals", [])
    features = data.get("features", [])
    
    # Group by multiindex_id
    items_dict: dict[int, dict[str, Any]] = {}
    
    # Process predictions
    for pred in predictions:
        multiindex_id = pred["multiindex_id"]
        if multiindex_id not in items_dict:
            items_dict[multiindex_id] = {
                "multiindex_id": multiindex_id,
                "metadata": {
                    "barcode": pred.get("barcode"),
                    "artist": pred.get("artist"),
                    "album": pred.get("album"),
                    "cover_type": pred.get("cover_type"),
                    "price_category": pred.get("price_category"),
                    "release_type": pred.get("release_type"),
                    "recording_decade": pred.get("recording_decade"),
                    "release_decade": pred.get("release_decade"),
                    "style": pred.get("style"),
                    "recording_year": pred.get("recording_year"),
                },
                "predictions": [],
                "actuals": [],
                "features": []
            }
        
        # Add prediction
        items_dict[multiindex_id]["predictions"].append({
            "prediction_month": pred["prediction_month"],
            "model_id": pred["model_id"],
            "quantiles": {
                "q05": float(pred["quantile_05"]),
                "q25": float(pred["quantile_25"]),
                "q50": float(pred["quantile_50"]),
                "q75": float(pred["quantile_75"]),
                "q95": float(pred["quantile_95"]),
            }
        })
    
    # Process actuals
    for actual in actuals:
        multiindex_id = actual["multiindex_id"]
        if multiindex_id in items_dict:
            items_dict[multiindex_id]["actuals"].append({
                "month": actual["month"],
                "sales_count": float(actual["sales_count"])
            })
    
    # Process features
    for feature in features:
        multiindex_id = feature["multiindex_id"]
        if multiindex_id in items_dict:
            items_dict[multiindex_id]["features"].append({
                "month": feature["month"],
                "availability": float(feature.get("availability", 0.0)),
                "confidence": float(feature.get("confidence", 0.0)),
                "masked_mean_sales_items": float(feature.get("masked_mean_sales_items", 0.0)),
                "masked_mean_sales_rub": float(feature.get("masked_mean_sales_rub", 0.0)),
                "lost_sales_rub": float(feature.get("lost_sales_rub", 0.0)),
            })
    
    # Convert to list of PredictionHistoryItem
    items = []
    all_dates: list[date] = []
    
    for item_data in items_dict.values():
        # Collect all dates for metadata
        for pred in item_data["predictions"]:
            pred_date = pred["prediction_month"]
            if isinstance(pred_date, str):
                all_dates.append(date.fromisoformat(pred_date))
            elif isinstance(pred_date, datetime):
                all_dates.append(pred_date.date())
            else:
                all_dates.append(pred_date)
        
        for actual in item_data["actuals"]:
            actual_date = actual["month"]
            if isinstance(actual_date, str):
                all_dates.append(date.fromisoformat(actual_date))
            elif isinstance(actual_date, datetime):
                all_dates.append(actual_date.date())
            else:
                all_dates.append(actual_date)
        
        for feature in item_data["features"]:
            feature_date = feature["month"]
            if isinstance(feature_date, str):
                all_dates.append(date.fromisoformat(feature_date))
            elif isinstance(feature_date, datetime):
                all_dates.append(feature_date.date())
            else:
                all_dates.append(feature_date)
        
        items.append(PredictionHistoryItem(**item_data))
    
    # Calculate metadata
    date_range = DateRangeMetadata(
        min=min(all_dates) if all_dates else date.today(),
        max=max(all_dates) if all_dates else date.today()
    )
    
    metadata = PredictionHistoryMetadata(
        total_items=len(items),
        date_range=date_range
    )
    
    return PredictionHistoryResponse(items=items, metadata=metadata)


@router.get("/history", response_model=PredictionHistoryResponse, summary="Get predictions history")
async def get_predictions_history(
    params: PredictionHistoryParams = Depends(get_prediction_history_params),
    dal: DataAccessLayer = Depends(get_dal_for_general_user),
    x_api_key_valid: dict[str, Any] = Depends(get_unified_auth),
):
    """
    Retrieve historical predictions and related data for dashboard visualization.
    
    Returns predictions, actual sales, and report features grouped by product (multiindex_id).
    All dates must be in YYYY-MM-01 format (first day of month).
    """
    try:
        # Query database
        data = await dal.get_predictions_history(
            multiindex_id=params.multiindex_id,
            prediction_month_from=params.prediction_month_from,
            prediction_month_to=params.prediction_month_to,
            include_actuals=params.include_actuals,
            include_features=params.include_features,
        )
        
        # Check if no predictions found
        if not data.get("predictions"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "message": "No predictions found for the specified criteria",
                    "code": "no_predictions_found",
                    "details": {
                        "multiindex_id": params.multiindex_id,
                        "prediction_month_from": params.prediction_month_from.isoformat() if params.prediction_month_from else None,
                        "prediction_month_to": params.prediction_month_to.isoformat() if params.prediction_month_to else None
                    }
                }
            )
        
        # Transform to response format
        response = _transform_to_response(data)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get predictions history: {e}", exc_info=True)
        error = ErrorDetail(
            message="Failed to retrieve predictions history",
            code="internal_server_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"error": str(e)},
            exception=e,
        )
        error.log_error()
        raise HTTPException(
            status_code=error.status_code,
            detail=error.to_response_model().model_dump()
        ) from e

