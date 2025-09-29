import pandas as pd
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from deployment.app.models.api_models import ReportParams, ReportType
from deployment.app.services.report_service import generate_report


@pytest.fixture
def mock_dal():
    """Provides a mock for the DataAccessLayer."""
    mock = MagicMock()
    mock.get_active_model.return_value = {"model_id": "test_model"}
    # Default mock values, can be overridden in tests
    mock.get_prediction_results_by_month.return_value = [{"job_id": "job1"}]
    mock.get_predictions.return_value = [
        {"multiindex_id": 1, "quantile_50": 15.0},
        {"multiindex_id": 2, "quantile_50": 8.0},
    ]
    return mock


@patch('deployment.app.db.feature_storage.load_features')
def test_generate_report_success(mock_load_features, mock_dal):
    """Test successful report generation from pre-calculated features."""
    # Arrange
    params = ReportParams(
        report_type=ReportType.PREDICTION_REPORT,
        prediction_month=date(2023, 1, 1),
        filters={"artist": "Test Artist"},
    )

    # Mock for load_features to return daily data
    mock_load_features.return_value = {
        "report_features": pd.DataFrame({
            "multiindex_id": [1, 1, 2, 2],
            "data_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 2)],
            "masked_mean_sales_items": [10.0, 11.0, 5.0, 5.4],
            "lost_sales": [100, 120, 50, 50],
            "availability": [0.8, 0.9, 1.0, 1.0],
            "confidence": [0.95, 0.95, 0.9, 0.9]
        })
    }

    # Act
    report_df = generate_report(params, mock_dal)

    # Assert
    assert not report_df.empty
    assert len(report_df) == 2
    assert "Средние продажи (шт)" in report_df.columns
    assert "Потерянные продажи (руб)" in report_df.columns
    assert "quantile_50" in report_df.columns

    # Check aggregation results
    report_item1 = report_df[report_df['multiindex_id'] == 1].iloc[0]
    assert report_item1['Средние продажи (шт)'] == pytest.approx(10.5)  # (10.0 + 11.0) / 2
    assert report_item1['Потерянные продажи (руб)'] == pytest.approx(220)  # 100 + 120

    report_item2 = report_df[report_df['multiindex_id'] == 2].iloc[0]
    assert report_item2['Средние продажи (шт)'] == pytest.approx(5.2)  # (5.0 + 5.4) / 2
    assert report_item2['Потерянные продажи (руб)'] == pytest.approx(100)  # 50 + 50

    # Assert mock calls
    mock_dal.get_prediction_results_by_month.assert_called_once_with(
        prediction_month=date(2023, 1, 1), model_id="test_model"
    )

    # Check get_predictions call arguments
    mock_dal.get_predictions.assert_called_once()
    call_args, call_kwargs = mock_dal.get_predictions.call_args
    assert set(call_kwargs['job_ids']) == {"job1"}
    assert call_kwargs['model_id'] == "test_model"
    assert call_kwargs['prediction_month'] == date(2023, 1, 1)

    mock_load_features.assert_called_once()
    load_features_kwargs = mock_load_features.call_args.kwargs
    assert load_features_kwargs['dal'] == mock_dal
    assert load_features_kwargs['start_date'] == '2023-01-01'
    assert load_features_kwargs['end_date'] == '2023-01-31'


def test_generate_report_no_prediction_results(mock_dal):
    """Test report generation when no prediction results are found."""
    # Arrange
    mock_dal.get_prediction_results_by_month.return_value = []
    params = ReportParams(
        report_type=ReportType.PREDICTION_REPORT, prediction_month=date(2023, 2, 1)
    )

    # Act & Assert
    with pytest.raises(ValueError, match="No predictions found"):
        generate_report(params, mock_dal)


@patch('deployment.app.db.feature_storage.load_features')
def test_generate_report_no_feature_data(mock_load_features, mock_dal):
    """Test report generation when no feature data is found."""
    # Arrange
    params = ReportParams(
        report_type=ReportType.PREDICTION_REPORT,
        prediction_month=date(2023, 1, 1),
    )
    mock_load_features.return_value = {}  # No 'report_features' key

    # Act
    report_df = generate_report(params, mock_dal)

    # Assert
    assert not report_df.empty
    assert len(report_df) == 2
    assert "Средние продажи (шт)" in report_df.columns
    # The value should be NaN as there were no features to merge
    assert pd.isna(report_df["Средние продажи (шт)"]).all()
    assert pd.isna(report_df["Потерянные продажи (руб)"]).all()


def test_generate_report_invalid_type():
    """Test that an unsupported report type raises a ValueError."""
    # Arrange
    params = ReportParams(report_type=ReportType.PREDICTION_REPORT, prediction_month=date(2023, 1, 1))
    # Manually set an invalid report type to test the function's validation, bypassing pydantic's validation
    object.__setattr__(params, 'report_type', 'invalid_report_type')
    mock_dal = MagicMock()

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported report type"):
        generate_report(params, mock_dal)


def test_generate_report_no_month():
    """Test that a missing prediction month raises a ValueError."""
    # Arrange
    params = ReportParams(report_type=ReportType.PREDICTION_REPORT, prediction_month=None)
    mock_dal = MagicMock()

    # Act & Assert
    with pytest.raises(ValueError, match="Prediction month must be provided"):
        generate_report(params, mock_dal)
