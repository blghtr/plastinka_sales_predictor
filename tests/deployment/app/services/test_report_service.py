from datetime import date
from unittest.mock import MagicMock

import pytest

from deployment.app.models.api_models import ReportParams, ReportType
from deployment.app.services.report_service import generate_report


@pytest.fixture
def mock_dal():
    """Provides a mock for the DataAccessLayer."""
    mock = MagicMock()
    mock.get_report_features.return_value = [
        {"barcode": "123", "artist": "Test Artist", "album": "Test Album", "avg_sales_items": 10.5, "quantile_50": 15.0},
        {"barcode": "456", "artist": "Another Artist", "album": "Another Album", "avg_sales_items": 5.2, "quantile_50": 8.0},
    ]
    return mock


def test_generate_report_success(mock_dal):
    """Test successful report generation from pre-calculated features."""
    # Arrange
    params = ReportParams(
        report_type=ReportType.PREDICTION_REPORT,
        prediction_month=date(2023, 1, 1),
        filters={"artist": "Test Artist"}
    )

    # Act
    report_df = generate_report(params, mock_dal)

    # Assert
    assert not report_df.empty
    assert len(report_df) == 2 # The mock returns 2 records
    assert "avg_sales_items" in report_df.columns
    assert "quantile_50" in report_df.columns
    mock_dal.get_report_features.assert_called_once_with(
        prediction_month=date(2023, 1, 1),
        model_id=mock_dal.get_active_model().__getitem__(),
        filters={"artist": "Test Artist"}
    )

def test_generate_report_no_data(mock_dal):
    """Test report generation when no pre-calculated data is found."""
    # Arrange
    mock_dal.get_report_features.return_value = []
    params = ReportParams(
        report_type=ReportType.PREDICTION_REPORT,
        prediction_month=date(2023, 2, 1)
    )

    # Act
    report_df = generate_report(params, mock_dal)

    # Assert
    assert not report_df.empty
    assert "message" in report_df.columns
    assert "No report data found" in report_df["message"][0]

def test_generate_report_invalid_type():
    """Test that an unsupported report type raises a ValueError."""
    # Arrange
    # Create a valid ReportParams first, then modify the report_type to test the function validation
    params = ReportParams(report_type=ReportType.PREDICTION_REPORT, prediction_month=date(2023, 1, 1))
    # Manually set an invalid report type to test the function's validation
    params.report_type = "invalid_report_type"
    mock_dal = MagicMock()

    # Act & Assert
    with pytest.raises(ValueError, match="Unsupported report type"):
        generate_report(params, mock_dal)

def test_generate_report_no_month():
    """Test that a missing prediction month raises a ValueError."""
    # Arrange
    params = ReportParams(report_type=ReportType.PREDICTION_REPORT)
    mock_dal = MagicMock()

    # Act & Assert
    with pytest.raises(ValueError, match="Prediction month must be provided"):
        generate_report(params, mock_dal)
