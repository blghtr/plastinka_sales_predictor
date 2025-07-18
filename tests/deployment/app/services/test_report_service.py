from datetime import datetime
from unittest.mock import MagicMock

from deployment.app.services.report_service import (
    _find_training_jobs_for_prediction_month,
)


def test_find_training_jobs_for_prediction_month_success():
    """
    Tests that the function correctly uses the DAL to find jobs
    associated with prediction results for a given month.
    """
    # Arrange
    mock_dal = MagicMock()
    prediction_month = datetime(2023, 10, 1)
    model_id = "model-123"

    # Mock the DAL calls
    mock_dal.get_prediction_results_by_month.return_value = [
        {"job_id": "job-1", "model_id": model_id},
        {"job_id": "job-2", "model_id": model_id},
        {"job_id": "job-1", "model_id": "other-model"}, # Duplicate job_id
    ]
    mock_dal.get_job.side_effect = [
        {"job_id": "job-1", "status": "completed"},
        {"job_id": "job-2", "status": "failed"}, # This job should be filtered out
    ]

    # Act
    jobs = _find_training_jobs_for_prediction_month(
        prediction_month, mock_dal, model_id
    )

    # Assert
    assert len(jobs) == 1
    assert jobs[0]["job_id"] == "job-1"

    # Verify DAL calls
    mock_dal.get_prediction_results_by_month.assert_called_once_with(
        prediction_month=prediction_month, model_id=model_id
    )
    # It should be called for each *unique* job ID
    assert mock_dal.get_job.call_count == 2
    mock_dal.get_job.assert_any_call("job-1")
    mock_dal.get_job.assert_any_call("job-2")


def test_find_training_jobs_no_results():
    """
    Tests that the function returns an empty list when no prediction results are found.
    """
    # Arrange
    mock_dal = MagicMock()
    prediction_month = datetime(2023, 10, 1)
    mock_dal.get_prediction_results_by_month.return_value = []

    # Act
    jobs = _find_training_jobs_for_prediction_month(prediction_month, mock_dal)

    # Assert
    assert jobs == []
    mock_dal.get_prediction_results_by_month.assert_called_once()
    mock_dal.get_job.assert_not_called() 