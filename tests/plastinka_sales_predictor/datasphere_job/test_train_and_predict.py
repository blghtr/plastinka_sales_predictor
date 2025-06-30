"""
Comprehensive tests for plastinka_sales_predictor/datasphere_job/train_and_predict.py

This test suite covers all functions in the train_and_predict module with comprehensive mocking
of external dependencies. Tests are organized by function groups and include both success and
failure scenarios.

Testing Approach:
- Mock all external dependencies (PlastinkaTrainingTSDataset, model training, file I/O)
- Test individual functions in isolation
- Test main pipeline integration
- Verify error handling and validation
- Test CLI interface

All external imports and dependencies are mocked to ensure test isolation.
"""

import json
import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, call, mock_open, patch

import click.testing
import numpy as np
import pandas as pd
import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from plastinka_sales_predictor.datasphere_job import train_and_predict


class TestTrainModel:
    """Test suite for train_model function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict._train_model')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.prepare_for_training')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.extract_early_stopping_callback')
    def test_train_model_success(self, mock_extract_callback, mock_prepare, mock_train_model):
        """Test successful model training with validation and final training phases."""
        # Arrange
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {
            'model_config': {'n_epochs': 10},
            'model_id': 'test_model'
        }

        # Mock model objects
        mock_model_1 = MagicMock()
        mock_model_1.trainer.current_epoch = 8
        mock_model_1.trainer.callback_metrics = {
            'val_loss': MagicMock(item=lambda: 0.5),
            'val_accuracy': MagicMock(item=lambda: 0.85)
        }

        mock_model_2 = MagicMock()
        mock_model_2.trainer.callback_metrics = {
            'train_loss': MagicMock(item=lambda: 0.3),
            'train_accuracy': MagicMock(item=lambda: 0.9)
        }

        # Mock early stopping callback
        mock_callback = MagicMock()
        mock_callback.wait_count = 2
        mock_extract_callback.return_value = mock_callback

        # Setup mocks
        mock_prepare.side_effect = [
            ('prepared_train', 'prepared_val'),  # First call with validation
            ('prepared_full',)  # Second call with full dataset
        ]
        mock_train_model.side_effect = [mock_model_1, mock_model_2]
        mock_train_dataset.setup_dataset.return_value = 'full_train_ds'
        mock_train_dataset._n_time_steps = 100

        # Act
        model, metrics = train_and_predict.train_model(mock_train_dataset, mock_val_dataset, config)

        # Assert
        assert model == mock_model_2
        assert 'validation' in metrics
        assert 'training' in metrics
        assert metrics['validation']['val_loss'] == 0.5
        assert metrics['validation']['val_accuracy'] == 0.85
        assert metrics['training']['train_loss'] == 0.3
        assert metrics['training']['train_accuracy'] == 0.9

        # Verify function calls
        assert mock_train_model.call_count == 2
        mock_extract_callback.assert_called_once()

        # Verify effective epochs calculation
        expected_effective_epochs = max(1, int((8 - 1 - 2) * 1.1))
        assert config['model_config']['n_epochs'] == expected_effective_epochs

    def test_train_model_exception_handling(self):
        """Test train_model handles exceptions properly."""
        # Arrange
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {'model_config': {'n_epochs': 10}, 'model_id': 'test_model'}

        with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.prepare_for_training') as mock_prepare:
            mock_prepare.side_effect = RuntimeError("Training failed")

            # Act & Assert
            with pytest.raises(RuntimeError, match="Training failed"):
                train_and_predict.train_model(mock_train_dataset, mock_val_dataset, config)


class TestPredictSales:
    """Test suite for predict_sales function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.get_predictions_df')
    def test_predict_sales_success(self, mock_get_predictions_df):
        """Test successful sales prediction."""
        # Arrange
        mock_model = MagicMock()
        mock_dataset = MagicMock()

        # Setup dataset mock
        mock_dataset.to_dict.return_value = {
            'series': [['data1'], ['data2']],
            'future_covariates': ['future1', 'future2'],
            'past_covariates': ['past1', 'past2'],
            'labels': ['label1', 'label2']
        }

        # Setup model prediction
        mock_predictions = [MagicMock(), MagicMock()]
        mock_model.predict.return_value = mock_predictions

        # Setup get_predictions_df
        expected_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_get_predictions_df.return_value = expected_df

        # Act
        result = train_and_predict.predict_sales(mock_model, mock_dataset)

        # Assert
        assert result.equals(expected_df)
        mock_model.predict.assert_called_once_with(
            1,
            num_samples=500,
            series=[['data1'], ['data2']],  # series as returned by mock
            future_covariates=['future1', 'future2'],
            past_covariates=['past1', 'past2']
        )

    def test_predict_sales_exception_handling(self):
        """Test predict_sales handles exceptions properly."""
        # Arrange
        mock_model = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.to_dict.return_value = {'series': [], 'future_covariates': [], 'past_covariates': [], 'labels': []}
        mock_model.predict.side_effect = RuntimeError("Prediction failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Prediction failed"):
            train_and_predict.predict_sales(mock_model, mock_dataset)


class TestGetPredictionsDF:
    """Test suite for get_predictions_df function."""

    def test_get_predictions_df_success(self):
        """Test successful conversion of predictions to DataFrame."""
        # Arrange
        mock_preds = []
        for i in range(5):  # Create 5 predictions to match 5 labels
            mock_pred = MagicMock()
            # Each prediction should be a 2D column vector (5 rows, 1 column)
            # This way hstack will create a (5, 5) matrix where each column is a prediction
            mock_pred.data_array.return_value.values = np.array([[i * 10 + j] for j in range(5)])
            mock_preds.append(mock_pred)

        series = [['series1'], ['series2'], ['series3'], ['series4'], ['series5']]
        future_covariates = [['future1'], ['future2'], ['future3'], ['future4'], ['future5']]
        # labels should match the number of predictions (5)
        labels = [('label1', 'value1'), ('label1', 'value2'), ('label2', 'value1'), ('label2', 'value2'), ('label3', 'value1')]
        index_names_mapping = {'name1': 0, 'name2': 1}
        mock_scaler = MagicMock()
        mock_scaler.inverse_transform.side_effect = lambda x: x * 2  # Simple transformation

        # Act
        result = train_and_predict.get_predictions_df(
            mock_preds, series, future_covariates, labels,
            index_names_mapping, mock_scaler
        )

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # Five rows (after transpose and reset_index)
        assert list(result.columns) == ['name1', 'name2', '0.05', '0.25', '0.5', '0.75', '0.95']

        # Verify index names mapping was used correctly
        assert 'name1' in result.columns
        assert 'name2' in result.columns

    def test_get_predictions_df_no_scaler(self):
        """Test get_predictions_df with no scaler (scaler=None)."""
        # Arrange
        mock_pred = MagicMock()
        # Each prediction should be a 2D column vector (5 rows, 1 column)
        mock_pred.data_array.return_value.values = np.array([[1], [2], [3], [4], [5]])

        series = [['series1']]
        future_covariates = [['future1']]
        # labels should match the number of rows in the data (5)
        labels = [('label1', 'value1'), ('label1', 'value2'), ('label2', 'value1'), ('label2', 'value2'), ('label3', 'value1')]
        index_names_mapping = {'name1': 0, 'name2': 1}  # Need both levels for the tuple

        # Act
        result = train_and_predict.get_predictions_df(
            [mock_pred], series, future_covariates, labels,
            index_names_mapping, scaler=None
        )

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # Five rows (after transpose and reset_index)
        assert list(result.columns) == ['name1', 'name2', '0.05', '0.25', '0.5', '0.75', '0.95']
        # Verify that predictions are processed without scaling


class TestValidationFunctions:
    """Test suite for all validation functions."""

    def test_validate_input_directory_success(self):
        """Test validate_input_directory with valid directory."""
        with patch('os.path.isdir', return_value=True):
            # Should not raise any exception
            train_and_predict.validate_input_directory("/valid/path")

    def test_validate_input_directory_failure(self):
        """Test validate_input_directory with invalid directory."""
        with patch('os.path.isdir', return_value=False):
            with pytest.raises(SystemExit):
                train_and_predict.validate_input_directory("/invalid/path")

    def test_validate_config_file_success(self):
        """Test validate_config_file with existing file."""
        with patch('os.path.exists', return_value=True):
            train_and_predict.validate_config_file("/valid/config.json")

    def test_validate_config_file_failure(self):
        """Test validate_config_file with missing file."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(SystemExit):
                train_and_predict.validate_config_file("/missing/config.json")

    def test_validate_dataset_file_success(self):
        """Test validate_dataset_file with existing file."""
        with patch('os.path.exists', return_value=True):
            train_and_predict.validate_dataset_file("/valid/dataset.dill", "train")

    def test_validate_dataset_file_failure(self):
        """Test validate_dataset_file with missing file."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(SystemExit):
                train_and_predict.validate_dataset_file("/missing/dataset.dill", "validation")

    def test_validate_dataset_objects_success(self):
        """Test validate_dataset_objects with valid objects."""
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        train_and_predict.validate_dataset_objects(train_dataset, val_dataset)

    def test_validate_dataset_objects_failure(self):
        """Test validate_dataset_objects with None objects."""
        with pytest.raises(SystemExit):
            train_and_predict.validate_dataset_objects(None, MagicMock())

        with pytest.raises(SystemExit):
            train_and_predict.validate_dataset_objects(MagicMock(), None)

    def test_validate_config_parameters_success(self):
        """Test validate_config_parameters with valid config."""
        config = {'lags': 12}
        train_and_predict.validate_config_parameters(config)

    def test_validate_config_parameters_missing_lags(self):
        """Test validate_config_parameters with missing lags."""
        config = {}
        with pytest.raises(SystemExit):
            train_and_predict.validate_config_parameters(config)

    def test_validate_config_parameters_invalid_lags(self):
        """Test validate_config_parameters with invalid lags."""
        config = {'lags': -1}
        with pytest.raises(SystemExit):
            train_and_predict.validate_config_parameters(config)

        config = {'lags': 'invalid'}
        with pytest.raises(SystemExit):
            train_and_predict.validate_config_parameters(config)

    def test_validate_prediction_window_success(self):
        """Test validate_prediction_window with valid parameters."""
        train_and_predict.validate_prediction_window(100, 20)

    def test_validate_prediction_window_failure(self):
        """Test validate_prediction_window with invalid parameters."""
        with pytest.raises(SystemExit):
            train_and_predict.validate_prediction_window(10, 20)  # L < length

    def test_validate_file_created_success(self):
        """Test validate_file_created with existing file."""
        with patch('os.path.exists', return_value=True):
            train_and_predict.validate_file_created("/existing/file.txt", "Test")

    def test_validate_file_created_failure(self):
        """Test validate_file_created with missing file."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(SystemExit):
                train_and_predict.validate_file_created("/missing/file.txt", "Test")

    def test_validate_archive_files_success(self):
        """Test validate_archive_files with all files existing."""
        files = ["/file1.txt", "/file2.txt"]
        with patch('os.path.exists', return_value=True):
            train_and_predict.validate_archive_files(files)

    def test_validate_archive_files_failure(self):
        """Test validate_archive_files with missing files."""
        files = ["/file1.txt", "/missing.txt"]
        with patch('os.path.exists', side_effect=[True, False]):
            with pytest.raises(SystemExit):
                train_and_predict.validate_archive_files(files)


class TestLoadConfiguration:
    """Test suite for load_configuration function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_input_directory')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_config_file')
    def test_load_configuration_success(self, mock_validate_config, mock_validate_input):
        """Test successful configuration loading."""
        # Arrange
        input_dir = "/test/input"
        config_data = {"model_id": "test", "lags": 12}
        mock_validate_input.return_value = input_dir

        with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
            # Act
            result = train_and_predict.load_configuration(input_dir)

            # Assert
            assert result[0] == config_data
            mock_validate_input.assert_called_once_with(input_dir)
            mock_validate_config.assert_called_once_with(os.path.join(input_dir, "config.json"))

    def test_load_configuration_json_error(self):
        """Test load_configuration with invalid JSON."""
        input_dir = "/test/input"

        with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_input_directory'):
            with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_config_file'):
                with patch('builtins.open', mock_open(read_data="invalid json")):
                    with pytest.raises(SystemExit):
                        train_and_predict.load_configuration(input_dir)

    def test_load_configuration_file_error(self):
        """Test load_configuration with file read error."""
        input_dir = "/test/input"

        with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_input_directory'):
            with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_config_file'):
                with patch('builtins.open', side_effect=OSError("File read error")):
                    with pytest.raises(SystemExit):
                        train_and_predict.load_configuration(input_dir)


class TestLoadDatasets:
    """Test suite for load_datasets function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_dataset_file')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_dataset_objects')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.PlastinkaTrainingTSDataset')
    def test_load_datasets_success(self, mock_dataset_class, mock_validate_objects, mock_validate_file):
        """Test successful dataset loading."""
        # Arrange
        input_dir = "/test/input"
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_dataset_class.from_dill.side_effect = [mock_train_dataset, mock_val_dataset]

        # Act
        train_ds, val_ds = train_and_predict.load_datasets(input_dir)

        # Assert
        assert train_ds == mock_train_dataset
        assert val_ds == mock_val_dataset

        # Verify validation calls
        assert mock_validate_file.call_count == 2
        mock_validate_objects.assert_called_once_with(mock_train_dataset, mock_val_dataset)

        # Verify dataset loading calls
        expected_train_path = os.path.join(input_dir, "train.dill")
        expected_val_path = os.path.join(input_dir, "val.dill")
        mock_dataset_class.from_dill.assert_any_call(expected_train_path)
        mock_dataset_class.from_dill.assert_any_call(expected_val_path)

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_dataset_file')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.PlastinkaTrainingTSDataset')
    def test_load_datasets_exception(self, mock_dataset_class, mock_validate_file):
        """Test load_datasets with loading exception."""
        # Arrange
        input_dir = "/test/input"
        mock_dataset_class.from_dill.side_effect = Exception("Loading failed")

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.load_datasets(input_dir)


class TestRunTraining:
    """Test suite for run_training function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.train_model')
    @patch('time.monotonic')
    def test_run_training_success(self, mock_time, mock_train_model):
        """Test successful training execution."""
        # Arrange
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {"model_id": "test"}

        mock_model = MagicMock()
        mock_metrics = {"val_loss": 0.5, "train_loss": 0.3}
        mock_train_model.return_value = (mock_model, mock_metrics)

        mock_time.side_effect = [100.0, 150.0]  # 50 seconds duration

        # Act
        model, metrics, duration = train_and_predict.run_training(mock_train_dataset, mock_val_dataset, config)

        # Assert
        assert model == mock_model
        assert metrics == mock_metrics
        assert duration == 50.0
        mock_train_model.assert_called_once_with(mock_train_dataset, mock_val_dataset, config)

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.train_model')
    def test_run_training_no_model_returned(self, mock_train_model):
        """Test run_training when no model is returned."""
        # Arrange
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {"model_id": "test"}

        mock_train_model.return_value = (None, {})

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.run_training(mock_train_dataset, mock_val_dataset, config)

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.train_model')
    def test_run_training_exception(self, mock_train_model):
        """Test run_training with training exception."""
        # Arrange
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {"model_id": "test"}

        mock_train_model.side_effect = Exception("Training failed")

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.run_training(mock_train_dataset, mock_val_dataset, config)


class TestRunPrediction:
    """Test suite for run_prediction function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_config_parameters')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_prediction_window')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.predict_sales')
    def test_run_prediction_success(self, mock_predict, mock_validate_window, mock_validate_config):
        """Test successful prediction execution."""
        # Arrange
        mock_model = MagicMock()
        mock_train_dataset = MagicMock()
        mock_train_dataset._n_time_steps = 100
        mock_predict_dataset = MagicMock()
        mock_train_dataset.setup_dataset.return_value = mock_predict_dataset

        config = {"lags": 12}
        expected_predictions = pd.DataFrame({"pred": [1, 2, 3]})
        mock_predict.return_value = expected_predictions

        # Act
        result = train_and_predict.run_prediction(mock_model, mock_train_dataset, config)

        # Assert
        assert result.equals(expected_predictions)
        mock_validate_config.assert_called_once_with(config)
        mock_validate_window.assert_called_once_with(100, 13)  # lags + 1
        mock_train_dataset.setup_dataset.assert_called_once_with(window=(87, 100))  # 100 - 13, 100
        mock_predict.assert_called_once_with(mock_model, mock_predict_dataset)

        # Verify minimum_sales_months is set
        assert mock_predict_dataset.minimum_sales_months == 1

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_config_parameters')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.predict_sales')
    def test_run_prediction_no_result(self, mock_predict, mock_validate_config):
        """Test run_prediction when no predictions returned."""
        # Arrange
        mock_model = MagicMock()
        mock_train_dataset = MagicMock()
        mock_train_dataset._n_time_steps = 100
        mock_train_dataset.setup_dataset.return_value = MagicMock()

        config = {"lags": 12}
        mock_predict.return_value = None

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.run_prediction(mock_model, mock_train_dataset, config)

    def test_run_prediction_missing_config(self):
        """Test run_prediction with missing config parameters."""
        # Arrange
        mock_model = MagicMock()
        mock_train_dataset = MagicMock()
        config = {}  # Missing 'lags'

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.run_prediction(mock_model, mock_train_dataset, config)


class TestPrepareMetrics:
    """Test suite for prepare_metrics function."""

    def test_prepare_metrics_complete(self):
        """Test prepare_metrics with complete training metrics."""
        # Arrange
        training_metrics = {
            'validation': {'val_loss': 0.5, 'val_accuracy': 0.85},
            'training': {'train_loss': 0.3, 'train_accuracy': 0.9}
        }
        duration = 123.45

        # Act
        result = train_and_predict.prepare_metrics(training_metrics, duration)

        # Assert
        assert result['val_loss'] == 0.5
        assert result['val_accuracy'] == 0.85
        assert result['train_loss'] == 0.3
        assert result['train_accuracy'] == 0.9
        assert result['training_duration_seconds'] == 123.45

    def test_prepare_metrics_partial(self):
        """Test prepare_metrics with partial metrics."""
        # Arrange
        training_metrics = {
            'validation': {'val_loss': 0.5}
        }
        duration = 60.0

        # Act
        result = train_and_predict.prepare_metrics(training_metrics, duration)

        # Assert
        assert result['val_loss'] == 0.5
        assert result['training_duration_seconds'] == 60.0
        assert 'train_loss' not in result

    def test_prepare_metrics_empty(self):
        """Test prepare_metrics with empty metrics."""
        # Arrange
        training_metrics = {}
        duration = 30.5

        # Act
        result = train_and_predict.prepare_metrics(training_metrics, duration)

        # Assert
        assert result['training_duration_seconds'] == 30.5
        assert len(result) == 1


class TestSaveModelFile:
    """Test suite for save_model_file function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_file_created')
    @patch('os.path.getsize')
    def test_save_model_file_success(self, mock_getsize, mock_validate):
        """Test successful model saving."""
        # Arrange
        mock_model = MagicMock()
        output_path = "/test/model.onnx"
        mock_getsize.return_value = 1024

        # Act
        train_and_predict.save_model_file(mock_model, output_path)

        # Assert
        mock_model.to_onnx.assert_called_once_with(output_path)
        mock_validate.assert_called_once_with(output_path, "Model")
        mock_getsize.assert_called_once_with(output_path)

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_file_created')
    def test_save_model_file_exception(self, mock_validate):
        """Test save_model_file with saving exception."""
        # Arrange
        mock_model = MagicMock()
        mock_model.to_onnx.side_effect = Exception("Save failed")
        output_path = "/test/model.onnx"

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.save_model_file(mock_model, output_path)


class TestSavePredictionsFile:
    """Test suite for save_predictions_file function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_file_created')
    def test_save_predictions_file_success(self, mock_validate):
        """Test successful predictions saving."""
        # Arrange
        predictions = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        output_path = "/test/predictions.csv"

        with patch.object(predictions, 'to_csv') as mock_to_csv:
            # Act
            train_and_predict.save_predictions_file(predictions, output_path)

            # Assert
            mock_to_csv.assert_called_once_with(output_path, index=False)
            mock_validate.assert_called_once_with(output_path, "Predictions")

    def test_save_predictions_file_exception(self):
        """Test save_predictions_file with saving exception."""
        # Arrange
        predictions = pd.DataFrame({"col1": [1, 2]})
        output_path = "/test/predictions.csv"

        with patch.object(predictions, 'to_csv', side_effect=Exception("Save failed")):
            # Act & Assert
            with pytest.raises(SystemExit):
                train_and_predict.save_predictions_file(predictions, output_path)


class TestSaveMetricsFile:
    """Test suite for save_metrics_file function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_file_created')
    def test_save_metrics_file_success(self, mock_validate):
        """Test successful metrics saving."""
        # Arrange
        metrics = {"loss": 0.5, "accuracy": 0.9}
        output_path = "/test/metrics.json"

        with patch('builtins.open', mock_open()) as mock_file:
            with patch('json.dump') as mock_json_dump:
                # Act
                train_and_predict.save_metrics_file(metrics, output_path)

                # Assert
                mock_file.assert_called_once_with(output_path, 'w')
                mock_json_dump.assert_called_once_with(metrics, mock_file.return_value.__enter__.return_value, indent=2)
                mock_validate.assert_called_once_with(output_path, "Metrics")

    def test_save_metrics_file_exception(self):
        """Test save_metrics_file with saving exception."""
        # Arrange
        metrics = {"loss": 0.5}
        output_path = "/test/metrics.json"

        with patch('builtins.open', side_effect=Exception("Save failed")):
            # Act & Assert
            with pytest.raises(SystemExit):
                train_and_predict.save_metrics_file(metrics, output_path)


class TestCreateOutputArchive:
    """Test suite for create_output_archive function."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_archive_files')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_file_created')
    @patch('os.path.getsize')
    @patch('zipfile.ZipFile')
    def test_create_output_archive_success(self, mock_zipfile, mock_getsize, mock_validate_created, mock_validate_files):
        """Test successful archive creation."""
        # Arrange
        temp_dir = "/temp/output"
        output_path = "/output/result.zip"
        mock_getsize.return_value = 2048

        mock_zip_context = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_context

        # Act
        train_and_predict.create_output_archive(temp_dir, output_path)

        # Assert
        expected_files = [
            os.path.join(temp_dir, "model.onnx"),
            os.path.join(temp_dir, "predictions.csv"),
            os.path.join(temp_dir, "metrics.json")
        ]
        mock_validate_files.assert_called_once_with(expected_files)
        mock_zipfile.assert_called_once_with(output_path, 'w', zipfile.ZIP_DEFLATED)

        # Verify all files are added to archive
        expected_write_calls = [
            call(expected_files[0], "model.onnx"),
            call(expected_files[1], "predictions.csv"),
            call(expected_files[2], "metrics.json")
        ]
        mock_zip_context.write.assert_has_calls(expected_write_calls)

        mock_validate_created.assert_called_once_with(output_path, "Output archive")
        mock_getsize.assert_called_once_with(output_path)

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.validate_archive_files')
    def test_create_output_archive_exception(self, mock_validate_files):
        """Test create_output_archive with archiving exception."""
        # Arrange
        temp_dir = "/temp/output"
        output_path = "/output/result.zip"

        with patch('zipfile.ZipFile', side_effect=Exception("Archive failed")):
            # Act & Assert
            with pytest.raises(SystemExit):
                train_and_predict.create_output_archive(temp_dir, output_path)


class TestMainFunction:
    """Test suite for main CLI function."""

    @patch('click.Path.convert')  # Mock Click's path validation
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.load_configuration')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.load_datasets')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.run_training')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.run_prediction')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.prepare_metrics')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.save_model_file')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.save_predictions_file')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.save_metrics_file')
    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.create_output_archive')
    @patch('tempfile.TemporaryDirectory')
    def test_main_success(self, mock_tempdir, mock_create_archive, mock_save_metrics,
                         mock_save_predictions, mock_save_model, mock_prepare_metrics,
                         mock_run_prediction, mock_run_training, mock_load_datasets,
                         mock_load_config, mock_path_convert):
        """Test successful main function execution."""
        # Arrange
        mock_path_convert.return_value = '/test/input'  # Mock path validation to return valid path
        mock_temp_context = MagicMock()
        mock_temp_context.__enter__.return_value = "/temp/outputs"
        mock_tempdir.return_value = mock_temp_context

        # Setup mocks
        config = {"model_id": "test", "lags": 12}
        mock_load_config.return_value = (config, '/test/input')

        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        # Configure reset_window to return a different mock for prediction
        mock_full_train_ds = MagicMock()
        mock_train_dataset.reset_window.return_value = mock_full_train_ds
        mock_load_datasets.return_value = (mock_train_dataset, mock_val_dataset)

        mock_model = MagicMock()
        training_metrics = {"val_loss": 0.5}
        duration = 60.0
        mock_run_training.return_value = (mock_model, training_metrics, duration)

        predictions = pd.DataFrame({"pred": [1, 2, 3]})
        mock_run_prediction.return_value = predictions

        final_metrics = {"val_loss": 0.5, "training_duration_seconds": 60.0}
        mock_prepare_metrics.return_value = final_metrics

        # Act
        runner = click.testing.CliRunner()
        result = runner.invoke(train_and_predict.main, ['--input', '/test/input', '--output', 'result.zip'])

        # Assert
        assert result.exit_code == 0

        # Verify function call sequence
        mock_load_config.assert_called_once_with('/test/input')
        mock_load_datasets.assert_called_once_with('/test/input')
        mock_run_training.assert_called_once_with(mock_train_dataset, mock_val_dataset, config)
        mock_run_prediction.assert_called_once_with(mock_model, mock_full_train_ds, config)
        mock_prepare_metrics.assert_called_once_with(training_metrics, duration)

        # Verify save operations
        expected_model_path = os.path.join("/temp/outputs", "model.onnx")
        expected_pred_path = os.path.join("/temp/outputs", "predictions.csv")
        expected_metrics_path = os.path.join("/temp/outputs", "metrics.json")

        mock_save_model.assert_called_once_with(mock_model, expected_model_path)
        mock_save_predictions.assert_called_once_with(predictions, expected_pred_path)
        mock_save_metrics.assert_called_once_with(final_metrics, expected_metrics_path)
        # Extract the actual arguments and check basename for cross-platform compatibility
        create_archive_call_args = mock_create_archive.call_args[0]
        assert create_archive_call_args[0] == "/temp/outputs"
        assert os.path.basename(create_archive_call_args[1]) == "result.zip"

    def test_main_default_output(self):
        """Test main function with default output parameter."""
        runner = click.testing.CliRunner()

        # Mock all the pipeline functions to avoid actual execution
        with patch('click.Path.convert') as mock_path_convert:  # Mock path validation
            mock_path_convert.return_value = '/test/input'
            with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.load_configuration') as mock_load_config:
                mock_load_config.return_value = {"model_id": "test", "lags": 12}
                with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.load_datasets') as mock_load_datasets:
                    # Fix: load_datasets should return a tuple of (train_dataset, val_dataset)
                    mock_train_dataset = MagicMock()
                    mock_val_dataset = MagicMock()
                    mock_load_datasets.return_value = (mock_train_dataset, mock_val_dataset)
                    with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.run_training') as mock_run_training:
                        # Fix: run_training should return a tuple of (model, metrics, duration)
                        mock_model = MagicMock()
                        mock_run_training.return_value = (mock_model, {"val_loss": 0.5}, 60.0)
                        with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.run_prediction') as mock_run_prediction:
                            mock_run_prediction.return_value = pd.DataFrame({"pred": [1, 2, 3]})
                            with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.prepare_metrics') as mock_prepare_metrics:
                                mock_prepare_metrics.return_value = {"val_loss": 0.5, "training_duration_seconds": 60.0}
                                with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.save_model_file'):
                                    with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.save_predictions_file'):
                                        with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.save_metrics_file'):
                                            with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.create_output_archive') as mock_archive:
                                                with patch('tempfile.TemporaryDirectory') as mock_tempdir:
                                                    mock_temp_context = MagicMock()
                                                    mock_temp_context.__enter__.return_value = "/temp/outputs"
                                                    mock_tempdir.return_value = mock_temp_context

                                                    # Act
                                                    result = runner.invoke(train_and_predict.main, ['--input', '/test/input'])

                                                    # Assert
                                                    assert result.exit_code == 0
                                                    # Verify default output is used
                                                    # Check with basename for cross-platform compatibility
                                    create_archive_call_args = mock_archive.call_args[0]
                                    assert create_archive_call_args[0] == "/temp/outputs"
                                    assert os.path.basename(create_archive_call_args[1]) == "output.zip"


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""

    @patch('plastinka_sales_predictor.datasphere_job.train_and_predict.configure_logger')
    def test_module_imports_successfully(self, mock_logger):
        """Test that the module can be imported without errors."""
        # This test verifies that all imports and module-level code work correctly
        assert hasattr(train_and_predict, 'main')
        assert hasattr(train_and_predict, 'train_model')
        assert hasattr(train_and_predict, 'predict_sales')
        assert hasattr(train_and_predict, 'DEFAULT_CONFIG_PATH')
        assert hasattr(train_and_predict, 'DEFAULT_MODEL_OUTPUT_REF')

    def test_constants_defined(self):
        """Test that all expected constants are defined."""
        assert train_and_predict.DEFAULT_CONFIG_PATH == 'config/model_config.json'
        assert train_and_predict.DEFAULT_MODEL_OUTPUT_REF == 'model.onnx'
        assert train_and_predict.DEFAULT_PREDICTION_OUTPUT_REF == 'predictions.csv'
        assert train_and_predict.DEFAULT_METRICS_OUTPUT_REF == 'metrics.json'
