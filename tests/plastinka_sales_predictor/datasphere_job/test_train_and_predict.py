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
import importlib

# Import the module under test
from plastinka_sales_predictor.datasphere_jobs.train import train_and_predict


# Pytest fixtures for mocking external dependencies
@pytest.fixture
def mock_extract_callback():
    """Mock for extract_early_stopping_callback function."""
    return MagicMock()


@pytest.fixture
def mock_prepare():
    """Mock for prepare_for_training function."""
    return MagicMock()


@pytest.fixture
def mock_train_model():
    """Mock for _train_model function."""
    return MagicMock()


@pytest.fixture
def mock_get_predictions_df():
    """Mock for get_predictions_df function."""
    return MagicMock()


@pytest.fixture
def mock_validate_config():
    """Mock for validate_config_file function."""
    return MagicMock()


@pytest.fixture
def mock_validate_input():
    """Mock for validate_input_directory function."""
    return MagicMock()


@pytest.fixture
def mock_inference_dataset_class():
    """Mock for PlastinkaInferenceTSDataset class."""
    return MagicMock()


@pytest.fixture
def mock_training_dataset_class():
    """Mock for PlastinkaTrainingTSDataset class."""
    return MagicMock()


@pytest.fixture
def mock_dataset_class():
    """Mock for dataset class."""
    return MagicMock()


@pytest.fixture
def mock_validate_file():
    """Mock for validate_dataset_file function."""
    return MagicMock()


@pytest.fixture
def mock_time():
    """Mock for time module."""
    return MagicMock()


@pytest.fixture
def mock_predict():
    """Mock for predict_sales function."""
    return MagicMock()


@pytest.fixture
def mock_getsize():
    """Mock for os.path.getsize function."""
    return MagicMock()


@pytest.fixture
def mock_validate():
    """Mock for validate_file_created function."""
    return MagicMock()


@pytest.fixture
def mock_validate_created():
    """Mock for validate_file_created function."""
    return MagicMock()


@pytest.fixture
def mock_zipfile():
    """Mock for zipfile module."""
    return MagicMock()


@pytest.fixture
def mock_validate_files():
    """Mock for validate_archive_files function."""
    return MagicMock()


@pytest.fixture
def mock_tempdir():
    """Mock for tempfile.TemporaryDirectory."""
    return MagicMock()


@pytest.fixture
def mock_create_archive():
    """Mock for create_output_archive function."""
    return MagicMock()


@pytest.fixture
def mock_save_metrics():
    """Mock for save_metrics_file function."""
    return MagicMock()


@pytest.fixture
def mock_save_predictions():
    """Mock for save_predictions_file function."""
    return MagicMock()


@pytest.fixture
def mock_save_model():
    """Mock for save_model_file function."""
    return MagicMock()


@pytest.fixture
def mock_prepare_metrics():
    """Mock for prepare_metrics function."""
    return MagicMock()


@pytest.fixture
def mock_run_prediction():
    """Mock for run_prediction function."""
    return MagicMock()


@pytest.fixture
def mock_run_training():
    """Mock for run_training function."""
    return MagicMock()


@pytest.fixture
def mock_load_datasets():
    """Mock for load_datasets function."""
    return MagicMock()


@pytest.fixture
def mock_load_config():
    """Mock for load_configuration function."""
    return MagicMock()


@pytest.fixture
def mock_path_convert():
    """Mock for path conversion functions."""
    return MagicMock()


class TestTrainModel:
    """Test suite for train_model function."""

    def test_train_model_success(
        self, monkeypatch, mock_extract_callback, mock_prepare, mock_train_model
    ):
        """Test successful model training with validation and final training phases."""
        # Arrange
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict._train_model", mock_train_model)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.prepare_for_training", mock_prepare)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.extract_early_stopping_callback", mock_extract_callback)
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {"model_config": {"n_epochs": 10}, "model_id": "test_model"}

        # Mock model objects
        mock_model_1 = MagicMock()
        mock_model_1.trainer.current_epoch = 8
        mock_model_1.trainer.callback_metrics = {
            "val_loss": MagicMock(item=lambda: 0.5),
            "val_accuracy": MagicMock(item=lambda: 0.85),
        }

        mock_model_2 = MagicMock()
        mock_model_2.trainer.callback_metrics = {
            "train_loss": MagicMock(item=lambda: 0.3),
            "train_accuracy": MagicMock(item=lambda: 0.9),
        }

        # Mock early stopping callback
        mock_callback = MagicMock()
        mock_callback.wait_count = 2
        mock_extract_callback.return_value = mock_callback

        # Setup mocks
        # Define explicit mock return values for prepare_for_training for the first call
        prepared_train_ds_mock_1 = MagicMock(name="prepared_train_ds_1")
        prepared_val_ds_mock_1 = MagicMock(name="prepared_val_ds_1")
        callbacks_mock_1 = MagicMock(name="callbacks_1")
        lr_scheduler_cls_mock_1 = MagicMock(name="lr_scheduler_cls_1")
        lr_shed_config_mock_1 = MagicMock(name="lr_shed_config_1")
        optimizer_config_mock_1 = MagicMock(name="optimizer_config_1")
        model_config_mock_1 = MagicMock(name="model_config_1")
        likelihood_mock_1 = MagicMock(name="likelihood_1")
        model_id_mock_1 = MagicMock(name="model_id_1")

        # Define explicit mock return values for prepare_for_training for the second call
        prepared_full_ds_mock_2 = MagicMock(name="prepared_full_ds_2")
        callbacks_mock_2 = MagicMock(name="callbacks_2")
        lr_scheduler_cls_mock_2 = MagicMock(name="lr_scheduler_cls_2")
        lr_shed_config_mock_2 = MagicMock(name="lr_shed_config_2")
        optimizer_config_mock_2 = MagicMock(name="optimizer_config_2")
        model_config_mock_2 = MagicMock(name="model_config_2")
        likelihood_mock_2 = MagicMock(name="likelihood_2")
        model_id_mock_2 = MagicMock(name="model_id_2")
        
        mock_prepare.side_effect = [
            (prepared_train_ds_mock_1, prepared_val_ds_mock_1, callbacks_mock_1, lr_scheduler_cls_mock_1, lr_shed_config_mock_1, optimizer_config_mock_1, model_config_mock_1, likelihood_mock_1, model_id_mock_1),
            (prepared_full_ds_mock_2, None, callbacks_mock_2, lr_scheduler_cls_mock_2, lr_shed_config_mock_2, optimizer_config_mock_2, model_config_mock_2, likelihood_mock_2, model_id_mock_2),
        ]
        mock_train_model.side_effect = [mock_model_1, mock_model_2]
        mock_train_dataset._n_time_steps = 100
        mock_full_train_ds = MagicMock(name='full_train_ds_mock')
        mock_train_dataset.setup_dataset.return_value = mock_full_train_ds
        mock_train_dataset.reset_window.return_value = mock_full_train_ds

        # Act
        model, metrics = train_and_predict.train_model(
            mock_train_dataset, mock_val_dataset, config
        )

        # Assert
        assert model == mock_model_2
        assert "val_loss" in metrics and "val_accuracy" in metrics
        assert metrics["train_loss"] == 0.3
        assert metrics["train_accuracy"] == 0.9

        # Verify function calls
        mock_prepare.assert_has_calls(
            [
                call(config, mock_train_dataset, mock_val_dataset),  # First call
                call(config, mock_full_train_ds),  # Second call
            ]
        )
        mock_train_model.assert_has_calls(
            [
                call(prepared_train_ds_mock_1, prepared_val_ds_mock_1, callbacks_mock_1, lr_scheduler_cls_mock_1, lr_shed_config_mock_1, optimizer_config_mock_1, model_config_mock_1, likelihood_mock_1, model_id_mock_1, model_name="TiDE__n_epochs_search"),  # First call
                call(prepared_full_ds_mock_2, None, callbacks_mock_2, lr_scheduler_cls_mock_2, lr_shed_config_mock_2, optimizer_config_mock_2, model_config_mock_2, likelihood_mock_2, model_id_mock_2, model_name="TiDE__full_train"),  # Second call
            ]
        )

        # Verify effective epochs calculation
        expected_effective_epochs = max(1, int((8 - 1 - 2) * 1.1))
        assert config["model_config"]["n_epochs"] == expected_effective_epochs

    def test_train_model_exception_handling(self, monkeypatch):
        """Test train_model handles exceptions properly."""
        # Arrange
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {"model_config": {"n_epochs": 10}, "model_id": "test_model"}

        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.prepare_for_training",
            MagicMock(side_effect=RuntimeError("Training failed"))
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Training failed"):
            train_and_predict.train_model(
                mock_train_dataset, mock_val_dataset, config
            )


class TestPredictSales:
    """Test suite for predict_sales function."""

    def test_predict_sales_success(self, monkeypatch, mock_get_predictions_df):
        """Test successful sales prediction."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.get_predictions_df",
            mock_get_predictions_df
        )
        mock_model = MagicMock()
        mock_dataset = MagicMock()

        # Setup dataset mock for new API
        mock_dataset.__len__.return_value = 2
        mock_dataset._project_index.side_effect = [
            (0, 10, 20),  # array_index, start_index, end_index for idx=0
            (1, 15, 25),  # array_index, start_index, end_index for idx=1
        ]
        mock_dataset._idx2multiidx = {
            0: ("label1", "value1"),
            1: ("label2", "value2"),
        }
        mock_dataset._index_names_mapping = {"name1": 0, "name2": 1}
        mock_dataset.scaler = None

        # Setup model prediction
        mock_predictions = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        mock_model.predict_from_dataset.return_value = (mock_predictions, None, None)

        # Setup get_predictions_df
        expected_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_get_predictions_df.return_value = expected_df

        # Act
        result = train_and_predict.predict_sales(mock_model, mock_dataset)

        # Assert
        assert result.equals(expected_df)
        mock_model.predict_from_dataset.assert_called_once_with(
            1,
            mock_dataset,
            num_samples=500,
            values_only=True,
            mc_dropout=True
        )

    def test_predict_sales_exception_handling(self):
        """Test predict_sales handles exceptions properly."""
        # Arrange
        mock_model = MagicMock()
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 0
        mock_model.predict_from_dataset.side_effect = RuntimeError("Prediction failed")

        # Act & Assert
        with pytest.raises(RuntimeError, match="Prediction failed"):
            train_and_predict.predict_sales(mock_model, mock_dataset)


class TestGetPredictionsDF:
    """Test suite for get_predictions_df function."""

    def test_get_predictions_df_success(self):
        """Test successful conversion of predictions to DataFrame."""
        # Arrange - predictions are now numpy arrays directly
        mock_preds = []
        for i in range(5):  # Create 5 predictions to match 5 labels
            # Each prediction should be a 2D column vector (5 rows, 1 column)
            # This way hstack will create a (5, 5) matrix where each column is a prediction
            mock_preds.append(np.array([[i * 10 + j] for j in range(5)]))

        # labels should match the number of predictions (5)
        labels = [
            ("label1", "value1"),
            ("label1", "value2"),
            ("label2", "value1"),
            ("label2", "value2"),
            ("label3", "value1"),
        ]
        index_names_mapping = {"name1": 0, "name2": 1}
        mock_scaler = MagicMock()
        mock_scaler.inverse_transform.side_effect = (
            lambda x: x * 2
        )  # Simple transformation

        # Act
        result = train_and_predict.get_predictions_df(
            mock_preds,
            labels,
            index_names_mapping,
            mock_scaler,
        )

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # Five rows (after transpose and reset_index)
        assert list(result.columns) == [
            "name1",
            "name2",
            "0.05",
            "0.25",
            "0.5",
            "0.75",
            "0.95",
        ]

        # Verify index names mapping was used correctly
        assert "name1" in result.columns
        assert "name2" in result.columns

    def test_get_predictions_df_no_scaler(self):
        """Test get_predictions_df with no scaler (scaler=None)."""
        # Arrange - predictions are now numpy arrays directly
        mock_pred = np.array([[1], [2], [3], [4], [5]])

        # labels should match the number of rows in the data (5)
        labels = [
            ("label1", "value1"),
            ("label1", "value2"),
            ("label2", "value1"),
            ("label2", "value2"),
            ("label3", "value1"),
        ]
        index_names_mapping = {"name1": 0, "name2": 1}  # Need both levels for the tuple

        # Act
        result = train_and_predict.get_predictions_df(
            [mock_pred],
            labels,
            index_names_mapping,
            scaler=None,
        )

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # Five rows (after transpose and reset_index)
        assert list(result.columns) == [
            "name1",
            "name2",
            "0.05",
            "0.25",
            "0.5",
            "0.75",
            "0.95",
        ]
        # Verify that predictions are processed without scaling


class TestValidationFunctions:
    """Test suite for all validation functions."""

    def test_validate_input_directory_success(self, monkeypatch):
        """Test validate_input_directory with valid directory."""
        monkeypatch.setattr("os.path.isdir", lambda x: True)
        # Should not raise any exception
        train_and_predict.validate_input_directory("/valid/path")

    def test_validate_input_directory_failure(self, monkeypatch):
        """Test validate_input_directory with invalid directory."""
        monkeypatch.setattr("os.path.isdir", lambda x: False)
        with pytest.raises(SystemExit):
            train_and_predict.validate_input_directory("/invalid/path")

    def test_validate_config_file_success(self, monkeypatch):
        """Test validate_config_file with existing file."""
        monkeypatch.setattr("os.path.exists", lambda x: True)
        train_and_predict.validate_config_file("/valid/config.json")

    def test_validate_config_file_failure(self, monkeypatch):
        """Test validate_config_file with missing file."""
        monkeypatch.setattr("os.path.exists", lambda x: False)
        with pytest.raises(SystemExit):
            train_and_predict.validate_config_file("/missing/config.json")

    def test_validate_dataset_file_success(self, monkeypatch):
        """Test validate_dataset_file with existing file."""
        monkeypatch.setattr("os.path.exists", lambda x: True)
        train_and_predict.validate_dataset_file("/valid/dataset.dill", "train")

    def test_validate_dataset_file_failure(self, monkeypatch):
        """Test validate_dataset_file with missing file."""
        monkeypatch.setattr("os.path.exists", lambda x: False)
        with pytest.raises(SystemExit):
            train_and_predict.validate_dataset_file(
                "/missing/dataset.dill", "validation"
            )

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
        config = {"lags": 12}
        train_and_predict.validate_config_parameters(config)

    def test_validate_config_parameters_missing_lags(self):
        """Test validate_config_parameters with missing lags."""
        config = {}
        with pytest.raises(SystemExit):
            train_and_predict.validate_config_parameters(config)

    def test_validate_config_parameters_invalid_lags(self):
        """Test validate_config_parameters with invalid lags."""
        config = {"lags": -1}
        with pytest.raises(SystemExit):
            train_and_predict.validate_config_parameters(config)

        config = {"lags": "invalid"}
        with pytest.raises(SystemExit):
            train_and_predict.validate_config_parameters(config)

    def test_validate_prediction_window_success(self):
        """Test validate_prediction_window with valid parameters."""
        train_and_predict.validate_prediction_window(100, 20)

    def test_validate_prediction_window_failure(self):
        """Test validate_prediction_window with invalid parameters."""
        with pytest.raises(SystemExit):
            train_and_predict.validate_prediction_window(10, 20)  # L < length

    def test_validate_file_created_success(self, monkeypatch):
        """Test validate_file_created with existing file."""
        monkeypatch.setattr("os.path.exists", lambda x: True)
        train_and_predict.validate_file_created("/existing/file.txt", "Test")

    def test_validate_file_created_failure(self, monkeypatch):
        """Test validate_file_created with missing file."""
        monkeypatch.setattr("os.path.exists", lambda x: False)
        with pytest.raises(SystemExit):
            train_and_predict.validate_file_created("/missing/file.txt", "Test")

    def test_validate_archive_files_success(self, monkeypatch):
        """Test validate_archive_files with all files existing."""
        files = ["/file1.txt", "/file2.txt"]
        monkeypatch.setattr("os.path.exists", lambda x: True)
        train_and_predict.validate_archive_files(files)

    def test_validate_archive_files_failure(self, monkeypatch):
        """Test validate_archive_files with missing files."""
        files = ["/file1.txt", "/missing.txt"]
        monkeypatch.setattr("os.path.exists", MagicMock(side_effect=[True, False]))
        with pytest.raises(SystemExit):
            train_and_predict.validate_archive_files(files)


class TestLoadConfiguration:
    """Test suite for load_configuration function."""

    def test_load_configuration_success(
        self, monkeypatch, mock_validate_config, mock_validate_input
    ):
        """Test successful configuration loading."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_input_directory",
            mock_validate_input,
        )
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_config_file",
            mock_validate_config,
        )
        input_dir = "/test/input"
        config_data = {"model_id": "test", "lags": 12}
        mock_validate_input.return_value = input_dir

        monkeypatch.setattr("builtins.open", mock_open(read_data=json.dumps(config_data)))
        # Act
        result = train_and_predict.load_configuration(input_dir)

        # Assert
        assert result[0] == config_data
        mock_validate_input.assert_called_once_with(input_dir)
        mock_validate_config.assert_called_once_with(
            os.path.join(input_dir, "config.json")
        )

    def test_load_configuration_json_error(self, monkeypatch):
        """Test load_configuration with invalid JSON."""
        input_dir = "/test/input"

        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_input_directory",
            MagicMock(),
        )
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_config_file",
            MagicMock(),
        )
        monkeypatch.setattr("builtins.open", mock_open(read_data="invalid json"))
        with pytest.raises(SystemExit):
            train_and_predict.load_configuration(input_dir)

    def test_load_configuration_file_error(self, monkeypatch):
        """Test load_configuration with file read error."""
        input_dir = "/test/input"

        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_input_directory",
            MagicMock(),
        )
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_config_file",
            MagicMock(),
        )
        monkeypatch.setattr("builtins.open", MagicMock(side_effect=OSError("File read error")))
        with pytest.raises(SystemExit):
            train_and_predict.load_configuration(input_dir)


class TestLoadDatasets:
    """Test suite for load_datasets function."""

    def test_load_datasets_success(
        self, monkeypatch, mock_inference_dataset_class, mock_training_dataset_class, mock_validate_file
    ):
        """Test successful dataset loading."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_dataset_file",
            mock_validate_file,
        )
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.PlastinkaTrainingTSDataset",
            mock_training_dataset_class,
        )
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.PlastinkaInferenceTSDataset",
            mock_inference_dataset_class,
        )
        input_dir = "/test/input"
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_inference_dataset = MagicMock()
        
        # Configure the class methods to return different datasets
        mock_training_dataset_class.from_dill.side_effect = [
            mock_train_dataset,
            mock_val_dataset,
        ]
        mock_inference_dataset_class.from_dill.return_value = mock_inference_dataset

        # Act
        train_ds, val_ds, inference_ds = train_and_predict.load_datasets(input_dir)

        # Assert
        assert train_ds == mock_train_dataset
        assert val_ds == mock_val_dataset
        assert inference_ds == mock_inference_dataset

        # Verify validation calls (now 3 files)
        assert mock_validate_file.call_count == 3

        # Verify dataset loading calls
        expected_train_path = os.path.join(input_dir, "train.dill")
        expected_val_path = os.path.join(input_dir, "val.dill")
        expected_inference_path = os.path.join(input_dir, "inference.dill")
        
        mock_training_dataset_class.from_dill.assert_any_call(expected_train_path)
        mock_training_dataset_class.from_dill.assert_any_call(expected_val_path)
        mock_inference_dataset_class.from_dill.assert_called_once_with(expected_inference_path)

    def test_load_datasets_exception(self, monkeypatch, mock_dataset_class, mock_validate_file):
        """Test load_datasets with loading exception."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_dataset_file",
            mock_validate_file,
        )
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.PlastinkaTrainingTSDataset",
            mock_dataset_class,
        )
        input_dir = "/test/input"
        mock_dataset_class.from_dill.side_effect = Exception("Loading failed")

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.load_datasets(input_dir)


class TestRunTraining:
    """Test suite for run_training function."""

    def test_run_training_success(self, monkeypatch, mock_time, mock_train_model):
        """Test successful training execution."""
        # Arrange
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.train_model", mock_train_model)
        monkeypatch.setattr("time.monotonic", mock_time)
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {"model_id": "test"}

        mock_model = MagicMock()
        mock_metrics = {"val_loss": 0.5, "train_loss": 0.3}
        mock_train_model.return_value = (mock_model, mock_metrics)

        mock_time.side_effect = [100.0, 150.0]  # 50 seconds duration

        # Act
        model, metrics, duration = train_and_predict.run_training(
            mock_train_dataset, mock_val_dataset, config
        )

        # Assert
        assert model == mock_model
        assert metrics == mock_metrics
        assert duration == 50.0
        mock_train_model.assert_called_once_with(
            mock_train_dataset, mock_val_dataset, config
        )

    def test_run_training_no_model_returned(self, monkeypatch, mock_train_model):
        """Test run_training when no model is returned."""
        # Arrange
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.train_model", mock_train_model)
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {"model_id": "test"}

        mock_train_model.return_value = (None, {})

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.run_training(mock_train_dataset, mock_val_dataset, config)

    def test_run_training_exception(self, monkeypatch, mock_train_model):
        """Test run_training with training exception."""
        # Arrange
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.train_model", mock_train_model)
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        config = {"model_id": "test"}

        mock_train_model.side_effect = Exception("Training failed")

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.run_training(mock_train_dataset, mock_val_dataset, config)


class TestRunPrediction:
    """Test suite for run_prediction function."""

    def test_run_prediction_success(
        self, monkeypatch, mock_predict, mock_validate_config
    ):
        """Test successful prediction execution."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_config_parameters",
            mock_validate_config,
        )
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.predict_sales", mock_predict)
        mock_model = MagicMock()
        mock_inference_dataset = MagicMock()

        config = {"lags": 12}
        expected_predictions = pd.DataFrame({"pred": [1, 2, 3]})
        mock_predict.return_value = expected_predictions

        # Act
        result = train_and_predict.run_prediction(
            mock_model, mock_inference_dataset, config
        )

        # Assert
        assert result.equals(expected_predictions)
        mock_validate_config.assert_called_once_with(config)
        mock_predict.assert_called_once_with(mock_model, mock_inference_dataset)

    def test_run_prediction_no_result(self, monkeypatch, mock_predict, mock_validate_config):
        """Test run_prediction when no predictions returned."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_config_parameters",
            mock_validate_config,
        )
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.predict_sales", mock_predict)
        mock_model = MagicMock()
        mock_inference_dataset = MagicMock()

        config = {"lags": 12}
        mock_predict.return_value = None

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.run_prediction(mock_model, mock_inference_dataset, config)

    def test_run_prediction_missing_config(self):
        """Test run_prediction with missing config parameters."""
        # Arrange
        mock_model = MagicMock()
        mock_inference_dataset = MagicMock()
        config = {}  # Missing 'lags'

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.run_prediction(mock_model, mock_inference_dataset, config)


class TestSaveModelFile:
    """Test suite for save_model_file function."""

    def test_save_model_file_success(self, monkeypatch, mock_getsize, mock_validate):
        """Test successful model saving."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_file_created",
            mock_validate,
        )
        monkeypatch.setattr("os.path.getsize", mock_getsize)
        mock_model = MagicMock()
        output_path = "/test/model.onnx"
        mock_getsize.return_value = 1024

        # Act
        train_and_predict.save_model_file(mock_model, output_path)

        # Assert
        mock_model.to_onnx.assert_called_once_with(output_path)
        mock_validate.assert_called_once_with(output_path, "Model")
        mock_getsize.assert_called_once_with(output_path)

    def test_save_model_file_exception(self, monkeypatch, mock_validate):
        """Test save_model_file with saving exception."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_file_created",
            mock_validate,
        )
        mock_model = MagicMock()
        mock_model.to_onnx.side_effect = Exception("Save failed")
        output_path = "/test/model.onnx"

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.save_model_file(mock_model, output_path)


class TestSavePredictionsFile:
    """Test suite for save_predictions_file function."""

    def test_save_predictions_file_success(self, monkeypatch, mock_validate):
        """Test successful predictions saving."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_file_created",
            mock_validate,
        )
        predictions = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        output_path = "/test/predictions.csv"

        mock_to_csv = MagicMock()
        monkeypatch.setattr(predictions, "to_csv", mock_to_csv)

        # Act
        train_and_predict.save_predictions_file(predictions, output_path)

        # Assert
        mock_to_csv.assert_called_once_with(output_path, index=False)
        mock_validate.assert_called_once_with(output_path, "Predictions")

    def test_save_predictions_file_exception(self, monkeypatch):
        """Test save_predictions_file with saving exception."""
        # Arrange
        predictions = pd.DataFrame({"col1": [1, 2]})
        output_path = "/test/predictions.csv"

        monkeypatch.setattr(predictions, "to_csv", MagicMock(side_effect=Exception("Save failed")))

        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.save_predictions_file(predictions, output_path)


class TestSaveMetricsFile:
    """Test suite for save_metrics_file function."""

    def test_save_metrics_file_success(self, monkeypatch, mock_validate):
        """Test successful metrics saving."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_file_created",
            mock_validate,
        )
        metrics = {"loss": 0.5, "accuracy": 0.9}
        output_path = "/test/metrics.json"

        mock_file = mock_open()
        monkeypatch.setattr("builtins.open", mock_file)
        mock_json_dump = MagicMock()
        monkeypatch.setattr("json.dump", mock_json_dump)

        # Act
        train_and_predict.save_metrics_file(metrics, output_path)

        # Assert
        mock_file.assert_called_once_with(output_path, "w")
        mock_json_dump.assert_called_once_with(
            metrics, mock_file.return_value.__enter__.return_value, indent=2
        )
        mock_validate.assert_called_once_with(output_path, "Metrics")

    def test_save_metrics_file_exception(self, monkeypatch):
        """Test save_metrics_file with saving exception."""
        # Arrange
        metrics = {"loss": 0.5}
        output_path = "/test/metrics.json"

        monkeypatch.setattr("builtins.open", MagicMock(side_effect=Exception("Save failed")))
        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.save_metrics_file(metrics, output_path)


class TestCreateOutputArchive:
    """Test suite for create_output_archive function."""

    def test_create_output_archive_success(
        self, monkeypatch, mock_zipfile, mock_getsize, mock_validate_created, mock_validate_files
    ):
        """Test successful archive creation."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_archive_files",
            mock_validate_files,
        )
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_file_created",
            mock_validate_created,
        )
        monkeypatch.setattr("os.path.getsize", mock_getsize)
        monkeypatch.setattr("zipfile.ZipFile", mock_zipfile)
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
            os.path.join(temp_dir, "metrics.json"),
        ]
        mock_validate_files.assert_called_once_with(expected_files)
        mock_zipfile.assert_called_once_with(output_path, "w", zipfile.ZIP_DEFLATED)

        # Verify all files are added to archive
        expected_write_calls = [
            call(expected_files[0], "model.onnx"),
            call(expected_files[1], "predictions.csv"),
            call(expected_files[2], "metrics.json"),
        ]
        mock_zip_context.write.assert_has_calls(expected_write_calls)

        mock_validate_created.assert_called_once_with(output_path, "Output archive")
        mock_getsize.assert_called_once_with(output_path)

    def test_create_output_archive_exception(self, monkeypatch, mock_validate_files):
        """Test create_output_archive with archiving exception."""
        # Arrange
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.validate_archive_files",
            mock_validate_files,
        )
        temp_dir = "/temp/output"
        output_path = "/output/result.zip"

        monkeypatch.setattr("zipfile.ZipFile", MagicMock(side_effect=Exception("Archive failed")))
        # Act & Assert
        with pytest.raises(SystemExit):
            train_and_predict.create_output_archive(temp_dir, output_path)


class TestMainFunction:
    """Test suite for main CLI function."""

    def test_main_success(
        self,
        monkeypatch,
        mock_tempdir,
        mock_create_archive,
        mock_save_metrics,
        mock_save_predictions,
        mock_save_model,
        mock_run_prediction,
        mock_run_training,
        mock_load_datasets,
        mock_load_config,
        mock_path_convert,
    ):
        """Test successful main function execution."""
        # Arrange
        monkeypatch.setattr("click.Path.convert", mock_path_convert)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.load_configuration", mock_load_config)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.load_datasets", mock_load_datasets)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.run_training", mock_run_training)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.run_prediction", mock_run_prediction)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.save_model_file", mock_save_model)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.save_predictions_file", mock_save_predictions)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.save_metrics_file", mock_save_metrics)
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.create_output_archive", mock_create_archive)
        monkeypatch.setattr("tempfile.TemporaryDirectory", mock_tempdir)
        mock_path_convert.return_value = (
            "/test/input"  # Mock path validation to return valid path
        )
        mock_temp_context = MagicMock()
        mock_temp_context.__enter__.return_value = "/temp/outputs"
        mock_tempdir.return_value = mock_temp_context

        # Setup mocks
        config = {"model_id": "test", "lags": 12}
        mock_load_config.return_value = (config, "/test/input")

        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_inference_dataset = MagicMock()
        mock_load_datasets.return_value = (mock_train_dataset, mock_val_dataset, mock_inference_dataset)

        mock_model = MagicMock()
        training_metrics = {"val_loss": 0.5, "train_loss": 0.3} # Include train_loss for consistency
        duration = 60.0
        mock_run_training.return_value = (mock_model, training_metrics, duration)

        predictions = pd.DataFrame({"pred": [1, 2, 3]})
        mock_run_prediction.return_value = predictions

        final_metrics = {"val_loss": 0.5, "train_loss": 0.3, "training_duration_seconds": 60.0} # Ensure consistency here too
        mock_save_metrics.return_value = None # Explicitly set return value for clarity

        # Act
        runner = click.testing.CliRunner()
        result = runner.invoke(
            train_and_predict.main, ["--input", "/test/input", "--output", "result.zip"]
        )

        # Assert
        assert result.exit_code == 0

        # Verify function call sequence
        mock_load_config.assert_called_once_with("/test/input")
        mock_load_datasets.assert_called_once_with("/test/input")
        mock_run_training.assert_called_once_with(
            mock_train_dataset, mock_val_dataset, config
        )
        mock_run_prediction.assert_called_once_with(
            mock_model, mock_inference_dataset, config
        )

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

    def test_main_default_output(self, monkeypatch):
        """Test main function with default output parameter."""
        runner = click.testing.CliRunner()

        # Mock all the pipeline functions to avoid actual execution
        mock_path_convert = MagicMock()
        monkeypatch.setattr("click.Path.convert", mock_path_convert)  # Mock path validation
        mock_path_convert.return_value = "/test/input"

        mock_load_config = MagicMock()
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.load_configuration", mock_load_config)
        mock_load_config.return_value = {"model_id": "test", "lags": 12}

        mock_load_datasets = MagicMock()
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.load_datasets", mock_load_datasets)
        # Fix: load_datasets should return a tuple of (train_dataset, val_dataset, inference_dataset)
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()
        mock_inference_dataset = MagicMock()
        mock_load_datasets.return_value = (
            mock_train_dataset,
            mock_val_dataset,
            mock_inference_dataset,
        )

        mock_run_training = MagicMock()
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.run_training", mock_run_training)
        # Fix: run_training should return a tuple of (model, metrics, duration)
        mock_model = MagicMock()
        mock_run_training.return_value = (
            mock_model,
            {"val_loss": 0.5, "train_loss": 0.3}, # Include train_loss for consistency
            60.0,
        )

        mock_run_prediction = MagicMock()
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.run_prediction", mock_run_prediction)
        mock_run_prediction.return_value = pd.DataFrame({"pred": [1, 2, 3]})

        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.save_model_file", MagicMock())
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.save_predictions_file", MagicMock())
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.save_metrics_file", MagicMock())

        mock_archive = MagicMock()
        monkeypatch.setattr("plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.create_output_archive", mock_archive)

        mock_tempdir = MagicMock()
        monkeypatch.setattr("tempfile.TemporaryDirectory", mock_tempdir)

        mock_temp_context = MagicMock()
        mock_temp_context.__enter__.return_value = "/temp/outputs"
        mock_tempdir.return_value = (
            mock_temp_context
        )

        # Act
        result = runner.invoke(
            train_and_predict.main,
            ["--input", "/test/input"],
        )

        # Assert
        assert result.exit_code == 0
        # Verify default output is used
        # Check with basename for cross-platform compatibility
        create_archive_call_args = mock_archive.call_args[0]
        assert (
            os.path.basename(create_archive_call_args[1]) == "output.zip"
        )
        assert create_archive_call_args[0] == "/temp/outputs"
        assert os.path.basename(create_archive_call_args[1]) == "output.zip" # Corrected this line


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_module_imports_successfully(self, monkeypatch):
        """Test that the module can be imported without errors."""
        monkeypatch.setattr(
            "plastinka_sales_predictor.datasphere_jobs.train.train_and_predict.configure_logger",
            MagicMock()
        )
        # This test verifies that all imports and module-level code work correctly
        assert hasattr(train_and_predict, "main")
        assert hasattr(train_and_predict, "train_model")
        assert hasattr(train_and_predict, "predict_sales")
        assert hasattr(train_and_predict, "DEFAULT_CONFIG_PATH")
        assert hasattr(train_and_predict, "DEFAULT_MODEL_OUTPUT_REF")

    def test_constants_defined(self):
        """Test that all expected constants are defined."""
        assert train_and_predict.DEFAULT_CONFIG_PATH == "config/model_config.json"
        assert train_and_predict.DEFAULT_MODEL_OUTPUT_REF == "model.onnx"
        assert train_and_predict.DEFAULT_PREDICTION_OUTPUT_REF == "predictions.csv"
        assert train_and_predict.DEFAULT_METRICS_OUTPUT_REF == "metrics.json"
