import json
import os
import sys
import tempfile
import time
import zipfile

import click
import numpy as np
import pandas as pd

# plastinka_sales_predictor imports are now at the top level
from plastinka_sales_predictor import (
    configure_logger,
    extract_early_stopping_callback,
    prepare_for_training,
)
from plastinka_sales_predictor import train_model as _train_model
from plastinka_sales_predictor.data_preparation import PlastinkaTrainingTSDataset, PlastinkaInferenceTSDataset

DEFAULT_CONFIG_PATH = "config/model_config.json"
DEFAULT_MODEL_OUTPUT_REF = "model.onnx"
DEFAULT_PREDICTION_OUTPUT_REF = "predictions.csv"
DEFAULT_METRICS_OUTPUT_REF = "metrics.json"
DEFAULT_MODEL_NAME = "TiDE"


logger = configure_logger()


def train_model(
    train_dataset: PlastinkaTrainingTSDataset,
    val_dataset: PlastinkaTrainingTSDataset,
    config: dict,
) -> tuple:
    """
    Trains an TiDE model using prepared features.

    Returns:
        tuple: (model, metrics_dict) where:
            - model is the trained TiDEModel
            - metrics_dict is a dictionary with validation metrics from first training phase
              and training metrics from final training phase
    """
    logger.info("Starting model training...")
    logger.info("Initializing TiDEModel...")

    # Handle both test and production configuration formats
    if "nn_model_config" in config:
        config["model_config"] = config.pop("nn_model_config")
    # If model_config already exists, use it as is (for tests)

    # Instantiate the model
    config["model_config"]["n_epochs"] = 200  # hardcoded for now
    metrics_dict = {}

    # Train the model
    try:
        logger.info("Train first time to determine effective epochs")

        # First training run - with validation dataset
        model = _train_model(
            *prepare_for_training(config, train_dataset, val_dataset),
            model_name=f"{DEFAULT_MODEL_NAME}__n_epochs_search",
        )

        # Capture validation metrics from first training phase
        val_metrics = {}
        if hasattr(model, "trainer") and hasattr(model.trainer, "callback_metrics"):
            for k, v in model.trainer.callback_metrics.items():
                if k.startswith("val"):
                    val_metrics[k] = v.item() if hasattr(v, "item") else v

        metrics_dict.update(val_metrics)
        logger.info(
            f"Captured validation metrics from first training phase: {len(val_metrics)} metrics"
        )

        early_stopping = extract_early_stopping_callback(model.trainer)
        effective_epochs = (model.trainer.current_epoch - 1) - early_stopping.wait_count
        effective_epochs *= 1.1
        effective_epochs = max(1, int(effective_epochs))
        config["model_config"]["n_epochs"] = effective_epochs
        logger.info(f"Effective epochs: {effective_epochs}")

        if hasattr(train_dataset, "reset_window"):
            full_train_ds = train_dataset.reset_window()
        else:
            full_train_ds = train_dataset.setup_dataset(
                window=(0, train_dataset._n_time_steps)
            )

        logger.info("Train model on full dataset")
        # Second training run - with full training dataset
        model = _train_model(
            *prepare_for_training(config, full_train_ds),
            model_name=f"{DEFAULT_MODEL_NAME}__full_train",
        )

        # Capture training metrics from the second training phase
        train_metrics = {}
        if hasattr(model, "trainer") and hasattr(model.trainer, "callback_metrics"):
            for k, v in model.trainer.callback_metrics.items():
                if k.startswith("train"):
                    train_metrics[k] = v.item() if hasattr(v, "item") else v

        metrics_dict.update(train_metrics)
        logger.info(
            f"Captured training metrics from second training phase: {len(train_metrics)} metrics"
        )

        logger.info("Model trained successfully.")

    except Exception as e:
        logger.error(f"Error during model fitting: {e}", exc_info=True)
        raise

    return model, metrics_dict


def predict_sales(model, predict_dataset: PlastinkaInferenceTSDataset) -> pd.DataFrame:
    """Generates sales predictions using the trained model."""
    logger.info("Starting sales prediction...")

    predictions = None
    try:
        # Generate predictions using predict_from_dataset
        predictions, _, _ = model.predict_from_dataset(
            1,
            predict_dataset,
            num_samples=500,
            values_only=True,
            mc_dropout=True
        )

        logger.info("Prediction generation complete.")

        # Extract labels from dataset for creating the DataFrame
        labels = []
        for idx in range(len(predict_dataset)):
            array_index, _, _ = predict_dataset._project_index(idx)
            item_multiidx = predict_dataset._idx2multiidx[array_index]
            labels.append(item_multiidx)

        predictions_df = get_predictions_df(
            predictions,
            labels,
            predict_dataset._index_names_mapping,
            predict_dataset.scaler,
        )

        logger.info("Prediction dataframe successfully formed.")
        return predictions_df
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise


def get_predictions_df(
    preds,
    labels,
    index_names_mapping,
    scaler=None,
):
    def _maybe_inverse_transform(array):
        if scaler is not None:
            return scaler.inverse_transform(array)
        return array

    preds_, labels_ = [], []
    for p, label in zip(preds, labels, strict=False):
        # preds is already numpy.ndarray, no need for .data_array().values
        preds_.append(_maybe_inverse_transform(p))
        labels_.append(label)

    preds_ = np.hstack(preds_).squeeze()

    level_names = {i: n for n, i in index_names_mapping.items()}
    level_names = [level_names[i] for i in range(len(level_names))]

    preds_df = pd.DataFrame(
        preds_, index=pd.MultiIndex.from_tuples(labels, names=level_names)
    )

    # Используем строковые названия колонок для квантилей
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    preds_df = preds_df.clip(0).quantile(quantiles, axis=1).T

    # Преобразуем названия колонок квантилей в строки
    preds_df.columns = [str(col) for col in preds_df.columns]

    preds_df = preds_df.reset_index()

    return preds_df


def validate_input_directory(input_path: str) -> str:
    """
    Validate input and extract if it's a ZIP archive.

    Args:
        input_path: Path to input directory or ZIP file

    Returns:
        Path to directory containing input files

    Raises:
        SystemExit: If validation fails
    """
    # Check if input is a ZIP file
    if zipfile.is_zipfile(input_path):
        logger.info(f"Input is a ZIP file: {input_path}. Extracting...")
        temp_dir = tempfile.mkdtemp(prefix="plastinka_input_")
        try:
            with zipfile.ZipFile(input_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            logger.info(f"ZIP file extracted to: {temp_dir}")
            return temp_dir
        except Exception as e:
            logger.error(f"Failed to extract ZIP file {input_path}: {e}")
            sys.exit(1)

    # Check if it's a directory
    elif os.path.isdir(input_path):
        logger.info(f"Input is a directory: {input_path}")
        return input_path

    else:
        logger.error(
            f"Input path is neither a directory nor a valid ZIP file: {input_path}"
        )
        sys.exit(1)


def validate_config_file(config_file_path: str) -> None:
    """
    Validate that config file exists.

    Args:
        config_file_path: Path to config.json file

    Raises:
        SystemExit: If validation fails
    """
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file not found: {config_file_path}")
        sys.exit(1)


def validate_dataset_file(file_path: str, file_type: str) -> None:
    """
    Validate that dataset file exists.

    Args:
        file_path: Path to dataset file
        file_type: Type of dataset (e.g., "train", "validation")

    Raises:
        SystemExit: If validation fails
    """
    if not os.path.exists(file_path):
        logger.error(f"{file_type.capitalize()} dataset file not found: {file_path}")
        sys.exit(1)


def validate_dataset_objects(train_dataset, val_dataset) -> None:
    """
    Validate that dataset objects are not None.

    Args:
        train_dataset: Training dataset object
        val_dataset: Validation dataset object

    Raises:
        SystemExit: If validation fails
    """
    if train_dataset is None or val_dataset is None:
        logger.error("Failed to load datasets. Dataset objects are None.")
        sys.exit(1)


def validate_config_parameters(config: dict) -> None:
    """
    Validate required configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        SystemExit: If validation fails
    """
    if "lags" not in config:
        logger.error("Configuration missing required 'lags' parameter")
        sys.exit(1)

    lags_value = config["lags"]
    if not isinstance(lags_value, int) or lags_value < 1:
        logger.error(
            f"Invalid 'lags' value in configuration: {lags_value}. Must be a positive integer."
        )
        sys.exit(1)


def validate_prediction_window(L: int, length: int) -> None:
    """
    Validate prediction window parameters.

    Args:
        L: Total number of time steps in dataset
        length: Required window length

    Raises:
        SystemExit: If validation fails
    """
    if L - length < 0:
        logger.error(
            f"Invalid window: dataset has {L} time steps but needs {length} for prediction window"
        )
        sys.exit(1)


def validate_file_created(file_path: str, file_type: str) -> None:
    """
    Validate that a file was successfully created.

    Args:
        file_path: Path to the file
        file_type: Type of file for error message

    Raises:
        SystemExit: If validation fails
    """
    if not os.path.exists(file_path):
        logger.error(f"{file_type} file was not created at {file_path}")
        sys.exit(1)


def validate_archive_files(files_to_archive: list) -> None:
    """
    Validate that all files required for archiving exist.

    Args:
        files_to_archive: List of file paths to validate

    Raises:
        SystemExit: If validation fails
    """
    for file_path in files_to_archive:
        if not os.path.exists(file_path):
            logger.error(f"Required file missing for archiving: {file_path}")
            sys.exit(1)


def load_configuration(input_path: str) -> tuple:
    """
    Load and validate configuration from input directory or ZIP file.

    Args:
        input_path: Path to directory or ZIP file containing config.json

    Returns:
        Tuple of (configuration_dictionary, actual_input_directory)

    Raises:
        SystemExit: If configuration loading fails
    """
    # Validate and get actual input directory (extract ZIP if needed)
    input_dir = validate_input_directory(input_path)

    config_file_path = os.path.join(input_dir, "config.json")
    validate_config_file(config_file_path)

    try:
        with open(config_file_path) as f:
            config = json.load(f)

        logger.info(f"Configuration loaded successfully from {config_file_path}")
        return config, input_dir

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_file_path}: {e}")
        sys.exit(1)


def load_datasets(input_dir: str) -> tuple:
    """
    Load and validate training, validation, and inference datasets.

    Args:
        input_dir: Directory containing train.dill, val.dill, and inference.dill files

    Returns:
        Tuple of (train_dataset, val_dataset, inference_dataset)

    Raises:
        SystemExit: If dataset loading fails
    """
    train_dill_path = os.path.join(input_dir, "train.dill")
    val_dill_path = os.path.join(input_dir, "val.dill")
    inference_dill_path = os.path.join(input_dir, "inference.dill")

    # Validate dataset files exist
    validate_dataset_file(train_dill_path, "train")
    validate_dataset_file(val_dill_path, "validation")
    validate_dataset_file(inference_dill_path, "inference")

    try:
        logger.info(f"Loading train_dataset from {train_dill_path}...")
        train_dataset = PlastinkaTrainingTSDataset.from_dill(train_dill_path)
        logger.info("Train dataset loaded successfully.")

        logger.info(f"Loading val_dataset from {val_dill_path}...")
        val_dataset = PlastinkaTrainingTSDataset.from_dill(val_dill_path)
        logger.info("Validation dataset loaded successfully.")

        logger.info(f"Loading inference_dataset from {inference_dill_path}...")
        inference_dataset = PlastinkaInferenceTSDataset.from_dill(inference_dill_path)
        logger.info("Inference dataset loaded successfully.")

        # Validate loaded dataset objects
        if train_dataset is None or val_dataset is None or inference_dataset is None:
            logger.error("Failed to load datasets. One or more dataset objects are None.")
            sys.exit(1)

        logger.info("All datasets validated successfully.")
        return train_dataset, val_dataset, inference_dataset

    except Exception as e:
        logger.error(f"Error loading datasets: {e}", exc_info=True)
        sys.exit(1)


def run_training(train_dataset, val_dataset, config: dict) -> tuple:
    """
    Execute model training and return model with metrics.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary

    Returns:
        Tuple of (model, training_metrics, training_duration_seconds)

    Raises:
        SystemExit: If training fails
    """
    try:
        logger.info("Starting model training phase...")
        start_time = time.monotonic()
        model, training_metrics = train_model(train_dataset, val_dataset, config)
        end_time = time.monotonic()
        training_duration_seconds = end_time - start_time
        logger.info(
            f"Model training phase completed in {training_duration_seconds:.2f} seconds."
        )
        logger.info(
            f"Obtained metrics from training process: validation metrics: {len(training_metrics.get('validation', {}))} items, training metrics: {len(training_metrics.get('training', {}))} items"
        )

        if not model:
            logger.error("Model training failed to return a model object. Exiting.")
            sys.exit(1)

        return model, training_metrics, training_duration_seconds

    except Exception as e:
        logger.error(f"Pipeline failed during model training: {e}")
        sys.exit(1)


def run_prediction(model, inference_dataset, config: dict):
    """
    Execute prediction using trained model.

    Args:
        model: Trained model
        inference_dataset: Inference dataset for prediction
        config: Configuration dictionary

    Returns:
        Predictions dataframe

    Raises:
        SystemExit: If prediction fails
    """
    try:
        # Validate configuration parameters
        validate_config_parameters(config)

        predictions = predict_sales(model, inference_dataset)
        logger.info("Predictions generated successfully.")

        if predictions is None:
            logger.error("Prediction failed to return results. Exiting.")
            sys.exit(1)

        return predictions

    except KeyError as e:
        logger.error(f"Configuration missing required parameter: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed during prediction: {e}", exc_info=True)
        sys.exit(1)


def save_model_file(model, model_output: str) -> None:
    """
    Save model to ONNX format with validation.

    Args:
        model: Trained model to save
        model_output: Output file path

    Raises:
        SystemExit: If saving fails
    """
    try:
        model.to_onnx(model_output)
        logger.info(f"Model saved: {model_output}")

        # Validate model file was created
        validate_file_created(model_output, "Model")

        model_size = os.path.getsize(model_output)
        logger.info(f"Model file size: {model_size} bytes")

    except Exception as e:
        logger.error(f"Failed to save model: {e}", exc_info=True)
        sys.exit(1)


def save_predictions_file(predictions, prediction_output: str) -> None:
    """
    Save predictions to CSV format with validation.

    Args:
        predictions: Predictions dataframe
        prediction_output: Output file path

    Raises:
        SystemExit: If saving fails
    """
    try:
        predictions.to_csv(prediction_output, index=False)
        logger.info(f"Predictions saved: {prediction_output}")

        # Validate predictions file was created
        validate_file_created(prediction_output, "Predictions")

        pred_rows = len(predictions)
        logger.info(f"Predictions file contains {pred_rows} rows")

    except Exception as e:
        logger.error(f"Failed to save predictions: {e}", exc_info=True)
        sys.exit(1)


def save_metrics_file(metrics: dict, metrics_output: str) -> None:
    """
    Save metrics to JSON format with validation.

    Args:
        metrics: Metrics dictionary
        metrics_output: Output file path

    Raises:
        SystemExit: If saving fails
    """
    try:
        with open(metrics_output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved: {metrics_output}")

        # Validate metrics file was created
        validate_file_created(metrics_output, "Metrics")
        logger.info(f"Metrics saved: {len(metrics)} metrics")

    except Exception as e:
        logger.error(f"Failed to save metrics: {e}", exc_info=True)
        sys.exit(1)


def create_output_archive(temp_output_dir: str, output: str) -> None:
    """
    Create output archive with all result files.

    Args:
        temp_output_dir: Directory containing output files
        output: Output archive path

    Raises:
        SystemExit: If archive creation fails
    """
    try:
        model_output = os.path.join(temp_output_dir, DEFAULT_MODEL_OUTPUT_REF)
        prediction_output = os.path.join(temp_output_dir, DEFAULT_PREDICTION_OUTPUT_REF)
        metrics_output = os.path.join(temp_output_dir, DEFAULT_METRICS_OUTPUT_REF)

        files_to_archive = [model_output, prediction_output, metrics_output]
        archived_files = []

        # Validate all required files exist before archiving
        validate_archive_files(files_to_archive)

        logger.info(f"Creating output archive: {output}")
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from temp_output_dir to the archive
            for file_path in files_to_archive:
                archive_name = os.path.basename(file_path)
                zipf.write(file_path, archive_name)
                archived_files.append(archive_name)
                logger.info(f"Added {file_path} to archive as {archive_name}")

        # Validate archive was created
        validate_file_created(output, "Output archive")

        archive_size = os.path.getsize(output)
        logger.info(
            f"Output archive created successfully: {output} ({archive_size} bytes)"
        )
        logger.info(
            f"Archive contains {len(archived_files)} files: {', '.join(archived_files)}"
        )

        # Note: No manual cleanup needed - tempfile.TemporaryDirectory() handles it automatically
        logger.info(
            f"Temporary directory {temp_output_dir} will be cleaned up automatically"
        )

    except Exception as e:
        logger.error(f"Failed to create output archive: {e}", exc_info=True)
        sys.exit(1)


@click.command()
@click.option(
    "--input",
    "input_dir",
    required=True,
    type=click.Path(exists=True),
    help="Path to input directory containing config.json and dataset files",
)
@click.option(
    "--output", type=str, default="output.zip", help="Name of output archive file"
)
def main(input_dir, output):
    """
    Run the training and prediction pipeline.

    This script assumes that the `plastinka_sales_predictor` wheel has already
    been installed in the execution environment by the DataSphere job runner.
    The installation logic has been removed from this script and should be
    placed in the `cmd` field of the job's `config.yaml`.
    """
    logger.info("Starting datasphere job...")
    logger.info(f"Loading inputs from {input_dir}...")

    # Resolve output path to be absolute so it's created in the working directory, not temp directory
    output_path = os.path.abspath(output)
    logger.info(f"Archive will be created at: {output_path}")

    # Use temporary directory context manager for proper resource management
    with tempfile.TemporaryDirectory(
        prefix="plastinka_temp_outputs_"
    ) as temp_output_dir:
        logger.info(f"Created temporary output directory: {temp_output_dir}")
        logger.info(
            f"Output configuration: temp_dir={temp_output_dir}, archive={output}"
        )

        # Set default output paths in temp directory
        model_output = os.path.join(temp_output_dir, DEFAULT_MODEL_OUTPUT_REF)
        prediction_output = os.path.join(temp_output_dir, DEFAULT_PREDICTION_OUTPUT_REF)
        metrics_output = os.path.join(temp_output_dir, DEFAULT_METRICS_OUTPUT_REF)

        # 1. Load configuration (and get actual input directory)
        config, actual_input_dir = load_configuration(input_dir)

        # 2. Load datasets
        train_dataset, val_dataset, inference_dataset = load_datasets(actual_input_dir)

        # 3. Train model
        model, training_metrics, training_duration_seconds = run_training(
            train_dataset, val_dataset, config
        )

        # 4. Generate predictions using inference dataset
        predictions = run_prediction(model, inference_dataset, config)

        # 5. Prepare final metrics
        training_metrics["training_duration_seconds"] = training_duration_seconds

        # 6. Save all outputs
        save_model_file(model, model_output)
        save_predictions_file(predictions, prediction_output)
        save_metrics_file(training_metrics, metrics_output)

        # 7. Create output archive
        create_output_archive(temp_output_dir, output_path)

    logger.info("Train and predict pipeline finished.")


if __name__ == "__main__":
    main()
