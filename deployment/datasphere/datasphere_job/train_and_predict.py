import json
import os
import sys
from datetime import datetime
import click
import numpy as np
import pandas as pd
from plastinka_sales_predictor import (
    configure_logger,
    prepare_for_training,
    train_model as _train_model,
    extract_early_stopping_callback
)
from plastinka_sales_predictor.data_preparation import PlastinkaTrainingTSDataset
from ..prepare_datasets import get_datasets
import time

DEFAULT_OUTPUT_DIR = 'datasets/'
DEFAULT_CONFIG_PATH = 'config/model_config.json'
DEFAULT_MODEL_OUTPUT_REF = 'model.onnx'
DEFAULT_PREDICTION_OUTPUT_REF = 'predictions.csv'
DEFAULT_METRICS_OUTPUT_REF = 'metrics.json'


logger = configure_logger(child_logger_name='train_predict_job_preparation')


# Custom Click parameter type for dictionary input
class DictParamType(click.ParamType):
    name = "dictionary"

    def convert(self, value, param, ctx):
        if value is None:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            self.fail(f"'{value}' is not a valid JSON dictionary", param, ctx)


# Function to get output paths with defaults
def get_output_paths(output_dict):
    output_dir = output_dict.get("dir", DEFAULT_OUTPUT_DIR)
    model_output = output_dict.get("model", DEFAULT_MODEL_OUTPUT_REF)
    prediction_output = output_dict.get("prediction", DEFAULT_PREDICTION_OUTPUT_REF)
    metrics_output = output_dict.get("metrics", DEFAULT_METRICS_OUTPUT_REF)
    
    # Ensure output_dir exists
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir, model_output, prediction_output, metrics_output


def train_model(
        train_dataset: PlastinkaTrainingTSDataset, 
        val_dataset: PlastinkaTrainingTSDataset, 
        config: dict
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

    # Instantiate the model
    config['model_config']['n_epochs'] = 200
    model_id = config['model_id']

    metrics_dict = {}

    # Train the model
    try:
        logger.info("Train first time to determine effective epochs")
        config['model_id'] = model_id + '_n_epochs_search'
        
        # First training run - with validation dataset
        model = _train_model(
            *prepare_for_training(
                config,
                train_dataset,
                val_dataset
            )
        )
        
        # Capture validation metrics from first training phase
        val_metrics = {}
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'callback_metrics'):
            for k, v in model.trainer.callback_metrics.items():
                if k.startswith('val'):
                    val_metrics[k] = v.item() if hasattr(v, 'item') else v
        
        metrics_dict['validation'] = val_metrics
        logger.info(f"Captured validation metrics from first training phase: {len(val_metrics)} metrics")
        
        early_stopping = extract_early_stopping_callback(model.trainer)
        effective_epochs = (
            model.trainer.current_epoch - 1
        ) - early_stopping.wait_count
        effective_epochs *= 1.1
        effective_epochs = max(1, int(effective_epochs))
        config['model_config']['n_epochs'] = effective_epochs
        logger.info(f"Effective epochs: {effective_epochs}")

        full_train_ds = train_dataset.setup_dataset(
            window=(0, train_dataset._n_time_steps),
        )

        logger.info("Train model on full dataset")
        config['model_id'] = model_id
        
        # Second training run - with full training dataset
        model = _train_model(
            *prepare_for_training(
                config,
                full_train_ds
            )
        )
        
        # Capture training metrics from the second training phase
        train_metrics = {}
        if hasattr(model, 'trainer') and hasattr(model.trainer, 'callback_metrics'):
            for k, v in model.trainer.callback_metrics.items():
                if k.startswith('train'):
                    train_metrics[k] = v.item() if hasattr(v, 'item') else v
        
        metrics_dict['training'] = train_metrics
        logger.info(f"Captured training metrics from second training phase: {len(train_metrics)} metrics")
        
        logger.info("Model trained successfully.")

    except Exception as e:
        logger.error(f"Error during model fitting: {e}", exc_info=True)
        raise

    return model, metrics_dict

def predict_sales(model, predict_dataset: PlastinkaTrainingTSDataset) -> pd.DataFrame:
    """Generates sales predictions using the trained model."""
    logger.info("Starting sales prediction...")
    
    # Extract inputs from dataset object
    data_dict = predict_dataset.to_dict()
    series = data_dict['series']
    targets = [s[-1:] for s in series]
    series = [s[:-1] for s in series]
    future_covariates = data_dict['future_covariates']
    past_covariates = data_dict['past_covariates']
    labels = data_dict['labels']

    predictions = None
    try:
        # Generate predictions
        predictions = model.predict(
            1, 
            num_samples=500, 
            series=series, 
            future_covariates=future_covariates, 
            past_covariates=past_covariates
        )

        logger.info("Prediction generation complete.")
    
        predictions_df = get_predictions_df(
            predictions,
            series,
            targets,
            future_covariates,
            labels,
        )

        logger.info("Prediction dataframe successfully formed.")
        return predictions_df
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise


def get_predictions_df(
        preds,
        series,
        targets, 
        future_covariates, 
        labels,
        index_names_mapping,
        scaler=None, 
):
    def _maybe_inverse_transform(array):
        if scaler is not None:
            return scaler.inverse_transform(array)
        return array
    
    preds_, labels_ = [], []
    for p, _, _, _, l in zip(preds, series, targets, future_covariates, labels):
        preds_.append(_maybe_inverse_transform(p.data_array().values))
        labels_.append(l)
        
    preds_ = np.hstack(preds_).squeeze()

    level_names = {i: n for n, i in index_names_mapping.items()}
    level_names = [level_names[i] for i in range(len(level_names))]

    preds_df = pd.DataFrame(
        preds_, 
        index=pd.MultiIndex.from_tuples(
            labels,
            names=level_names
        )
    )

    # Используем строковые названия колонок для квантилей
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
    preds_df = preds_df.clip(0).quantile(quantiles, axis=1).T
    
    # Преобразуем названия колонок квантилей в строки
    preds_df.columns = [str(col) for col in preds_df.columns]
    
    preds_df = preds_df.reset_index()

    return preds_df


@click.command()
@click.option('--input', required=True, type=click.Path(exists=True), help='Path to input configuration JSON file')
@click.option('--output', type=DictParamType(), default='{}', help='JSON dictionary of output paths (keys: dir, model, prediction, metrics)')
def main(input, output):
    """Run the training and prediction pipeline with the specified configuration."""
    logger.info("Starting datasphere job...")
    logger.info(f'Loading inputs from {input}...')
    
    # Parse output dictionary
    output_dict = output if isinstance(output, dict) else {}
    output_dir, model_output, prediction_output, metrics_output = get_output_paths(
        output_dict
    )
    
    logger.info(f"Output configuration: dir={output_dir}, model={model_output}, prediction={prediction_output}, metrics={metrics_output}")
    
    # Load configuration from the specified input file
    with open(input, 'r') as f:
        input_dict = json.load(f)
        config = input_dict['config']
        start_date = datetime.strptime(input_dict["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(input_dict["end_date"], "%Y-%m-%d")

    train_dataset, val_dataset = get_datasets(
        start_date=start_date, 
        end_date=end_date,
        config=config
    )

    if not train_dataset or not val_dataset:
        logger.error("Feature preparation returned empty results. Exiting pipeline.")
        sys.exit(1)

    # 3. Train model
    training_duration_seconds = 0
    try:
        logger.info("Starting model training phase...")
        start_time = time.monotonic()
        model, training_metrics = train_model(train_dataset, val_dataset, config)
        end_time = time.monotonic()
        training_duration_seconds = end_time - start_time
        logger.info(f"Model training phase completed in {training_duration_seconds:.2f} seconds.")
        logger.info(f"Obtained metrics from training process: validation metrics: {len(training_metrics.get('validation', {}))} items, training metrics: {len(training_metrics.get('training', {}))} items")
    
    except Exception as e:
        logger.error(f"Pipeline failed during model training: {e}")
        sys.exit(1)

    if not model:
        logger.error("Model training failed to return a model object. Exiting.")
        sys.exit(1)

    # 4. Predict using the trained model
    try:
        L = train_dataset._n_time_steps
        length = config['lags'] + 1
        predict_dataset = train_dataset.setup_dataset(
            window=(L - length, L),
        )
        predict_dataset.minimum_sales_months = 1

        predictions = predict_sales(model, predict_dataset)
    except Exception as e:
        logger.error(f"Pipeline failed during prediction: {e}")
        sys.exit(1)

    if predictions is None:
        logger.error("Prediction failed to return results. Exiting.")
        sys.exit(1)

    # Extract metrics, adding training duration and those from training phases
    metrics = {}
    
    # Add metrics from both training phases
    if 'validation' in training_metrics:
        metrics.update(training_metrics['validation'])
        logger.info(f"Added {len(training_metrics['validation'])} validation metrics from first training phase")
    
    if 'training' in training_metrics:
        metrics.update(training_metrics['training'])
        logger.info(f"Added {len(training_metrics['training'])} training metrics from second training phase")
    
    # Add training duration to metrics
    metrics["training_duration_seconds"] = round(training_duration_seconds, 2)

    # 6. Save model
    model_output_path = os.path.join(output_dir, model_output) if not os.path.isabs(model_output) else model_output
    model.to_onnx(model_output_path)
    logger.info(f"Model saved: {model_output_path}")

    # 7. Save predictions
    prediction_output_path = os.path.join(output_dir, prediction_output) if not os.path.isabs(prediction_output) else prediction_output
    predictions.to_csv(prediction_output_path, index=False)
    logger.info(f"Predictions saved: {prediction_output_path}")

    # 8. Save metrics
    metrics_output_path = os.path.join(output_dir, metrics_output) if not os.path.isabs(metrics_output) else metrics_output
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f)
    logger.info(f"Metrics saved: {metrics_output_path}")
    
    logger.info("Train and predict pipeline finished.")

if __name__ == '__main__':
    main() 