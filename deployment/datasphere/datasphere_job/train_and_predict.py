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
    train_model as _train_model
)
from plastinka_sales_predictor.data_preparation import PlastinkaTrainingTSDataset
from darts import TimeSeries
from darts.models import TiDEModel
from pytorch_lightning.callbacks import EarlyStopping
from deployment.datasphere.prepare_datasets import get_datasets

DEFAULT_OUTPUT_DIR = 'datasets/'
DEFAULT_CONFIG_PATH = 'config/model_config.json'
DEFAULT_MODEL_OUTPUT_REF = 'model.onnx'
DEFAULT_PREDICTION_OUTPUT_REF = 'predictions.csv'


logger = configure_logger(child_logger_name='train_predict_job_preparation')


def train_model(
        train_dataset: PlastinkaTrainingTSDataset, 
        val_dataset: PlastinkaTrainingTSDataset, 
        config: dict
) -> TiDEModel:
    """Trains an TiDE model using prepared features."""
    logger.info("Starting model training...")
    logger.info("Initializing TiDEModel...")

    # Instantiate the model
    config['model_config']['n_epochs'] = 200
    model_id = config['model_id']

    # Train the model
    try:
        logger.info("Train first time to determine effective epochs")
        config['model_id'] = model_id + '_n_epochs_search'
        model = _train_model(
            *prepare_for_training(
                config,
                train_dataset,
                val_dataset
            )
        )
        
        for callback in model.trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                early_stopping = callback
                break

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

        logger.info("Train model")
        config['model_id'] = model_id
        model = _train_model(
            *prepare_for_training(
                config,
                full_train_ds
            )
        )
        logger.info("Model trained successfully.")

    except Exception as e:
        logger.error(f"Error during model fitting: {e}", exc_info=True)
        raise

    return model

def predict_sales(model, predict_dataset: PlastinkaTrainingTSDataset) -> TimeSeries:
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
    
    preds_, targets_, labels_, fcovs_, past_sales_ = [], [], [], [], []
    for p, s, t, fc, l in zip(preds, series, targets, future_covariates, labels):
        preds_.append(_maybe_inverse_transform(p.data_array().values))
        past_sales_.append(_maybe_inverse_transform(s.values()))
        targets_.append(_maybe_inverse_transform(t.values()))
        labels_.append(l)
        fcovs_.append(fc.values()[0, :2])
        
    preds_ = np.hstack(preds_).squeeze()
    past_sales_ = np.hstack(past_sales_).T
    targets_ = np.hstack(targets_).squeeze()
    fcovs_ = np.vstack(fcovs_)

    level_names = {i: n for n, i in index_names_mapping.items()}
    level_names = [level_names[i] for i in range(len(level_names))]

    preds_df = pd.DataFrame(
        preds_, 
        index=pd.MultiIndex.from_tuples(
            labels,
            names=level_names
        )
    )

    preds_df = preds_df.clip(0).quantile((0.05, 0.25, 0.5, 0.75, 0.95), axis=1).T
    preds_df = preds_df.reset_index()

    return preds_df


@click.command()
def main():
    logger.info("Starting datasphere job...")
    logger.info('Loading inputs...')
    with open(DEFAULT_CONFIG_PATH, 'r') as f:
        model_config = json.load(f)

    datasets = {}
    for dataset_name in ['train', 'val']:
        dataset_path = os.path.join(DEFAULT_OUTPUT_DIR, f'{dataset_name}.dill')
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset {dataset_name} not found. Exiting pipeline.")
            sys.exit(1)
        with open(dataset_path, 'rb') as f:
            datasets[dataset_name] = PlastinkaTrainingTSDataset.from_dill(dataset_path)

    train_dataset, val_dataset = datasets['train'], datasets['val']
    if not train_dataset or not val_dataset:
        logger.error("Feature preparation returned empty results. Exiting pipeline.")
        sys.exit(1)

    # 3. Train model
    try:
        model = train_model(train_dataset, val_dataset, model_config)
    except Exception as e:
        logger.error(f"Pipeline failed during model training: {e}")
        sys.exit(1)

    if not model:
        logger.error("Model training failed to return a model object. Exiting.")
        sys.exit(1)

    # 4. Predict using the trained model
    try:
        L = train_dataset._n_time_steps
        length = model_config['lags'] + 1
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

    # TODO: 5. calculate metrics

    # 6. Save model
    model.to_onnx(DEFAULT_MODEL_OUTPUT_REF)
    logger.info(f"Model saved: {DEFAULT_MODEL_OUTPUT_REF}")

    # 7. Save predictions
    predictions.to_csv(DEFAULT_PREDICTION_OUTPUT_REF, index=False)
    logger.info(f"Predictions saved: {DEFAULT_PREDICTION_OUTPUT_REF}")

    # TODO: 8. Save metrics (placeholder)
    
    logger.info("Train and predict pipeline finished.")

if __name__ == '__main__':
    main() 