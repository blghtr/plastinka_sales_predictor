import logging
import uuid
from plastinka_sales_predictor import (
    PlastinkaTrainingTSDataset,
    GlobalLogMinMaxScaler,
    configure_logger,
    setup_dataset,
    prepare_for_training,
    train_tide
)
from warnings import filterwarnings
import json
from copy import deepcopy
from pytorch_lightning.callbacks import (
    EarlyStopping
)
from pathlib import Path
import dill
import click
filterwarnings('ignore')

DEFAULT_CONFIG_PATH = '../configs/'
DEFAULT_DS_PATH = '../datasets/train.dill'
DEFAULT_OUTPUT_DIR = '../models/'


logger = configure_logger(
    child_logger_name='train',
)


def train_fn():
    return None


@click.command()
@click.option(
    '--config_path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    default=None,
    help='Path to the config file or directory with config files'
)
@click.option(
    '--ds_path',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    default=None,
    help='Path to the dataset file'
)
@click.option(
    '--output_dir',
    type=click.Path(path_type=Path),
    default=None,
    help='Path to the output directory'
)
def train(config_path, ds_path, output_dir):
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if ds_path is None:
        ds_path = DEFAULT_DS_PATH

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    if config_path.is_dir():
        logger.info("Config path is a directory, getting all json files")
        config_path = list(config_path.glob('*.json'))
        logger.info(f"Got {len(config_path)} config files")
    else:
        config_path = [config_path]

    for config_file in config_path:
        try:
            logger.info(f"Training model with config {config_file}")
            with open(config_file, 'r') as f:
                config = json.load(f)

            with open(ds_path, 'rb') as f:
                ds = dill.load(f)

            model_id = config.setdefault('model_id', str(uuid.uuid4()))

            model_filename = Path(output_dir, f'{model_id}.pt')
            model_filename.parent.mkdir(parents=True, exist_ok=True)
            model_filename = str(output_dir)

            ds = PlastinkaTrainingTSDataset.from_dill(ds_path)
            ds_copy = deepcopy(ds)
            L = ds_copy.L
            lags = config['lags']
            length = lags + 1

            temp_train, temp_val = deepcopy(ds_copy), deepcopy(ds_copy)
            scaler = GlobalLogMinMaxScaler()
            
            temp_train = setup_dataset(
                ds=temp_train,
                input_chunk_length=lags,
                output_chunk_length=1,
                window=(0, length),
                scaler=scaler
            )

            temp_val = setup_dataset(
                ds=temp_val,
                input_chunk_length=lags,
                output_chunk_length=1,
                window=(L - length, L),
                scaler=scaler
            )
            
            logger.info("Train first time to determine effective epochs")
            model = train_tide(
                *prepare_for_training(
                    config,
                    temp_train,
                    temp_val
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
            
            full_train_ds = deepcopy(ds_copy)
            full_train_ds = setup_dataset(
                ds=full_train_ds,
                input_chunk_length=lags,
                output_chunk_length=1,
                window=(0, L),
                scaler=scaler
            )

            logging.info("Train model")

            model = train_tide(
                *prepare_for_training(
                    config,
                    full_train_ds
                )
            )

            model.save(model_filename)
            logger.info(f"Model saved to {model_filename}")

        except Exception:
            logger.error(
                f"Error training model with config {config_file}",
                exc_info=True
            )


if __name__ == '__main__':
    train()
