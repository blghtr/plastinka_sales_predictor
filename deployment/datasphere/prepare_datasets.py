import sys
import pandas as pd
from datetime import timedelta
from plastinka_sales_predictor import configure_logger
from deployment.app.db import feature_storage
from plastinka_sales_predictor.data_preparation import (
    PlastinkaTrainingTSDataset,
    MultiColumnLabelBinarizer,
    GlobalLogMinMaxScaler,
    get_stock_features,
    get_monthly_sales_pivot,
)
import os


DEFAULT_OUTPUT_DIR = './datasets/' # Default location if not specified
DEFAULT_CUTOFF_DATE = '30-09-2022'
DEFAULT_STATIC_FEATURES = [
    'Конверт',
    'Тип',
    'Ценовая категория',
    'Стиль',
    'Год записи',
    'Год выпуска'
]
DEFAULT_PAST_COVARIATES_FNAMES = [
    'Тип',
    'Конверт',
    'Стиль',
    'Ценовая категория'
]

logger = configure_logger(child_logger_name='train_predict_job_preparation')


def load_data(start_date=None, end_date=None):
    """Loads features using feature_storage.load_features (factory pattern)."""
    features = None
    try:
        logger.info("Loading features from database...")
        features = feature_storage.load_features(
            store_type='sql',
            start_date=start_date,
            end_date=end_date,
        )
        logger.info("Features loaded successfully via factory.")
    except Exception as e:
        logger.error(f"Error loading data via factory: {e}", exc_info=True)
        raise
    return features


def prepare_datasets(raw_features: dict, end_date: str, config: dict, output_dir: str | None = None) -> tuple[
    PlastinkaTrainingTSDataset, PlastinkaTrainingTSDataset
]:
    """
    Loads data from DB, prepares features, creates full dataset.
    Saves train and validation datasets to the specified output_dir.
    """
    # Determine the save directory
    save_directory = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    os.makedirs(save_directory, exist_ok=True) # Ensure the directory exists

    end_date_str = end_date
    logger.info("Starting feature preparation...")

    # Validate required raw features exist
    if 'sales' not in raw_features or 'stock' not in raw_features:
        raise ValueError("Missing required raw feature data.")

    features = raw_features.copy()

    try:
        # 1. Stock Features (using get_stock_features from prepare_datasets.py)
        features['stock_features'] = get_stock_features(
            features['stock'], features['change'] 
        )

        # 2. Determine Cutoff Date if not provided
        if end_date is None:
            latest_date = features['sales'].index.get_level_values('_date').max()
            if (latest_date + timedelta(days=1)).month == latest_date.month:
                end_date_str = latest_date.replace(day=1)
            else:
                end_date_str = latest_date + timedelta(days=1)
            
            end_date = end_date_str.strftime('%d-%m-%Y')
            logger.info(f"Determined cutoff date: {end_date}")
        end_date = pd.to_datetime(end_date_str, dayfirst=True)

        # 3. Filter data based on cutoff date
        rounded_sales = features['sales'][
            features['sales'].index.get_level_values('_date') < end_date
        ]
        rounded_stocks = features['stock_features'][
            features['stock_features'].index < end_date
        ]
        # 4. Create Sales Pivot Table
        sales_pivot = get_monthly_sales_pivot(rounded_sales)

        # 5. Initialize Transformers (as per prepare_datasets.py example)
        static_transformer = MultiColumnLabelBinarizer()
        scaler = GlobalLogMinMaxScaler()

        input_chunk_length = sales_pivot.shape[0] - 1
        output_chunk_length = 1

        logger.info('Creating full dataset...')
        dataset = PlastinkaTrainingTSDataset(
            stock_features=rounded_stocks,
            monthly_sales=sales_pivot,
            static_transformer=None,
            static_features=DEFAULT_STATIC_FEATURES,
            scaler=None,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            save_dir=None,
            dataset_name='full',
            past_covariates_span=14,
            past_covariates_fnames=DEFAULT_PAST_COVARIATES_FNAMES,
            minimum_sales_months=2
        )

        lags = config['lags']
        length = lags + 1
        train_end = max(length, dataset.L - length)
        logger.info('Creating train dataset...')
        train_dataset = dataset.setup_dataset(
            input_chunk_length=lags,
            output_chunk_length=1,
            window=(0, train_end),
            scaler=scaler,
            transformer=static_transformer
        )
        train_dataset.save(
            save_dir=save_directory,
            dataset_name='train'
        )

        logger.info('Creating val dataset...')
        val_dataset = dataset.setup_dataset(
            input_chunk_length=lags,
            output_chunk_length=1,
            window=(dataset.L - length, dataset.L),
            scaler=scaler,
            transformer=static_transformer
        )
        val_dataset.save(
            save_dir=save_directory,
            dataset_name='val'
        )
        

        logger.info('Datasets created.')
    except Exception as e:
        logger.error(f"Error during feature preparation: {e}", exc_info=True)
        raise

    return train_dataset, val_dataset


def get_datasets(start_date=None, end_date=None, config=None, output_dir: str | None = None):
    """
    Loads data from DB, prepares features, creates and saves datasets to output_dir.
    """
    raw_features = load_data(start_date, end_date)
    # Pass the output_dir down to the preparation function
    return prepare_datasets(raw_features, end_date, config, output_dir)

