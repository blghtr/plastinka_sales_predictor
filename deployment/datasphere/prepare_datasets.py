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
DEFAULT_STATIC_FEATURES = [
    'cover_type',
    'release_type',
    'price_category',
    'style',
    'record_year',
    'release_decade'
]
DEFAULT_PAST_COVARIATES_FNAMES = [
    'release_type',
    'cover_type',
    'style',
    'price_category'
]

logger = configure_logger(child_logger_name='train_predict_job_preparation')


def load_data(start_date=None, end_date=None, feature_types=None):
    """
    Loads features using feature_storage.load_features (factory pattern).
    
    Args:
        start_date: Optional start date for filtering data
        end_date: Optional end date for filtering data
        feature_types: Optional list of feature types to load (e.g., ['sales', 'stock'])
                      If None, all available features will be loaded
    
    Returns:
        Dictionary of loaded features
    """
    features = None

    try:
        logger.info(f"Loading features from database{' (selected types)' if feature_types else ''}...")
        features = feature_storage.load_features(
            store_type='sql',
            start_date=start_date,
            end_date=end_date,
            feature_types=feature_types
        )
        logger.info("Features loaded successfully via factory.")
    except Exception as e:
        logger.error(f"Error loading data via factory: {e}", exc_info=True)
        raise
    return features


def prepare_datasets(raw_features: dict, config: dict, output_dir: str | None = None) -> tuple[
    PlastinkaTrainingTSDataset, PlastinkaTrainingTSDataset
]:
    """
    Loads data from DB, prepares features, creates full dataset.
    Saves train and validation datasets to the specified output_dir.
    
    Args:
        raw_features: Dictionary of raw features loaded from the database.
        config: Dictionary containing training configuration.
        output_dir: Directory to save the prepared datasets.
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Determine the save directory
    save_directory = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
    os.makedirs(save_directory, exist_ok=True) # Ensure the directory exists

    logger.info("Starting feature preparation...")

    # Validate required raw features exist
    if 'sales' not in raw_features or 'stock' not in raw_features:
        raise ValueError("Missing required raw feature data.")

    features = raw_features.copy()

    try:
        # 1. Stock Features (using get_stock_features from prepare_datasets.py)
        features['stock_features'] = get_stock_features(
            features['stock'], features.get('change', pd.DataFrame())
        )

        # 2. Create Sales Pivot Table
        sales_pivot = get_monthly_sales_pivot(features['sales'])

        # 3. Initialize Transformers (as per prepare_datasets.py example)
        static_transformer = MultiColumnLabelBinarizer()
        scaler = GlobalLogMinMaxScaler()

        input_chunk_length = sales_pivot.shape[0] - 1
        output_chunk_length = 1

        logger.info('Creating full dataset...')
        dataset = PlastinkaTrainingTSDataset(
            stock_features=features['stock_features'],
            monthly_sales=sales_pivot,
            static_transformer=static_transformer,
            static_features=DEFAULT_STATIC_FEATURES,
            scaler=scaler,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            save_dir=None,
            dataset_name='full',
            past_covariates_span=14,
            past_covariates_fnames=DEFAULT_PAST_COVARIATES_FNAMES,
            minimum_sales_months=2,
        )

        lags = config.lags
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
            dataset_name='train',
            save_dir=save_directory
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
            dataset_name='val',
            save_dir=save_directory
        )
        logger.info('Datasets created.')
    except Exception as e:
        logger.error(f"Error during feature preparation: {e}", exc_info=True)
        raise

    return train_dataset, val_dataset


def get_datasets(start_date=None, end_date=None, config=None, output_dir: str | None = None, feature_types=None):
    """
    Loads data from DB, prepares features, creates and saves datasets to output_dir.
    
    Args:
        start_date: Start date for data loading.
        end_date: End date for data loading.
        config: Dictionary containing training configuration.
        output_dir: Directory to save the prepared datasets.
        feature_types: Optional list of feature types to load (e.g., ['sales', 'stock'])
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    raw_features = load_data(start_date, end_date, feature_types)
    # Pass the output_dir down to the preparation function
    return prepare_datasets(raw_features, config, output_dir)

