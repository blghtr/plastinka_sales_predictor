import logging

import pandas as pd

from deployment.app.db import feature_storage
from plastinka_sales_predictor.data_preparation import (
    GlobalLogMinMaxScaler,
    MultiColumnLabelBinarizer,
    PlastinkaTrainingTSDataset,
    get_monthly_sales_pivot,
    get_stock_features,
)

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

# Use application-wide logger configured in deployment.app.logger_config
logger = logging.getLogger(__name__)


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
        logger.info(f"Loading with parameters: start_date={start_date}, end_date={end_date}, feature_types={feature_types}")
        logger.info("About to call feature_storage.load_features...")

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


def prepare_datasets(raw_features: dict, config: dict, save_directory: str) -> tuple[
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
    logger.info("Starting feature preparation...")

    # Validate required raw features exist
    if 'sales' not in raw_features or 'stock' not in raw_features:
        raise ValueError("Missing required raw feature data.")

    try:
        features = raw_features.copy()
    except Exception as e:
        logger.error(f"Error copying raw_features: {e}", exc_info=True)
        raise

    logger.info("Raw features validation passed, proceeding with feature engineering...")

    try:
        # 1. Stock Features (using get_stock_features from prepare_datasets.py)
        features['stock_features'] = get_stock_features(
            features['stock'], features.get('change', pd.DataFrame())
        )
        logger.info("Step 1 completed: Stock features created successfully.")

        # 2. Create Sales Pivot Table
        sales_pivot = get_monthly_sales_pivot(features['sales'])
        logger.info("Step 2 completed: Sales pivot table created successfully.")

        # 3. Initialize Transformers (as per prepare_datasets.py example)
        static_transformer = MultiColumnLabelBinarizer()
        scaler = GlobalLogMinMaxScaler()

        dataset_length = sales_pivot.shape[0]
        output_chunk_length = 1
        lags = config.lags
        length = lags + 1
        train_end = max(length, dataset_length - length)

        if dataset_length <= length:
            raise ValueError(
                'Dataset length is less than the length of the lags. '
                f'Please provide more data. Dataset length: {dataset_length}, lags: {lags}'
            )

        train_dataset = PlastinkaTrainingTSDataset(
            stock_features=features['stock_features'],
            monthly_sales=sales_pivot,
            static_transformer=static_transformer,
            static_features=DEFAULT_STATIC_FEATURES,
            scaler=scaler,
            input_chunk_length=lags,
            output_chunk_length=output_chunk_length,
            start=0,
            end=train_end,
            save_dir=save_directory,
            dataset_name='train',
            past_covariates_span=lags,
            past_covariates_fnames=DEFAULT_PAST_COVARIATES_FNAMES,
            minimum_sales_months=2,
        )

        val_dataset = train_dataset.setup_dataset(
            window=(dataset_length - length, dataset_length),
        )
        val_dataset.save(
            dataset_name='val'
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
    logger.info('get_datasets function started...')

    try:
        raw_features = load_data(start_date, end_date, feature_types)

        logger.info('Raw features loaded, calling prepare_datasets...')

        # Pass the output_dir down to the preparation function
        train_dataset, val_dataset = prepare_datasets(raw_features, config, output_dir)

        logger.info('prepare_datasets completed, returning datasets from get_datasets...')

        return train_dataset, val_dataset
    except Exception as e:
        logger.error(f"Error in get_datasets: {e}", exc_info=True)
        raise



