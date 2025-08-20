import logging
from collections.abc import Sequence
from warnings import warn

import pandas as pd

from deployment.app.config import get_settings
from deployment.app.db import feature_storage
from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.db.schema import MULTIINDEX_NAMES
from plastinka_sales_predictor.data_preparation import (
    GlobalLogMinMaxScaler,
    MultiColumnLabelBinarizer,
    PlastinkaInferenceTSDataset,
    PlastinkaTrainingTSDataset,
    get_monthly_sales_pivot,
)

DEFAULT_OUTPUT_DIR = "./datasets/"  # Default location if not specified
DEFAULT_STATIC_FEATURES = [
    "cover_type",
    "release_type",
    "price_category",
    "style",
    "record_year",
    "release_decade",
]
DEFAULT_PAST_COVARIATES_FNAMES = [
    "release_type",
    "cover_type",
    "style",
    "price_category",
]

# Use application-wide logger configured in deployment.app.logger_config
logger = logging.getLogger(__name__)


def load_data(start_date=None, end_date=None, feature_types=None, dal=None):
    """
    Loads features using feature_storage.load_features (factory pattern).

    Args:
        start_date: Optional start date for filtering data
        end_date: Optional end date for filtering data
        feature_types: Optional list of feature types to load (e.g., ['sales', 'stock'])
                      If None, all available features will be loaded
        connection: Database connection object

    Returns:
        Dictionary of loaded features
    """
    features = None

    try:
        # Ensure we're loading all required feature types
        required_types = ["sales", "stock", "movement", "prices"]
        if feature_types is None:
            feature_types = required_types
        else:
            # Make sure required types are included
            for req_type in required_types:
                if req_type not in feature_types:
                    feature_types.append(req_type)

        logger.info(
            f"Loading features from database with types: {feature_types}..."
        )
        logger.info(
            f"Loading with parameters: start_date={start_date}, end_date={end_date}"
        )
        logger.info("About to call feature_storage.load_features...")

        features = feature_storage.load_features(
            store_type="sql",
            start_date=start_date,
            end_date=end_date,
            feature_types=feature_types,
            dal=dal,
        )
        logger.info("Features loaded successfully via factory.")

    except Exception as e:
        logger.error(f"Error loading data via factory: {e}", exc_info=True)
        raise

    return features


def prepare_datasets(
    raw_features: dict,
    stock_features: pd.DataFrame,
    save_directory: str,
    datasets_to_generate: Sequence[str] = ("train", "inference"),
) -> tuple[PlastinkaTrainingTSDataset | None, PlastinkaInferenceTSDataset | None]:
    """
    Loads data from DB, prepares features, creates full dataset.
    Saves train and validation datasets to the specified output_dir.

    Args:
        raw_features: Dictionary of raw features loaded from the database.
        stock_features: DataFrame of stock features loaded from the database.
        output_dir: Directory to save the prepared datasets.
        datasets_to_generate: List of dataset types to generate (e.g., ["train", "val", "inference"])

    Returns:
        Tuple of (train_dataset, val_dataset, inference_dataset)
    """
    logger.info("Starting feature preparation...")
    
    # Validate required raw features exist
    if (
        "sales" not in raw_features
        or "stock" not in raw_features
        or "movement" not in raw_features
    ):
        raise ValueError("Missing required raw feature data.")

    features = raw_features.copy()
    logger.info(
        "Raw features validation passed, proceeding with feature engineering..."
    )

    try:
        # 1. Create Sales Pivot Table
        sales_pivot = get_monthly_sales_pivot(features["sales"])
        logger.info("Sales pivot table created successfully.")

        # 2. Initialize Transformers (as per prepare_datasets.py example)
        static_transformer = MultiColumnLabelBinarizer()
        scaler = GlobalLogMinMaxScaler()

        # 3. Create datasets
        output_chunk_length = 1
        dataset_length = sales_pivot.shape[0]
        default_input_chunk_length = dataset_length - output_chunk_length

        if default_input_chunk_length <= 0:
            warn(
                "Dataset length is not enough for training with evaluation: \n"
                f"dataset_length: {dataset_length}, output_chunk_length: {output_chunk_length}", stacklevel=2
            )

        # Common dataset parameters
        dataset_params = {
            "stock_features": stock_features,
            "monthly_sales": sales_pivot,
            "static_transformer": static_transformer,
            "static_features": DEFAULT_STATIC_FEATURES,
            "scaler": scaler,
            "input_chunk_length": default_input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "save_dir": save_directory,
            "past_covariates_span": default_input_chunk_length,
            "past_covariates_fnames": DEFAULT_PAST_COVARIATES_FNAMES,
            "minimum_sales_months": 2, # TODO: make this configurable
        }

        train_dataset = None
        inference_dataset = None

        if "train" in datasets_to_generate:
            train_dataset = PlastinkaTrainingTSDataset(
                **dataset_params,
                dataset_name="train",
            )

        if "inference" in datasets_to_generate:
            # Inference dataset uses the full data range for its initial setup
            # The window will be adjusted later in __init__
            if train_dataset:
                inference_dataset = PlastinkaInferenceTSDataset(
                    **dataset_params,
                    dataset_name="inference",
                )
                logger.info("Inference dataset created and saved.")

        logger.info("Datasets created.")

    except Exception as e:
        logger.error(f"Error during feature preparation: {e}", exc_info=True)
        raise

    return train_dataset, inference_dataset


def get_datasets(
    start_date=None,
    end_date=None,
    config=None,
    output_dir: str | None = None,
    feature_types=None,
    datasets_to_generate: Sequence[str] = ("train", "val"),
    connection=None,
):
    """
    Loads data from DB, prepares features, creates and saves datasets to output_dir.

    Args:
        start_date: Start date for data loading.
        end_date: End date for data loading.
        config: Dictionary containing training configuration.
        output_dir: Directory to save the prepared datasets.
        feature_types: Optional list of feature types to load (e.g., ['sales', 'stock'])
        datasets_to_generate: List of dataset types to generate (e.g., ["train", "val", "inference"])
        connection: Database connection object.

    Returns:
        Tuple of (train_dataset, inference_dataset)
    """
    logger.info("get_datasets function started...")

    dal = DataAccessLayer(connection=connection)
    try:
        raw_features = load_data(start_date, end_date, feature_types, dal=dal)
        logger.info("Raw features loaded successfully.")
        feature_values = ["availability", "confidence"]
        stock_features_long = feature_storage.load_report_features(
            store_type="sql",
            dal=dal,
            start_date=config.get("start_date", None),
            end_date=config.get("end_date", None),
            prediction_month=config.get("prediction_month", None),
            feature_subset=feature_values
        )
        logger.info("Stock features loaded successfully in long format.")

        # Transform from long to wide format
        stock_features_long['data_date'] = pd.to_datetime(
            stock_features_long['data_date']
        )
        
        stock_features_wide = stock_features_long.pivot_table(
            index='data_date',
            columns=MULTIINDEX_NAMES,
            values=feature_values
        )
        logger.info(f"Transformed stock features to wide format with shape: {stock_features_wide.shape}")

        train_dataset, inference_dataset = prepare_datasets(
            raw_features,
            stock_features_wide, # Pass the transformed wide dataframe
            output_dir,
            datasets_to_generate,
        )

        logger.info(
            "prepare_datasets completed, returning datasets from get_datasets..."
        )

        return train_dataset, inference_dataset
    except Exception as e:
        logger.error(f"Error in get_datasets: {e}", exc_info=True)
        raise

