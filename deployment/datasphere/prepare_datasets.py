import logging
import os
from typing import Sequence

import pandas as pd
import numpy as np

from deployment.app.db import feature_storage
from deployment.app.config import get_settings
from deployment.app.models.api_models import TrainingConfig
from plastinka_sales_predictor.data_preparation import (
    GlobalLogMinMaxScaler,
    MultiColumnLabelBinarizer,
    PlastinkaTrainingTSDataset,
    PlastinkaInferenceTSDataset,
    OrdinalEncoder,
    get_monthly_sales_pivot,
    get_stock_features,
    COLTYPES,
    GROUP_KEYS,
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
        logger.info(
            f"Loading features from database{' (selected types)' if feature_types else ''}..."
        )
        logger.info(
            f"Loading with parameters: start_date={start_date}, end_date={end_date}, feature_types={feature_types}"
        )
        logger.info("About to call feature_storage.load_features...")

        features = feature_storage.load_features(
            store_type="sql",
            start_date=start_date,
            end_date=end_date,
            feature_types=feature_types,
        )
        logger.info("Features loaded successfully via factory.")

    except Exception as e:
        logger.error(f"Error loading data via factory: {e}", exc_info=True)
        raise

    return features


# type: ignore[override]
def prepare_datasets(
    raw_features: dict,
    config: dict | None,
    save_directory: str,
    datasets_to_generate: Sequence[str] = ("train", "val"),
) -> tuple[PlastinkaTrainingTSDataset, PlastinkaTrainingTSDataset]:
    """
    Loads data from DB, prepares features, creates full dataset.
    Saves train and validation datasets to the specified output_dir.

    Args:
        raw_features: Dictionary of raw features loaded from the database.
        config: Dictionary containing training configuration.
        output_dir: Directory to save the prepared datasets.
        datasets_to_generate: List of dataset types to generate (e.g., ["train", "val", "inference"])

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("Starting feature preparation...")

    # Validate required raw features exist
    if (
        "sales" not in raw_features
        or "stock" not in raw_features
        or "change" not in raw_features
    ):
        raise ValueError("Missing required raw feature data.")

    try:
        features = raw_features.copy()
    except Exception as e:
        logger.error(f"Error copying raw_features: {e}", exc_info=True)
        raise

    logger.info(
        "Raw features validation passed, proceeding with feature engineering..."
    )

    try:
        # 1. Stock Features (using get_stock_features from prepare_datasets.py)
        features["stock_features"] = get_stock_features(
            features["stock"], features["change"]
        )
        logger.info("Step 1 completed: Stock features created successfully.")

        # 2. Create Sales Pivot Table
        sales_pivot = get_monthly_sales_pivot(features["sales"])
        logger.info("Step 2 completed: Sales pivot table created successfully.")

        # 3. Initialize Transformers (as per prepare_datasets.py example)
        static_transformer = MultiColumnLabelBinarizer()
        scaler = GlobalLogMinMaxScaler()

        dataset_length = sales_pivot.shape[0]
        output_chunk_length = 1

        # Determine lags: prefer value from config; else derive automatically
        if config and hasattr(config, "lags") and getattr(config, "lags"):
            lags = getattr(config, "lags")
        else:
            # Choose lags so that at least one training window exists, cap at 12
            lags = max(1, min(12, dataset_length - 2))

        length = lags + 1
        train_end = max(length, dataset_length - length)

        if dataset_length <= length:
            raise ValueError(
                "Dataset length is less than the length of the lags. "
                f"Please provide more data. Dataset length: {dataset_length}, lags: {lags}"
            )

        # Common dataset parameters
        dataset_params = {
            "stock_features": features["stock_features"],
            "monthly_sales": sales_pivot,
            "static_transformer": static_transformer,
            "static_features": DEFAULT_STATIC_FEATURES,
            "scaler": scaler,
            "input_chunk_length": lags,
            "output_chunk_length": output_chunk_length,
            "save_dir": save_directory,
            "past_covariates_span": lags,
            "past_covariates_fnames": DEFAULT_PAST_COVARIATES_FNAMES,
            "minimum_sales_months": 2,
        }

        train_dataset = None
        val_dataset = None
        inference_dataset = None

        if "train" in datasets_to_generate:
            train_dataset = PlastinkaTrainingTSDataset(
                **dataset_params,
                start=0,
                end=train_end,
                dataset_name="train",
            )

        if "val" in datasets_to_generate:
            if train_dataset is not None:
                val_dataset = train_dataset.setup_dataset(
                    window=(dataset_length - length, dataset_length),
                )
                val_dataset.save(dataset_name="val")
            else:
                # Create val dataset directly if train dataset wasn't created
                val_dataset = PlastinkaTrainingTSDataset(
                    **dataset_params,
                    start=dataset_length - length,
                    end=dataset_length,
                    dataset_name="val",
                )

        if "inference" in datasets_to_generate:
            # Inference dataset uses the full data range for its initial setup
            # The window will be adjusted later in train_and_predict.py
            inference_dataset = PlastinkaInferenceTSDataset(
                **dataset_params,
                start=0,  # Start from the beginning of the available data
                end=None,  # Use the full length, will be padded internally
                dataset_name="inference",
            )
            logger.info("Inference dataset created and saved.")

        logger.info("Datasets created.")

    except Exception as e:
        logger.error(f"Error during feature preparation: {e}", exc_info=True)
        raise

    return train_dataset, val_dataset, inference_dataset


def get_datasets(
    start_date=None,
    end_date=None,
    config=None,
    output_dir: str | None = None,
    feature_types=None,
    datasets_to_generate: Sequence[str] = ("train", "val"),
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

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("get_datasets function started...")

    try:
        raw_features = load_data(start_date, end_date, feature_types)

        logger.info("Raw features loaded, calling prepare_datasets...")

        # Pass the output_dir and datasets_to_generate down to the preparation function
        train_dataset, val_dataset, inference_dataset = prepare_datasets(
            raw_features,
            config,
            output_dir,
            datasets_to_generate,
        )

        logger.info(
            "prepare_datasets completed, returning datasets from get_datasets..."
        )

        return train_dataset, val_dataset, inference_dataset
    except Exception as e:
        logger.error(f"Error in get_datasets: {e}", exc_info=True)
        raise
