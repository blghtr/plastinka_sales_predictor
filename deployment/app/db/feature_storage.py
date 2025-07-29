import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.db.schema import MULTIINDEX_NAMES

logger = logging.getLogger(__name__)


class SQLFeatureStore:
    """Store for saving and loading pandas DataFrames to/from SQL database using DataAccessLayer."""

    def __init__(self, dal: DataAccessLayer, run_id: int | None = None):
        if not isinstance(dal, DataAccessLayer):
            raise TypeError("A DataAccessLayer instance is required.")
        self._dal = dal
        self.run_id = run_id
        # The connection is now managed by the DAL instance
        self.db_conn = self._dal.connection if hasattr(self._dal, 'connection') else None


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Connection is managed by the DAL, so no need to close it here.
        pass

    def create_run(self, cutoff_date: str, source_files: str) -> int:
        """Create a new processing run and store its ID using the DAL."""
        self.run_id = self._dal.create_processing_run(
            start_time=datetime.now(),
            status="running",
            cutoff_date=cutoff_date,
            source_files=source_files,
        )
        return self.run_id

    def complete_run(self, status: str = "completed") -> None:
        """Mark the current run as completed using the DAL."""
        if self.run_id:
            self._dal.update_processing_run(
                run_id=self.run_id,
                status=status,
                end_time=datetime.now(),
            )

    def save_features(
        self, features: dict[str, pd.DataFrame], append: bool = False
    ) -> None:
        """Save all feature DataFrames to SQL database via the DAL."""
        for feature_type, df in features.items():
            if hasattr(df, "shape"):
                self._save_feature(feature_type, df, append)

        if self.run_id:
            self._dal.update_processing_run(
                run_id=self.run_id,
                status="features_saved",
            )

    def _get_feature_config(self):
        """Return standardized configuration for different feature types."""
        return {
            "sales": {"table": "fact_sales", "is_date_in_index": True},
            "stock": {"table": "fact_stock", "is_date_in_index": True},
            "change": {"table": "fact_stock_changes", "is_date_in_index": True},
            "prices": {"table": "fact_prices", "is_date_in_index": True},
        }

    def _get_or_create_multiindex_ids_batch(self, unique_tuples: list[tuple]) -> dict[tuple, int]:
        """Get or create multi-index IDs in a batch using the DAL."""
        id_map = {}
        for t in unique_tuples:
            # Assuming MULTIINDEX_NAMES order matches the tuple order
            params = dict(zip(MULTIINDEX_NAMES, t, strict=False))
            multiindex_id = self._dal.get_or_create_multiindex_id(**params)
            id_map[t] = multiindex_id
        return id_map

    def _save_feature(
        self, feature_type: str, df: pd.DataFrame, append: bool = False
    ) -> None:
        """Save a feature DataFrame to the appropriate SQL table using the DAL."""
        config = self._get_feature_config().get(feature_type)
        if not config:
            logger.warning(f"Unknown feature type '{feature_type}'")
            return

        table = config["table"]

        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"Unexpected DataFrame format for {feature_type}: expected DatetimeIndex in index")
            return

        unique_tuples = [tuple(col) for col in df.columns]
        id_map = self._get_or_create_multiindex_ids_batch(unique_tuples)

        params_list = []
        for date_idx, row in df.iterrows():
            date_str = self._convert_to_date_str(date_idx)
            for col_idx, value in row.items():
                if pd.notna(value):
                    multiindex_id = id_map.get(tuple(col_idx))
                    if multiindex_id:
                        params_list.append((multiindex_id, date_str, value))

        if not params_list:
            logger.warning(f"No valid data to save for feature type '{feature_type}'")
            return

        if feature_type == "stock":
            query = f"INSERT OR IGNORE INTO {table} (multiindex_id, data_date, value) VALUES (?, ?, ?)"
            self._dal.execute_many_with_batching(query, params_list)
            logger.info(f"Saved/ignored {len(params_list)} records to {table} (non-overwrite mode)")
        else:
            if not append:
                self._dal.delete_features_by_table(table)

            self._dal.insert_features_batch(table, params_list)
            logger.info(f"Saved {len(params_list)} records to {table}")

    def _convert_to_date_str(self, date_value: Any) -> str:
        """Convert various date formats to a standard date string."""
        if hasattr(date_value, "strftime"):
            return date_value.strftime("%Y-%m-%d")
        elif isinstance(date_value, str):
            try:
                parsed_date = pd.to_datetime(date_value)
                return parsed_date.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                return str(date_value)
        else:
            return str(date_value)

    def _convert_to_int(self, value: Any, default: int = 0) -> int:
        """Safely convert any value to integer with proper handling of np.float64."""
        if pd.isna(value):
            return default
        if isinstance(value, np.floating | float):
            return int(np.round(value))
        elif isinstance(value, np.integer | int):
            return int(value)
        else:
            try:
                return int(float(value))
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {value} of type {type(value)} to int, using {default}")
                return default

    def _convert_to_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert any value to float."""
        if pd.isna(value):
            return default
        if isinstance(value, np.floating | float | np.integer | int):
            return float(value)
        else:
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {value} of type {type(value)} to float, using {default}")
                return default

    def load_features(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        feature_types: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load features from SQL database via the DAL."""
        features = {}
        available_types = self._get_feature_config().keys()

        if feature_types:
            types_to_load = [ft for ft in feature_types if ft in available_types]
            if len(types_to_load) < len(feature_types):
                unknown_types = set(feature_types) - set(types_to_load)
                logger.warning(f"Unknown feature type(s) requested: {unknown_types}")
        else:
            types_to_load = list(available_types)

        for feature_type in types_to_load:
            df = self._load_feature(feature_type, start_date, end_date)
            if df is not None and not df.empty:
                features[feature_type] = df
        return features

    def _build_multiindex_from_mapping(
        self, multiindex_ids: list[int]
    ) -> tuple[pd.MultiIndex, list[bool]]:
        """Build a pandas MultiIndex from dim_multiindex_mapping using IDs via the DAL."""
        empty_index = pd.MultiIndex(levels=[[]] * 10, codes=[[]] * 10, names=MULTIINDEX_NAMES)
        if not multiindex_ids:
            return empty_index, []

        # Use the DAL method for getting multiindex mapping
        unique_ids = list(set(multiindex_ids))
        
        # Get the mapping data using the DAL method
        mapping_data = self._dal.get_multiindex_mapping_by_ids(unique_ids)
        
        # Create reverse mapping from ID to tuple
        id_to_tuple = {}
        for row in mapping_data:
            multiindex_id = row["multiindex_id"]
            tuple_data = tuple(row[name] for name in MULTIINDEX_NAMES)
            id_to_tuple[multiindex_id] = tuple_data

        index_tuples = []
        mask = []
        for multiindex_id in multiindex_ids:
            if multiindex_id in id_to_tuple:
                index_tuples.append(id_to_tuple[multiindex_id])
                mask.append(True)
            else:
                mask.append(False)

        n_missing = sum(1 for m in mask if not m)
        if n_missing > 0:
            logger.warning(f"Missing {n_missing} multiindex_ids in database")

        if not index_tuples:
            return empty_index, mask

        return pd.MultiIndex.from_tuples(index_tuples, names=MULTIINDEX_NAMES), mask

    def _load_feature(
        self,
        feature_type: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Load a single feature DataFrame from the database via the DAL."""
        config = self._get_feature_config().get(feature_type)
        if not config:
            logger.warning(f"Unknown feature type '{feature_type}'")
            return None

        table = config["table"]

        data = self._dal.get_features_by_date_range(table, start_date, end_date)

        if not data:
            return None

        try:
            df = pd.DataFrame(data, columns=["multiindex_id", "data_date", "value"])
            df["data_date"] = pd.to_datetime(df["data_date"])

            if df.empty:
                return pd.DataFrame()

            sorted_df = df.sort_values(by="multiindex_id")

            # Ensure alignment before pivot
            pivot_df = sorted_df.pivot_table(
                index="data_date",
                columns="multiindex_id",
                values="value"
            ).fillna(0)

            # Map pivot columns to multi-index
            pivot_df.columns, _ = self._build_multiindex_from_mapping(list(pivot_df.columns))
            pivot_df.index.name = "_date"
            return pivot_df

        except Exception as e:
            logger.error(f"Error processing feature {feature_type}: {e}", exc_info=True)
            return None


class FeatureStoreFactory:
    """Factory for creating feature store instances."""

    @staticmethod
    def get_store(store_type: str = "sql", dal: DataAccessLayer = None, run_id: int | None = None, **kwargs) -> Any:
        """Get a feature store instance based on type."""
        if store_type == "sql":
            if not dal:
                raise ValueError("DataAccessLayer instance is required for SQLFeatureStore.")
            return SQLFeatureStore(dal=dal, run_id=run_id)
        else:
            raise ValueError(f"Unsupported feature store type: {store_type}")


def save_features(
    features: dict[str, pd.DataFrame],
    cutoff_date: str,
    source_files: str,
    store_type: str = "sql",
    dal: DataAccessLayer = None,
    **kwargs,
) -> int:
    """Helper function to save features using a specific store type, requiring a DAL instance."""
    if not dal:
        raise ValueError("A DataAccessLayer instance must be provided.")

    # The DAL now manages transactions, so we don't need a separate db_transaction context.
    # We assume the DAL is configured with a connection that can handle transactions.
    store = FeatureStoreFactory.get_store(store_type=store_type, dal=dal, **kwargs)
    run_id = store.create_run(cutoff_date, source_files)
    store.save_features(features)
    store.complete_run()
    return run_id


def load_features(
    store_type: str = "sql",
    start_date: str | None = None,
    end_date: str | None = None,
    feature_types: list[str] | None = None,
    dal: DataAccessLayer = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Helper function to load features, requiring a DAL instance."""
    if not dal:
        raise ValueError("A DataAccessLayer instance must be provided.")

    store = FeatureStoreFactory.get_store(store_type=store_type, dal=dal, **kwargs)
    features = store.load_features(
        start_date=start_date, end_date=end_date, feature_types=feature_types
    )
    return features

