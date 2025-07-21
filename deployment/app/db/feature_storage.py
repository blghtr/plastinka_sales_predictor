import logging
import sqlite3  # Import sqlite3
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import re # Added for date validation
from datetime import date # Added for date validation
from deployment.app.db.schema import MULTIINDEX_NAMES
from deployment.app.db.database import (
    create_processing_run,
    execute_many,
    execute_query,
    get_or_create_multiindex_ids_batch,
    update_processing_run,
)

logger = logging.getLogger(__name__)


class SQLFeatureStore:
    """Store for saving and loading pandas DataFrames to/from SQL database"""

    def __init__(
        self, run_id: int | None = None, connection: sqlite3.Connection = None
    ):
        self.run_id = run_id
        self.db_conn = connection  # Store the connection
        self._conn_created_internally = False
        if not self.db_conn:
            # If no connection provided, create one internally (for non-test use)
            from deployment.app.db.database import (
                get_db_connection,  # Import locally to avoid circular dependency issues
            )

            self.db_conn = get_db_connection()
            self._conn_created_internally = True
        
        # Initialize DAL for batch operations
        from deployment.app.db.data_access_layer import DataAccessLayer, UserContext, UserRoles
        self._dal = DataAccessLayer(UserContext([UserRoles.SYSTEM]))

    def __enter__(self):
        # Connection is already established in __init__
        # If self.db_conn is None here, it means get_db_connection() failed in __init__
        # or an invalid connection was passed.
        # For now, we assume __init__ handles connection setup.
        # If more robust error handling for failed connection in __init__ is needed,
        # it could be added here or __init__ could raise an error.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Close the connection only if it was created internally by this instance
        if self._conn_created_internally and self.db_conn:
            try:
                self.db_conn.close()
            except Exception as e:
                logger.error(
                    f"Error closing internally managed DB connection: {e}",
                    exc_info=True,
                )
        # Do not suppress exceptions, return None or False (implicitly)

    def create_run(self, cutoff_date: str, source_files: str) -> int:
        """Create a new processing run and store its ID"""
        self.run_id = create_processing_run(
            start_time=datetime.now(),
            status="running",
            cutoff_date=cutoff_date,
            source_files=source_files,
            connection=self.db_conn,  # Pass connection
        )
        return self.run_id

    def complete_run(self, status: str = "completed") -> None:
        """Mark the current run as completed"""
        if self.run_id:
            update_processing_run(
                run_id=self.run_id,
                status=status,
                end_time=datetime.now(),
                connection=self.db_conn,  # Pass connection
            )

    def save_features(
        self, features: dict[str, pd.DataFrame], append: bool = False
    ) -> None:
        """Save all feature DataFrames to SQL database"""
        for feature_type, df in features.items():
            if hasattr(df, "shape"):  # Check if it's actually a DataFrame
                self._save_feature(feature_type, df, append)

        # Update run status
        if self.run_id:
            update_processing_run(
                run_id=self.run_id,
                status="features_saved",
                connection=self.db_conn,  # Pass connection
            )

    def _get_feature_config(self):
        """Return standardized configuration for different feature types"""
        config = {
            "sales": {"table": "fact_sales", "is_date_in_index": True},
            "stock": {"table": "fact_stock", "is_date_in_index": True},
            "change": {"table": "fact_stock_changes", "is_date_in_index": True},
            "prices": {"table": "fact_prices", "is_date_in_index": True},
        }
        return config

    def _save_feature(
        self, feature_type: str, df: pd.DataFrame, append: bool = False
    ) -> None:
        """Save a feature DataFrame to the appropriate SQL table using configuration"""
        config = self._get_feature_config().get(feature_type)
        if not config:
            logger.warning(f"Unknown feature type '{feature_type}'")
            return

        table = config["table"]
        
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"Unexpected DataFrame format for {feature_type}: expected DatetimeIndex in index")
            return

        # 1. Collect all unique multi-index tuples from the DataFrame columns
        unique_tuples = [tuple(col) for col in df.columns]

        # 2. Get all multi-index IDs in one batch
        id_map = get_or_create_multiindex_ids_batch(unique_tuples, self.db_conn)

        # 3. Prepare the list of parameters for insertion
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

        # Conditional logic for saving based on feature type
        if feature_type == "stock":
            # For stock, never delete existing data and ignore new data if PK exists.
            query = f"INSERT OR IGNORE INTO {table} (multiindex_id, data_date, value) VALUES (?, ?, ?)"
            execute_many(query, params_list, self.db_conn)
            logger.info(f"Saved/ignored {len(params_list)} records to {table} (non-overwrite mode)")
        else:
            # For other types (sales, prices, change), maintain original behavior (overwrite or append)
            # 4. Clear existing data if not appending
            if not append:
                cursor = self.db_conn.cursor()
                cursor.execute(f"DELETE FROM {table}")

            # 5. Insert new data in a single batch
            query = f"INSERT OR REPLACE INTO {table} (multiindex_id, data_date, value) VALUES (?, ?, ?)"
            execute_many(query, params_list, self.db_conn)
            logger.info(f"Saved {len(params_list)} records to {table}")

    def _convert_to_date_str(self, date_value: Any) -> str:
        """Convert various date formats to a standard date string."""
        if hasattr(date_value, "strftime"):
            return date_value.strftime("%Y-%m-%d")
        elif isinstance(date_value, str):
            try:
                # Try to parse string as date
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
                logger.warning(
                    f"Could not convert {value} of type {type(value)} to int, using {default}"
                )
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
                logger.warning(
                    f"Could not convert {value} of type {type(value)} to float, using {default}"
                )
                return default

    def load_features(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        feature_types: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load features from SQL database with optional date range filtering and feature type selection

        Args:
            start_date: Optional start date for filtering data
            end_date: Optional end date for filtering data
            feature_types: Optional list of feature types to load (e.g., ['sales', 'stock'])
                          If None, all available features will be loaded

        Returns:
            Dictionary of loaded features
        """
        features = {}

        # Determine which feature types to load
        if feature_types:
            # Filter to only the requested feature types that exist in config
            available_types = self._get_feature_config().keys()
            types_to_load = [ft for ft in feature_types if ft in available_types]
            if len(types_to_load) < len(feature_types):
                unknown_types = set(feature_types) - set(types_to_load)
                logger.warning(f"Unknown feature type(s) requested: {unknown_types}")
        else:
            # Load all available feature types
            types_to_load = self._get_feature_config().keys()

        for feature_type in types_to_load:
            df = self._load_feature(feature_type, start_date, end_date)
            if df is not None and not df.empty:
                features[feature_type] = df

        return features

    def _build_multiindex_from_mapping(
        self, multiindex_ids: list[int]
    ) -> tuple[pd.MultiIndex, list[bool]]:
        """
        Build a pandas MultiIndex from dim_multiindex_mapping using IDs.
        
        Args:
            multiindex_ids: List of multiindex IDs to retrieve
            
        Returns:
            tuple: (MultiIndex, mask) where mask indicates which IDs were found
        """
        empty_index = pd.MultiIndex(
            levels=[[]] * 10,
            codes=[[]] * 10,
            names=MULTIINDEX_NAMES,
        )
        if not multiindex_ids:
            return empty_index, []

        # Query with multiindex_id included to maintain mapping
        # Use batching to avoid SQLite variable limit
        unique_ids = list(set(multiindex_ids))
        query_template = """
        SELECT multiindex_id, barcode, artist, album, cover_type, price_category, 
               release_type, recording_decade, release_decade, style, record_year
        FROM dim_multiindex_mapping
        WHERE multiindex_id IN ({placeholders})
        """

        mapping_data = self._dal.execute_query_with_batching(
            query_template, unique_ids, connection=self.db_conn
        )

        # Create mapping from ID to tuple for O(1) lookup
        id_to_tuple = {}
        if mapping_data:
            for row in mapping_data:
                multiindex_id = row["multiindex_id"]
                tuple_data = tuple(
                    row[name] for name in MULTIINDEX_NAMES
                )
                id_to_tuple[multiindex_id] = tuple_data

        # Build result in original order with mask for missing IDs
        index_tuples = []
        mask = []
        n_missing = 0
        for multiindex_id in multiindex_ids:
            if multiindex_id in id_to_tuple:
                index_tuples.append(id_to_tuple[multiindex_id])
                mask.append(True)
            else:
                mask.append(False)
                n_missing += 1
        if n_missing > 0:
            logger.warning(f"Missing {n_missing} multiindex_ids in database")

        if not index_tuples:
            return empty_index, mask

        multiindex = pd.MultiIndex.from_tuples(
            index_tuples,
            names=MULTIINDEX_NAMES,
        )
        
        return multiindex, mask

    def _load_feature(
        self,
        feature_type: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Load feature from database using configuration"""
        config = self._get_feature_config().get(feature_type)
        if not config:
            logger.warning(f"Unknown feature type '{feature_type}'")
            return None

        table = config["table"]

        # Build query with standardized column names
        query = f"SELECT multiindex_id, data_date, value FROM {table}"
        params = []

        # Apply date filters only for time-series features (sales and changes)
        # Stock and prices should not be filtered by date
        if feature_type in ['sales', 'change']:
            if start_date:
                query += " WHERE data_date >= ?"
                params.append(start_date)

            if end_date:
                query += f"{' AND' if start_date else ' WHERE'} data_date <= ?"
                params.append(end_date)

        data = execute_query(
            query, tuple(params), fetchall=True, connection=self.db_conn
        )

        if not data:
            return None

        try:
            # Process results with standardized column names
            df = pd.DataFrame(data)
            df["data_date"] = pd.to_datetime(df["data_date"])

            # Получаем все уникальные multiindex_id в отсортированном порядке
            sorted_df = df.sort_values(by="multiindex_id")
            all_ids = sorted_df["multiindex_id"].tolist()
            full_index, mask = self._build_multiindex_from_mapping(all_ids)
            valid_df = sorted_df.loc[mask].reset_index(drop=True)
            
            # Разложить MultiIndex на отдельные колонки
            multiindex_df = full_index.to_frame(index=False)
            mapped_multiindex_df = pd.concat([multiindex_df, valid_df], axis=1)
            mapped_multiindex_df.drop(columns=["multiindex_id"], inplace=True)
            

            pivot_df = mapped_multiindex_df.pivot(
                index="data_date",
                columns=list(multiindex_df.columns),
                values="value"
            ).fillna(0)
            pivot_df.index.name = "_date"
            return pivot_df
        
        except Exception as e:
            logger.error(f"Error processing feature {feature_type}: {e}", exc_info=True)
            return None


class FeatureStoreFactory:
    """Factory for creating feature store instances"""

    @staticmethod
    def get_store(store_type: str = "sql", run_id: int | None = None, **kwargs) -> Any:
        """Get a feature store instance based on type"""
        if store_type == "sql":
            # Pass any additional kwargs (like connection) to the SQL store
            return SQLFeatureStore(run_id=run_id, **kwargs)
        # Add other store types here if needed (e.g., 'file', 'redis')
        # elif store_type == 'file':
        #     return FileFeatureStore(...)
        else:
            raise ValueError(f"Unsupported feature store type: {store_type}")


def save_features(
    features: dict[str, pd.DataFrame],
    cutoff_date: str,
    source_files: str,
    store_type: str = "sql",
    **kwargs,
) -> int:
    """Helper function to save features using a specific store type"""

    # Use an existing connection if provided, otherwise use db_transaction
    if "connection" in kwargs and kwargs["connection"] is not None:
        conn = kwargs.pop("connection")
        store = FeatureStoreFactory.get_store(
            store_type=store_type, connection=conn, **kwargs
        )
        run_id = store.create_run(cutoff_date, source_files)
        store.save_features(features)
        store.complete_run()
        # NOTE: COMMIT/ROLLBACK is the responsibility of the caller when using a provided connection
        return run_id
    else:
        # Use db_transaction context manager for automatic commit/rollback
        from deployment.app.db.database import (
            db_transaction,  # Import locally to avoid potential circular deps
        )

        with db_transaction() as conn:
            store = FeatureStoreFactory.get_store(
                store_type=store_type, connection=conn, **kwargs
            )
            run_id = store.create_run(cutoff_date, source_files)
            store.save_features(features)
            store.complete_run()
            # COMMIT/ROLLBACK handled by db_transaction context manager
        return run_id


def load_features(
    store_type: str = "sql",
    start_date: str | None = None,
    end_date: str | None = None,
    feature_types: list[str] | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Helper function to load features using a specific store type

    Args:
        store_type: Type of feature store to use (default: 'sql')
        start_date: Optional start date for filtering data
        end_date: Optional end date for filtering data
        feature_types: Optional list of feature types to load (e.g., ['sales', 'stock'])
                      If None, all available features will be loaded
        **kwargs: Additional arguments to pass to the feature store

    Returns:
        Dictionary of loaded features
    """
    store = FeatureStoreFactory.get_store(store_type=store_type, **kwargs)
    features = store.load_features(
        start_date=start_date, end_date=end_date, feature_types=feature_types
    )
    return features
