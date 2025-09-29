import warnings
import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.db.database import EXPECTED_REPORT_FEATURES, EXPECTED_REPORT_FEATURES_SET
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

    def create_run(self, source_files: str) -> int:
        """Create a new processing run and store its ID using the DAL."""
        self.run_id = self._dal.create_processing_run(
            start_time=datetime.now(),
            status="running",
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
        self, 
        features: dict[str, pd.DataFrame], 
        append: bool = True
    ) -> None:
        """Save all feature DataFrames to SQL database via the DAL."""
        # Save features
        for feature_type, df in features.items():
            if hasattr(df, "shape"):
                self._save_feature(feature_type, df, append)

        if self.run_id:
            self._dal.update_processing_run(
                run_id=self.run_id,
                status="features_saved",
            )

    def _save_report_features(self, df: pd.DataFrame) -> None:
        """Saves a special DataFrame with features for reports."""

        # 1. Reset index to access its levels as columns
        # Ensure there are no duplicates in the index
        df_reset = (
            df[~df.index.duplicated(keep='first')]
            .reset_index()
        )

        # 2. Check for the presence of all necessary columns
        id_map_cols = set(MULTIINDEX_NAMES)
        expected_cols = set(['_date']) | id_map_cols
        missing_cols = expected_cols - set(df_reset.columns)
        if missing_cols:
            logger.error(
                f"Missing columns in report_features: {missing_cols}"
            )
            raise ValueError(
                f"Missing columns in report_features: {missing_cols}"
            )
        
        # 3. Add missing features for the report
        missing_report_features = EXPECTED_REPORT_FEATURES_SET.difference(
            df_reset.columns
        )
        df_reset = df_reset.assign(**{
            feature: 0.0 for feature in missing_report_features
        })
        expected_cols.update(EXPECTED_REPORT_FEATURES_SET)

        # 4. Check that there are no extra columns
        forbidden_keys = set(df_reset.columns) - expected_cols
        if forbidden_keys:
            logger.warning(
                "Forbidden columns in report_features will be skipped: "
                f"{forbidden_keys}"
            )
        df_reset = df_reset.loc[:, list(expected_cols)]

        # 5. Check that there are no empty values
        df_reset[EXPECTED_REPORT_FEATURES] = df_reset[
                EXPECTED_REPORT_FEATURES
            ].fillna(0.0)

        # 6. Get IDs for all multi-indexes in one batch
        expected_features = expected_cols - id_map_cols
        features_data = df_reset[list(expected_features)]
        index_elems = df_reset[MULTIINDEX_NAMES]
        unique_tuples = (
            index_elems
            .itertuples(index=False, name=None)
        )

        features_data.loc[:, 'multiindex_id'] = self._dal.get_or_create_multiindex_ids_batch(
            list(unique_tuples)
        )
        features_data.loc[:, 'data_date'] = features_data['_date'].dt.strftime('%Y-%m-%d')
        features_data.loc[:, 'created_at'] = datetime.now().isoformat()
        features_data = features_data.drop(columns=['_date'])

        # 4. Form data for insertion
        params_list = list(
            features_data[
                ['data_date', 'multiindex_id'] + 
                EXPECTED_REPORT_FEATURES + 
                ['created_at']
            ]
            .astype(object)
            .itertuples(index=False, name=None)
        )

        self._dal.insert_report_features(params_list)
        logger.info(f"Saved {len(params_list)} records to report_features table.")

    def _get_feature_config(self):
        """
        Returns configuration for loading and formatting features.

        Returns:
            dict: Dictionary where keys are feature group names,
                  and values are configuration dictionaries:
                  - 'table' (str): Table name in the database.
                  - 'value_columns' (list[str]): Columns with feature values.
                  - 'output' (str): Desired output DataFrame format:
                    - 'pivoted': "date x product" matrix (index - date,
                      columns - MultiIndex with product attributes).
                    - 'flat': "Flat" table where each row is
                      one record (date, product) with all features in columns.
                  - 'rename_map' (dict, optional): Map for renaming
                    columns (e.g., {'value': 'sales'}).
        """
        return {
            "sales": {
                "table": "fact_sales",
                "value_columns": ["value"],
                "rename_map": {"value": "sales"},
                "output": "pivoted",
            },
            "movement": {
                "table": "fact_stock_movement",
                "value_columns": ["value"],
                "rename_map": {"value": "movement"},
                "output": "pivoted",
            },
            "report_features": {
                "table": "report_features",
                "value_columns": EXPECTED_REPORT_FEATURES,
                "output": "flat",
            },
        }

    def _save_feature(
        self, feature_type: str, df: pd.DataFrame | pd.Series, append: bool = True
    ) -> None:
        """Save a feature DataFrame to the appropriate SQL table using the DAL."""
        expected_cols = [
            'multiindex_id',
            'data_date',
            'value',
        ]
        config = self._get_feature_config().get(feature_type)
        if not config:
            logger.warning(f"Unknown feature type '{feature_type}'")
            return
        
        table = config["table"]
        is_time_agnostic = config.get("is_time_agnostic", False)

        if not append:
            self._dal.delete_features_by_table(table)

        if feature_type == "report_features":
            self._save_report_features(df)
            return
        
        if isinstance(df, pd.Series):
            df = df.to_frame(name='value')

        if isinstance(df, pd.DataFrame):
            if df.empty:
                logger.warning(f"Empty DataFrame provided for {feature_type}, skipping save")
                return
                
            if not isinstance(df.index, pd.MultiIndex):
                logger.error(
                    f"Unexpected DataFrame format for {feature_type}: "
                    f"expected MultiIndex in index, got {type(df.index)}"
                )
                raise ValueError(f"Unexpected DataFrame format for {feature_type}: expected MultiIndex in index, got {type(df.index)}")
            
            nan_rows = df[df.isnull().any(axis=1)]
            if not nan_rows.empty:
                logger.warning(
                    f"Found {len(nan_rows)} rows with NaN values in '{feature_type}'. "
                    f"These product rows will be dropped: {nan_rows.index.to_list()}"
                )
            df = df.dropna()
            if df.empty:
                logger.warning(f"Empty DataFrame after dropna for {feature_type}, skipping save")
                return
                
            df = df.loc[~df.index.duplicated(keep='first')]
            # Normalize tuples to strings for consistent identity mapping
            original_unique_tuples = df.index.to_list()

            multiindex_id = (
                self
                ._dal
                .get_or_create_multiindex_ids_batch(
                    original_unique_tuples
                )
            )

            if not isinstance(df.columns, pd.DatetimeIndex) and not is_time_agnostic:
                logger.warning(
                    f"Unexpected DataFrame format for {feature_type}: "
                    f"expected DatetimeIndex in columns, got {type(df.columns)}"
                )
                raise ValueError(f"Unexpected DataFrame format for {feature_type}: expected DatetimeIndex in columns, got {type(df.columns)}")
            
            df = df.reset_index(drop=True)
            if is_time_agnostic:
                if len(df.columns) != 1:
                    msg = (
                        f"Unexpected DataFrame format for {feature_type}: "
                        f"expected single feature column, got {len(df.columns)}"
                    )
                    logger.error(msg)
                    raise ValueError(msg)

                df = df.rename(columns={df.columns[0]: 'value'})
                df['data_date'] = pd.Timestamp('today').normalize()
                df['multiindex_id'] = multiindex_id
            else:
                df = (
                    df
                    .assign(multiindex_id=multiindex_id)
                    .melt(
                        id_vars=['multiindex_id'],
                        var_name="data_date",
                        value_name="value"
                    )
                )
                
            df['data_date'] = pd.to_datetime(df['data_date'], errors='coerce').dt.strftime('%Y-%m-%d')
            df = df.loc[df.value.ne(0.0)]

            params_list = []
            if not df.empty:
                params_list = list(
                    df[expected_cols]
                    .astype(object)
                    .itertuples(index=False, name=None)
                )
            
            if params_list:
                logger.info(f"Attempting to save {len(params_list)} records for {feature_type} into {table}")
                self._dal.insert_features_batch(table, params_list)
                self._dal.commit()
                logger.info(f"Saved {len(params_list)} records to {table}")
            else:
                logger.warning(f"No data to save for {feature_type}")

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
    ) -> dict[str, pd.DataFrame]:
        """
        Loads, processes and formats all features according to configuration.

        Execution process:
        1. Data for all feature groups is loaded from the database. All
           unique `multiindex_id` are collected.
        2. With one batch query for all `multiindex_id`, product attributes
           (barcode, artist, etc.) are extracted as a single DataFrame.
        3. For each feature group, source data is enriched with product
           attributes through `merge`.
        4. The result is formatted into 'pivoted' or 'flat' view according
           to the 'output' key in the configuration.

        Args:
            start_date: Start date for data range (YYYY-MM-DD).
            end_date: End date for data range (YYYY-MM-DD).

        Returns:
            Dictionary where keys are group/feature names, and values are
            final pandas DataFrames in the specified format.
        """
        configs = self._get_feature_config()
        raw_dfs = {}
        all_multiindex_ids = set()

        # 1. Load raw data for all groups and collect IDs
        for group_name, config in configs.items():
            # This DAL method needs to be created.
            data = self._dal.get_feature_dataframe(
                table_name=config["table"],
                columns=config["value_columns"],
                start_date=start_date,
                end_date=end_date,
            )
            if not data:
                logger.warning(f"No data found for feature group '{group_name}'")
                continue

            df = pd.DataFrame(data)
            if 'multiindex_id' in df.columns:
                all_multiindex_ids.update(df['multiindex_id'].unique())
                raw_dfs[group_name] = df
            else:
                logger.warning(f"Feature group '{group_name}' loaded without 'multiindex_id' column.")


        if not raw_dfs:
            return {}

        # 2. Get attributes for all collected IDs in one batch
        # Convert numpy types to Python int to avoid SQLite type issues
        ids_list = [int(id_val) for id_val in all_multiindex_ids if id_val is not None]
        multiindex_tuples = self._dal.get_multiindex_mapping_by_ids(ids_list)
        if not multiindex_tuples:
            logger.error("Could not retrieve multi-index mapping for any of the found IDs.")
            return {}
        attributes_df = pd.DataFrame(multiindex_tuples, dtype=str)

        # 3. & 4. Format each group according to its config
        final_features = {}
        for group_name, raw_df in raw_dfs.items():
            config = configs[group_name]
            formatted_group = self._format_feature_group(raw_df, attributes_df, config)
            final_features.update(formatted_group)

        return final_features

    def _format_feature_group(
        self,
        raw_df: pd.DataFrame,
        attributes_df: pd.DataFrame,
        config: dict,
    ) -> dict[str, pd.DataFrame]:
        """
        Finally formats DataFrame for one feature group.

        Enriches raw data with product attributes and brings the result
        to 'pivoted' or 'flat' format according to configuration.

        Args:
            raw_df: "Raw" DataFrame loaded from database (contains
                    multiindex_id, data_date, and value columns).
            attributes_df: DataFrame with product attributes (multiindex_id,
                           barcode, artist, ...).
            config: Configuration dictionary for this feature group.

        Returns:
            Dictionary where key is feature/group name, and value is
            final DataFrame.
        """
        if raw_df.empty:
            return {}
            
        # Ensure data_date is in datetime format for processing
        raw_df['data_date'] = pd.to_datetime(raw_df['data_date'])

        # Enrich raw data with product attributes
        # Force key columns to a consistent numeric type to prevent merge errors
        raw_df['multiindex_id'] = pd.to_numeric(raw_df['multiindex_id'], errors='coerce').astype('Int64')
        attributes_df['multiindex_id'] = pd.to_numeric(attributes_df['multiindex_id'], errors='coerce').astype('Int64')

        # Drop rows where the key could not be converted, as they cannot be merged
        raw_df.dropna(subset=['multiindex_id'], inplace=True)
        attributes_df.dropna(subset=['multiindex_id'], inplace=True)

        enriched_df = pd.merge(raw_df, attributes_df, on='multiindex_id', how='left')
        
        output_format = config.get("output", "pivoted")
        result = {}
        group_name = config['table']

        if output_format == "flat":
            # For 'flat' format, the enriched DataFrame is the result.
            # enriched_df = enriched_df.drop(columns=['multiindex_id'])
            result[group_name] = enriched_df.fillna(0)
            logger.info(f"Formatted '{group_name}' as a flat DataFrame with {len(enriched_df)} rows.")

        elif output_format == "pivoted":
            # For 'pivoted' format, we melt and pivot for each feature.
            id_vars = [col for col in attributes_df.columns if col != 'multiindex_id'] + ['data_date']
            value_vars = config["value_columns"]
            
            rename_map = config.get("rename_map", {})
            if rename_map:
                enriched_df = enriched_df.rename(columns=rename_map)
                value_vars = [rename_map.get(v, v) for v in value_vars]

            long_df = enriched_df.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name="feature_name",
                value_name="value"
            )

            for feature_name, group_df in long_df.groupby("feature_name"):
                try:
                    pivoted_df = group_df.pivot_table(
                        index="data_date",
                        columns=MULTIINDEX_NAMES,
                        values="value"
                    ).fillna(0)
                    pivoted_df.index.name = "_date"
                    result[feature_name] = pivoted_df.fillna(0)
                    logger.info(f"Formatted '{feature_name}' as a pivoted DataFrame.")
                except Exception as e:
                    logger.error(f"Failed to pivot feature '{feature_name}': {e}", exc_info=True)
        
        return result


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
    source_files: str,
    store_type: str = "sql",
    dal: DataAccessLayer = None,
    append: bool = True,
    **kwargs,
) -> int:
    """Helper function to save features using a specific store type, requiring a DAL instance."""
    if not dal:
        raise ValueError("A DataAccessLayer instance must be provided.")

    # The DAL now manages transactions, so we don't need a separate db_transaction context.
    # We assume the DAL is configured with a connection that can handle transactions.
    store = FeatureStoreFactory.get_store(
        store_type=store_type, 
        dal=dal, 
        **kwargs
    )
    run_id = store.create_run(source_files)
    store.save_features(features, append=append)
    store.complete_run()
    return run_id


def load_features(
    store_type: str = "sql",
    start_date: str | None = None,
    end_date: str | None = None,
    dal: DataAccessLayer = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Helper function to load features, requiring a DAL instance."""
    if not dal:
        raise ValueError("A DataAccessLayer instance must be provided.")

    store = FeatureStoreFactory.get_store(store_type=store_type, dal=dal, **kwargs)
    features = store.load_features(
        start_date=start_date, end_date=end_date
    )
    return features


def load_report_features(
    store_type: str = "sql",
    multiidx_ids: list[int] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    feature_subset: list[str] | None = None,
    dal: DataAccessLayer = None,
    **kwargs,
) -> pd.DataFrame:
    """Helper function to load report features, requiring a DAL instance."""
    if not dal:
        raise ValueError("A DataAccessLayer instance must be provided.")

    store = FeatureStoreFactory.get_store(store_type=store_type, dal=dal, **kwargs)
    return store.load_report_features(
        multiidx_ids=multiidx_ids,
        start_date=start_date,
        end_date=end_date,
        feature_subset=feature_subset,
    )


