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
        # Сохраняем  фичи
        for feature_type, df in features.items():
            if hasattr(df, "shape"):
                self._save_feature(feature_type, df, append)

        if self.run_id:
            self._dal.update_processing_run(
                run_id=self.run_id,
                status="features_saved",
            )

    def _save_report_features(self, df: pd.DataFrame) -> None:
        """Сохраняет специальный датафрейм с фичами для отчетов."""

        # 1. Сбрасываем индекс, чтобы получить доступ к его уровням как к колонкам
        # Гарантируем, что нет дубликатов в индексе
        df_reset = (
            df[~df.index.duplicated(keep='first')]
            .reset_index()
        )

        # 2. Проверяем наличие всех необходимых колонок
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
        
        # 3. Добавляем отсутствующие фичи для отчета
        missing_report_features = EXPECTED_REPORT_FEATURES_SET.difference(
            df_reset.columns
        )
        df_reset = df_reset.assign(**{
            feature: 0.0 for feature in missing_report_features
        })
        expected_cols.update(EXPECTED_REPORT_FEATURES_SET)

        # 4. Проверяем, что нет лишних колонок
        forbidden_keys = set(df_reset.columns) - expected_cols
        if forbidden_keys:
            logger.warning(
                "Forbidden columns in report_features will be skipped: "
                f"{forbidden_keys}"
            )
        df_reset = df_reset.loc[:, list(expected_cols)]

        # 5. Проверяем, что нет пустых значений
        df_reset[EXPECTED_REPORT_FEATURES] = df_reset[
                EXPECTED_REPORT_FEATURES
            ].fillna(0.0)

        # 6. Получаем ID для всех мульти-индексов за один раз
        expected_features = expected_cols - id_map_cols
        features_data = df_reset[list(expected_features)]
        index_elems = df_reset[MULTIINDEX_NAMES]
        unique_tuples = (
            index_elems
            .itertuples(index=False, name=None)
        )
        id_map = self._dal.get_or_create_multiindex_ids_batch(list(unique_tuples))
        features_data['multiindex_id'] = (
            index_elems
            .agg(tuple, axis=1)
            .map(id_map)
        )
        features_data['data_date'] = features_data['_date'].dt.strftime('%Y-%m-%d')
        features_data['created_at'] = datetime.now().isoformat()
        features_data = features_data.drop(columns=['_date'])

        # 4. Формируем данные для вставки
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
        logger.info(f"Сохранено {len(params_list)} записей в таблицу report_features.")


    def _get_feature_config(self):
        """Return standardized configuration for different feature types."""
        return {
            "sales": {"table": "fact_sales", "is_time_agnostic": False},
            "movement": {"table": "fact_stock_movement", "is_time_agnostic": False},
            # --- Новая конфигурация для фичей отчета ---
            "report_features": {
                "table": "report_features",
                "is_time_agnostic": False,
                "value_columns": EXPECTED_REPORT_FEATURES,
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
        is_time_agnostic = config["is_time_agnostic"]

        # TODO: подумать, как реализовать безопаснее
        if not append:
            self._dal.delete_features_by_table(table)

        if feature_type == "report_features":
            self._save_report_features(df)
            return
        
        if isinstance(df, pd.Series):
            df = df.to_frame(name='value')

        if isinstance(df, pd.DataFrame):
            # Check for empty DataFrame first
            if df.empty:
                logger.warning(f"Empty DataFrame provided for {feature_type}, skipping save")
                return
                
            if not isinstance(df.index, pd.MultiIndex):
                logger.error(
                    f"Unexpected DataFrame format for {feature_type}: "
                    f"expected MultiIndex in index, got {type(df.index)}"
                )
                raise ValueError(f"Unexpected DataFrame format for {feature_type}: expected MultiIndex in index, got {type(df.index)}")
            
            # Check if DataFrame is empty after dropna
            df = df.dropna()
            if df.empty:
                logger.warning(f"Empty DataFrame after dropna for {feature_type}, skipping save")
                return
                
            df = df.loc[~df.index.duplicated(keep='first')]
            unique_tuples = df.index.to_list()
            id_map = self._dal.get_or_create_multiindex_ids_batch(unique_tuples)
            multiindex_id = (
                df
                .index
                .to_frame()
                .agg(tuple, axis=1)
                .map(id_map)
                .astype(int)
                .values
            )

            if not isinstance(df.columns, pd.DatetimeIndex) and not is_time_agnostic:
                logger.warning(
                    f"Unexpected DataFrame format for {feature_type}: "
                    f"expected DatetimeIndex in columns, got {type(df.columns)}"
                )
                raise ValueError(f"Unexpected DataFrame format for {feature_type}: expected DatetimeIndex in columns, got {type(df.columns)}")
            
            df = df.reset_index(drop=True)
            if is_time_agnostic:
                # value
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

            if not df.empty:
                params_list = list(
                    df[expected_cols]
                    .astype(object)
                    .itertuples(index=False, name=None)
                )
            else:
                logger.warning(f"No data to save for {feature_type}")
                params_list = []
        self._dal.insert_features_batch(table, params_list)
        logger.info(f"Saved {len(params_list)} records to {table}")

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

    def load_report_features(
        self,
        multiidx_ids: list[int] | None = None,
        prediction_month: date | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        feature_subset: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load report features from database and convert to DataFrame format.
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            prediction_month: Specific prediction month (optional)
            
        Returns:
            DataFrame with report features in MultiIndex format
        """
        try:
            report_data = self._dal.get_report_features(
                prediction_month=prediction_month,
                multiidx_ids=multiidx_ids,
                start_date=start_date,
                end_date=end_date,
                feature_subset=feature_subset,
            )

            # Convert to DataFrame format
            df = pd.DataFrame(report_data)
            
            logger.info(f"Loaded {len(df)} report features records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading report features: {e}")
            return pd.DataFrame()

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
        is_time_agnostic = config["is_time_agnostic"]
        if is_time_agnostic:
            start_date = None
            end_date = None
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
    store = FeatureStoreFactory.get_store(store_type=store_type, dal=dal, **kwargs)
    run_id = store.create_run(source_files)
    store.save_features(features, append=append)
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


def load_report_features(
    store_type: str = "sql",
    multiidx_ids: list[int] | None = None,
    prediction_month: date | None = None,
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
        prediction_month=prediction_month,
        multiidx_ids=multiidx_ids,
        start_date=start_date,
        end_date=end_date,
        feature_subset=feature_subset,
    )


