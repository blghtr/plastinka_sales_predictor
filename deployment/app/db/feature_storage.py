import logging
import sqlite3  # Import sqlite3
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from deployment.app.db.database import (
    create_processing_run,
    execute_many,
    execute_query,
    get_or_create_multiindex_id,
    update_processing_run,
)

logger = logging.getLogger(__name__)

class SQLFeatureStore:
    """Store for saving and loading pandas DataFrames to/from SQL database"""

    def __init__(self, run_id: int | None = None, connection: sqlite3.Connection = None):
        self.run_id = run_id
        self.db_conn = connection # Store the connection
        self._conn_created_internally = False
        if not self.db_conn:
            # If no connection provided, create one internally (for non-test use)
            from deployment.app.db.database import (
                get_db_connection,  # Import locally to avoid circular dependency issues
            )
            self.db_conn = get_db_connection()
            self._conn_created_internally = True

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
                logger.error(f"Error closing internally managed DB connection: {e}", exc_info=True)
        # Do not suppress exceptions, return None or False (implicitly)

    def create_run(self, cutoff_date: str, source_files: str) -> int:
        """Create a new processing run and store its ID"""
        self.run_id = create_processing_run(
            start_time=datetime.now(),
            status="running",
            cutoff_date=cutoff_date,
            source_files=source_files,
            connection=self.db_conn # Pass connection
        )
        return self.run_id

    def complete_run(self, status: str = "completed") -> None:
        """Mark the current run as completed"""
        if self.run_id:
            update_processing_run(
                run_id=self.run_id,
                status=status,
                end_time=datetime.now(),
                connection=self.db_conn # Pass connection
            )

    def save_features(self, features: dict[str, pd.DataFrame], append: bool = False) -> None:
        """Save all feature DataFrames to SQL database"""
        for feature_type, df in features.items():
            if hasattr(df, 'shape'):  # Check if it's actually a DataFrame
                self._save_feature(feature_type, df, append)

        # Update run status
        if self.run_id:
            update_processing_run(
                run_id=self.run_id,
                status="features_saved",
                connection=self.db_conn # Pass connection
            )

    def _get_feature_config(self):
        """Return standardized configuration for different feature types"""
        return {
            'stock': {
                'table': 'fact_stock',
                'is_date_in_index': False
            },
            'prices': {
                'table': 'fact_prices',
                'is_date_in_index': False
            },
            'sales': {
                'table': 'fact_sales',
                'is_date_in_index': True
            },
            'change': {
                'table': 'fact_stock_changes',
                'is_date_in_index': True
            }
        }

    def _save_feature(self, feature_type: str, df: pd.DataFrame, append: bool = False) -> None:
        """Save a feature DataFrame to the appropriate SQL table using configuration"""
        config = self._get_feature_config().get(feature_type)
        if not config:
            logger.warning(f"Unknown feature type '{feature_type}'")
            return

        table = config['table']
        is_date_in_index = config['is_date_in_index']
        params_list = []

        for idx, row in df.iterrows():
            try:
                # Process data based on date position (in index or not)
                if is_date_in_index:
                    # Date in index
                    date_str = self._convert_to_date_str(idx[0])
                    multiindex_id = self._get_multiindex_id(idx[1:])
                    # Try to get the value from feature_type column
                    value = row.get(feature_type, 0.0)
                else:
                    # Date not in index (stock/prices)
                    multiindex_id = self._get_multiindex_id(idx)

                    if feature_type == 'stock' and not row.empty:
                        date_str = self._convert_to_date_str(row.index[0])
                        value = row.iloc[0]
                    else:
                        # For prices use current date
                        date_str = datetime.now().strftime('%Y-%m-%d')
                        # Try to get from column named like feature_type+'s'
                        value = row.get(feature_type, 0.0)  # e.g., "prices" column

                # Convert to float (all values are stored as REAL)
                value_converted = self._convert_to_float(value)

                params_list.append((
                    multiindex_id,
                    date_str,
                    value_converted
                ))
            except Exception as e:
                logger.error(f"Error processing {feature_type} data row: {e}", exc_info=True)

        # Batch insert data
        if params_list:
            execute_many(
                f"INSERT OR REPLACE INTO {table} (multiindex_id, data_date, value) VALUES (?, ?, ?)",
                params_list,
                connection=self.db_conn # Pass connection
            )

    def _convert_to_date_str(self, date_value: Any) -> str:
        """Convert various date formats to a standard date string."""
        if hasattr(date_value, 'strftime'):
            return date_value.strftime('%Y-%m-%d')
        elif isinstance(date_value, str):
            try:
                # Try to parse string as date
                parsed_date = pd.to_datetime(date_value)
                return parsed_date.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                return str(date_value)
        else:
            return str(date_value)

    def _convert_to_int(self, value: Any, default: int = 0) -> int:
        """Safely convert any value to integer with proper handling of np.float64."""
        if pd.isna(value):
            return default

        if isinstance(value, (np.floating, float)):
            return int(np.round(value))
        elif isinstance(value, (np.integer, int)):
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

        if isinstance(value, (np.floating, float, np.integer, int)):
            return float(value)
        else:
            try:
                return float(value)
            except (ValueError, TypeError):
                logger.warning(f"Could not convert {value} of type {type(value)} to float, using {default}")
                return default

    def _get_multiindex_id(self, idx) -> int:
        """Retrieve or create multiindex_id based on index values"""
        # Ensure index is a tuple of the correct length (10 elements)
        # Pad with None if the index is shorter (e.g., during loading)
        if len(idx) < 10:
            idx = tuple(list(idx) + [None] * (10 - len(idx)))
        elif len(idx) > 10:
            idx = tuple(idx[:10]) # Take only the first 10 elements if longer
        else:
            idx = tuple(idx)

        # Extract components, handling potential None values
        barcode, artist, album, cover_type, price_category, release_type, \
            recording_decade, release_decade, style, record_year = idx

        return get_or_create_multiindex_id(
            barcode=str(barcode) if barcode is not None else 'UNKNOWN',
            artist=str(artist) if artist is not None else 'UNKNOWN',
            album=str(album) if album is not None else 'UNKNOWN',
            cover_type=str(cover_type) if cover_type is not None else 'UNKNOWN',
            price_category=str(price_category) if price_category is not None else 'UNKNOWN',
            release_type=str(release_type) if release_type is not None else 'UNKNOWN',
            recording_decade=str(recording_decade) if recording_decade is not None else 'UNKNOWN',
            release_decade=str(release_decade) if release_decade is not None else 'UNKNOWN',
            style=str(style) if style is not None else 'UNKNOWN',
            record_year=self._convert_to_int(record_year, default=0),
            connection=self.db_conn # Pass connection
        )

    def load_features(
            self,
            start_date: str | None = None,
            end_date: str | None = None,
            feature_types: list[str] | None = None
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

    def _build_multiindex_from_mapping(self, multiindex_ids: list[int]) -> pd.MultiIndex:
        """Build a pandas MultiIndex from dim_multiindex_mapping using IDs."""
        if not multiindex_ids:
            return pd.MultiIndex(levels=[[]]*10, codes=[[]]*10, names=[
                'barcode', 'artist', 'album', 'cover_type', 'price_category',
                'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
            ])

        placeholders = ', '.join('?' * len(multiindex_ids))
        query = f"""
        SELECT barcode, artist, album, cover_type, price_category, release_type, 
               recording_decade, release_decade, style, record_year
        FROM dim_multiindex_mapping
        WHERE multiindex_id IN ({placeholders})
        ORDER BY multiindex_id -- Ensure consistent order for rebuilding index
        """

        mapping_data = execute_query(query, tuple(multiindex_ids), fetchall=True, connection=self.db_conn)

        if not mapping_data:
            return pd.MultiIndex(levels=[[]]*10, codes=[[]]*10, names=[
                'barcode', 'artist', 'album', 'cover_type', 'price_category',
                'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
            ])

        # Convert list of dicts to list of tuples for MultiIndex creation
        index_tuples = [tuple(row.values()) for row in mapping_data]

        return pd.MultiIndex.from_tuples(index_tuples, names=[
            'barcode', 'artist', 'album', 'cover_type', 'price_category',
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ])

    def _load_feature(self, feature_type: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame | None:
        """Load feature from database using configuration"""
        config = self._get_feature_config().get(feature_type)
        if not config:
            logger.warning(f"Unknown feature type '{feature_type}'")
            return None

        table = config['table']

        # Build query with standardized column names
        query = f"SELECT multiindex_id, data_date, value FROM {table}"
        params = []

        # Add date filters
        if start_date:
            query += " WHERE data_date >= ?"
            params.append(start_date)

        if end_date:
            query += f"{' AND' if start_date else ' WHERE'} data_date <= ?"
            params.append(end_date)

        # For prices, sort by date to get latest price later
        if feature_type == 'prices':
            query += " ORDER BY multiindex_id, data_date DESC"

        # Execute query
        data = execute_query(query, tuple(params), fetchall=True, connection=self.db_conn)
        if not data:
            return None

        # Process results with standardized column names
        df = pd.DataFrame(data)
        df['data_date'] = pd.to_datetime(df['data_date'])

        # Handle different processing for different feature types
        if feature_type == 'prices':
            # Get the latest price for each multiindex_id
            latest_prices = df.loc[df.groupby('multiindex_id')['data_date'].idxmax()]

            # Rebuild the original MultiIndex
            original_index = self._build_multiindex_from_mapping(latest_prices['multiindex_id'].tolist())

            # Create a DataFrame with the correct index and the price column
            result_df = pd.DataFrame({'prices': latest_prices['value'].values}, index=original_index)

        elif feature_type == 'stock':
            # Pivot table to get dates as columns
            pivot_df = df.pivot_table(index='multiindex_id', columns='data_date', values='value')

            # Rebuild the original MultiIndex
            original_index = self._build_multiindex_from_mapping(pivot_df.index.tolist())
            pivot_df.index = original_index
            result_df = pivot_df

        else:  # 'sales' or 'change'
            # For features with date in index
            all_ids = df['multiindex_id'].unique().tolist()
            full_index = self._build_multiindex_from_mapping(all_ids)

            # Map multiindex_id back to the full index tuple
            id_to_tuple_map = {id_val: index_tuple for id_val, index_tuple in zip(all_ids, full_index, strict=False)}
            df['full_index'] = df['multiindex_id'].map(id_to_tuple_map)

            # Convert to the required structure with date in index
            # Pivot with full index information
            pivot_df = df.pivot_table(index='full_index', columns='data_date', values='value')

            # Stack to get MultiIndex format
            stacked_df = pivot_df.stack().reset_index()
            column_name = feature_type  # 'sales' or 'change'
            stacked_df.rename(columns={0: column_name, 'data_date': '_date'}, inplace=True)

            # Create the final MultiIndex (date, barcode, artist, ...)
            final_index = pd.MultiIndex.from_tuples(
                [(row['_date'], *row['full_index']) for _, row in stacked_df.iterrows()],
                names=['_date'] + list(full_index.names)
            )

            result_df = pd.DataFrame({column_name: stacked_df[column_name].values}, index=final_index)

        return result_df


class FeatureStoreFactory:
    """Factory for creating feature store instances"""

    @staticmethod
    def get_store(store_type: str = 'sql', run_id: int | None = None, **kwargs) -> Any:
        """Get a feature store instance based on type"""
        if store_type == 'sql':
            # Pass any additional kwargs (like connection) to the SQL store
            return SQLFeatureStore(run_id=run_id, **kwargs)
        # Add other store types here if needed (e.g., 'file', 'redis')
        # elif store_type == 'file':
        #     return FileFeatureStore(...)
        else:
            raise ValueError(f"Unsupported feature store type: {store_type}")

def save_features(features: dict[str, pd.DataFrame],
                 cutoff_date: str,
                 source_files: str,
                 store_type: str = 'sql', **kwargs) -> int:
    """Helper function to save features using a specific store type"""

    # Use an existing connection if provided, otherwise use db_transaction
    if 'connection' in kwargs and kwargs['connection'] is not None:
        conn = kwargs.pop('connection')
        store = FeatureStoreFactory.get_store(store_type=store_type, connection=conn, **kwargs)
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
            store = FeatureStoreFactory.get_store(store_type=store_type, connection=conn, **kwargs)
            run_id = store.create_run(cutoff_date, source_files)
            store.save_features(features)
            store.complete_run()
            # COMMIT/ROLLBACK handled by db_transaction context manager
        return run_id

def load_features(store_type: str = 'sql',
                  start_date: str | None = None,
                  end_date: str | None = None,
                  feature_types: list[str] | None = None,
                  **kwargs) -> dict[str, pd.DataFrame]:
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
    return store.load_features(start_date=start_date, end_date=end_date, feature_types=feature_types)
