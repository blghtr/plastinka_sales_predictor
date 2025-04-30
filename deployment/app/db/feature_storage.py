import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import json
from pathlib import Path
import sqlite3 # Import sqlite3

from deployment.app.db.database import (
    execute_query, execute_many, get_or_create_multiindex_id,
    create_processing_run, update_processing_run
)

class SQLFeatureStore:
    """Store for saving and loading pandas DataFrames to/from SQL database"""
    
    def __init__(self, run_id: Optional[int] = None, connection: sqlite3.Connection = None):
        self.run_id = run_id
        self.db_conn = connection # Store the connection
        self._conn_created_internally = False
        if not self.db_conn:
            # If no connection provided, create one internally (for non-test use)
            from deployment.app.db.database import get_db_connection # Import locally to avoid circular dependency issues
            self.db_conn = get_db_connection()
            self._conn_created_internally = True
        
    def __del__(self):
        # Close the connection only if it was created internally by this instance
        if self._conn_created_internally and self.db_conn:
            self.db_conn.close()

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
    
    def save_features(self, features: Dict[str, pd.DataFrame], append: bool = False) -> None:
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
    
    def _save_feature(self, feature_type: str, df: pd.DataFrame, append: bool = False) -> None:
        """Save a single feature DataFrame to the appropriate SQL table"""
        if feature_type == 'stock':
            self._save_stock_feature(df, append)
        elif feature_type == 'prices':
            self._save_prices_feature(df, append)
        elif feature_type == 'sales':
            self._save_sales_feature(df, append)
        elif feature_type == 'change':
            self._save_change_feature(df, append)
        else:
            print(f"Warning: Unknown feature type '{feature_type}'")
    
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
                print(f"Warning: Could not convert {value} of type {type(value)} to int, using {default}")
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
                print(f"Warning: Could not convert {value} of type {type(value)} to float, using {default}")
                return default
    
    def _handle_row(self, operation_fn, idx, row, *args):
        """Generic error handling wrapper for row processing."""
        try:
            return operation_fn(idx, row, *args)
        except Exception as e:
            feature_type = args[0] if args else "unknown"
            print(f"Error processing {feature_type} data row: {e}")
            return None  # Skip this row
    
    def _create_row_processor(self, feature_type, date_col=None, value_col=None, is_date_in_index=False, value_conversion='float'):
        """
        Создает функцию для обработки строк разных типов данных.
        
        Args:
            feature_type: Тип признака (stock, prices, sales, change)
            date_col: Имя столбца с датой (если дата не в индексе)
            value_col: Имя столбца со значением
            is_date_in_index: Флаг, указывающий, что дата находится в индексе (первый уровень)
            value_conversion: Тип конвертации значения ('int' или 'float')
        
        Returns:
            Функция для обработки строк данного типа признаков
        """
        def process_row(idx, row, *args):
            # Получаем multiindex_id, обрабатывая индекс по-разному в зависимости от is_date_in_index
            if is_date_in_index:
                # Для индексов, где дата - первый уровень (sales, change)
                date_str = self._convert_to_date_str(idx[0])
                multiindex_id = self._get_multiindex_id(idx[1:])
            else:
                # Для индексов без даты в индексе (stock, prices)
                multiindex_id = self._get_multiindex_id(idx)
                
                # Для цен используем текущую дату
                if feature_type == 'prices':
                    date_str = datetime.now().strftime('%Y-%m-%d')
            
            # Получаем значение и конвертируем его в правильный тип
            if feature_type == 'stock':
                # Упрощенная логика для stock с одной колонкой-датой (согласно JSON)
                if not row.empty:
                    col_date = row.index[0]  # Имя единственной колонки (это Timestamp)
                    value = row.iloc[0]      # Значение в этой колонке

                    col_date_str = self._convert_to_date_str(col_date)
                    # Конвертируем значение в float, как указано в JSON
                    value_float = self._convert_to_float(value)

                    # Возвращаем список из одного кортежа
                    return [(
                        multiindex_id,
                        col_date_str,
                        value_float
                    )]
                else:
                    # Обработка случая пустой строки, если необходимо
                    return []
            else:
                # Для других типов данных (prices, sales, change)
                if value_col:
                    # Получаем значение из указанного столбца
                    if value_conversion == 'int':
                        value_converted = self._convert_to_int(row.get(value_col, 0))
                    else:  # 'float'
                        value_converted = self._convert_to_float(row.get(value_col, 0.0))
                    
                    # Для sales, change, prices используем date_str, определенный выше
                    return [(
                        multiindex_id,
                        date_str,
                        value_converted
                    )]
                
            return []
        
        return process_row
    
    def _save_stock_feature(self, df: pd.DataFrame, append: bool = False) -> None:
        """Save stock feature to fact_stock table"""
        # Convert MultiIndex DataFrame to list of tuples for batch insert
        params_list = []
        
        # Создаем процессор строк для stock
        process_stock_row = self._create_row_processor(
            feature_type='stock',
            is_date_in_index=False
        )
        
        # Process each row with error handling
        for idx, row in df.iterrows():
            processed_rows = self._handle_row(process_stock_row, idx, row, 'stock')
            if processed_rows:
                params_list.extend(processed_rows)
        
        # Batch insert data
        if params_list:
            execute_many(
                "INSERT OR REPLACE INTO fact_stock (multiindex_id, snapshot_date, quantity) VALUES (?, ?, ?)",
                params_list,
                connection=self.db_conn # Pass connection
            )
    
    def _save_prices_feature(self, df: pd.DataFrame, append: bool = False) -> None:
        """Save prices feature to fact_prices table"""
        params_list = []
        
        process_prices_row = self._create_row_processor(
            feature_type='prices',
            value_col='prices', # Specify the column name for prices
            is_date_in_index=False,
            value_conversion='float'
        )
        
        for idx, row in df.iterrows():
            processed_rows = self._handle_row(process_prices_row, idx, row, 'prices')
            if processed_rows:
                params_list.extend(processed_rows)
                
        if params_list:
            execute_many(
                "INSERT OR REPLACE INTO fact_prices (multiindex_id, price_date, price) VALUES (?, ?, ?)",
                params_list,
                connection=self.db_conn # Pass connection
            )
            
    def _save_sales_feature(self, df: pd.DataFrame, append: bool = False) -> None:
        """Save sales feature to fact_sales table"""
        params_list = []
        
        # Sales data has date in the first level of the index
        process_sales_row = self._create_row_processor(
            feature_type='sales',
            value_col='sales', # Assuming the column name is 'sales'
            is_date_in_index=True,
            value_conversion='float'
        )
        
        for idx, row in df.iterrows():
            processed_rows = self._handle_row(process_sales_row, idx, row, 'sales')
            if processed_rows:
                params_list.extend(processed_rows)
                
        if params_list:
            execute_many(
                "INSERT OR REPLACE INTO fact_sales (multiindex_id, sale_date, quantity) VALUES (?, ?, ?)",
                params_list,
                connection=self.db_conn # Pass connection
            )
            
    def _save_change_feature(self, df: pd.DataFrame, append: bool = False) -> None:
        """Save stock change feature to fact_stock_changes table"""
        params_list = []
        
        # Stock change data also has date in the first level of the index
        process_change_row = self._create_row_processor(
            feature_type='change',
            value_col='change', # Assuming the column name is 'change'
            is_date_in_index=True,
            value_conversion='float'
        )
        
        for idx, row in df.iterrows():
            processed_rows = self._handle_row(process_change_row, idx, row, 'change')
            if processed_rows:
                params_list.extend(processed_rows)
                
        if params_list:
            execute_many(
                "INSERT OR REPLACE INTO fact_stock_changes (multiindex_id, change_date, quantity_change) VALUES (?, ?, ?)",
                params_list,
                connection=self.db_conn # Pass connection
            )
            
    
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

    def load_features(self, cutoff_date: Optional[str] = None, 
                     run_id: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load all features from SQL database up to a cutoff date or for a specific run"""
        target_run_id = run_id if run_id is not None else self.run_id
        
        features = {
            'stock': self._load_stock_feature(cutoff_date, target_run_id),
            'prices': self._load_prices_feature(cutoff_date, target_run_id),
            'sales': self._load_sales_feature(cutoff_date, target_run_id),
            'change': self._load_change_feature(cutoff_date, target_run_id),
        }
        
        # Remove empty DataFrames
        features = {k: v for k, v in features.items() if v is not None and not v.empty}
        
        return features

    def _build_multiindex_from_mapping(self, multiindex_ids: List[int]) -> pd.MultiIndex:
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

    def _load_stock_feature(self, cutoff_date: Optional[str] = None, 
                           run_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load stock feature from fact_stock table"""
        query = "SELECT multiindex_id, snapshot_date, quantity FROM fact_stock"
        params = []
        
        if cutoff_date:
            query += " WHERE snapshot_date <= ?"
            params.append(cutoff_date)
            
        data = execute_query(query, tuple(params), fetchall=True, connection=self.db_conn)
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
        
        # Pivot table to get dates as columns
        pivot_df = df.pivot_table(index='multiindex_id', columns='snapshot_date', values='quantity')
        
        # Rebuild the original MultiIndex
        original_index = self._build_multiindex_from_mapping(pivot_df.index.tolist())
        pivot_df.index = original_index
        
        return pivot_df

    def _load_prices_feature(self, cutoff_date: Optional[str] = None, 
                            run_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load prices feature from fact_prices table"""
        # Prices usually represent the latest known price, so cutoff might not apply directly
        # We might want the latest price before or on the cutoff date
        # For simplicity, let's just load all for now, filtering can be added if needed
        
        query = "SELECT multiindex_id, price_date, price FROM fact_prices ORDER BY multiindex_id, price_date DESC"
        data = execute_query(query, fetchall=True, connection=self.db_conn)
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        
        # Get the latest price for each multiindex_id
        latest_prices = df.loc[df.groupby('multiindex_id')['price_date'].idxmax()]
        
        # Rebuild the original MultiIndex
        original_index = self._build_multiindex_from_mapping(latest_prices['multiindex_id'].tolist())
        
        # Create a DataFrame with the correct index and the price column
        result_df = pd.DataFrame({'prices': latest_prices['price'].values}, index=original_index)
        
        return result_df

    def _load_sales_feature(self, cutoff_date: Optional[str] = None, 
                           run_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load sales feature from fact_sales table"""
        query = "SELECT multiindex_id, sale_date, quantity FROM fact_sales"
        params = []
        
        if cutoff_date:
            query += " WHERE sale_date <= ?"
            params.append(cutoff_date)
            
        data = execute_query(query, tuple(params), fetchall=True, connection=self.db_conn)
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        
        # Pivot table to get dates as columns
        # We need to get the mapping first to create the full index
        all_ids = df['multiindex_id'].unique().tolist()
        full_index = self._build_multiindex_from_mapping(all_ids)
        
        # Map multiindex_id back to the full index tuple for pivoting
        id_to_tuple_map = {id_val: index_tuple for id_val, index_tuple in zip(all_ids, full_index)}
        df['full_index'] = df['multiindex_id'].map(id_to_tuple_map)
        
        # Pivot with the full index information
        pivot_df = df.pivot_table(index='full_index', columns='sale_date', values='quantity')
        
        # Name the value column
        pivot_df.columns.name = None # Remove name from columns index
        pivot_df.rename_axis(None, axis=1, inplace=True)
        # We need to structure it like the input: index=(date, multiindex...), column='sales'
        stacked_df = pivot_df.stack().reset_index()
        stacked_df.rename(columns={0: 'sales', 'level_0': 'full_index', 'level_1': 'sale_date'}, inplace=True)
        
        # Create the final MultiIndex (date, barcode, artist, ...)
        final_index = pd.MultiIndex.from_tuples(
            [(row['sale_date'], *row['full_index']) for idx, row in stacked_df.iterrows()],
            names=['date'] + list(full_index.names)
        )
        
        result_df = pd.DataFrame({'sales': stacked_df['sales'].values}, index=final_index)
        
        return result_df

    def _load_change_feature(self, cutoff_date: Optional[str] = None, 
                            run_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load stock change feature from fact_stock_changes table"""
        query = "SELECT multiindex_id, change_date, quantity_change FROM fact_stock_changes"
        params = []
        
        if cutoff_date:
            query += " WHERE change_date <= ?"
            params.append(cutoff_date)
            
        data = execute_query(query, tuple(params), fetchall=True, connection=self.db_conn)
        
        if not data:
            return None
            
        df = pd.DataFrame(data)
        df['change_date'] = pd.to_datetime(df['change_date'])
        
        # Similar pivoting and restructuring as sales data
        all_ids = df['multiindex_id'].unique().tolist()
        full_index = self._build_multiindex_from_mapping(all_ids)
        id_to_tuple_map = {id_val: index_tuple for id_val, index_tuple in zip(all_ids, full_index)}
        df['full_index'] = df['multiindex_id'].map(id_to_tuple_map)
        
        pivot_df = df.pivot_table(index='full_index', columns='change_date', values='quantity_change')
        
        stacked_df = pivot_df.stack().reset_index()
        stacked_df.rename(columns={0: 'change', 'level_0': 'full_index', 'level_1': 'change_date'}, inplace=True)
        
        final_index = pd.MultiIndex.from_tuples(
            [(row['change_date'], *row['full_index']) for idx, row in stacked_df.iterrows()],
            names=['date'] + list(full_index.names)
        )
        
        result_df = pd.DataFrame({'change': stacked_df['change'].values}, index=final_index)
        
        return result_df

class FeatureStoreFactory:
    """Factory for creating feature store instances"""
    
    @staticmethod
    def get_store(store_type: str = 'sql', run_id: Optional[int] = None, **kwargs) -> Any:
        """Get a feature store instance based on type"""
        if store_type == 'sql':
            # Pass any additional kwargs (like connection) to the SQL store
            return SQLFeatureStore(run_id=run_id, **kwargs)
        # Add other store types here if needed (e.g., 'file', 'redis')
        # elif store_type == 'file':
        #     return FileFeatureStore(...)
        else:
            raise ValueError(f"Unsupported feature store type: {store_type}")

def save_features(features: Dict[str, pd.DataFrame], 
                 cutoff_date: str, 
                 source_files: str,
                 store_type: str = 'sql', **kwargs) -> int:
    """Helper function to save features using a specific store type"""
    store = FeatureStoreFactory.get_store(store_type=store_type, **kwargs)
    run_id = store.create_run(cutoff_date, source_files)
    store.save_features(features)
    store.complete_run()
    return run_id

def load_features(store_type: str = 'sql', 
                 cutoff_date: Optional[str] = None, 
                 run_id: Optional[int] = None, **kwargs) -> Dict[str, pd.DataFrame]:
    """Helper function to load features using a specific store type"""
    store = FeatureStoreFactory.get_store(store_type=store_type, run_id=run_id, **kwargs)
    return store.load_features(cutoff_date=cutoff_date, run_id=run_id) 