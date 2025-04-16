import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from deployment.app.db.database import (
    execute_query, execute_many, get_or_create_multiindex_id,
    create_processing_run, update_processing_run
)

class SQLFeatureStore:
    """Store for saving and loading pandas DataFrames to/from SQL database"""
    
    def __init__(self, run_id: Optional[int] = None):
        self.run_id = run_id
        
    def create_run(self, cutoff_date: str, source_files: str) -> int:
        """Create a new processing run and store its ID"""
        self.run_id = create_processing_run(
            start_time=datetime.now(),
            status="running",
            cutoff_date=cutoff_date,
            source_files=source_files
        )
        return self.run_id
        
    def complete_run(self, status: str = "completed") -> None:
        """Mark the current run as completed"""
        if self.run_id:
            update_processing_run(
                run_id=self.run_id,
                status=status,
                end_time=datetime.now()
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
                status="features_saved"
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
        elif feature_type == 'availability' or feature_type == 'confidence':
            self._save_computed_feature(feature_type, df, append)
        else:
            print(f"Warning: Unknown feature type '{feature_type}'")
    
    def _save_stock_feature(self, df: pd.DataFrame, append: bool = False) -> None:
        """Save stock feature to fact_stock table"""
        # If not appending, clear existing data for this run
        if not append and self.run_id:
            execute_query("DELETE FROM fact_stock WHERE 1=1")
            
        # Convert MultiIndex DataFrame to list of tuples for batch insert
        params_list = []
        
        for idx, row in df.iterrows():
            # Get or create multiindex_id
            multiindex_id = self._get_multiindex_id(idx)
            
            # Extract date and value from row
            for col, value in row.items():
                # Convert timestamp to date string if needed
                date_str = col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col)
                
                params_list.append((
                    multiindex_id,
                    date_str,
                    int(value) if not pd.isna(value) else 0
                ))
                
        # Batch insert
        if params_list:
            execute_many(
                """
                INSERT OR REPLACE INTO fact_stock 
                (multiindex_id, snapshot_date, quantity)
                VALUES (?, ?, ?)
                """,
                params_list
            )
    
    def _save_prices_feature(self, df: pd.DataFrame, append: bool = False) -> None:
        """Save prices feature to fact_prices table"""
        # If not appending, clear existing data for this run
        if not append and self.run_id:
            execute_query("DELETE FROM fact_prices WHERE 1=1")
            
        # Convert DataFrame to list of tuples for batch insert
        params_list = []
        
        for idx, row in df.iterrows():
            # Get or create multiindex_id
            multiindex_id = self._get_multiindex_id(idx)
            
            # Extract date (current date) and price value
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            params_list.append((
                multiindex_id,
                current_date,
                float(row['prices']) if not pd.isna(row['prices']) else 0.0
            ))
                
        # Batch insert
        if params_list:
            execute_many(
                """
                INSERT OR REPLACE INTO fact_prices 
                (multiindex_id, price_date, price)
                VALUES (?, ?, ?)
                """,
                params_list
            )
    
    def _save_sales_feature(self, df: pd.DataFrame, append: bool = False) -> None:
        """Save sales feature to fact_sales table"""
        # If not appending, clear existing data for this run
        if not append and self.run_id:
            execute_query("DELETE FROM fact_sales WHERE 1=1")
            
        # Convert DataFrame to list of tuples for batch insert
        params_list = []
        
        for idx, row in df.iterrows():
            # For sales, the first index level is the date
            date_str = idx[0].strftime('%Y-%m-%d') if hasattr(idx[0], 'strftime') else str(idx[0])
            
            # Get or create multiindex_id using the remaining index levels
            multiindex_id = self._get_multiindex_id(idx[1:])
            
            params_list.append((
                multiindex_id,
                date_str,
                int(row['sales']) if not pd.isna(row['sales']) else 0
            ))
                
        # Batch insert
        if params_list:
            execute_many(
                """
                INSERT OR REPLACE INTO fact_sales 
                (multiindex_id, sale_date, quantity)
                VALUES (?, ?, ?)
                """,
                params_list
            )
    
    def _save_change_feature(self, df: pd.DataFrame, append: bool = False) -> None:
        """Save stock change feature to fact_stock_changes table"""
        # If not appending, clear existing data for this run
        if not append and self.run_id:
            execute_query("DELETE FROM fact_stock_changes WHERE 1=1")
            
        # Convert DataFrame to list of tuples for batch insert
        params_list = []
        
        for idx, row in df.iterrows():
            # For changes, the first index level is the date
            date_str = idx[0].strftime('%Y-%m-%d') if hasattr(idx[0], 'strftime') else str(idx[0])
            
            # Get or create multiindex_id using the remaining index levels
            multiindex_id = self._get_multiindex_id(idx[1:])
            
            params_list.append((
                multiindex_id,
                date_str,
                int(row['change']) if not pd.isna(row['change']) else 0
            ))
                
        # Batch insert
        if params_list:
            execute_many(
                """
                INSERT OR REPLACE INTO fact_stock_changes 
                (multiindex_id, change_date, quantity_change)
                VALUES (?, ?, ?)
                """,
                params_list
            )
    
    def _save_computed_feature(self, feature_type: str, df: pd.DataFrame, append: bool = False) -> None:
        """Save computed feature (availability or confidence) to computed_features table"""
        # If not appending, clear existing data for this run
        if not append and self.run_id:
            execute_query(
                "DELETE FROM computed_features WHERE feature_type = ? AND run_id = ?", 
                (feature_type, self.run_id)
            )
            
        # Convert DataFrame to list of tuples for batch insert
        params_list = []
        
        for col_idx, col in enumerate(df.columns):
            multiindex_id = self._get_multiindex_id(col)
            
            for row_idx, date in enumerate(df.index):
                date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
                value = df.iloc[row_idx, col_idx]
                
                params_list.append((
                    multiindex_id,
                    date_str,
                    feature_type,
                    float(value) if not pd.isna(value) else 0.0,
                    self.run_id
                ))
                
        # Batch insert
        if params_list:
            execute_many(
                """
                INSERT OR REPLACE INTO computed_features 
                (multiindex_id, feature_date, feature_type, feature_value, run_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                params_list
            )
    
    def _get_multiindex_id(self, idx) -> int:
        """Extract fields from index and get/create multiindex ID"""
        # Convert tuple or pandas.Index to list for easier handling
        idx_values = list(idx) if isinstance(idx, (tuple, list)) else [idx]
        
        # Extract each component with proper defaults
        barcode = str(idx_values[0]) if len(idx_values) > 0 else ""
        artist = str(idx_values[1]) if len(idx_values) > 1 else ""
        album = str(idx_values[2]) if len(idx_values) > 2 else ""
        cover_type = str(idx_values[3]) if len(idx_values) > 3 else ""
        price_category = str(idx_values[4]) if len(idx_values) > 4 else ""
        release_type = str(idx_values[5]) if len(idx_values) > 5 else ""
        recording_decade = str(idx_values[6]) if len(idx_values) > 6 else ""
        release_decade = str(idx_values[7]) if len(idx_values) > 7 else ""
        style = str(idx_values[8]) if len(idx_values) > 8 else ""
        
        # Special handling for record_year (must be integer)
        try:
            record_year = int(idx_values[9]) if len(idx_values) > 9 else 0
        except (ValueError, TypeError):
            record_year = 0
            
        # Get or create the multiindex ID
        return get_or_create_multiindex_id(
            barcode, artist, album, cover_type, price_category,
            release_type, recording_decade, release_decade, style, record_year
        )
    
    def load_features(self, cutoff_date: Optional[str] = None, 
                     run_id: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load features from SQL database"""
        # Use provided run_id or current run_id
        run_id = run_id or self.run_id
        
        # Load individual feature types
        features = {}
        
        # Load stock data
        stock_df = self._load_stock_feature(cutoff_date, run_id)
        if stock_df is not None and not stock_df.empty:
            features['stock'] = stock_df
            
        # Load prices data
        prices_df = self._load_prices_feature(cutoff_date, run_id)
        if prices_df is not None and not prices_df.empty:
            features['prices'] = prices_df
            
        # Load sales data
        sales_df = self._load_sales_feature(cutoff_date, run_id)
        if sales_df is not None and not sales_df.empty:
            features['sales'] = sales_df
            
        # Load change data
        change_df = self._load_change_feature(cutoff_date, run_id)
        if change_df is not None and not change_df.empty:
            features['change'] = change_df
            
        # Load computed features (availability and confidence)
        availability_df = self._load_computed_feature('availability', cutoff_date, run_id)
        if availability_df is not None and not availability_df.empty:
            features['availability'] = availability_df
            
        confidence_df = self._load_computed_feature('confidence', cutoff_date, run_id)
        if confidence_df is not None and not confidence_df.empty:
            features['confidence'] = confidence_df
            
        return features
    
    def _load_stock_feature(self, cutoff_date: Optional[str] = None, 
                           run_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load stock feature from fact_stock table"""
        # Construct query to get stock data with multiindex
        query = """
        SELECT m.*, s.snapshot_date, s.quantity
        FROM fact_stock s
        JOIN dim_multiindex_mapping m ON s.multiindex_id = m.multiindex_id
        """
        
        # Add cutoff date filter if provided
        params = []
        if cutoff_date:
            query += " WHERE s.snapshot_date <= ?"
            params.append(cutoff_date)
            
        # Execute query
        conn = self._get_db_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return None
            
        # Convert to desired format with MultiIndex
        index_cols = [
            'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ]
        
        # Pivot to get dates as columns
        pivot_df = df.pivot(
            index=index_cols,
            columns='snapshot_date',
            values='quantity'
        )
        
        return pivot_df
    
    def _load_prices_feature(self, cutoff_date: Optional[str] = None, 
                            run_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load prices feature from fact_prices table"""
        # Construct query to get prices data with multiindex
        query = """
        SELECT m.*, p.price
        FROM fact_prices p
        JOIN dim_multiindex_mapping m ON p.multiindex_id = m.multiindex_id
        """
        
        # Add cutoff date filter if provided
        params = []
        if cutoff_date:
            query += " WHERE p.price_date <= ?"
            params.append(cutoff_date)
            
        # Execute query
        conn = self._get_db_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return None
            
        # Convert to desired format with MultiIndex
        index_cols = [
            'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ]
        
        # Set index and rename price column
        result_df = df[index_cols + ['price']].set_index(index_cols)
        result_df.columns = ['prices']
        
        return result_df
    
    def _load_sales_feature(self, cutoff_date: Optional[str] = None, 
                           run_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load sales feature from fact_sales table"""
        # Construct query to get sales data with multiindex
        query = """
        SELECT m.*, s.sale_date, s.quantity as sales
        FROM fact_sales s
        JOIN dim_multiindex_mapping m ON s.multiindex_id = m.multiindex_id
        """
        
        # Add cutoff date filter if provided
        params = []
        if cutoff_date:
            query += " WHERE s.sale_date <= ?"
            params.append(cutoff_date)
            
        # Execute query
        conn = self._get_db_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return None
            
        # Convert sale_date to datetime
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        
        # Convert to desired format with MultiIndex including date
        index_cols = [
            'sale_date', 'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ]
        
        # Set index
        result_df = df[index_cols + ['sales']].set_index(index_cols)
        
        return result_df
    
    def _load_change_feature(self, cutoff_date: Optional[str] = None, 
                            run_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load change feature from fact_stock_changes table"""
        # Construct query to get stock changes data with multiindex
        query = """
        SELECT m.*, c.change_date, c.quantity_change as change
        FROM fact_stock_changes c
        JOIN dim_multiindex_mapping m ON c.multiindex_id = m.multiindex_id
        """
        
        # Add cutoff date filter if provided
        params = []
        if cutoff_date:
            query += " WHERE c.change_date <= ?"
            params.append(cutoff_date)
            
        # Execute query
        conn = self._get_db_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return None
            
        # Convert change_date to datetime
        df['change_date'] = pd.to_datetime(df['change_date'])
        
        # Convert to desired format with MultiIndex including date
        index_cols = [
            'change_date', 'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ]
        
        # Set index
        result_df = df[index_cols + ['change']].set_index(index_cols)
        
        return result_df
    
    def _load_computed_feature(self, feature_type: str, cutoff_date: Optional[str] = None, 
                              run_id: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load computed feature from computed_features table"""
        # Construct query to get computed features with multiindex
        query = """
        SELECT m.*, f.feature_date, f.feature_value
        FROM computed_features f
        JOIN dim_multiindex_mapping m ON f.multiindex_id = m.multiindex_id
        WHERE f.feature_type = ?
        """
        
        # Add filters
        params = [feature_type]
        
        if cutoff_date:
            query += " AND f.feature_date <= ?"
            params.append(cutoff_date)
            
        if run_id:
            query += " AND f.run_id = ?"
            params.append(run_id)
            
        # Execute query
        conn = self._get_db_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return None
            
        # Convert feature_date to datetime
        df['feature_date'] = pd.to_datetime(df['feature_date'])
        
        # Prepare for pivot
        index_cols = [
            'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ]
        
        # Create MultiIndex from index columns
        multi_index = pd.MultiIndex.from_frame(df[index_cols])
        unique_indices = multi_index.drop_duplicates()
        
        # Pivot to get features with dates as index and multiindex as columns
        pivot_df = df.pivot(
            index='feature_date',
            columns=index_cols,
            values='feature_value'
        )
        
        return pivot_df
    
    def _get_db_connection(self):
        """Get SQLite connection with pandas-compatible settings"""
        from sqlite3 import connect
        return connect("deployment/data/plastinka.db")


class FeatureStoreFactory:
    """Factory for creating feature stores"""
    
    @staticmethod
    def get_store(store_type: str = 'sql', run_id: Optional[int] = None) -> Any:
        """Get feature store of specified type"""
        if store_type.lower() == 'sql':
            return SQLFeatureStore(run_id)
        else:
            raise ValueError(f"Unknown feature store type: {store_type}")
            
            
# Function to replace process_data's save_features functionality
def save_features(features: Dict[str, pd.DataFrame], 
                 cutoff_date: str, 
                 source_files: str,
                 store_type: str = 'sql') -> int:
    """Save features to storage and return run ID"""
    # Create feature store
    store = FeatureStoreFactory.get_store(store_type)
    
    # Create processing run
    run_id = store.create_run(cutoff_date, source_files)
    
    # Save features
    store.save_features(features)
    
    # Mark run as complete
    store.complete_run()
    
    return run_id 