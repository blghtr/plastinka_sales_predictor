import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sqlite3
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from deployment.app.db.feature_storage import SQLFeatureStore, FeatureStoreFactory, save_features
from deployment.app.db.schema import SCHEMA_SQL # Import schema SQL

class TestFeatureStorage(unittest.TestCase):
    """Test class for feature_storage.py data type conversion and storage."""
    
    def setUp(self):
        """Set up test environment with a temporary SQLite database."""
        # Create a temporary database file path
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
        
        # Create a single connection for the test case
        self.conn = sqlite3.connect(self.temp_db_file)
        self.conn.row_factory = sqlite3.Row # Use Row factory for easier access
        self.cursor = self.conn.cursor()
        
        # Create schema directly on this connection
        self._create_test_schema()
        
        # Create feature store instance, passing the connection
        self.store = SQLFeatureStore(run_id=1, connection=self.conn)
    
    def tearDown(self):
        """Clean up after tests are complete."""
        # Close the connection
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
        
        # Delete the temporary database file
        if os.path.exists(self.temp_db_file):
            os.unlink(self.temp_db_file)
    
    def _create_test_schema(self):
        """Create the required tables for testing using SCHEMA_SQL."""
        # Execute the imported schema SQL
        self.cursor.executescript(SCHEMA_SQL)
        
        # Insert initial test data required for foreign key constraints
        self.cursor.execute("""
        INSERT INTO dim_multiindex_mapping (multiindex_id, barcode, artist, album, cover_type, price_category, 
                                            release_type, recording_decade, release_decade, style, record_year)
        VALUES (1, '1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                'Studio', '2010s', '2010s', 'Rock', 2015)
        """)
        
        # Insert a test processing run
        self.cursor.execute("""
        INSERT INTO processing_runs (run_id, start_time, status, cutoff_date, source_files)
        VALUES (1, '2023-01-01T00:00:00', 'running', '2023-01-01', 'test_files.csv')
        """)
        
        # Commit changes
        self.conn.commit()
    
    def test_convert_to_int(self):
        """Test the _convert_to_int helper method."""
        # Test with integer input
        self.assertEqual(self.store._convert_to_int(5), 5)
        
        # Test with float input
        self.assertEqual(self.store._convert_to_int(5.7), 6)  # Should round up
        self.assertEqual(self.store._convert_to_int(5.3), 5)  # Should round down
        
        # Test with numpy float64
        self.assertEqual(self.store._convert_to_int(np.float64(5.7)), 6)
        
        # Test with string input
        self.assertEqual(self.store._convert_to_int("5"), 5)
        
        # Test with invalid string
        self.assertEqual(self.store._convert_to_int("invalid"), 0)
        
        # Test with None/NaN
        self.assertEqual(self.store._convert_to_int(None), 0)
        self.assertEqual(self.store._convert_to_int(np.nan), 0)
        self.assertEqual(self.store._convert_to_int(pd.NA), 0)
        
        # Test with custom default
        self.assertEqual(self.store._convert_to_int(np.nan, default=-1), -1)
    
    def test_convert_to_float(self):
        """Test the _convert_to_float helper method."""
        # Test with float input
        self.assertEqual(self.store._convert_to_float(5.7), 5.7)
        
        # Test with integer input
        self.assertEqual(self.store._convert_to_float(5), 5.0)
        
        # Test with numpy float64
        self.assertEqual(self.store._convert_to_float(np.float64(5.7)), 5.7)
        
        # Test with string input
        self.assertEqual(self.store._convert_to_float("5.7"), 5.7)
        
        # Test with invalid string
        self.assertEqual(self.store._convert_to_float("invalid"), 0.0)
        
        # Test with None/NaN
        self.assertEqual(self.store._convert_to_float(None), 0.0)
        self.assertEqual(self.store._convert_to_float(np.nan), 0.0)
        self.assertEqual(self.store._convert_to_float(pd.NA), 0.0)
        
        # Test with custom default
        self.assertEqual(self.store._convert_to_float(np.nan, default=-1.5), -1.5)
    
    def test_convert_to_date_str(self):
        """Test the _convert_to_date_str helper method."""
        # Test with datetime
        test_date = datetime(2023, 1, 15)
        self.assertEqual(self.store._convert_to_date_str(test_date), "2023-01-15")
        
        # Test with string date
        self.assertEqual(self.store._convert_to_date_str("2023-01-15"), "2023-01-15")
        
        # Test with invalid string date (should return the string as is)
        self.assertEqual(self.store._convert_to_date_str("invalid-date"), "invalid-date")
        
        # Test with non-date object
        self.assertEqual(self.store._convert_to_date_str(123), "123")

    def test_save_stock_feature_with_mixed_types(self):
        """Test saving stock feature with mixed data types."""
        # Create a test DataFrame with mixed data types
        today = datetime.now().date()
        yesterday = (datetime.now() - timedelta(days=1)).date()
        
        # Create index tuple that matches the test data
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        index = pd.MultiIndex.from_tuples([idx_tuple], names=[
            'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ])
        
        # Create DataFrame with integer, float, string, and NaN values
        df = pd.DataFrame({
            today: [10],             # int
            yesterday: [15.7],       # float
            '2023-01-01': ['20'],    # string 
            '2023-01-02': [np.nan],  # NaN
        }, index=index)
        
        # Save the feature (uses self.store with the shared connection)
        self.store._save_stock_feature(df)
        
        # Verify data was saved correctly using the test cursor
        query = "SELECT snapshot_date, quantity FROM fact_stock WHERE multiindex_id = 1 ORDER BY snapshot_date"
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Check that we have 4 rows as expected
        self.assertEqual(len(results), 4)
        
        # Check individual values (results are tuples)
        self.assertEqual(results[0][0], '2023-01-01') # date
        self.assertEqual(results[0][1], 20)         # quantity
        
        self.assertEqual(results[1][0], '2023-01-02')
        self.assertEqual(results[1][1], 0)
        
        self.assertEqual(results[2][0], yesterday.strftime('%Y-%m-%d'))
        self.assertEqual(results[2][1], 16)
        
        self.assertEqual(results[3][0], today.strftime('%Y-%m-%d'))
        self.assertEqual(results[3][1], 10)
    
    def test_save_prices_feature_with_mixed_types(self):
        """Test saving prices feature with mixed data types."""
        # Create a multi-index matching the test data
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        idx = pd.MultiIndex.from_tuples([idx_tuple], names=[
            'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ])
        
        # Create DataFrame with various price values
        df = pd.DataFrame({'prices': [123.45]}, index=idx) # float price
        df_int = pd.DataFrame({'prices': [100]}, index=idx) # integer
        df_str = pd.DataFrame({'prices': ['99.99']}, index=idx) # string
        df_nan = pd.DataFrame({'prices': [np.nan]}, index=idx) # NaN
        
        # Save each variant using the test store
        self.store._save_prices_feature(df)
        self.store._save_prices_feature(df_int, append=True) # Appending doesn't make sense for prices, but test behavior
        self.store._save_prices_feature(df_str, append=True)
        self.store._save_prices_feature(df_nan, append=True)
        
        # Verify data was saved correctly using the test cursor
        # Prices are saved with INSERT OR REPLACE, so only the last one should exist
        # The date used is the current date when saving
        today_str = datetime.now().strftime('%Y-%m-%d')
        query = "SELECT price_date, price FROM fact_prices WHERE multiindex_id = 1"
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        
        # Should have only 1 record (the last one written)
        self.assertEqual(len(results), 1)
        
        # Check date and value (should be from the NaN save, resulting in 0.0)
        self.assertEqual(results[0][0], today_str)
        self.assertEqual(results[0][1], 0.0)
    
    def test_handle_row_with_error(self):
        """Test the error handling wrapper for row processing."""
        def operation_that_fails(idx, row, feature_type):
            raise ValueError("Test error")
        
        def operation_that_succeeds(idx, row, feature_type):
            return True
        
        # Should return None when operation fails
        result = self.store._handle_row(operation_that_fails, "test_idx", {}, "test_feature")
        self.assertIsNone(result)
        
        # Should return operation result when it succeeds
        result = self.store._handle_row(operation_that_succeeds, "test_idx", {}, "test_feature")
        self.assertTrue(result)
    
    def test_get_multiindex_id(self):
        """Test the _get_multiindex_id method with various index formats."""
        # Test with tuple matching existing data
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        result = self.store._get_multiindex_id(idx_tuple)
        self.assertEqual(result, 1)  # Should match our pre-inserted test data
        
        # Test with list
        idx_list = list(idx_tuple)
        result = self.store._get_multiindex_id(idx_list)
        self.assertEqual(result, 1)
        
        # Test with pandas MultiIndex (pass the tuple from it)
        multi_idx = pd.MultiIndex.from_tuples([idx_tuple])
        result = self.store._get_multiindex_id(multi_idx[0]) 
        self.assertEqual(result, 1)
        
        # Test with missing values - should create a new multiindex
        partial_idx = ('999', 'New Artist', 'New Album', None, None, None, None, None, None, None)
        result = self.store._get_multiindex_id(partial_idx)
        self.assertGreater(result, 1)  # Should be a new ID > 1
        
        # Verify the new entry was created
        self.cursor.execute("SELECT * FROM dim_multiindex_mapping WHERE multiindex_id = ?", (result,))
        new_entry = self.cursor.fetchone()
        self.assertIsNotNone(new_entry)
        self.assertEqual(new_entry['barcode'], '999')
        self.assertEqual(new_entry['artist'], 'New Artist')
        self.assertEqual(new_entry['album'], 'New Album')
        self.assertEqual(new_entry['record_year'], 0) # Default for missing year

class TestFeatureStoreFactory(unittest.TestCase):
    """Test class for the FeatureStoreFactory."""
    
    def test_get_store_sql(self):
        """Test that the factory returns a SQLFeatureStore instance when requested."""
        # Test needs a real connection for SQLFeatureStore init
        conn = sqlite3.connect(':memory:')
        store = FeatureStoreFactory.get_store(store_type='sql', run_id=123, connection=conn)
        self.assertIsInstance(store, SQLFeatureStore)
        self.assertEqual(store.run_id, 123)
        self.assertEqual(store.db_conn, conn)
        conn.close()
    
    def test_get_store_default(self):
        """Test that the factory defaults to SQLFeatureStore when no type is specified."""
        # Test needs a real connection for SQLFeatureStore init
        conn = sqlite3.connect(':memory:')
        store = FeatureStoreFactory.get_store(run_id=456, connection=conn)
        self.assertIsInstance(store, SQLFeatureStore)
        self.assertEqual(store.run_id, 456)
        self.assertEqual(store.db_conn, conn)
        conn.close()
    
    def test_get_store_unsupported(self):
        """Test that the factory raises a ValueError for unsupported store types."""
        with self.assertRaises(ValueError):
            FeatureStoreFactory.get_store(store_type='unsupported_type')

class TestSaveFeaturesFunction(unittest.TestCase):
    """Test class for the save_features helper function."""
    
    @patch('deployment.app.db.feature_storage.SQLFeatureStore')
    def test_save_features_creates_store(self, mock_store_class):
        """Test that save_features creates a store with the factory."""
        # Setup mock
        mock_instance = MagicMock()
        mock_instance.create_run.return_value = 789
        mock_store_class.return_value = mock_instance
        
        # Create test data
        features = {
            'stock': pd.DataFrame({'2023-01-01': [10]})
        }
        
        # Call function
        run_id = save_features(
            features=features,
            cutoff_date='2023-01-01',
            source_files='test.csv',
            store_type='sql'
        )
        
        # Assertions
        self.assertEqual(run_id, 789)
        mock_instance.create_run.assert_called_once_with(
            '2023-01-01', # cutoff_date as positional
            'test.csv'    # source_files as positional
        )
        mock_instance.save_features.assert_called_once_with(features)
        mock_instance.complete_run.assert_called_once()

if __name__ == '__main__':
    unittest.main() 