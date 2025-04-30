import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sqlite3
import sys
import tempfile
from unittest.mock import patch, MagicMock, ANY

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from deployment.app.db.feature_storage import SQLFeatureStore, FeatureStoreFactory, save_features, load_features
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
        
        # Clean fact tables before each test method
        self.cursor.execute("DELETE FROM fact_stock")
        self.cursor.execute("DELETE FROM fact_prices")
        self.cursor.execute("DELETE FROM fact_sales")
        self.cursor.execute("DELETE FROM fact_stock_changes")
        self.conn.commit()
    
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
        
        # Save data for each date column separately to simulate real usage
        data_to_save = {
            today: 10,             # int
            yesterday: 15.7,       # float
            pd.to_datetime('2023-01-01').date(): '20',    # string 
            pd.to_datetime('2023-01-02').date(): np.nan,  # NaN
        }
        
        for date_col, value in data_to_save.items():
            df_single_date = pd.DataFrame({date_col: [value]}, index=index)
            # Clear previous saves for this test for simplicity, 
            # otherwise INSERT OR REPLACE would just update
            self.conn.execute("DELETE FROM fact_stock WHERE multiindex_id = 1 AND snapshot_date = ?", (date_col.strftime('%Y-%m-%d'),))
            self.conn.commit()
            self.store._save_stock_feature(df_single_date)
            
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
        self.assertEqual(results[2][1], 15.7)
        
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

    def test_connection_management(self):
        """Test that the store manages connections correctly."""
        # 1. Test with provided connection (should not close it)
        store_with_conn = SQLFeatureStore(connection=self.conn)
        self.assertFalse(store_with_conn._conn_created_internally)
        del store_with_conn # Explicitly delete to trigger __del__
        # Check if the original connection is still open (it should be)
        try:
            self.conn.execute("SELECT 1")
        except sqlite3.ProgrammingError as e:
            self.fail(f"Connection should still be open, but got error: {e}")

        # 2. Test without provided connection (should create and close)
        # Mock get_db_connection to return our test db connection for this part
        with patch('deployment.app.db.database.get_db_connection') as mock_get_conn: 
            # Use a separate connection for this test to avoid closing the main one
            temp_conn_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
            temp_conn = sqlite3.connect(temp_conn_path)
            mock_get_conn.return_value = temp_conn
            
            store_internal_conn = SQLFeatureStore()
            self.assertTrue(store_internal_conn._conn_created_internally)
            internal_conn_obj = store_internal_conn.db_conn # Keep a reference
            
            del store_internal_conn # Trigger __del__
            
            # Check if the internally created connection is closed
            with self.assertRaises(sqlite3.ProgrammingError):
                internal_conn_obj.execute("SELECT 1") # Should raise error if closed
            
            # Clean up the temporary file for this specific test
            if os.path.exists(temp_conn_path):
                os.unlink(temp_conn_path)

    def test_create_and_complete_run(self):
        """Test creating and completing a processing run."""
        # Use a store instance without a pre-set run_id
        store = SQLFeatureStore(connection=self.conn)
        
        # Create a run
        run_id = store.create_run(cutoff_date='2023-02-01', source_files='run_test.csv')
        self.assertIsNotNone(run_id)
        self.assertEqual(store.run_id, run_id)
        
        # Verify run in DB
        self.cursor.execute("SELECT status, cutoff_date, source_files FROM processing_runs WHERE run_id = ?", (run_id,))
        run_data = self.cursor.fetchone()
        self.assertIsNotNone(run_data)
        self.assertEqual(run_data['status'], 'running')
        self.assertEqual(run_data['cutoff_date'], '2023-02-01')
        self.assertEqual(run_data['source_files'], 'run_test.csv')
        
        # Complete the run
        store.complete_run(status='success')
        
        # Verify completion in DB
        self.cursor.execute("SELECT status, end_time FROM processing_runs WHERE run_id = ?", (run_id,))
        run_data = self.cursor.fetchone()
        self.assertIsNotNone(run_data)
        self.assertEqual(run_data['status'], 'success')
        self.assertIsNotNone(run_data['end_time'])
        
    def test_save_sales_feature(self):
        """Test saving sales feature data."""
        # Create test data with date in index
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        date1 = pd.to_datetime('2023-03-10')
        date2 = pd.to_datetime('2023-03-11')
        index = pd.MultiIndex.from_tuples([
            (date1, *idx_tuple),
            (date2, *idx_tuple)
        ], names=['date'] + list(self.store._build_multiindex_from_mapping([1]).names)) # Reuse names
        
        df = pd.DataFrame({'sales': [5.0, 3.0]}, index=index)
        
        # Save the feature
        self.store._save_sales_feature(df)
        
        # Verify data
        self.cursor.execute("SELECT sale_date, quantity FROM fact_sales WHERE multiindex_id = 1 ORDER BY sale_date")
        results = self.cursor.fetchall()
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['sale_date'], '2023-03-10')
        self.assertEqual(results[0]['quantity'], 5.0)
        self.assertEqual(results[1]['sale_date'], '2023-03-11')
        self.assertEqual(results[1]['quantity'], 3.0)

    def test_save_change_feature(self):
        """Test saving stock change feature data."""
        # Create test data similar to sales
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        date1 = pd.to_datetime('2023-04-01')
        date2 = pd.to_datetime('2023-04-02')
        index = pd.MultiIndex.from_tuples([
            (date1, *idx_tuple),
            (date2, *idx_tuple)
        ], names=['date'] + list(self.store._build_multiindex_from_mapping([1]).names))
        
        df = pd.DataFrame({'change': [-2.0, 1.0]}, index=index)
        
        # Save the feature
        self.store._save_change_feature(df)
        
        # Verify data
        self.cursor.execute("SELECT change_date, quantity_change FROM fact_stock_changes WHERE multiindex_id = 1 ORDER BY change_date")
        results = self.cursor.fetchall()
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['change_date'], '2023-04-01')
        self.assertEqual(results[0]['quantity_change'], -2.0)
        self.assertEqual(results[1]['change_date'], '2023-04-02')
        self.assertEqual(results[1]['quantity_change'], 1.0)
        
    def test_build_multiindex_from_mapping(self):
        """Test rebuilding the MultiIndex from the database."""
        # Use the existing multiindex_id=1
        multiindex = self.store._build_multiindex_from_mapping([1])
        
        self.assertIsInstance(multiindex, pd.MultiIndex)
        self.assertEqual(len(multiindex), 1)
        expected_names = ['barcode', 'artist', 'album', 'cover_type', 'price_category', 
                          'release_type', 'recording_decade', 'release_decade', 'style', 'record_year']
        self.assertListEqual(list(multiindex.names), expected_names)
        
        # Check values of the first (and only) index tuple
        expected_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                          'Studio', '2010s', '2010s', 'Rock', 2015)
        self.assertTupleEqual(multiindex[0], expected_tuple)
        
        # Test with multiple IDs (add one first)
        new_idx_id = self.store._get_multiindex_id(('987', 'Art2', 'Alb2', 'LP', 'High', 'Live', '2000s', '2000s', 'Jazz', 2005))
        multiindex_multi = self.store._build_multiindex_from_mapping([1, new_idx_id])
        
        self.assertEqual(len(multiindex_multi), 2)
        self.assertTupleEqual(multiindex_multi[0], expected_tuple)
        self.assertTupleEqual(multiindex_multi[1], ('987', 'Art2', 'Alb2', 'LP', 'High', 'Live', '2000s', '2000s', 'Jazz', 2005))

        # Test with empty list
        empty_multiindex = self.store._build_multiindex_from_mapping([])
        self.assertTrue(empty_multiindex.empty)
        self.assertListEqual(list(empty_multiindex.names), expected_names)
        
    def test_load_stock_feature(self):
        """Test loading stock features."""
        # Save some data first (using previously tested save method)
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        index = pd.MultiIndex.from_tuples([idx_tuple], names=[
            'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ])
        date1 = pd.to_datetime('2023-05-01')
        date2 = pd.to_datetime('2023-05-02')
        # Simulate saving data for two different dates
        df_save1 = pd.DataFrame({date1: [10]}, index=index)
        df_save2 = pd.DataFrame({date2: [12]}, index=index)
        self.store._save_stock_feature(df_save1)
        self.store._save_stock_feature(df_save2)
        
        # Load the feature
        loaded_df = self.store._load_stock_feature()
        
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), 1) # Should have 1 row for the unique index
        pd.testing.assert_index_equal(loaded_df.index, index)
        self.assertListEqual(list(loaded_df.columns), [date1, date2])
        self.assertEqual(loaded_df.iloc[0, 0], 10)
        self.assertEqual(loaded_df.iloc[0, 1], 12)
        
        # Test loading with cutoff date
        loaded_df_cutoff = self.store._load_stock_feature(cutoff_date='2023-05-01')
        self.assertListEqual(list(loaded_df_cutoff.columns), [date1])
        self.assertEqual(loaded_df_cutoff.iloc[0, 0], 10)
        
        # Test loading when no data exists
        self.conn.execute("DELETE FROM fact_stock")
        self.conn.commit()
        loaded_empty = self.store._load_stock_feature()
        self.assertIsNone(loaded_empty)

    def test_load_prices_feature(self):
        """Test loading prices features."""
        # Save some price data
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        index = pd.MultiIndex.from_tuples([idx_tuple], names=[
            'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ])
        df_price1 = pd.DataFrame({'prices': [1500.50]}, index=index)
        df_price2 = pd.DataFrame({'prices': [1600.00]}, index=index) # Newer price
        
        self.store._save_prices_feature(df_price1) # Older price first
        self.store._save_prices_feature(df_price2) # Newer price replaces
        
        # Load prices
        loaded_df = self.store._load_prices_feature()
        
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), 1)
        pd.testing.assert_index_equal(loaded_df.index, index)
        self.assertListEqual(list(loaded_df.columns), ['prices'])
        self.assertEqual(loaded_df.iloc[0, 0], 1600.00) # Should load the latest price
        
        # Test loading when no data exists
        self.conn.execute("DELETE FROM fact_prices")
        self.conn.commit()
        loaded_empty = self.store._load_prices_feature()
        self.assertIsNone(loaded_empty)

    def test_load_sales_feature(self):
        """Test loading sales features."""
        # Save some sales data
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        date1 = pd.to_datetime('2023-06-10')
        date2 = pd.to_datetime('2023-06-11')
        mi_names = ['date'] + list(self.store._build_multiindex_from_mapping([1]).names)
        index = pd.MultiIndex.from_tuples([
            (date1, *idx_tuple),
            (date2, *idx_tuple)
        ], names=mi_names)
        df_save = pd.DataFrame({'sales': [5.0, 3.0]}, index=index)
        self.store._save_sales_feature(df_save)

        # Load the feature
        loaded_df = self.store._load_sales_feature()

        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), 2)
        # Compare index carefully due to potential ordering issues
        expected_index = pd.MultiIndex.from_tuples([
            (date1, *idx_tuple),
            (date2, *idx_tuple)
        ], names=mi_names)
        # Sort both before comparing
        loaded_df_sorted = loaded_df.sort_index()
        expected_df_sorted = pd.DataFrame({'sales': [5.0, 3.0]}, index=expected_index).sort_index()
        pd.testing.assert_frame_equal(loaded_df_sorted, expected_df_sorted)

        # Test loading with cutoff date
        loaded_df_cutoff = self.store._load_sales_feature(cutoff_date='2023-06-10')
        self.assertEqual(len(loaded_df_cutoff), 1)
        self.assertEqual(loaded_df_cutoff.index[0][0], date1) # Check date part of index
        self.assertEqual(loaded_df_cutoff.iloc[0, 0], 5.0)
        
        # Test loading when no data exists
        self.conn.execute("DELETE FROM fact_sales")
        self.conn.commit()
        loaded_empty = self.store._load_sales_feature()
        self.assertIsNone(loaded_empty)

    def test_load_change_feature(self):
        """Test loading stock change features."""
        # Save some change data
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        date1 = pd.to_datetime('2023-07-01')
        date2 = pd.to_datetime('2023-07-02')
        mi_names = ['date'] + list(self.store._build_multiindex_from_mapping([1]).names)
        index = pd.MultiIndex.from_tuples([
            (date1, *idx_tuple),
            (date2, *idx_tuple)
        ], names=mi_names)
        df_save = pd.DataFrame({'change': [-2.0, 1.0]}, index=index)
        self.store._save_change_feature(df_save)

        # Load the feature
        loaded_df = self.store._load_change_feature()

        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), 2)
        # Compare index carefully
        expected_index = pd.MultiIndex.from_tuples([
            (date1, *idx_tuple),
            (date2, *idx_tuple)
        ], names=mi_names)
        loaded_df_sorted = loaded_df.sort_index()
        expected_df_sorted = pd.DataFrame({'change': [-2.0, 1.0]}, index=expected_index).sort_index()
        pd.testing.assert_frame_equal(loaded_df_sorted, expected_df_sorted)

        # Test loading with cutoff date
        loaded_df_cutoff = self.store._load_change_feature(cutoff_date='2023-07-01')
        self.assertEqual(len(loaded_df_cutoff), 1)
        self.assertEqual(loaded_df_cutoff.index[0][0], date1)
        self.assertEqual(loaded_df_cutoff.iloc[0, 0], -2.0)

        # Test loading when no data exists
        self.conn.execute("DELETE FROM fact_stock_changes")
        self.conn.commit()
        loaded_empty = self.store._load_change_feature()
        self.assertIsNone(loaded_empty)

    def test_load_features_all_types(self):
        """Test the main load_features method loading all types."""
        # --- Arrange: Save data for all types ---
        idx_tuple = ('1234567890', 'Test Artist', 'Test Album', 'CD', 'Standard', 
                     'Studio', '2010s', '2010s', 'Rock', 2015)
        base_index = pd.MultiIndex.from_tuples([idx_tuple], names=[
            'barcode', 'artist', 'album', 'cover_type', 'price_category', 
            'release_type', 'recording_decade', 'release_decade', 'style', 'record_year'
        ])
        date_index_names = ['date'] + list(base_index.names)
        
        # Stock
        stock_date = pd.to_datetime('2023-08-01')
        df_stock = pd.DataFrame({stock_date: [50]}, index=base_index)
        self.store._save_stock_feature(df_stock)
        
        # Prices
        df_prices = pd.DataFrame({'prices': [2000.0]}, index=base_index)
        self.store._save_prices_feature(df_prices)
        
        # Sales
        sales_date = pd.to_datetime('2023-08-10')
        sales_index = pd.MultiIndex.from_tuples([(sales_date, *idx_tuple)], names=date_index_names)
        df_sales = pd.DataFrame({'sales': [2.0]}, index=sales_index)
        self.store._save_sales_feature(df_sales)
        
        # Change
        change_date = pd.to_datetime('2023-08-11')
        change_index = pd.MultiIndex.from_tuples([(change_date, *idx_tuple)], names=date_index_names)
        df_change = pd.DataFrame({'change': [-1.0]}, index=change_index)
        self.store._save_change_feature(df_change)
        
        # --- Act: Load all features ---
        loaded_features = self.store.load_features()
        
        # --- Assert ---
        self.assertIn('stock', loaded_features)
        self.assertIn('prices', loaded_features)
        self.assertIn('sales', loaded_features)
        self.assertIn('change', loaded_features)
        
        # Check stock
        # Remove column index name generated by pivot before comparing
        loaded_features['stock'].columns.name = None
        pd.testing.assert_frame_equal(
            loaded_features['stock'], 
            df_stock, 
            check_like=True, 
            check_dtype=False
        ) # Use check_like and ignore dtype
        
        # Check prices
        pd.testing.assert_frame_equal(loaded_features['prices'], df_prices, check_dtype=False)
        
        # Check sales
        pd.testing.assert_frame_equal(loaded_features['sales'], df_sales, check_dtype=False)
        
        # Check change
        pd.testing.assert_frame_equal(loaded_features['change'], df_change, check_dtype=False)

        # Test loading with cutoff (only stock and prices should remain)
        loaded_features_cutoff = self.store.load_features(cutoff_date='2023-08-05')
        self.assertIn('stock', loaded_features_cutoff)
        self.assertIn('prices', loaded_features_cutoff) # Prices are latest, not filtered by date here
        self.assertNotIn('sales', loaded_features_cutoff)
        self.assertNotIn('change', loaded_features_cutoff)
        self.assertEqual(len(loaded_features_cutoff['stock'].columns), 1) # Only stock_date <= cutoff
        self.assertEqual(loaded_features_cutoff['stock'].columns[0], stock_date)

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

class TestLoadFeaturesFunction(unittest.TestCase):
    """Test class for the load_features helper function."""
    
    @patch('deployment.app.db.feature_storage.SQLFeatureStore')
    def test_load_features_creates_store_and_calls_load(self, mock_store_class):
        """Test that load_features creates a store and calls its load method."""
        # Setup mock
        mock_instance = MagicMock(spec=SQLFeatureStore) # Use spec for attribute checking
        expected_features = {'stock': pd.DataFrame({'test': [1]})}
        mock_instance.load_features.return_value = expected_features
        mock_store_class.return_value = mock_instance
        
        # Call function
        result = load_features(
            store_type='sql',
            cutoff_date='2023-09-01',
            run_id=999,
            connection=MagicMock() # Provide a mock connection for factory call if needed
        )
        
        # Assertions
        # Factory is called implicitly by load_features
        # Check that SQLFeatureStore was initialized with the correct args within load_features
        # Note: The actual factory call isn't directly checked here, 
        # but that the store instance was created and used correctly.
        mock_store_class.assert_called_once_with(run_id=999, connection=ANY) # Use ANY for connection
        mock_instance.load_features.assert_called_once_with(cutoff_date='2023-09-01', run_id=999)
        self.assertEqual(result, expected_features)

if __name__ == '__main__':
    unittest.main()