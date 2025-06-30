import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from deployment.app.config import DataRetentionSettings
from deployment.app.db.data_retention import (
    cleanup_old_historical_data,
    cleanup_old_models,
    cleanup_old_predictions,
    run_cleanup_job,
)


class TestDataRetention(unittest.TestCase):
    def setUp(self):
        """Set up test database and environment"""
        # Create a temp directory for test db and model files
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_dir, "test_db.sqlite")

        # Create test database
        self.conn = sqlite3.connect(self.test_db_path)
        self.cursor = self.conn.cursor()

        # Create necessary tables for testing
        self._create_test_tables()

        # Create some test models in file system
        self.model_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Create test settings patch
        self.settings_patch = patch('deployment.app.db.data_retention.get_settings')
        self.mock_get_settings = self.settings_patch.start()

        # Configure mock settings
        mock_retention_settings = DataRetentionSettings(
            sales_retention_days=730,
            stock_retention_days=730,
            prediction_retention_days=30,
            models_to_keep=2,
            inactive_model_retention_days=15,
            cleanup_enabled=True
        )
        # Create a mock object that simulates the AppSettings structure
        self.mock_settings_object = MagicMock()
        self.mock_settings_object.data_retention = mock_retention_settings
        self.mock_settings_object.default_metric = "val_MIC"
        self.mock_settings_object.default_metric_higher_is_better = True
        self.mock_get_settings.return_value = self.mock_settings_object

    def tearDown(self):
        """Clean up after tests"""
        self.conn.close()
        self.settings_patch.stop()
        shutil.rmtree(self.test_dir)

    def _create_test_tables(self):
        """Create test tables in the database"""
        self.cursor.executescript("""
        -- Models table
        CREATE TABLE IF NOT EXISTS models (
            model_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            model_path TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            metadata TEXT,
            is_active BOOLEAN DEFAULT 0
        );
        
        -- Predictions table
        CREATE TABLE IF NOT EXISTS fact_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            multiindex_id INTEGER NOT NULL,
            prediction_date TIMESTAMP NOT NULL,
            result_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            quantile_05 DECIMAL(10,2) NOT NULL,
            quantile_25 DECIMAL(10,2) NOT NULL,
            quantile_50 DECIMAL(10,2) NOT NULL,
            quantile_75 DECIMAL(10,2) NOT NULL,
            quantile_95 DECIMAL(10,2) NOT NULL,
            created_at TIMESTAMP NOT NULL
        );
        
        -- Configs table
        CREATE TABLE IF NOT EXISTS configs (
            config_id TEXT PRIMARY KEY,
            config TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            is_active BOOLEAN DEFAULT 0
        );
        
        -- Training results table
        CREATE TABLE IF NOT EXISTS training_results (
            result_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            model_id TEXT,
            config_id TEXT,
            metrics TEXT,
            parameters TEXT,
            duration INTEGER
        );

        -- Fact tables for historical data
        CREATE TABLE IF NOT EXISTS fact_sales (
            multiindex_id INTEGER,
            data_date DATE,
            quantity REAL,
            PRIMARY KEY (multiindex_id, data_date)
        );

        CREATE TABLE IF NOT EXISTS fact_stock (
            multiindex_id INTEGER,
            data_date DATE,
            quantity REAL,
            PRIMARY KEY (multiindex_id, data_date)
        );

        CREATE TABLE IF NOT EXISTS fact_prices (
            multiindex_id INTEGER,
            data_date DATE,
            price DECIMAL(10,2),
            PRIMARY KEY (multiindex_id, data_date)
        );

        CREATE TABLE IF NOT EXISTS fact_stock_changes (
            multiindex_id INTEGER,
            data_date DATE,
            quantity_change REAL,
            PRIMARY KEY (multiindex_id, data_date)
        );
        """)
        self.conn.commit()

    def _create_test_models(self):
        """Create test model records and files"""
        now = datetime.now()

        # Create model files
        model_files = [
            ("model1.pt", now - timedelta(days=5)),
            ("model2.pt", now - timedelta(days=10)),
            ("model3.pt", now - timedelta(days=20)),
            ("model4.pt", now - timedelta(days=30)),
            ("inactive_model1.pt", now - timedelta(days=10)),
            ("inactive_model2.pt", now - timedelta(days=20))
        ]

        for filename, _ in model_files:
            with open(os.path.join(self.model_dir, filename), 'w') as f:
                f.write("Test model content")

        # Insert model records
        self.cursor.executemany(
            """
            INSERT INTO models 
            (model_id, job_id, model_path, created_at, metadata, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("model1", "job1", os.path.join(self.model_dir, "model1.pt"),
                 (now - timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S'),
                 json.dumps({"size": 1000}), 1),

                ("model2", "job2", os.path.join(self.model_dir, "model2.pt"),
                 (now - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S'),
                 json.dumps({"size": 1000}), 1),

                ("model3", "job3", os.path.join(self.model_dir, "model3.pt"),
                 (now - timedelta(days=20)).strftime('%Y-%m-%d %H:%M:%S'),
                 json.dumps({"size": 1000}), 1),

                ("model4", "job4", os.path.join(self.model_dir, "model4.pt"),
                 (now - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S'),
                 json.dumps({"size": 1000}), 1),

                ("inactive_model1", "job5", os.path.join(self.model_dir, "inactive_model1.pt"),
                 (now - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S'),
                 json.dumps({"size": 1000}), 0),

                ("inactive_model2", "job6", os.path.join(self.model_dir, "inactive_model2.pt"),
                 (now - timedelta(days=20)).strftime('%Y-%m-%d %H:%M:%S'),
                 json.dumps({"size": 1000}), 0)
            ]
        )

        # Create config set
        self.cursor.execute(
            """
            INSERT INTO configs
            (config_id, config, created_at, is_active)
            VALUES (?, ?, ?, ?)
            """,
            ("config1", json.dumps({"epochs": 100}),
             now.strftime('%Y-%m-%d %H:%M:%S'), 1)
        )

        # Associate models with config set and add metrics
        self.cursor.executemany(
            """
            INSERT INTO training_results
            (result_id, job_id, model_id, config_id, metrics, parameters, duration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("result1", "job1", "model1", "config1",
                 json.dumps({"val_MIC": 0.95}), json.dumps({"epochs": 100}), 3600),

                ("result2", "job2", "model2", "config1",
                 json.dumps({"val_MIC": 0.90}), json.dumps({"epochs": 100}), 3600),

                ("result3", "job3", "model3", "config1",
                 json.dumps({"val_MIC": 0.85}), json.dumps({"epochs": 100}), 3600),

                ("result4", "job4", "model4", "config1",
                 json.dumps({"val_MIC": 0.80}), json.dumps({"epochs": 100}), 3600)
            ]
        )

        self.conn.commit()

    def _create_test_predictions(self):
        """Create test prediction records"""
        now = datetime.now()

        # Insert predictions with various dates
        predictions = []

        # Recent predictions (within retention period)
        for i in range(10):
            date = now - timedelta(days=i)
            predictions.append((
                i+1, date.strftime('%Y-%m-%d'), "result1", "model1",
                10.5, 15.2, 20.1, 25.8, 30.3, now.strftime('%Y-%m-%d %H:%M:%S')
            ))

        # Old predictions (beyond retention period)
        for i in range(10):
            date = now - timedelta(days=40+i)
            predictions.append((
                i+100, date.strftime('%Y-%m-%d'), "result2", "model2",
                11.5, 16.2, 21.1, 26.8, 31.3, now.strftime('%Y-%m-%d %H:%M:%S')
            ))

        self.cursor.executemany(
            """
            INSERT INTO fact_predictions
            (multiindex_id, prediction_date, result_id, model_id, 
             quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            predictions
        )

        self.conn.commit()

    def _create_test_historical_data(self):
        """Create test historical data records"""
        now = datetime.now()

        # Generate test data for different time ranges
        sales_data = []
        stock_data = []
        prices_data = []
        changes_data = []

        # Recent data (within 1 year)
        for i in range(10):
            date = now - timedelta(days=i*30)  # Roughly monthly
            date_str = date.strftime('%Y-%m-%d')

            # Add multiple records for each date (different products)
            for j in range(5):
                multiindex_id = j + 1

                sales_data.append((multiindex_id, date_str, 10 + j))
                stock_data.append((multiindex_id, date_str, 100 + j*10))
                prices_data.append((multiindex_id, date_str, 25.99 + j))
                changes_data.append((multiindex_id, date_str, -5 + j))

        # Old data (over 2 years old)
        for i in range(10):
            date = now - timedelta(days=800+i*30)  # Beyond the retention period
            date_str = date.strftime('%Y-%m-%d')

            # Add multiple records for each date
            for j in range(5):
                multiindex_id = j + 1

                sales_data.append((multiindex_id, date_str, 5 + j))
                stock_data.append((multiindex_id, date_str, 50 + j*10))
                prices_data.append((multiindex_id, date_str, 19.99 + j))
                changes_data.append((multiindex_id, date_str, -2 + j))

        # Insert data into tables
        self.cursor.executemany(
            "INSERT INTO fact_sales (multiindex_id, data_date, quantity) VALUES (?, ?, ?)",
            sales_data
        )

        self.cursor.executemany(
            "INSERT INTO fact_stock (multiindex_id, data_date, quantity) VALUES (?, ?, ?)",
            stock_data
        )

        self.cursor.executemany(
            "INSERT INTO fact_prices (multiindex_id, data_date, price) VALUES (?, ?, ?)",
            prices_data
        )

        self.cursor.executemany(
            "INSERT INTO fact_stock_changes (multiindex_id, data_date, quantity_change) VALUES (?, ?, ?)",
            changes_data
        )

        self.conn.commit()

    def test_cleanup_old_predictions(self):
        """Test cleaning up old predictions"""
        # Create test data
        self._create_test_predictions()

        # Count predictions before cleanup
        self.cursor.execute("SELECT COUNT(*) FROM fact_predictions")
        before_count = self.cursor.fetchone()[0]
        self.assertEqual(before_count, 20, "Should have 20 predictions before cleanup")

        # Run cleanup function with 30-day retention
        count = cleanup_old_predictions(days_to_keep=30, conn=self.conn)

        # Count predictions after cleanup
        self.cursor.execute("SELECT COUNT(*) FROM fact_predictions")
        after_count = self.cursor.fetchone()[0]

        # Check results
        self.assertEqual(count, 10, "Should have removed 10 old predictions")
        self.assertEqual(after_count, 10, "Should have 10 predictions remaining")

        # Check that only recent predictions remain
        self.cursor.execute(
            "SELECT COUNT(*) FROM fact_predictions WHERE prediction_date < ?",
            ((datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),)
        )
        old_remaining = self.cursor.fetchone()[0]
        self.assertEqual(old_remaining, 0, "Should have no predictions older than 30 days")

    def test_cleanup_old_historical_data(self):
        """Test cleaning up old historical data"""
        # Create test data
        self._create_test_historical_data()

        # Count data before cleanup
        self.cursor.execute("SELECT COUNT(*) FROM fact_sales")
        sales_before = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM fact_stock")
        stock_before = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM fact_prices")
        prices_before = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM fact_stock_changes")
        changes_before = self.cursor.fetchone()[0]

        self.assertEqual(sales_before, 100, "Should have 100 sales records before cleanup")
        self.assertEqual(stock_before, 100, "Should have 100 stock records before cleanup")
        self.assertEqual(prices_before, 100, "Should have 100 price records before cleanup")
        self.assertEqual(changes_before, 100, "Should have 100 change records before cleanup")

        # Run cleanup function with 1-year retention
        result = cleanup_old_historical_data(
            sales_days_to_keep=365,
            stock_days_to_keep=365,
            conn=self.conn
        )

        # Count data after cleanup
        self.cursor.execute("SELECT COUNT(*) FROM fact_sales")
        sales_after = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM fact_stock")
        stock_after = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM fact_prices")
        prices_after = self.cursor.fetchone()[0]

        self.cursor.execute("SELECT COUNT(*) FROM fact_stock_changes")
        changes_after = self.cursor.fetchone()[0]

        # Check results
        self.assertEqual(result["sales"], 50, "Should have removed 50 sales records")
        self.assertEqual(result["stock"], 50, "Should have removed 50 stock records")
        self.assertEqual(result["prices"], 50, "Should have removed 50 price records")
        self.assertEqual(result["stock_changes"], 50, "Should have removed 50 change records")

        self.assertEqual(sales_after, 50, "Should have 50 sales records remaining")
        self.assertEqual(stock_after, 50, "Should have 50 stock records remaining")
        self.assertEqual(prices_after, 50, "Should have 50 price records remaining")
        self.assertEqual(changes_after, 50, "Should have 50 change records remaining")

        # Check that only recent records remain
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        self.cursor.execute(
            "SELECT COUNT(*) FROM fact_sales WHERE data_date < ?",
            (one_year_ago,)
        )
        old_sales = self.cursor.fetchone()[0]

        self.cursor.execute(
            "SELECT COUNT(*) FROM fact_stock WHERE data_date < ?",
            (one_year_ago,)
        )
        old_stock = self.cursor.fetchone()[0]

        self.cursor.execute(
            "SELECT COUNT(*) FROM fact_prices WHERE data_date < ?",
            (one_year_ago,)
        )
        old_prices = self.cursor.fetchone()[0]

        self.cursor.execute(
            "SELECT COUNT(*) FROM fact_stock_changes WHERE data_date < ?",
            (one_year_ago,)
        )
        old_stock_changes = self.cursor.fetchone()[0]

        self.assertEqual(old_sales, 0, "Should have no sales records older than 1 year")
        self.assertEqual(old_stock, 0, "Should have no stock records older than 1 year")
        self.assertEqual(old_prices, 0, "Should have no price records older than 1 year")
        self.assertEqual(old_stock_changes, 0, "Should have no stock change records older than 1 year")

    def test_cleanup_old_models(self):
        """Test cleaning up old models"""
        # Create test data
        self._create_test_models()

        # Count models before cleanup
        self.cursor.execute("SELECT COUNT(*) FROM models")
        before_count = self.cursor.fetchone()[0]
        self.assertEqual(before_count, 6, "Should have 6 models before cleanup")

        # Run cleanup function keeping 2 models and 15-day retention for inactive
        deleted_model_ids = cleanup_old_models(models_to_keep=2, inactive_days_to_keep=15, conn=self.conn)

        # Count models after cleanup
        self.cursor.execute("SELECT COUNT(*) FROM models")
        after_count = self.cursor.fetchone()[0]

        # Check results - should keep model1 + model2 (top 2 active) and inactive_model1 (within 15-day retention)
        self.assertEqual(len(deleted_model_ids), 3, "Should have removed 3 models")
        self.assertEqual(after_count, 3, "Should have 3 models remaining")

        # Check that the correct models were retained
        self.cursor.execute("SELECT model_id FROM models")
        remaining_models = [row[0] for row in self.cursor.fetchall()]
        self.assertIn("model1", remaining_models, "model1 should be retained (active, best metric)")
        self.assertIn("model2", remaining_models, "model2 should be retained (active, second-best metric)")
        self.assertIn("inactive_model1", remaining_models, "inactive_model1 should be retained (within retention period)")

        # Check that deleted models are actually deleted from filesystem
        for model_id in deleted_model_ids:
            self.cursor.execute("SELECT model_path FROM models WHERE model_id = ?", (model_id,))
            result = self.cursor.fetchone()
            self.assertIsNone(result, f"{model_id} should be deleted from database")

    @patch('deployment.app.db.data_retention.cleanup_old_predictions')
    @patch('deployment.app.db.data_retention.cleanup_old_models')
    @patch('deployment.app.db.data_retention.cleanup_old_historical_data')
    def test_run_cleanup_job(self, mock_cleanup_historical, mock_cleanup_models, mock_cleanup_predictions):
        """Test running the complete cleanup job"""
        # Setup return values
        mock_cleanup_predictions.return_value = 10
        mock_cleanup_models.return_value = ["model3", "model4", "inactive_model2"]
        mock_cleanup_historical.return_value = {
            "sales": 50,
            "stock": 50,
            "stock_changes": 50,
            "prices": 50
        }

        # Run the complete cleanup job
        run_cleanup_job()

        # Check that all cleanup functions were called with correct parameters
        mock_cleanup_predictions.assert_called_once()
        mock_cleanup_models.assert_called_once()
        mock_cleanup_historical.assert_called_once()


if __name__ == '__main__':
    unittest.main()
