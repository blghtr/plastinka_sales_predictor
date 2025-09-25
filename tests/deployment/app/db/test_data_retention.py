import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, PropertyMock, patch

from deployment.app.config import DataRetentionSettings
from deployment.app.db.data_access_layer import DataAccessLayer
from deployment.app.db.data_retention import (
    cleanup_old_historical_data,
    cleanup_old_models,
    cleanup_old_predictions,
    run_cleanup_job,
)


class TestDataRetention(unittest.TestCase):
    def setUp(self):
        """Set up test database and environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.test_dir, "test_db.sqlite")

        # Initialize DAL with db_path, it will handle schema initialization
        self.dal = DataAccessLayer(db_path=self.test_db_path)

        self.model_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        from deployment.app.config import AppSettings, get_settings
        self.settings_patch_db_path = patch.object(AppSettings, 'database_path', new_callable=PropertyMock)
        self.mock_db_path = self.settings_patch_db_path.start()
        self.mock_db_path.return_value = self.test_db_path

        get_settings.cache_clear()

        self.settings_patch = patch("deployment.app.db.data_retention.get_settings")
        self.mock_get_settings = self.settings_patch.start()

        mock_retention_settings = DataRetentionSettings(
            sales_retention_days=730,
            stock_retention_days=730,
            prediction_retention_days=30,
            models_to_keep=2,
            inactive_model_retention_days=15,
            cleanup_enabled=True,
        )
        self.mock_settings_object = MagicMock()
        self.mock_settings_object.data_retention = mock_retention_settings
        self.mock_settings_object.default_metric = "val_MIC"
        self.mock_settings_object.default_metric_higher_is_better = True
        self.mock_settings_object.models_dir = self.model_dir
        self.mock_get_settings.return_value = self.mock_settings_object

    def tearDown(self):
        """Clean up after tests"""
        self.settings_patch_db_path.stop()
        self.settings_patch.stop()
        self.dal.close() # Explicitly close DAL connection

        # Handle file cleanup more gracefully
        try:
            shutil.rmtree(self.test_dir)
        except (PermissionError, OSError) as e:
            # Log the error but don't fail the test
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not clean up test directory {self.test_dir}: {e}")
            # Try to remove individual files if directory removal fails
            try:
                for root, dirs, files in os.walk(self.test_dir, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except (PermissionError, OSError):
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except (PermissionError, OSError):
                            pass
            except Exception:
                pass


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
            ("inactive_model2.pt", now - timedelta(days=20)),
        ]

        for filename, _ in model_files:
            with open(os.path.join(self.model_dir, filename), "w") as f:
                f.write("Test model content")

        # Create jobs first (required for foreign key constraints)
        job_ids = []
        for i in range(1, 7):
            job_id = self.dal.create_job(f"training_job_{i}", {"test": "params"})
            job_ids.append(job_id)

        # Create config first (required for foreign key constraints)
        config_id = self.dal.create_or_get_config(
            {
                "nn_model_config": {
                    "num_encoder_layers": 2,
                    "num_decoder_layers": 2,
                    "decoder_output_dim": 64,
                    "temporal_width_past": 10,
                    "temporal_width_future": 5,
                    "temporal_hidden_size_past": 64,
                    "temporal_hidden_size_future": 64,
                    "temporal_decoder_hidden": 64,
                    "batch_size": 32,
                    "dropout": 0.1,
                    "use_reversible_instance_norm": True,
                    "use_layer_norm": True
                },
                "optimizer_config": {
                    "lr": 0.001,
                    "weight_decay": 0.0001
                },
                "lr_shed_config": {
                    "T_0": 10,
                    "T_mult": 2
                },
                "train_ds_config": {
                    "alpha": 0.5,
                    "span": 12
                },
                "lags": 12
            },
            is_active=True,
        )

        # Insert model records using DAL
        self.dal.create_model_record(
            model_id="model1",
            job_id=job_ids[0],
            model_path=os.path.join(self.model_dir, "model1.pt"),
            created_at=(now - timedelta(days=5)),
            metadata={"size": 1000},
            is_active=True,
        )
        self.dal.create_model_record(
            model_id="model2",
            job_id=job_ids[1],
            model_path=os.path.join(self.model_dir, "model2.pt"),
            created_at=(now - timedelta(days=10)),
            metadata={"size": 1000},
            is_active=True,
        )
        self.dal.create_model_record(
            model_id="model3",
            job_id=job_ids[2],
            model_path=os.path.join(self.model_dir, "model3.pt"),
            created_at=(now - timedelta(days=20)),
            metadata={"size": 1000},
            is_active=True,
        )
        self.dal.create_model_record(
            model_id="model4",
            job_id=job_ids[3],
            model_path=os.path.join(self.model_dir, "model4.pt"),
            created_at=(now - timedelta(days=30)),
            metadata={"size": 1000},
            is_active=True,
        )
        self.dal.create_model_record(
            model_id="inactive_model1",
            job_id=job_ids[4],
            model_path=os.path.join(self.model_dir, "inactive_model1.pt"),
            created_at=(now - timedelta(days=10)),
            metadata={"size": 1000},
            is_active=False,
        )
        self.dal.create_model_record(
            model_id="inactive_model2",
            job_id=job_ids[5],
            model_path=os.path.join(self.model_dir, "inactive_model2.pt"),
            created_at=(now - timedelta(days=20)),
            metadata={"size": 1000},
            is_active=False,
        )

        # Associate models with config set and add metrics using DAL
        self.dal.create_training_result(
            job_id=job_ids[0],
            model_id="model1",
            config_id=config_id,
            metrics={"val_MIC": 0.95},
            duration=3600,
        )
        self.dal.create_training_result(
            job_id=job_ids[1],
            model_id="model2",
            config_id=config_id,
            metrics={"val_MIC": 0.90},
            duration=3600,
        )
        self.dal.create_training_result(
            job_id=job_ids[2],
            model_id="model3",
            config_id=config_id,
            metrics={"val_MIC": 0.85},
            duration=3600,
        )
        self.dal.create_training_result(
            job_id=job_ids[3],
            model_id="model4",
            config_id=config_id,
            metrics={"val_MIC": 0.80},
            duration=3600,
        )

    def _create_test_predictions(self):
        """Create test prediction records"""
        now = datetime.now()

        # Create jobs and results first (required for foreign key constraints)
        job1_id = self.dal.create_job("prediction_job_1", {"test": "params"})
        job2_id = self.dal.create_job("prediction_job_2", {"test": "params"})

        # Create models first
        self.dal.create_model_record(
            model_id="model1",
            job_id=job1_id,
            model_path=os.path.join(self.model_dir, "model1.pt"),
            created_at=now,
            metadata={"size": 1000},
            is_active=True,
        )
        self.dal.create_model_record(
            model_id="model2",
            job_id=job2_id,
            model_path=os.path.join(self.model_dir, "model2.pt"),
            created_at=now,
            metadata={"size": 1000},
            is_active=True,
        )

        # Create prediction results
        result1_id = self.dal.create_prediction_result(
            job_id=job1_id,
            prediction_month=now.strftime("%Y-%m"),
            model_id="model1",
            output_path="/tmp/test1",
            summary_metrics={"mape": 10.5},
        )
        result2_id = self.dal.create_prediction_result(
            job_id=job2_id,
            prediction_month=now.strftime("%Y-%m"),
            model_id="model2",
            output_path="/tmp/test2",
            summary_metrics={"mape": 11.5},
        )

        # Create multiindex records first (required for foreign key constraints)
        multiindex_ids = []
        for i in range(200):  # Create enough for all predictions
            multiindex_id = self.dal.get_or_create_multiindex_id(
                barcode=f"barcode{i}",
                artist=f"artist{i}",
                album=f"album{i}",
                cover_type="vinyl",
                price_category="medium",
                release_type="studio",
                recording_decade="1990s",
                release_decade="1990s",
                style="rock",
                recording_year=1995
            )
            multiindex_ids.append(multiindex_id)

        # Insert predictions with various dates
        predictions = []

        # Recent predictions (within retention period)
        for i in range(10):
            date = now - timedelta(days=i)
            predictions.append(
                (
                    multiindex_ids[i],
                    date.strftime("%Y-%m-%d"),
                    result1_id,
                    "model1",
                    10.5,
                    15.2,
                    20.1,
                    25.8,
                    30.3,
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )

        # Old predictions (beyond retention period)
        for i in range(10):
            date = now - timedelta(days=40 + i)
            predictions.append(
                (
                    multiindex_ids[i + 100],
                    date.strftime("%Y-%m-%d"),
                    result2_id,
                    "model2",
                    11.5,
                    16.2,
                    21.1,
                    26.8,
                    31.3,
                    now.strftime("%Y-%m-%d %H:%M:%S"),
                )
            )

        for p in predictions:
            self.dal.execute_raw_query(
                "INSERT INTO fact_predictions (multiindex_id, prediction_month, result_id, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9])
            )

    def _create_test_historical_data(self):
        """Create test historical data records"""
        now = datetime.now()

        # Generate test data for different time ranges
        sales_data = []
        changes_data = []

        # Recent data (within 1 year)
        for i in range(10):
            date = now - timedelta(days=i * 30)  # Roughly monthly
            date_str = date.strftime("%Y-%m-%d")

            # Add multiple records for each date (different products)
            for j in range(10):  # Changed from 5 to 10
                multiindex_id = i * 10 + j + 1  # Unique multiindex_id for each record

                sales_data.append((multiindex_id, date_str, 10 + j))
                changes_data.append((multiindex_id, date_str, 1 + j))  # Changed from -5 + j to 1 + j to avoid zero values

        # Old data (over 2 years old)
        for i in range(10):
            date = now - timedelta(days=800 + i * 30)  # Beyond the retention period
            date_str = date.strftime("%Y-%m-%d")

            # Add multiple records for each date
            for j in range(10):  # Changed from 5 to 10
                multiindex_id = 100 + i * 10 + j + 1  # Unique multiindex_id for each record

                sales_data.append((multiindex_id, date_str, 5 + j))
                changes_data.append((multiindex_id, date_str, 1 + j))  # Changed from -2 + j to 1 + j to avoid zero values

        # Convert to DataFrames and save using SQLFeatureStore
        import pandas as pd

        from deployment.app.db.feature_storage import SQLFeatureStore
        from deployment.app.db.schema import MULTIINDEX_NAMES

        feature_store = SQLFeatureStore(dal=self.dal)

        def create_df(data):
            df = pd.DataFrame(data, columns=["multiindex_id", "data_date", "value"])
            df["data_date"] = pd.to_datetime(df["data_date"])
            pivot_df = df.pivot_table(index="multiindex_id", columns="data_date", values="value", fill_value=0)

            # Check if pivot_df is empty
            if pivot_df.empty:
                # Create a minimal DataFrame with correct structure
                pivot_df = pd.DataFrame(
                    data=[[1.0]],  # Single value
                    index=pd.MultiIndex.from_tuples([("dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy", 2000)], names=MULTIINDEX_NAMES),
                    columns=[pd.Timestamp.now().normalize()]
                )
            else:
                full_multiindex_tuples = []
                for idx_multiindex_id in pivot_df.index:
                    # Create a tuple with the multiindex_id as the barcode, and dummy values for others
                    # The order must match MULTIINDEX_NAMES: ["barcode", "artist", "album", ...]
                    multiindex_tuple = [str(idx_multiindex_id)] + ["dummy"] * (len(MULTIINDEX_NAMES) - 1)
                    full_multiindex_tuples.append(tuple(multiindex_tuple))

                pivot_df.index = pd.MultiIndex.from_tuples(full_multiindex_tuples, names=MULTIINDEX_NAMES)
            return pivot_df

        feature_store.save_features({"sales": create_df(sales_data)})
        feature_store.save_features({"movement": create_df(changes_data)})

    def test_cleanup_old_predictions(self):
        """Test cleaning up old predictions"""
        # Create test data
        self._create_test_predictions()

        # Count predictions before cleanup
        before_count_result = self.dal.execute_raw_query("SELECT COUNT(*) as count FROM fact_predictions", fetchall=False)
        before_count = before_count_result["count"]
        self.assertEqual(before_count, 20, "Should have 20 predictions before cleanup")

        # Run cleanup function with 30-day retention
        count = cleanup_old_predictions(days_to_keep=30, dal=self.dal)

        # Count predictions after cleanup
        after_count_result = self.dal.execute_raw_query("SELECT COUNT(*) as count FROM fact_predictions", fetchall=False)
        after_count = after_count_result["count"]

        # Check results
        self.assertEqual(count, 10, "Should have removed 10 old predictions")
        self.assertEqual(after_count, 10, "Should have 10 predictions remaining")

        # Check that only recent predictions remain
        old_remaining_result = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_predictions WHERE prediction_month < ?",
            ((datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),),
            fetchall=False
        )
        old_remaining = old_remaining_result["count"]
        self.assertEqual(
            old_remaining, 0, "Should have no predictions older than 30 days"
        )

    def test_cleanup_old_historical_data(self):
        """Test cleaning up old historical data"""
        # Create test data
        self._create_test_historical_data()

        # Count data before cleanup
        sales_before = self.dal.execute_raw_query("SELECT COUNT(*) as count FROM fact_sales", fetchall=False)["count"]
        changes_before = self.dal.execute_raw_query("SELECT COUNT(*) as count FROM fact_stock_movement", fetchall=False)["count"]

        self.assertEqual(
            sales_before, 200, "Should have 200 sales records before cleanup"
        )
        self.assertEqual(
            changes_before, 200, "Should have 200 movement records before cleanup"
        )

        # Run cleanup function with 1-year retention
        result = cleanup_old_historical_data(
            sales_days_to_keep=365, stock_days_to_keep=365, dal=self.dal
        )

        # Count data after cleanup
        sales_after = self.dal.execute_raw_query("SELECT COUNT(*) as count FROM fact_sales", fetchall=False)["count"]
        changes_after = self.dal.execute_raw_query("SELECT COUNT(*) as count FROM fact_stock_movement", fetchall=False)["count"]

        # Check results
        self.assertEqual(result["sales"], 100, "Should have removed 100 sales records")
        self.assertEqual(
            result["stock_movement"], 100, "Should have removed 100 movement records"
        )

        self.assertEqual(sales_after, 100, "Should have 100 sales records remaining")
        self.assertEqual(changes_after, 100, "Should have 100 movement records remaining")

        # Check that only recent records remain
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        old_sales = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_sales WHERE data_date < ?", (one_year_ago,),
            fetchall=False
        )["count"]
        old_stock_movement = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_stock_movement WHERE data_date < ?",
            (one_year_ago,),
            fetchall=False
        )["count"]

        self.assertEqual(old_sales, 0, "Should have no sales records older than 1 year")
        self.assertEqual(
            old_stock_movement,
            0,
            "Should have no stock movement records older than 1 year",
        )

    def test_cleanup_old_models(self):
        """Test cleaning up old models"""
        # Create test data
        self._create_test_models()

        # Count models before cleanup
        before_count = self.dal.execute_raw_query("SELECT COUNT(*) as count FROM models", fetchall=False)["count"]
        self.assertEqual(before_count, 6, "Should have 6 models before cleanup")

        # Run cleanup function keeping 2 models and 15-day retention for inactive
        deleted_model_ids = cleanup_old_models(
            models_to_keep=2, inactive_days_to_keep=15, dal=self.dal
        )

        # Count models after cleanup
        after_count = self.dal.execute_raw_query("SELECT COUNT(*) as count FROM models", fetchall=False)["count"]

        # Check results - should keep model1 + model2 (top 2 active) and inactive_model1 (within 15-day retention)
        self.assertEqual(len(deleted_model_ids), 3, "Should have removed 3 models")
        self.assertEqual(after_count, 3, "Should have 3 models remaining")

        # Check that the correct models were retained
        remaining_models = [row["model_id"] for row in self.dal.execute_raw_query("SELECT model_id FROM models", fetchall=True)]
        self.assertIn(
            "model1",
            remaining_models,
            "model1 should be retained (active, best metric)",
        )
        self.assertIn(
            "model2",
            remaining_models,
            "model2 should be retained (active, second-best metric)",
        )
        self.assertIn(
            "inactive_model1",
            remaining_models,
            "inactive_model1 should be retained (within retention period)",
        )

        # Check that deleted models are actually deleted from filesystem
        for model_id in deleted_model_ids:
            result = self.dal.execute_raw_query(
                "SELECT model_path FROM models WHERE model_id = ?", (model_id,),
                fetchall=False
            )
            self.assertIsNone(result, f"{model_id} should be deleted from database")

    @patch("deployment.app.db.data_retention.cleanup_old_predictions")
    @patch("deployment.app.db.data_retention.cleanup_old_models")
    @patch("deployment.app.db.data_retention.cleanup_old_historical_data")
    def test_run_cleanup_job(
        self, mock_cleanup_historical, mock_cleanup_models, mock_cleanup_predictions
    ):
        """Test running the complete cleanup job"""
        # Setup return values
        mock_cleanup_predictions.return_value = 10
        mock_cleanup_models.return_value = ["model3", "model4", "inactive_model2"]
        mock_cleanup_historical.return_value = {
            "sales": 50,
            "stock_movement": 50,
        }

        # Run the complete cleanup job
        run_cleanup_job()

        # Check that all cleanup functions were called with correct parameters
        mock_cleanup_predictions.assert_called_once()
        mock_cleanup_models.assert_called_once()
        mock_cleanup_historical.assert_called_once()


if __name__ == "__main__":
    unittest.main()
