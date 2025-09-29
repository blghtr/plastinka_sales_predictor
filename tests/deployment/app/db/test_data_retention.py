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

        self.settings_patch_db_path = patch.object(
            AppSettings, "database_path", new_callable=PropertyMock
        )
        self.mock_db_path = self.settings_patch_db_path.start()
        self.mock_db_path.return_value = self.test_db_path

        get_settings.cache_clear()

        self.settings_patch = patch("deployment.app.db.data_retention.get_settings")
        self.mock_get_settings = self.settings_patch.start()

        mock_retention_settings = DataRetentionSettings(
            sales_retention_days=365,
            stock_retention_days=365,
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
        self.dal.close()  # Explicitly close DAL connection

        try:
            shutil.rmtree(self.test_dir)
        except (PermissionError, OSError) as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Could not clean up test directory {self.test_dir}: {e}")

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
                    "use_layer_norm": True,
                },
                "optimizer_config": {"lr": 0.001, "weight_decay": 0.0001},
                "lr_shed_config": {"T_0": 10, "T_mult": 2},
                "train_ds_config": {"alpha": 0.5, "span": 12},
                "lags": 12,
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

        # Create jobs and results first
        job1_id = self.dal.create_job("prediction_job_1", {"test": "params"})
        job2_id = self.dal.create_job("prediction_job_2", {"test": "params"})

        # Create models if they don't exist
        try:
            self.dal.create_model_record(
                model_id="model1",
                job_id=job1_id,
                model_path=os.path.join(self.model_dir, "model1.pt"),
                created_at=now,
                is_active=True,
            )
        except Exception:  # May already exist
            pass
        try:
            self.dal.create_model_record(
                model_id="model2",
                job_id=job2_id,
                model_path=os.path.join(self.model_dir, "model2.pt"),
                created_at=now,
                is_active=True,
            )
        except Exception:  # May already exist
            pass

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

        # Create multiindex records in batch
        multiindex_tuples = [
            (
                f"barcode{i}",
                f"artist{i}",
                f"album{i}",
                "vinyl",
                "medium",
                "studio",
                "1990s",
                "1990s",
                "rock",
                1995,
            )
            for i in range(200)
        ]
        multiindex_ids = self.dal.get_or_create_multiindex_ids_batch(multiindex_tuples)

        predictions = []
        # Recent predictions (10 records)
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
                    now.isoformat(),
                )
            )

        # Old predictions (10 records)
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
                    now.isoformat(),
                )
            )

        self.dal.connection.executemany(
            "INSERT INTO fact_predictions (multiindex_id, prediction_month, result_id, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            predictions,
        )
        self.dal.commit()

    def _create_test_historical_data(self):
        """Create test historical data records directly."""
        now = datetime.now()

        # 1. Create multi-index entries
        multiindex_tuples = [
            (
                f"barcode_hist_{i}",
                f"artist_hist_{i}",
                f"album_hist_{i}",
                "CD",
                "low",
                "live",
                "2010s",
                "2010s",
                "electronic",
                2015,
            )
            for i in range(200)
        ]
        multiindex_ids = self.dal.get_or_create_multiindex_ids_batch(multiindex_tuples)

        sales_params = []
        movement_params = []

        # 2. Generate data points
        # Recent data (100 records)
        for i in range(10):  # 10 dates
            date = now - timedelta(days=i * 30)
            date_str = date.strftime("%Y-%m-%d")
            for j in range(10):  # 10 products per date
                midx_id = multiindex_ids[i * 10 + j]
                sales_params.append((midx_id, date_str, 10.0 + j))
                movement_params.append((midx_id, date_str, 1.0 + j))

        # Old data (100 records)
        for i in range(10):  # 10 dates
            date = now - timedelta(days=800 + i * 30)
            date_str = date.strftime("%Y-%m-%d")
            for j in range(10):  # 10 products per date
                midx_id = multiindex_ids[100 + i * 10 + j]
                sales_params.append((midx_id, date_str, 5.0 + j))
                movement_params.append((midx_id, date_str, 1.0 + j))

        # 3. Insert data directly
        self.dal.insert_features_batch("fact_sales", sales_params)
        self.dal.insert_features_batch("fact_stock_movement", movement_params)
        self.dal.commit()

    def test_cleanup_old_predictions(self):
        """Test cleaning up old predictions"""
        self._create_test_predictions()

        before_count = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_predictions", fetchall=False
        )["count"]
        self.assertEqual(before_count, 20)

        count = cleanup_old_predictions(days_to_keep=30, dal=self.dal)

        after_count = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_predictions", fetchall=False
        )["count"]

        self.assertEqual(count, 10)
        self.assertEqual(after_count, 10)

    def test_cleanup_old_historical_data(self):
        """Test cleaning up old historical data"""
        self._create_test_historical_data()

        sales_before = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_sales", fetchall=False
        )["count"]
        changes_before = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_stock_movement", fetchall=False
        )["count"]

        self.assertEqual(sales_before, 200)
        self.assertEqual(changes_before, 200)

        result = cleanup_old_historical_data(
            sales_days_to_keep=365, stock_days_to_keep=365, dal=self.dal
        )

        sales_after = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_sales", fetchall=False
        )["count"]
        changes_after = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM fact_stock_movement", fetchall=False
        )["count"]

        self.assertEqual(result["sales"], 100)
        self.assertEqual(result["stock_movement"], 100)
        self.assertEqual(sales_after, 100)
        self.assertEqual(changes_after, 100)

    def test_cleanup_old_models(self):
        """Test cleaning up old models, including check for linked predictions."""
        self._create_test_models()

        # Add predictions linked to models that would otherwise be deleted
        job_id = self.dal.create_job("prediction_job_test", {})
        result_id = self.dal.create_prediction_result(
            job_id=job_id,
            prediction_month="2025-01",
            model_id="model4",
            output_path="/tmp/pred_test1",
            summary_metrics={"test": 1.0},
        )
        multiindex_id = self.dal.get_or_create_multiindex_ids_batch(
            [("p_bc", "p_art", "p_alb", "a", "b", "c", "d", "e", "f", 2000)]
        )[0]

        self.dal.connection.execute(
            "INSERT INTO fact_predictions (multiindex_id, prediction_month, result_id, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                multiindex_id,
                "2025-01-01",
                result_id,
                "model4",
                1,
                2,
                3,
                4,
                5,
                datetime.now().isoformat(),
            ),
        )

        job_id2 = self.dal.create_job("prediction_job_test2", {})
        result_id2 = self.dal.create_prediction_result(
            job_id=job_id2,
            prediction_month="2025-01",
            model_id="inactive_model2",
            output_path="/tmp/pred_test2",
            summary_metrics={"test": 1.0},
        )
        self.dal.connection.execute(
            "INSERT INTO fact_predictions (multiindex_id, prediction_month, result_id, model_id, quantile_05, quantile_25, quantile_50, quantile_75, quantile_95, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                multiindex_id,
                "2025-01-01",
                result_id2,
                "inactive_model2",
                1,
                2,
                3,
                4,
                5,
                datetime.now().isoformat(),
            ),
        )
        self.dal.commit()

        before_count = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM models", fetchall=False
        )["count"]
        self.assertEqual(before_count, 6)

        deleted_model_ids = cleanup_old_models(
            models_to_keep=2, inactive_days_to_keep=15, dal=self.dal
        )

        after_count = self.dal.execute_raw_query(
            "SELECT COUNT(*) as count FROM models", fetchall=False
        )["count"]

        # Only model3 should be deleted. model4 and inactive_model2 are protected by predictions.
        self.assertEqual(len(deleted_model_ids), 1)
        self.assertIn("model3", deleted_model_ids)
        self.assertEqual(after_count, 5)

        remaining_models_q = self.dal.execute_raw_query(
            "SELECT model_id FROM models", fetchall=True
        )
        remaining_models = {row["model_id"] for row in remaining_models_q}
        self.assertIn("model1", remaining_models)
        self.assertIn("model2", remaining_models)
        self.assertIn("inactive_model1", remaining_models)
        self.assertIn("model4", remaining_models)
        self.assertIn("inactive_model2", remaining_models)

    @patch("deployment.app.db.data_retention.cleanup_old_predictions")
    @patch("deployment.app.db.data_retention.cleanup_old_models")
    @patch("deployment.app.db.data_retention.cleanup_old_historical_data")
    def test_run_cleanup_job(
        self, mock_cleanup_historical, mock_cleanup_models, mock_cleanup_predictions
    ):
        """Test running the complete cleanup job"""
        mock_cleanup_predictions.return_value = 10
        mock_cleanup_models.return_value = ["model3"]
        mock_cleanup_historical.return_value = {"sales": 50, "stock_movement": 50}

        run_cleanup_job(dal=self.dal)

        mock_cleanup_predictions.assert_called_once()
        mock_cleanup_models.assert_called_once()
        mock_cleanup_historical.assert_called_once()


if __name__ == "__main__":
    unittest.main()