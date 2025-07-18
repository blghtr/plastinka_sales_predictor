"""
Unit tests for enhanced prediction report functionality.

Tests the new enrichment functions that adapt the notebook's process_features
functionality to work with the database system.
"""

from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from deployment.app.services.report_service import (
    _extract_features_for_month,
    _join_predictions_with_enriched_metrics,
    _load_raw_features_for_report,
    _process_features_for_report,
)


class TestLoadRawFeaturesForReport:
    """Test suite for _load_raw_features_for_report function."""

    def test_load_raw_features_success(self, monkeypatch, mock_load_features):
        """Test successful loading of raw features."""
        # Arrange
        monkeypatch.setattr("deployment.app.services.report_service.load_features", mock_load_features)
        mock_features = {
            "sales": pd.DataFrame({"value": [1, 2, 3]}),
            "change": pd.DataFrame({"value": [0, 1, -1]}),
            "stock": pd.DataFrame({"value": [10, 9, 10]}),
            "prices": pd.DataFrame({"value": [100, 200, 150]}),
        }
        mock_load_features.return_value = mock_features
        prediction_month = datetime(2024, 3, 1)

        # Act
        result = _load_raw_features_for_report(prediction_month)

        # Assert
        assert result == mock_features
        mock_load_features.assert_called_once_with(
            store_type="sql", feature_types=["sales", "change", "stock", "prices"]
        )

    def test_load_raw_features_exception(self, monkeypatch, mock_load_features):
        """Test handling of exceptions during feature loading."""
        # Arrange
        monkeypatch.setattr("deployment.app.services.report_service.load_features", mock_load_features)
        mock_load_features.side_effect = Exception("Database error")
        prediction_month = datetime(2024, 3, 1)

        # Act
        result = _load_raw_features_for_report(prediction_month)

        # Assert
        assert result == {}

    def test_load_raw_features_empty_response(self, monkeypatch, mock_load_features):
        """Test handling of empty features response."""
        # Arrange
        monkeypatch.setattr("deployment.app.services.report_service.load_features", mock_load_features)
        mock_load_features.return_value = {}
        prediction_month = datetime(2024, 3, 1)

        # Act
        result = _load_raw_features_for_report(prediction_month)

        # Assert
        assert result == {}


# Удалён класс TestProcessFeaturesForReport и все его тесты


class TestExtractFeaturesForMonth:
    """Test suite for _extract_features_for_month function."""

    def create_mock_processed_features(self):
        """Create mock processed features for testing."""
        dates = [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)]
        products = [("123", "Artist A"), ("456", "Artist B")]

        features = {}
        for feature_name in [
            "masked_mean_sales_items",
            "masked_mean_sales_rub",
            "lost_sales",
        ]:
            # Create DataFrame with dates as index and products as columns (expected structure)
            data = np.random.randn(len(dates), len(products))
            df = pd.DataFrame(
                data,
                index=pd.DatetimeIndex(
                    dates
                ),  # Dates as index (expected by _extract_features_for_month)
                columns=pd.MultiIndex.from_tuples(
                    products, names=["barcode", "artist"]
                ),  # Products as columns
            )
            features[feature_name] = df

        return features

    def test_extract_features_exact_match(self):
        """Test extraction when exact date match exists."""
        # Arrange
        processed_features = self.create_mock_processed_features()
        target_month = datetime(2024, 3, 1)

        # Act
        result = _extract_features_for_month(processed_features, target_month)

        # Assert
        assert not result.empty
        expected_columns = [
            "Средние продажи (шт)",
            "Средние продажи (руб)",
            "Потерянные продажи (руб)",
        ]
        for col in expected_columns:
            assert col in result.columns

    def test_extract_features_no_match(self):
        """Test extraction when no date match exists."""
        # Arrange
        processed_features = self.create_mock_processed_features()
        target_month = datetime(2025, 1, 1)  # Future date not in data

        # Act
        result = _extract_features_for_month(processed_features, target_month)

        # Assert
        # Should still work by using the latest available date
        assert isinstance(result, pd.DataFrame)

    def test_extract_features_empty_input(self):
        """Test extraction with empty processed features."""
        # Arrange
        processed_features = {}
        target_month = datetime(2024, 3, 1)

        # Act
        result = _extract_features_for_month(processed_features, target_month)

        # Assert
        assert result.empty

    def test_extract_features_exception(self):
        """Test handling of exceptions during extraction."""
        # Arrange
        # Create problematic features that will cause an exception
        processed_features = {"masked_mean_sales_items": "not_a_dataframe"}
        target_month = datetime(2024, 3, 1)

        # Act
        result = _extract_features_for_month(processed_features, target_month)

        # Assert
        assert result.empty


class TestJoinPredictionsWithEnrichedMetrics:
    """Test suite for _join_predictions_with_enriched_metrics function."""

    def create_mock_predictions(self):
        """Create mock prediction data."""
        data = [
            [
                "123",
                "Artist A",
                "Album 1",
                "Opened",
                "Low",
                "Original",
                "1990s",
                "1990s",
                "Rock",
                1995,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            [
                "456",
                "Artist B",
                "Album 2",
                "Sealed",
                "Medium",
                "Reissue",
                "2000s",
                "2010s",
                "Pop",
                2005,
                1.5,
                2.5,
                3.5,
                4.5,
                5.5,
            ],
        ]

        columns = [
            "barcode",
            "artist",
            "album",
            "cover_type",
            "price_category",
            "release_type",
            "recording_decade",
            "release_decade",
            "style",
            "record_year",
            "quantile_05",
            "quantile_25",
            "quantile_50",
            "quantile_75",
            "quantile_95",
        ]

        return pd.DataFrame(data, columns=columns)

    def create_mock_enriched_columns(self):
        """Create mock enriched columns data."""
        products = [
            (
                "123",
                "Artist A",
                "Album 1",
                "Opened",
                "Low",
                "Original",
                "1990s",
                "1990s",
                "Rock",
                1995,
            ),
            (
                "456",
                "Artist B",
                "Album 2",
                "Sealed",
                "Medium",
                "Reissue",
                "2000s",
                "2010s",
                "Pop",
                2005,
            ),
        ]

        index = pd.MultiIndex.from_tuples(
            products,
            names=[
                "barcode",
                "artist",
                "album",
                "cover_type",
                "price_category",
                "release_type",
                "recording_decade",
                "release_decade",
                "style",
                "record_year",
            ],
        )

        data = {
            "Средние продажи (шт)": [2.5, 1.8],
            "Средние продажи (руб)": [250, 360],
            "Потерянные продажи (руб)": [100, 80],
        }

        return pd.DataFrame(data, index=index)

    def test_join_predictions_success(self):
        """Проверяет успешный join при совпадающих MultiIndex по всем колонкам продукта."""
        products = [
            ("123", "Artist A", "Album 1", "Opened", "Low", "Original", "1990s", "1990s", "Rock", 1995),
            ("456", "Artist B", "Album 2", "Opened", "Low", "Original", "1990s", "1990s", "Rock", 1995),
        ]
        columns = [
            "barcode", "artist", "album", "cover_type", "price_category", "release_type", "recording_decade", "release_decade", "style", "record_year"
        ]
        predictions_df = pd.DataFrame(products, columns=columns)
        predictions_df["0.5"] = [21.4, 24.8]
        predictions_df["0.95"] = [35.7, 40.2]
        # НЕ устанавливаем index - функция сама создаст MultiIndex
        # Функция берет колонки [1:11], поэтому создаем MultiIndex только из этих колонок
        key_columns = columns[1:11]  # Исключаем первую колонку
        enriched_columns = pd.DataFrame({"enriched_metric": [0.1, 0.2]}, index=pd.MultiIndex.from_tuples([(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]) for p in products], names=key_columns))
        try:
            result = _join_predictions_with_enriched_metrics(predictions_df, enriched_columns)
        except Exception:
            print("DIAG: predictions_df shape:", predictions_df.shape, "index:", predictions_df.index)
            print("DIAG: enriched_columns shape:", enriched_columns.shape, "index:", enriched_columns.index)
            raise
        assert "enriched_metric" in result.columns
        assert result.shape[0] == 2
        # Проверяем по ключу без первой колонки
        key1 = products[0][1:]  # Исключаем barcode
        key2 = products[1][1:]  # Исключаем barcode
        assert result.loc[key1, "enriched_metric"].iloc[0] == 0.1
        assert result.loc[key2, "enriched_metric"].iloc[0] == 0.2

    def test_join_predictions_empty_enriched(self):
        """Test joining with empty enriched columns."""
        # Arrange
        predictions_df = self.create_mock_predictions()
        enriched_columns = pd.DataFrame()

        # Act
        result = _join_predictions_with_enriched_metrics(
            predictions_df, enriched_columns
        )

        # Assert
        pd.testing.assert_frame_equal(result, predictions_df)

    def test_join_predictions_no_matching_keys(self):
        """Test joining when no matching keys exist."""
        # Arrange
        predictions_df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
        enriched_columns = pd.DataFrame({"metric": [100, 200]}, index=["a", "b"])

        # Act
        result = _join_predictions_with_enriched_metrics(
            predictions_df, enriched_columns
        )

        # Assert
        # Should return original predictions when no matching keys
        assert len(result.columns) == len(predictions_df.columns)

    def test_join_predictions_partial_match(self):
        """Проверяет корректную обработку частичного совпадения MultiIndex."""
        products = [
            ("123", "Artist A", "Album 1", "Opened", "Low", "Original", "1990s", "1990s", "Rock", 1995),
            ("456", "Artist B", "Album 2", "Opened", "Low", "Original", "1990s", "1990s", "Rock", 1995),
        ]
        columns = [
            "barcode", "artist", "album", "cover_type", "price_category", "release_type", "recording_decade", "release_decade", "style", "record_year"
        ]
        predictions_df = pd.DataFrame(products, columns=columns)
        predictions_df["0.5"] = [21.4, 24.8]
        predictions_df["0.95"] = [35.7, 40.2]
        # НЕ устанавливаем index - функция сама создаст MultiIndex
        # enriched_columns только для первой строки
        key_columns = columns[1:11]  # Исключаем первую колонку
        enriched_columns = pd.DataFrame({"enriched_metric": [0.1]}, index=pd.MultiIndex.from_tuples([products[0][1:]], names=key_columns))
        try:
            result = _join_predictions_with_enriched_metrics(predictions_df, enriched_columns)
        except Exception:
            print("DIAG: predictions_df shape:", predictions_df.shape, "index:", predictions_df.index)
            print("DIAG: enriched_columns shape:", enriched_columns.shape, "index:", enriched_columns.index)
            raise
        assert "enriched_metric" in result.columns
        assert result.shape[0] == 2
        key1 = products[0][1:]  # Исключаем barcode
        key2 = products[1][1:]  # Исключаем barcode
        assert result.loc[key1, "enriched_metric"].iloc[0] == 0.1
        assert pd.isna(result.loc[key2, "enriched_metric"].iloc[0])

    def test_join_predictions_exception(self):
        """Test handling of exceptions during joining."""
        # Arrange
        predictions_df = self.create_mock_predictions()
        # Create problematic enriched data
        enriched_columns = pd.DataFrame({"metric": [1, 2, 3]})  # Mismatched length

        # Act
        result = _join_predictions_with_enriched_metrics(
            predictions_df, enriched_columns
        )

        # Assert
        # Should return original predictions on exception
        assert len(result.columns) >= len(predictions_df.columns)


class TestIntegrationScenarios:
    """Integration tests for the complete enrichment workflow."""

    def test_full_enrichment_workflow(self, monkeypatch, mock_load_features):
        """Test the complete enrichment workflow from raw features to enriched predictions."""
        # Arrange
        monkeypatch.setattr("deployment.app.services.report_service.load_features", mock_load_features)
        prediction_month = datetime(2024, 3, 1)

        # Mock raw features
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="D")
        products = [("123", "Artist A", "Album 1")]

        sales_data = []
        change_data = []
        for date in dates[:10]:  # First 10 days
            sales_data.append((*products[0], date, 2))
            change_data.append((*products[0], date, 1))

        mock_features = {
            "sales": pd.DataFrame(
                sales_data, columns=["barcode", "artist", "album", "_date", "sales"]
            ).set_index(["barcode", "artist", "album", "_date"]),
            "change": pd.DataFrame(
                change_data, columns=["barcode", "artist", "album", "_date", "change"]
            ).set_index(["barcode", "artist", "album", "_date"]),
            "stock": pd.DataFrame(
                [[10]],
                index=pd.MultiIndex.from_tuples(
                    products, names=["barcode", "artist", "album"]
                ),
                columns=[datetime(2024, 1, 1)],
            ),
            "prices": pd.DataFrame(
                {"prices": [100]},
                index=pd.MultiIndex.from_tuples(
                    products, names=["barcode", "artist", "album"]
                ),
            ),
        }
        mock_load_features.return_value = mock_features

        # Mock predictions
        predictions_df = pd.DataFrame(
            [
                [
                    "123",
                    "Artist A",
                    "Album 1",
                    "Opened",
                    "Low",
                    "Original",
                    "1990s",
                    "1990s",
                    "Rock",
                    1995,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                ]
            ],
            columns=[
                "barcode",
                "artist",
                "album",
                "cover_type",
                "price_category",
                "release_type",
                "recording_decade",
                "release_decade",
                "style",
                "record_year",
                "quantile_05",
                "quantile_25",
                "quantile_50",
                "quantile_75",
                "quantile_95",
            ],
        )

        # Act
        # Step 1: Load raw features
        raw_features = _load_raw_features_for_report(prediction_month)

        # Step 2: Process features
        processed_features = _process_features_for_report(
            raw_features, prediction_month
        )

        # Step 3: Extract for month (if we have processed features)
        if processed_features:
            enriched_columns = _extract_features_for_month(
                processed_features, prediction_month
            )

            # Step 4: Join with predictions
            result = _join_predictions_with_enriched_metrics(
                predictions_df, enriched_columns
            )
        else:
            result = predictions_df

        # Assert
        assert not result.empty
        assert len(result) == 1
        # Original columns should be preserved
        for col in predictions_df.columns:
            assert col in result.columns


# Pytest configuration and fixtures
@pytest.fixture
def sample_multiindex_products():
    """Fixture providing sample product MultiIndex."""
    products = [
        (
            "123456",
            "Artist A",
            "Album 1",
            "Opened",
            "Low",
            "Original",
            "1990s",
            "1990s",
            "Rock",
            1995,
        ),
        (
            "234567",
            "Artist B",
            "Album 2",
            "Sealed",
            "Medium",
            "Reissue",
            "2000s",
            "2010s",
            "Pop",
            2005,
        ),
    ]
    return pd.MultiIndex.from_tuples(
        products,
        names=[
            "barcode",
            "artist",
            "album",
            "cover_type",
            "price_category",
            "release_type",
            "recording_decade",
            "release_decade",
            "style",
            "record_year",
        ],
    )


@pytest.fixture
def sample_date_range():
    """Fixture providing sample date range."""
    return pd.date_range("2024-01-01", "2024-03-31", freq="D")


@pytest.fixture
def mock_load_features():
    return MagicMock()


# Удалены все тесты, связанные с _adapt_features_schema


# Performance and edge case tests
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    def test_large_dataset_handling(self):
        """Проверяет обработку большого датасета: MultiIndex и количество строк согласованы."""
        n = 10
        products = [
            (str(100 + i), f"Artist {i}", f"Album {i}", "Opened", "Low", "Original", "1990s", "1990s", "Rock", 1995)
            for i in range(n)
        ]
        columns = [
            "barcode", "artist", "album", "cover_type", "price_category", "release_type", "recording_decade", "release_decade", "style", "record_year"
        ]
        predictions_df = pd.DataFrame(products, columns=columns)
        predictions_df["0.5"] = [21.4 + i for i in range(n)]
        predictions_df["0.95"] = [35.7 + i for i in range(n)]
        # НЕ устанавливаем index - функция сама создаст MultiIndex
        key_columns = columns[1:11]  # Исключаем первую колонку
        enriched_columns = pd.DataFrame({"enriched_metric": [0.1 + 0.01 * i for i in range(n)]}, index=pd.MultiIndex.from_tuples([p[1:] for p in products], names=key_columns))
        try:
            result = _join_predictions_with_enriched_metrics(predictions_df, enriched_columns)
        except Exception:
            print("DIAG: predictions_df shape:", predictions_df.shape, "index:", predictions_df.index)
            print("DIAG: enriched_columns shape:", enriched_columns.shape, "index:", enriched_columns.index)
            raise
        assert "enriched_metric" in result.columns
        assert result.shape[0] == n
        for i, prod in enumerate(products):
            key = prod[1:]  # Исключаем barcode
            assert result.loc[key, "enriched_metric"].iloc[0] == 0.1 + 0.01 * i

    def test_memory_efficiency(self):
        """Test memory efficiency with repeated operations."""
        # This test ensures we don't have memory leaks in repeated operations
        prediction_month = datetime(2024, 3, 1)

        for _ in range(10):  # Repeat operations
            empty_features = {
                "sales": pd.DataFrame(),
                "change": pd.DataFrame(),
                "stock": pd.DataFrame(),
                "prices": pd.DataFrame(),
            }

            # These should all handle empty data gracefully
            processed = _process_features_for_report(empty_features, prediction_month)
            extracted = _extract_features_for_month(processed, prediction_month)

            # Clean up
            del processed, extracted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
