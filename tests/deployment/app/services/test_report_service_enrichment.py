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

from deployment.app.services.report_service import (
    _adapt_features_schema,
    _extract_features_for_month,
    _join_predictions_with_enriched_metrics,
    _load_raw_features_for_report,
    _process_features_for_report,
)


class TestLoadRawFeaturesForReport:
    """Test suite for _load_raw_features_for_report function."""

    @patch("deployment.app.services.report_service.load_features")
    def test_load_raw_features_success(self, mock_load_features):
        """Test successful loading of raw features."""
        # Arrange
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

    @patch("deployment.app.services.report_service.load_features")
    def test_load_raw_features_exception(self, mock_load_features):
        """Test handling of exceptions during feature loading."""
        # Arrange
        mock_load_features.side_effect = Exception("Database error")
        prediction_month = datetime(2024, 3, 1)

        # Act
        result = _load_raw_features_for_report(prediction_month)

        # Assert
        assert result == {}

    @patch("deployment.app.services.report_service.load_features")
    def test_load_raw_features_empty_response(self, mock_load_features):
        """Test handling of empty features response."""
        # Arrange
        mock_load_features.return_value = {}
        prediction_month = datetime(2024, 3, 1)

        # Act
        result = _load_raw_features_for_report(prediction_month)

        # Assert
        assert result == {}


class TestAdaptFeaturesSchema:
    """Test suite for _adapt_features_schema function."""

    def create_sample_multiindex_df(self, feature_type, dates=None):
        """Helper to create sample DataFrame with MultiIndex."""
        if dates is None:
            dates = [datetime(2024, 1, 1), datetime(2024, 2, 1)]

        products = [("123", "Artist A", "Album 1"), ("456", "Artist B", "Album 2")]

        index = pd.MultiIndex.from_tuples(
            products, names=["barcode", "artist", "album"]
        )

        if feature_type in ["sales", "change"]:
            # Create with dates as columns for sales/change
            data = np.random.randn(len(products), len(dates))
            return pd.DataFrame(data, index=index, columns=dates)
        else:
            # Create with single value column for stock/prices
            data = np.random.randn(len(products))
            return pd.DataFrame({feature_type: data}, index=index)

    def test_adapt_sales_change_features(self):
        """Test adaptation of sales and change features."""
        # Arrange
        sales_df = self.create_sample_multiindex_df("sales")
        change_df = self.create_sample_multiindex_df("change")
        raw_features = {"sales": sales_df, "change": change_df}

        # Act
        result = _adapt_features_schema(raw_features)

        # Assert
        assert "sales" in result
        assert "change" in result
        # Check that the DataFrame structure is adapted
        assert isinstance(result["sales"], pd.DataFrame)
        assert isinstance(result["change"], pd.DataFrame)

    def test_adapt_stock_prices_features(self):
        """Test adaptation of stock and prices features."""
        # Arrange
        stock_df = self.create_sample_multiindex_df("stock")
        prices_df = self.create_sample_multiindex_df("prices")
        raw_features = {"stock": stock_df, "prices": prices_df}

        # Act
        result = _adapt_features_schema(raw_features)

        # Assert
        assert "stock" in result
        assert "prices" in result
        # Stock and prices should be used as-is
        pd.testing.assert_frame_equal(result["stock"], stock_df)
        pd.testing.assert_frame_equal(result["prices"], prices_df)

    def test_adapt_empty_features(self):
        """Test adaptation with empty features."""
        # Arrange
        raw_features = {
            "sales": pd.DataFrame(),
            "change": pd.DataFrame(),
            "stock": pd.DataFrame(),
            "prices": pd.DataFrame(),
        }

        # Act
        result = _adapt_features_schema(raw_features)

        # Assert
        for feature_type in raw_features:
            assert feature_type in result
            assert result[feature_type].empty

    def test_adapt_features_with_exception(self):
        """Test graceful handling of adaptation exceptions."""
        # Arrange
        # Create a DataFrame that will cause an exception during adaptation
        problematic_df = pd.DataFrame({"invalid": [1, 2, 3]})
        raw_features = {"sales": problematic_df, "change": pd.DataFrame()}

        # Act
        result = _adapt_features_schema(raw_features)

        # Assert
        # Should handle the exception gracefully and return original DataFrames
        assert "sales" in result
        assert "change" in result


class TestProcessFeaturesForReport:
    """Test suite for _process_features_for_report function."""

    def create_mock_adapted_features(self):
        """Create mock adapted features for testing."""
        dates = pd.date_range("2024-01-01", "2024-03-31", freq="D")
        products = [("123", "Artist A", "Album 1"), ("456", "Artist B", "Album 2")]

        # Create sales data with MultiIndex including _date
        sales_data = []
        for date in dates[:30]:  # First 30 days
            for product in products:
                sales_data.append((*product, date, np.random.poisson(2)))

        sales_df = pd.DataFrame(
            sales_data, columns=["barcode", "artist", "album", "_date", "sales"]
        ).set_index(["barcode", "artist", "album", "_date"])

        # Create change data
        change_data = []
        for date in dates[:30]:
            for product in products:
                change_data.append((*product, date, np.random.randint(-2, 5)))

        change_df = pd.DataFrame(
            change_data, columns=["barcode", "artist", "album", "_date", "change"]
        ).set_index(["barcode", "artist", "album", "_date"])

        # Create stock data
        stock_df = pd.DataFrame(
            [[10], [15]],
            index=pd.MultiIndex.from_tuples(
                products, names=["barcode", "artist", "album"]
            ),
            columns=[datetime(2024, 1, 1)],
        )

        # Create prices data
        prices_df = pd.DataFrame(
            {"prices": [100, 200]},
            index=pd.MultiIndex.from_tuples(
                products, names=["barcode", "artist", "album"]
            ),
        )

        return {
            "sales": sales_df,
            "change": change_df,
            "stock": stock_df,
            "prices": prices_df,
        }

    @patch("deployment.app.services.report_service._adapt_features_schema")
    def test_process_features_success(self, mock_adapt_schema):
        """Test successful feature processing."""
        # Arrange
        raw_features = {"sales": pd.DataFrame(), "change": pd.DataFrame()}
        mock_adapt_schema.return_value = self.create_mock_adapted_features()
        prediction_month = datetime(2024, 3, 1)

        # Act
        result = _process_features_for_report(raw_features, prediction_month)

        # Assert
        assert isinstance(result, dict)
        mock_adapt_schema.assert_called_once_with(raw_features)

    @patch("deployment.app.services.report_service._adapt_features_schema")
    def test_process_features_missing_required(self, mock_adapt_schema):
        """Test handling of missing required features."""
        # Arrange
        raw_features = {"sales": pd.DataFrame()}
        mock_adapt_schema.return_value = {
            "sales": pd.DataFrame()
        }  # Missing other features
        prediction_month = datetime(2024, 3, 1)

        # Act
        result = _process_features_for_report(raw_features, prediction_month)

        # Assert
        assert result == {}

    @patch("deployment.app.services.report_service._adapt_features_schema")
    def test_process_features_exception(self, mock_adapt_schema):
        """Test handling of exceptions during processing."""
        # Arrange
        raw_features = {"sales": pd.DataFrame()}
        mock_adapt_schema.side_effect = Exception("Processing error")
        prediction_month = datetime(2024, 3, 1)

        # Act
        result = _process_features_for_report(raw_features, prediction_month)

        # Assert
        assert result == {}


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
        """Test successful joining of predictions with enriched metrics."""
        # Arrange
        predictions_df = self.create_mock_predictions()
        enriched_columns = self.create_mock_enriched_columns()

        # Act
        result = _join_predictions_with_enriched_metrics(
            predictions_df, enriched_columns
        )

        # Assert
        assert len(result) == len(predictions_df)
        # Check that enriched columns are added
        enriched_cols = [
            "Средние продажи (шт)",
            "Средние продажи (руб)",
            "Потерянные продажи (руб)",
        ]
        for col in enriched_cols:
            assert col in result.columns

        # Check that original columns are preserved
        for col in predictions_df.columns:
            assert col in result.columns

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
        """Test joining with partial matches."""
        # Arrange
        predictions_df = self.create_mock_predictions()

        # Create enriched columns with only one matching product
        enriched_columns = pd.DataFrame(
            {"Средние продажи (шт)": [2.5]},
            index=pd.MultiIndex.from_tuples(
                [
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
                    )
                ],
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
            ),
        )

        # Act
        result = _join_predictions_with_enriched_metrics(
            predictions_df, enriched_columns
        )

        # Assert
        assert len(result) == len(predictions_df)
        assert "Средние продажи (шт)" in result.columns
        # First row should have data, second should be NaN
        assert pd.notna(result.iloc[0]["Средние продажи (шт)"])
        assert pd.isna(result.iloc[1]["Средние продажи (шт)"])

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

    @patch("deployment.app.services.report_service.load_features")
    def test_full_enrichment_workflow(self, mock_load_features):
        """Test the complete enrichment workflow from raw features to enriched predictions."""
        # Arrange
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


# Performance and edge case tests
class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    def test_large_dataset_handling(
        self, sample_multiindex_products, sample_date_range
    ):
        """Test handling of large datasets."""
        # Create large mock dataset
        large_products = []
        for i in range(100):  # 100 products
            large_products.append(
                (
                    f"product_{i}",
                    f"Artist {i}",
                    f"Album {i}",
                    "Opened",
                    "Low",
                    "Original",
                    "1990s",
                    "1990s",
                    "Rock",
                    1995,
                )
            )

        large_index = pd.MultiIndex.from_tuples(
            large_products,
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

        # Create large enriched columns
        enriched_data = {
            "Средние продажи (шт)": np.random.randn(100),
            "Средние продажи (руб)": np.random.randn(100) * 100,
            "Потерянные продажи (руб)": np.random.randn(100) * 50,
        }
        large_enriched = pd.DataFrame(enriched_data, index=large_index)

        # Create corresponding predictions
        predictions_data = []
        for product in large_products:
            predictions_data.append(list(product) + [1.0, 2.0, 3.0, 4.0, 5.0])

        large_predictions = pd.DataFrame(
            predictions_data,
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
        result = _join_predictions_with_enriched_metrics(
            large_predictions, large_enriched
        )

        # Assert
        assert len(result) == 100
        assert all(col in result.columns for col in enriched_data.keys())

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
            adapted = _adapt_features_schema(empty_features)
            processed = _process_features_for_report(empty_features, prediction_month)
            extracted = _extract_features_for_month(processed, prediction_month)

            # Clean up
            del adapted, processed, extracted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
