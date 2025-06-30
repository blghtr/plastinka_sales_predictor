"""
Comprehensive tests for DataFrame comparison utilities.

This test suite covers all utility functions used for comparing DataFrames including
index comparison, numeric/categorical column comparison, comprehensive reporting, and
enhanced assertion methods. Tests are organized by function groups and include both
success and failure scenarios.

Testing Approach:
1. Index Comparison: Tests compare_indices function with various DataFrame index scenarios
2. Numeric Columns: Tests compare_numeric_columns function with statistical analysis
3. Categorical Columns: Tests compare_categorical_columns function with value differences
4. Comprehensive Reporting: Tests dataframe_comparison_report function for full analysis
5. Enhanced Assertions: Tests assert_dataframes_equal_with_details function with detailed errors
6. Integration Testing: Tests end-to-end DataFrame comparison functionality

All functions are tested with mocked data to ensure test isolation and performance.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

# Import utility functions to test - they're defined in this same module
# We'll reference them directly since they're in the same file


class TestCompareIndices:
    """Test suite for compare_indices function."""

    def test_compare_identical_indices(self):
        """Test comparison of DataFrames with identical indices."""
        # Arrange
        df1 = pd.DataFrame({"A": [1, 2, 3]}, index=[0, 1, 2])
        df2 = pd.DataFrame({"B": [4, 5, 6]}, index=[0, 1, 2])

        # Act
        result = compare_indices(df1, df2)

        # Assert
        assert result["missing"] == []
        assert result["extra"] == []

    def test_compare_missing_indices(self):
        """Test comparison with missing indices in actual DataFrame."""
        # Arrange
        actual = pd.DataFrame({"A": [1, 2]}, index=[0, 1])
        expected = pd.DataFrame({"B": [4, 5, 6]}, index=[0, 1, 2])

        # Act
        result = compare_indices(actual, expected)

        # Assert
        assert result["missing"] == [2]
        assert result["extra"] == []

    def test_compare_extra_indices(self):
        """Test comparison with extra indices in actual DataFrame."""
        # Arrange
        actual = pd.DataFrame({"A": [1, 2, 3, 4]}, index=[0, 1, 2, 3])
        expected = pd.DataFrame({"B": [4, 5]}, index=[0, 1])

        # Act
        result = compare_indices(actual, expected)

        # Assert
        assert result["missing"] == []
        assert result["extra"] == [2, 3]

    def test_compare_string_indices(self):
        """Test comparison with string indices."""
        # Arrange
        actual = pd.DataFrame({"A": [1, 2]}, index=["a", "b"])
        expected = pd.DataFrame({"B": [4, 5]}, index=["b", "c"])

        # Act
        result = compare_indices(actual, expected)

        # Assert
        assert result["missing"] == ["c"]
        assert result["extra"] == ["a"]


class TestCompareNumericColumns:
    """Test suite for compare_numeric_columns function."""

    def test_compare_identical_numeric_columns(self):
        """Test comparison of identical numeric columns."""
        # Arrange
        df1 = pd.DataFrame({"num1": [1, 2, 3], "num2": [4, 5, 6]})
        df2 = pd.DataFrame({"num1": [1, 2, 3], "num2": [4, 5, 6]})

        # Act
        result = compare_numeric_columns(df1, df2)

        # Assert
        assert result == {}  # No differences

    def test_compare_different_numeric_columns(self):
        """Test comparison of numeric columns with statistical differences."""
        # Arrange
        actual = pd.DataFrame({"values": [1, 2, 3, 4, 5]})
        expected = pd.DataFrame({"values": [2, 3, 4, 5, 6]})
        tolerance = 1e-6

        # Act
        result = compare_numeric_columns(actual, expected, tolerance=tolerance)

        # Assert
        assert "values" in result
        assert "mean" in result["values"]
        assert result["values"]["mean"]["actual"] == 3.0
        assert result["values"]["mean"]["expected"] == 4.0
        assert result["values"]["mean"]["diff"] == -1.0

    def test_compare_custom_columns_list(self):
        """Test comparison with specified column list."""
        # Arrange
        df1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6]})
        df2 = pd.DataFrame({"col1": [1, 2], "col2": [7, 8], "col3": [5, 6]})

        # Act
        result = compare_numeric_columns(df1, df2, columns=["col2"])

        # Assert
        assert "col2" in result
        assert "col1" not in result
        assert "col3" not in result

    def test_compare_with_tolerance(self):
        """Test comparison respects tolerance parameter."""
        # Arrange
        df1 = pd.DataFrame({"values": [1.0, 2.0]})
        df2 = pd.DataFrame({"values": [1.0001, 2.0001]})

        # Act - high tolerance, should show no differences
        result_high_tol = compare_numeric_columns(df1, df2, tolerance=1e-2)

        # Act - low tolerance, should show differences
        result_low_tol = compare_numeric_columns(df1, df2, tolerance=1e-6)

        # Assert
        assert result_high_tol == {}
        assert "values" in result_low_tol

    def test_compare_with_missing_columns(self):
        """Test comparison when columns don't exist in both DataFrames."""
        # Arrange
        df1 = pd.DataFrame({"col1": [1, 2]})
        df2 = pd.DataFrame({"col2": [3, 4]})

        # Act
        result = compare_numeric_columns(df1, df2)

        # Assert
        assert result == {}  # No shared columns


class TestCompareCategoricalColumns:
    """Test suite for compare_categorical_columns function."""

    def test_compare_identical_categorical_columns(self):
        """Test comparison of identical categorical columns."""
        # Arrange
        df1 = pd.DataFrame({"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]})
        df2 = pd.DataFrame({"cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]})

        # Act
        result = compare_categorical_columns(df1, df2)

        # Assert
        assert result == {}  # No differences

    def test_compare_different_categorical_values(self):
        """Test comparison with different categorical values."""
        # Arrange
        actual = pd.DataFrame({"category": ["A", "B", "C"]})
        expected = pd.DataFrame({"category": ["B", "C", "D"]})

        # Act
        result = compare_categorical_columns(actual, expected)

        # Assert
        assert "category" in result
        assert result["category"]["missing_values"] == ["D"]
        assert result["category"]["extra_values"] == ["A"]
        assert result["category"]["actual_cardinality"] == 3
        assert result["category"]["expected_cardinality"] == 3

    def test_compare_custom_categorical_columns(self):
        """Test comparison with specified categorical columns."""
        # Arrange
        df1 = pd.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"], "num": [1, 2]})
        df2 = pd.DataFrame({"cat1": ["A", "B"], "cat2": ["Y", "Z"], "num": [3, 4]})

        # Act
        result = compare_categorical_columns(df1, df2, columns=["cat2"])

        # Assert
        assert "cat2" in result
        assert "cat1" not in result

    def test_compare_with_null_values(self):
        """Test comparison handles null values properly."""
        # Arrange
        df1 = pd.DataFrame({"category": ["A", "B", np.nan]})
        df2 = pd.DataFrame({"category": ["A", np.nan, "C"]})

        # Act
        result = compare_categorical_columns(df1, df2)

        # Assert
        assert "category" in result
        assert result["category"]["missing_values"] == ["C"]
        assert result["category"]["extra_values"] == ["B"]


class TestDataframeComparisonReport:
    """Test suite for dataframe_comparison_report function."""

    def test_comprehensive_comparison_report(self):
        """Test generation of comprehensive comparison report."""
        # Arrange
        actual = pd.DataFrame(
            {"num_col": [1, 2, 3], "cat_col": ["A", "B", "C"], "extra_col": [7, 8, 9]},
            index=[0, 1, 2],
        )

        expected = pd.DataFrame(
            {
                "num_col": [2, 3, 4],
                "cat_col": ["B", "C", "D"],
                "missing_col": [10, 11, 12],
            },
            index=[1, 2, 3],
        )

        # Act
        report = dataframe_comparison_report(actual, expected)

        # Assert
        # Shape comparison
        assert report["shape"]["actual"] == (3, 3)
        assert report["shape"]["expected"] == (3, 3)
        assert report["shape"]["difference"] == (0, 0)

        # Column comparison
        assert "missing_col" in report["columns"]["missing"]
        assert "extra_col" in report["columns"]["extra"]
        assert "num_col" in report["columns"]["common"]
        assert "cat_col" in report["columns"]["common"]

        # Index comparison
        assert 0 in report["indices"]["extra"]
        assert 3 in report["indices"]["missing"]

        # Numeric comparison
        assert "num_col" in report["numeric"]

        # Categorical comparison
        assert "cat_col" in report["categorical"]

    def test_identical_dataframes_report(self):
        """Test report for identical DataFrames."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": ["X", "Y"]})

        # Act
        report = dataframe_comparison_report(df, df)

        # Assert
        assert report["shape"]["difference"] == (0, 0)
        assert report["columns"]["missing"] == []
        assert report["columns"]["extra"] == []
        assert report["indices"]["missing"] == []
        assert report["indices"]["extra"] == []
        assert report["numeric"] == {}
        assert report["categorical"] == {}


class TestAssertDataframesEqualWithDetails:
    """Test suite for assert_dataframes_equal_with_details function."""

    def test_assert_identical_dataframes_success(self):
        """Test assertion passes for identical DataFrames."""
        # Arrange
        df = pd.DataFrame({"A": [1, 2], "B": ["X", "Y"]})

        # Act & Assert - should not raise
        assert_dataframes_equal_with_details(df, df)

    def test_assert_different_dataframes_detailed_error(self):
        """Test assertion provides detailed error for different DataFrames."""
        # Arrange
        actual = pd.DataFrame({"A": [1, 2, 3]})
        expected = pd.DataFrame({"A": [2, 3, 4]})

        # Act & Assert
        with pytest.raises(AssertionError) as exc_info:
            assert_dataframes_equal_with_details(actual, expected)

        error_msg = str(exc_info.value)
        assert "DataFrame comparison failed" in error_msg
        assert "Detailed differences" in error_msg

    def test_assert_shape_differences_in_error(self):
        """Test assertion includes shape differences in error message."""
        # Arrange
        actual = pd.DataFrame({"A": [1, 2]})
        expected = pd.DataFrame({"A": [1, 2, 3]})

        # Act & Assert
        with pytest.raises(AssertionError) as exc_info:
            assert_dataframes_equal_with_details(actual, expected)

        error_msg = str(exc_info.value)
        assert "Shape difference" in error_msg
        assert "(2, 1)" in error_msg
        assert "(3, 1)" in error_msg

    def test_assert_column_differences_in_error(self):
        """Test assertion includes column differences in error message."""
        # Arrange
        actual = pd.DataFrame({"A": [1, 2], "extra": [3, 4]})
        expected = pd.DataFrame({"A": [1, 2], "missing": [5, 6]})

        # Act & Assert
        with pytest.raises(AssertionError) as exc_info:
            assert_dataframes_equal_with_details(actual, expected)

        error_msg = str(exc_info.value)
        assert "Missing columns" in error_msg
        assert "Extra columns" in error_msg
        assert "missing" in error_msg
        assert "extra" in error_msg

    def test_assert_check_dtype_parameter(self):
        """Test assertion respects check_dtype parameter."""
        # Arrange
        df1 = pd.DataFrame({"A": [1, 2]})  # int64
        df2 = pd.DataFrame({"A": [1.0, 2.0]})  # float64

        # Act & Assert - should not raise when check_dtype=False
        assert_dataframes_equal_with_details(df1, df2, check_dtype=False)

        # Act & Assert - should raise when check_dtype=True
        with pytest.raises(AssertionError):
            assert_dataframes_equal_with_details(df1, df2, check_dtype=True)


class TestIntegration:
    """Integration tests for DataFrame comparison utilities."""

    def test_module_imports_successfully(self):
        """Test that the module can be imported without errors."""
        # Arrange & Act & Assert
        import tests.plastinka_sales_predictor.data_preparation.test_utils

        assert tests.plastinka_sales_predictor.data_preparation.test_utils is not None

    def test_all_functions_available(self):
        """Test that all expected utility functions are available."""
        # Arrange
        expected_functions = [
            "compare_indices",
            "compare_numeric_columns",
            "compare_categorical_columns",
            "dataframe_comparison_report",
            "assert_dataframes_equal_with_details",
        ]

        # Act & Assert
        import tests.plastinka_sales_predictor.data_preparation.test_utils as utils_module

        for func_name in expected_functions:
            assert hasattr(utils_module, func_name), (
                f"Function {func_name} should be available"
            )
            assert callable(getattr(utils_module, func_name)), (
                f"Function {func_name} should be callable"
            )

    def test_dataframe_workflow_integration(self):
        """Test end-to-end DataFrame comparison workflow."""
        # Arrange
        df1 = pd.DataFrame({"nums": [1, 2, 3], "cats": ["A", "B", "C"]})
        df2 = pd.DataFrame({"nums": [1.1, 2.1, 3.1], "cats": ["A", "B", "D"]})

        # Act - generate full report
        report = dataframe_comparison_report(df1, df2)

        # Act - individual comparisons
        idx_diff = compare_indices(df1, df2)
        num_diff = compare_numeric_columns(df1, df2, tolerance=1e-6)
        cat_diff = compare_categorical_columns(df1, df2)

        # Assert - all components work together
        assert report["indices"] == idx_diff
        assert report["numeric"] == num_diff
        assert report["categorical"] == cat_diff

        # Verify specific differences detected
        assert "nums" in num_diff  # Should detect numeric differences
        assert "cats" in cat_diff  # Should detect categorical differences
        assert cat_diff["cats"]["missing_values"] == ["D"]
        assert cat_diff["cats"]["extra_values"] == ["C"]


# Utility functions that these tests are designed to test
def compare_indices(
    actual: pd.DataFrame, expected: pd.DataFrame
) -> dict[str, list[Any]]:
    """
    Compares indices between two DataFrames and returns differences.

    Args:
        actual: The actual DataFrame
        expected: The expected DataFrame

    Returns:
        Dict with keys 'missing' and 'extra' containing lists of indices
    """
    actual_indices = set(actual.index)
    expected_indices = set(expected.index)

    return {
        "missing": sorted(expected_indices - actual_indices),
        "extra": sorted(actual_indices - expected_indices),
    }


def compare_numeric_columns(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    columns: list[str] | None = None,
    tolerance: float = 1e-6,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Compares numeric columns between DataFrames and returns statistical differences.

    Args:
        actual: The actual DataFrame
        expected: The expected DataFrame
        columns: List of numeric columns to compare (if None, all shared numeric columns)
        tolerance: Tolerance for considering differences significant

    Returns:
        Nested dict with column -> stat -> difference information
    """
    # Identify shared numeric columns if not specified
    if columns is None:
        actual_numeric = actual.select_dtypes(include=np.number).columns
        expected_numeric = expected.select_dtypes(include=np.number).columns
        columns = sorted(set(actual_numeric).intersection(set(expected_numeric)))

    result = {}
    stats = ["mean", "median", "min", "max", "std"]

    for col in columns:
        if col in actual.columns and col in expected.columns:
            # Calculate stats for each DataFrame
            actual_stats = {
                "mean": actual[col].mean(),
                "median": actual[col].median(),
                "min": actual[col].min(),
                "max": actual[col].max(),
                "std": actual[col].std(),
            }

            expected_stats = {
                "mean": expected[col].mean(),
                "median": expected[col].median(),
                "min": expected[col].min(),
                "max": expected[col].max(),
                "std": expected[col].std(),
            }

            # Compare stats
            diff = {}
            for stat in stats:
                diff_value = actual_stats[stat] - expected_stats[stat]
                if abs(diff_value) > tolerance:
                    diff[stat] = {
                        "actual": actual_stats[stat],
                        "expected": expected_stats[stat],
                        "diff": diff_value,
                        "pct_diff": diff_value / expected_stats[stat] * 100
                        if expected_stats[stat] != 0
                        else float("inf"),
                    }

            if diff:
                result[col] = diff

    return result


def compare_categorical_columns(
    actual: pd.DataFrame, expected: pd.DataFrame, columns: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """
    Compares categorical columns between DataFrames and returns differences in values.

    Args:
        actual: The actual DataFrame
        expected: The expected DataFrame
        columns: List of categorical columns to compare (if None, all columns with object or category dtype)

    Returns:
        Dict with information about differences in categorical columns
    """
    # Identify categorical columns if not specified
    if columns is None:
        actual_cat = set(actual.select_dtypes(include=["object", "category"]).columns)
        expected_cat = set(
            expected.select_dtypes(include=["object", "category"]).columns
        )
        columns = sorted(actual_cat.union(expected_cat))

    result = {}

    for col in columns:
        if col in actual.columns and col in expected.columns:
            # Get unique values
            actual_values = set(actual[col].dropna().unique())
            expected_values = set(expected[col].dropna().unique())

            missing_values = expected_values - actual_values
            extra_values = actual_values - expected_values

            if missing_values or extra_values:
                result[col] = {
                    "missing_values": sorted(missing_values),
                    "extra_values": sorted(extra_values),
                    "actual_cardinality": len(actual_values),
                    "expected_cardinality": len(expected_values),
                }

    return result


def dataframe_comparison_report(
    actual: pd.DataFrame, expected: pd.DataFrame
) -> dict[str, Any]:
    """
    Generates a comprehensive comparison report between two DataFrames.

    Args:
        actual: The actual DataFrame
        expected: The expected DataFrame

    Returns:
        Dict with detailed comparison information
    """
    report = {
        "shape": {
            "actual": actual.shape,
            "expected": expected.shape,
            "difference": (
                actual.shape[0] - expected.shape[0],
                actual.shape[1] - expected.shape[1],
            ),
        },
        "columns": {
            "missing": sorted(set(expected.columns) - set(actual.columns)),
            "extra": sorted(set(actual.columns) - set(expected.columns)),
            "common": sorted(set(actual.columns).intersection(set(expected.columns))),
        },
    }

    # Compare indices
    report["indices"] = compare_indices(actual, expected)

    # Compare numeric columns
    report["numeric"] = compare_numeric_columns(actual, expected)

    # Compare categorical columns
    report["categorical"] = compare_categorical_columns(actual, expected)

    return report


def assert_dataframes_equal_with_details(
    actual: pd.DataFrame, expected: pd.DataFrame, check_dtype: bool = False
):
    """
    Asserts that two DataFrames are equal, with detailed error message if they're not.

    Args:
        actual: The actual DataFrame
        expected: The expected DataFrame
        check_dtype: Whether to check dtypes (passed to pd.testing.assert_frame_equal)

    Raises:
        AssertionError: If the DataFrames are not equal, with detailed information
    """
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=check_dtype)
    except AssertionError as e:
        # Generate detailed comparison report
        report = dataframe_comparison_report(actual, expected)

        # Format error message
        error_msg = f"DataFrame comparison failed:\n{str(e)}\n\nDetailed differences:\n"

        # Shape differences
        shape_diff = report["shape"]["difference"]
        if shape_diff != (0, 0):
            error_msg += f"- Shape difference: actual {report['shape']['actual']} vs expected {report['shape']['expected']}\n"

        # Column differences
        if report["columns"]["missing"] or report["columns"]["extra"]:
            error_msg += f"- Missing columns: {report['columns']['missing']}\n"
            error_msg += f"- Extra columns: {report['columns']['extra']}\n"

        # Index differences
        if report["indices"]["missing"] or report["indices"]["extra"]:
            error_msg += f"- Missing indices: {report['indices']['missing'][:10]}{'...' if len(report['indices']['missing']) > 10 else ''}\n"
            error_msg += f"- Extra indices: {report['indices']['extra'][:10]}{'...' if len(report['indices']['extra']) > 10 else ''}\n"

        # Numeric column differences
        if report["numeric"]:
            error_msg += "- Numeric column differences:\n"
            for col, stats in report["numeric"].items():
                error_msg += f"  - {col}:\n"
                for stat, values in stats.items():
                    error_msg += f"    - {stat}: actual={values['actual']:.6f}, expected={values['expected']:.6f}, diff={values['diff']:.6f} ({values['pct_diff']:.2f}%)\n"

        # Categorical column differences
        if report["categorical"]:
            error_msg += "- Categorical column differences:\n"
            for col, info in report["categorical"].items():
                error_msg += f"  - {col}: cardinality actual={info['actual_cardinality']}, expected={info['expected_cardinality']}\n"
                if info["missing_values"]:
                    error_msg += f"    - Missing values: {info['missing_values'][:5]}{'...' if len(info['missing_values']) > 5 else ''}\n"
                if info["extra_values"]:
                    error_msg += f"    - Extra values: {info['extra_values'][:5]}{'...' if len(info['extra_values']) > 5 else ''}\n"

        raise AssertionError(error_msg) from e


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
