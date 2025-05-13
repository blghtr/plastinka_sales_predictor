import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional


def compare_indices(actual: pd.DataFrame, expected: pd.DataFrame) -> Dict[str, List[Any]]:
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
        'missing': sorted(list(expected_indices - actual_indices)),
        'extra': sorted(list(actual_indices - expected_indices))
    }


def compare_numeric_columns(
    actual: pd.DataFrame, 
    expected: pd.DataFrame,
    columns: Optional[List[str]] = None,
    tolerance: float = 1e-6
) -> Dict[str, Dict[str, Dict[str, float]]]:
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
        columns = sorted(list(set(actual_numeric).intersection(set(expected_numeric))))
    
    result = {}
    stats = ['mean', 'median', 'min', 'max', 'std']
    
    for col in columns:
        if col in actual.columns and col in expected.columns:
            # Calculate stats for each DataFrame
            actual_stats = {
                'mean': actual[col].mean(),
                'median': actual[col].median(),
                'min': actual[col].min(),
                'max': actual[col].max(),
                'std': actual[col].std()
            }
            
            expected_stats = {
                'mean': expected[col].mean(),
                'median': expected[col].median(),
                'min': expected[col].min(),
                'max': expected[col].max(),
                'std': expected[col].std()
            }
            
            # Compare stats
            diff = {}
            for stat in stats:
                diff_value = actual_stats[stat] - expected_stats[stat]
                if abs(diff_value) > tolerance:
                    diff[stat] = {
                        'actual': actual_stats[stat],
                        'expected': expected_stats[stat],
                        'diff': diff_value,
                        'pct_diff': diff_value / expected_stats[stat] * 100 if expected_stats[stat] != 0 else float('inf')
                    }
            
            if diff:
                result[col] = diff
    
    return result


def compare_categorical_columns(
    actual: pd.DataFrame, 
    expected: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
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
        actual_cat = set(actual.select_dtypes(include=['object', 'category']).columns)
        expected_cat = set(expected.select_dtypes(include=['object', 'category']).columns)
        columns = sorted(list(actual_cat.union(expected_cat)))
    
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
                    'missing_values': sorted(list(missing_values)),
                    'extra_values': sorted(list(extra_values)),
                    'actual_cardinality': len(actual_values),
                    'expected_cardinality': len(expected_values)
                }
    
    return result


def dataframe_comparison_report(actual: pd.DataFrame, expected: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a comprehensive comparison report between two DataFrames.
    
    Args:
        actual: The actual DataFrame
        expected: The expected DataFrame
        
    Returns:
        Dict with detailed comparison information
    """
    report = {
        'shape': {
            'actual': actual.shape,
            'expected': expected.shape,
            'difference': (
                actual.shape[0] - expected.shape[0],
                actual.shape[1] - expected.shape[1]
            )
        },
        'columns': {
            'missing': sorted(list(set(expected.columns) - set(actual.columns))),
            'extra': sorted(list(set(actual.columns) - set(expected.columns))),
            'common': sorted(list(set(actual.columns).intersection(set(expected.columns))))
        }
    }
    
    # Compare indices
    report['indices'] = compare_indices(actual, expected)
    
    # Compare numeric columns
    report['numeric'] = compare_numeric_columns(actual, expected)
    
    # Compare categorical columns
    report['categorical'] = compare_categorical_columns(actual, expected)
    
    return report


def assert_dataframes_equal_with_details(actual: pd.DataFrame, expected: pd.DataFrame, check_dtype: bool = False):
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
        shape_diff = report['shape']['difference']
        if shape_diff != (0, 0):
            error_msg += f"- Shape difference: actual {report['shape']['actual']} vs expected {report['shape']['expected']}\n"
        
        # Column differences
        if report['columns']['missing'] or report['columns']['extra']:
            error_msg += f"- Missing columns: {report['columns']['missing']}\n"
            error_msg += f"- Extra columns: {report['columns']['extra']}\n"
        
        # Index differences
        if report['indices']['missing'] or report['indices']['extra']:
            error_msg += f"- Missing indices: {report['indices']['missing'][:10]}{'...' if len(report['indices']['missing']) > 10 else ''}\n"
            error_msg += f"- Extra indices: {report['indices']['extra'][:10]}{'...' if len(report['indices']['extra']) > 10 else ''}\n"
        
        # Numeric column differences
        if report['numeric']:
            error_msg += "- Numeric column differences:\n"
            for col, stats in report['numeric'].items():
                error_msg += f"  - {col}:\n"
                for stat, values in stats.items():
                    error_msg += f"    - {stat}: actual={values['actual']:.6f}, expected={values['expected']:.6f}, diff={values['diff']:.6f} ({values['pct_diff']:.2f}%)\n"
        
        # Categorical column differences
        if report['categorical']:
            error_msg += "- Categorical column differences:\n"
            for col, info in report['categorical'].items():
                error_msg += f"  - {col}: cardinality actual={info['actual_cardinality']}, expected={info['expected_cardinality']}\n"
                if info['missing_values']:
                    error_msg += f"    - Missing values: {info['missing_values'][:5]}{'...' if len(info['missing_values']) > 5 else ''}\n"
                if info['extra_values']:
                    error_msg += f"    - Extra values: {info['extra_values'][:5]}{'...' if len(info['extra_values']) > 5 else ''}\n"
        
        raise AssertionError(error_msg) 