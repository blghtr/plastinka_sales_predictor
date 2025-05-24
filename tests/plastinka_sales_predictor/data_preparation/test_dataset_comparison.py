import pandas as pd
import numpy as np
import pytest
import json
import pickle
from pathlib import Path
from plastinka_sales_predictor.data_preparation import PlastinkaTrainingTSDataset, MultiColumnLabelBinarizer, GlobalLogMinMaxScaler
from tests.plastinka_sales_predictor.data_preparation.test_utils import compare_numeric_columns, compare_categorical_columns
from darts import TimeSeries

# Define path to example data
ISOLATED_TESTS_BASE_DIR = Path("tests/example_data/isolated_tests")
GENERAL_EXAMPLES_DIR = Path("generated_general_examples")

def _load_artifact(file_path: Path):
    """Loads a pickled or JSON artifact based on its extension."""
    if not file_path.exists():
        pytest.fail(f"Artifact file not found: {file_path}")
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        pytest.fail(f"Unsupported artifact file extension: {file_path.suffix} for file {file_path}. Only .json and .pkl are supported.")

def compare_timeseries(actual_ts, expected_ts, tolerance=1e-6):
    """
    Compare two TimeSeries-like objects and return their differences.
    
    Args:
        actual_ts: The actual TimeSeries or dict representation
        expected_ts: The expected TimeSeries or dict representation
        tolerance: Tolerance for numeric differences
        
    Returns:
        Dict with difference information or None if no significant differences
    """
    # Extract values for comparison
    if isinstance(actual_ts, dict) and 'type' in actual_ts and actual_ts['type'] == 'TimeSeries':
        actual_values = np.array(actual_ts['values']) if actual_ts['values'] is not None else None
        actual_shape = tuple(actual_ts['shape']) if 'shape' in actual_ts and actual_ts['shape'] is not None else None
        actual_start = actual_ts.get('start')
        actual_end = actual_ts.get('end')
        actual_freq = actual_ts.get('freq')
    elif isinstance(actual_ts, TimeSeries):
        actual_values = actual_ts.values()
        actual_shape = actual_values.shape if actual_values is not None else None
        actual_start = actual_ts.start_time().isoformat() if actual_ts.has_datetime_index else None
        actual_end = actual_ts.end_time().isoformat() if actual_ts.has_datetime_index else None
        actual_freq = str(actual_ts.freq) if actual_ts.has_datetime_index else None
    else:
        return {"error": f"Unsupported type for actual: {type(actual_ts)}"}
    
    if isinstance(expected_ts, dict) and 'type' in expected_ts and expected_ts['type'] == 'TimeSeries':
        expected_values = np.array(expected_ts['values']) if expected_ts['values'] is not None else None
        expected_shape = tuple(expected_ts['shape']) if 'shape' in expected_ts and expected_ts['shape'] is not None else None
        expected_start = expected_ts.get('start')
        expected_end = expected_ts.get('end')
        expected_freq = expected_ts.get('freq')
    elif isinstance(expected_ts, TimeSeries):
        expected_values = expected_ts.values()
        expected_shape = expected_values.shape if expected_values is not None else None
        expected_start = expected_ts.start_time().isoformat() if expected_ts.has_datetime_index else None
        expected_end = expected_ts.end_time().isoformat() if expected_ts.has_datetime_index else None
        expected_freq = str(expected_ts.freq) if expected_ts.has_datetime_index else None
    else:
        return {"error": f"Unsupported type for expected: {type(expected_ts)}"}
    
    # Compare shape
    differences = {}
    if actual_shape != expected_shape:
        differences['shape'] = {
            'actual': actual_shape,
            'expected': expected_shape
        }
        return differences  # If shapes differ, no need to compare values
    
    # Compare values if both are not None
    if actual_values is not None and expected_values is not None:
        # Calculate differences
        abs_diff = np.abs(actual_values - expected_values)
        max_diff = np.max(abs_diff) if abs_diff.size > 0 else 0
        
        if max_diff > tolerance:
            # Calculate various statistics for the differences
            differences['values'] = {
                'max_diff': float(max_diff),
                'mean_diff': float(np.mean(abs_diff)),
                'std_diff': float(np.std(abs_diff)),
                'actual_mean': float(np.mean(actual_values)),
                'expected_mean': float(np.mean(expected_values)),
                'actual_std': float(np.std(actual_values)),
                'expected_std': float(np.std(expected_values))
            }
    
    # Compare metadata
    if actual_start != expected_start:
        differences['start'] = {
            'actual': actual_start,
            'expected': expected_start
        }
    
    if actual_end != expected_end:
        differences['end'] = {
            'actual': actual_end,
            'expected': expected_end
        }
    
    if actual_freq != expected_freq:
        differences['freq'] = {
            'actual': actual_freq,
            'expected': expected_freq
        }
    
    return differences if differences else None


def compare_dataset_values(actual_dict, expected_dict, tolerance=1e-6):
    """
    Compare the actual and expected dataset dictionaries with detailed feedback.
    
    Args:
        actual_dict: The actual dataset dictionary
        expected_dict: The expected dataset dictionary
        tolerance: Tolerance for numeric differences
        
    Returns:
        Dict with detailed comparison results
    """
    # First check if keys match
    actual_keys = set(actual_dict.keys())
    expected_keys = set(expected_dict.keys())
    
    comparison = {
        'missing_keys': sorted(list(expected_keys - actual_keys)),
        'extra_keys': sorted(list(actual_keys - expected_keys)),
        'length_differences': {},
        'sample_differences': {}
    }
    
    # Check lengths for all keys
    common_keys = actual_keys.intersection(expected_keys)
    for key in common_keys:
        actual_len = len(actual_dict[key])
        expected_len = len(expected_dict[key])
        
        if actual_len != expected_len:
            comparison['length_differences'][key] = {
                'actual': actual_len,
                'expected': expected_len,
                'diff': actual_len - expected_len,
                'pct_diff': (actual_len - expected_len) / expected_len * 100 if expected_len != 0 else float('inf')
            }
    
    # Sample check: For each list, compare a sample of items
    for key in common_keys:
        if len(actual_dict[key]) == 0 or len(expected_dict[key]) == 0:
            continue
            
        # Determine sample size (min of 5 or the full length)
        sample_size = min(5, len(actual_dict[key]), len(expected_dict[key]))
        
        # For each key, take the first n items as samples
        actual_samples = actual_dict[key][:sample_size]
        expected_samples = expected_dict[key][:sample_size]
        
        # Check item types and compare accordingly
        item_differences = []
        
        for i, (act, exp) in enumerate(zip(actual_samples, expected_samples)):
            # Handle TimeSeries objects
            if isinstance(act, TimeSeries) or (isinstance(act, dict) and 'type' in act and act['type'] == 'TimeSeries'):
                ts_diff = compare_timeseries(act, exp, tolerance)
                if ts_diff:
                    item_differences.append({
                        'index': i,
                        'type': 'TimeSeries',
                        'differences': ts_diff
                    })
            
            # PRIORITY: Handle list/tuple of mixed types (like labels)
            elif isinstance(act, (list, tuple)) and isinstance(exp, (list, tuple)) and \
                 not (isinstance(act, TimeSeries) or (isinstance(act, dict) and act.get('type') == 'TimeSeries')) and \
                 not (isinstance(exp, TimeSeries) or (isinstance(exp, dict) and exp.get('type') == 'TimeSeries')) and \
                 not isinstance(act, np.ndarray) and not isinstance(exp, np.ndarray) and \
                 not all(isinstance(x, (int, float, np.number)) for x in (act if isinstance(act, (list, tuple)) else act.flatten())) and \
                 not all(isinstance(x, (int, float, np.number)) for x in (exp if isinstance(exp, (list, tuple)) else exp.flatten())):
                
                act_comp = list(act)
                exp_comp = list(exp)
                if act_comp != exp_comp:
                    item_differences.append({
                        'index': i,
                        'type': 'list/tuple', 
                        'differences': {
                            'actual': str(act),
                            'expected': str(exp)
                        }
                    })

            # Handle numpy arrays or lists of PURELY numbers
            elif isinstance(act, (np.ndarray, list, tuple)) and all(isinstance(x, (int, float, np.number)) for x in (act if isinstance(act, (list, tuple)) else act.flatten())):
                # Convert to numpy arrays for comparison
                act_array = np.array(act)
                exp_array = np.array(exp)
                
                # Check shapes
                if act_array.shape != exp_array.shape:
                    item_differences.append({
                        'index': i,
                        'type': 'array',
                        'shape_difference': {
                            'actual': act_array.shape,
                            'expected': exp_array.shape
                        }
                    })
                    continue
                
                # Calculate differences
                abs_diff = np.abs(act_array - exp_array)
                max_diff = np.max(abs_diff) if abs_diff.size > 0 else 0
                
                if max_diff > tolerance:
                    item_differences.append({
                        'index': i,
                        'type': 'array',
                        'value_differences': {
                            'max_diff': float(max_diff),
                            'mean_diff': float(np.mean(abs_diff)),
                            'std_diff': float(np.std(abs_diff)),
                            'actual_mean': float(np.mean(act_array)),
                            'expected_mean': float(np.mean(exp_array)),
                            'actual_std': float(np.std(act_array)),
                            'expected_std': float(np.std(exp_array))
                        }
                    })
            
            # Handle scalar values (int, float)
            elif isinstance(act, (int, float, np.number)) and isinstance(exp, (int, float, np.number)):
                abs_diff = abs(float(act) - float(exp))
                if abs_diff > tolerance:
                    item_differences.append({
                        'index': i,
                        'type': 'scalar',
                        'differences': {
                            'actual': float(act) if isinstance(act, np.number) else act,
                            'expected': float(exp) if isinstance(exp, np.number) else exp,
                            'diff': abs_diff
                        }
                    })
            
            # Handle other scalar values (str, bool, etc.)
            # Convert to list if both are list-like (tuple or list) to allow tuple vs list comparison
            # Ensure they are not TimeSeries or np.ndarray as those are handled specifically earlier
            elif isinstance(act, (list, tuple)) and isinstance(exp, (list, tuple)) and \
                 not (isinstance(act, TimeSeries) or (isinstance(act, dict) and act.get('type') == 'TimeSeries')) and \
                 not (isinstance(exp, TimeSeries) or (isinstance(exp, dict) and exp.get('type') == 'TimeSeries')) and \
                 not isinstance(act, np.ndarray) and not isinstance(exp, np.ndarray):
                
                act_comp = list(act)
                exp_comp = list(exp)
                if act_comp != exp_comp:
                    item_differences.append({
                        'index': i,
                        'type': 'list/tuple', # Indicate it's a list/tuple comparison
                        'differences': {
                            'actual': str(act),   # Report original types
                            'expected': str(exp)
                        }
                    })
            elif act != exp:
                # This will now catch other non-numeric, non-list-like differences
                if (isinstance(act, str) and isinstance(exp, str)) or \
                   (isinstance(act, bool) and isinstance(exp, bool)) or \
                   (type(act) == type(exp)): # General catch for same-type scalars
                    item_differences.append({
                        'index': i,
                        'type': 'scalar',
                        'differences': {
                            'actual': str(act),
                            'expected': str(exp),
                            'diff': None # Diff may not be meaningful for all scalars
                        }
                    })
                else: # Different types that are not list/tuple
                    item_differences.append({
                        'index': i,
                        'type': 'scalar',
                        'differences': {
                            'actual': str(act),
                            'expected': str(exp)
                        }
                    })
            
            # Handle other types (e.g., dicts)
            elif isinstance(act, dict) and isinstance(exp, dict):
                # Compare dict keys
                act_keys = set(act.keys())
                exp_keys = set(exp.keys())
                
                missing_keys = sorted(list(exp_keys - act_keys))
                extra_keys = sorted(list(act_keys - exp_keys))
                
                if missing_keys or extra_keys:
                    item_differences.append({
                        'index': i,
                        'type': 'dict',
                        'key_differences': {
                            'missing': missing_keys,
                            'extra': extra_keys
                        }
                    })
                    continue
                
                # Compare common keys
                value_diffs = {}
                for k in act_keys.intersection(exp_keys):
                    if isinstance(act[k], (int, float, np.number)) and isinstance(exp[k], (int, float, np.number)):
                        if abs(act[k] - exp[k]) > tolerance:
                            value_diffs[k] = {
                                'actual': float(act[k]) if isinstance(act[k], np.number) else act[k],
                                'expected': float(exp[k]) if isinstance(exp[k], np.number) else exp[k],
                                'diff': float(act[k] - exp[k])
                            }
                    elif act[k] != exp[k]:
                        value_diffs[k] = {
                            'actual': str(act[k]),
                            'expected': str(exp[k])
                        }
                
                if value_diffs:
                    item_differences.append({
                        'index': i,
                        'type': 'dict',
                        'value_differences': value_diffs
                    })
        
        if item_differences:
            comparison['sample_differences'][key] = item_differences
    
    return comparison


def test_dataset_values_full_comparison(save_expected=False):
    """
    Comprehensive test that compares the full dataset values, not just lengths.
    
    Args:
        save_expected: If True, saves the current dataset as the new expected data
    """
    # Use the pkl files from isolated tests instead of CSV files
    try:
        # Try to load from training_dataset inputs first
        stock_features = _load_artifact(ISOLATED_TESTS_BASE_DIR / "training_dataset" / "inputs" / "stock_features.pkl")
        monthly_sales_pivot = _load_artifact(ISOLATED_TESTS_BASE_DIR / "training_dataset" / "inputs" / "monthly_sales_pivot.pkl")
    except Exception:
        try:
            # Fall back to getting data from get_stock_features and get_monthly_sales_pivot outputs
            stock_features = _load_artifact(ISOLATED_TESTS_BASE_DIR / "get_stock_features" / "outputs" / "stock_features.pkl")
            monthly_sales_pivot = _load_artifact(ISOLATED_TESTS_BASE_DIR / "get_monthly_sales_pivot" / "outputs" / "monthly_sales_pivot.pkl")
        except Exception:
            pytest.skip("Required pkl files not found in example data. Run data generation script to create them.")
            return
    
    static_transformer = MultiColumnLabelBinarizer()
    scaler = GlobalLogMinMaxScaler()
    # Parameters aligned with generate_pipeline_examples.py
    input_chunk_length = 6
    output_chunk_length = 3
    default_past_covariates_span = 3
    default_minimum_sales_months = 1
    aligned_static_features = ['Тип', 'Конверт', 'Стиль', 'Ценовая категория']

    dataset = PlastinkaTrainingTSDataset(
        stock_features=stock_features,
        monthly_sales=monthly_sales_pivot,
        static_transformer=static_transformer,
        static_features=aligned_static_features, # Aligned
        scaler=scaler,
        input_chunk_length=input_chunk_length, # Aligned
        output_chunk_length=output_chunk_length, # Aligned
        past_covariates_span=default_past_covariates_span, # Aligned
        past_covariates_fnames=['Тип','Конверт','Стиль','Ценовая категория'], # Matches default
        minimum_sales_months=default_minimum_sales_months # Aligned
    )
    
    dataset_dict = dataset.to_dict()
    expected_values_path = GENERAL_EXAMPLES_DIR / "PlastinkaTrainingTSDataset_values.json"
    
    if save_expected:
        # This functionality has been moved to the separate generate_dataset_values.py script
        pytest.skip("Use the generate_dataset_values.py script to generate expected values")
        return
    
    # Load expected values for comparison
    try:
        expected_dict = _load_artifact(expected_values_path)
    except Exception:
        pytest.skip(f"Expected values file not found: {expected_values_path}. Run generate_dataset_values.py to create it.")
        return
    
    # Create a sample dict with the same keys as the expected dict
    sample_dict = {}
    for key in expected_dict.keys():
        if key in dataset_dict and len(dataset_dict[key]) > 0:
            # Take the same number of samples as in the expected dict
            sample_size = min(len(expected_dict[key]), len(dataset_dict[key]))
            sample_dict[key] = dataset_dict[key][:sample_size]
        else:
            sample_dict[key] = []
    
    # Compare the dictionaries
    comparison = compare_dataset_values(sample_dict, expected_dict)
    
    # Check for significant differences
    has_differences = (
        comparison['missing_keys'] or 
        comparison['extra_keys'] or 
        comparison['length_differences'] or 
        comparison['sample_differences']
    )
    
    if has_differences:
        error_msg = "Differences found in dataset values:\n"
        
        if comparison['missing_keys']:
            error_msg += f"- Missing keys: {comparison['missing_keys']}\n"
        if comparison['extra_keys']:
            error_msg += f"- Extra keys: {comparison['extra_keys']}\n"
        
        if comparison['length_differences']:
            error_msg += "- Length differences:\n"
            for key, diff in comparison['length_differences'].items():
                error_msg += f"  - {key}: actual={diff['actual']}, expected={diff['expected']}, diff={diff['diff']} ({diff['pct_diff']:.2f}%)\n"
        
        if comparison['sample_differences']:
            error_msg += "- Sample value differences:\n"
            for key, item_diffs in comparison['sample_differences'].items():
                error_msg += f"  - {key} ({len(item_diffs)} differences):\n"
                
                for item_diff in item_diffs:
                    error_msg += f"    - Item {item_diff['index']} ({item_diff['type']}):\n"
                    
                    if item_diff['type'] == 'TimeSeries':
                        diffs = item_diff['differences']
                        
                        if 'shape' in diffs:
                            error_msg += f"      - Shape: actual={diffs['shape']['actual']}, expected={diffs['shape']['expected']}\n"
                        
                        if 'values' in diffs:
                            vals = diffs['values']
                            error_msg += f"      - Values: max_diff={vals['max_diff']:.6f}, mean_diff={vals['mean_diff']:.6f}\n"
                            error_msg += f"        - Mean: actual={vals['actual_mean']:.6f}, expected={vals['expected_mean']:.6f}\n"
                            error_msg += f"        - Std: actual={vals['actual_std']:.6f}, expected={vals['expected_std']:.6f}\n"
                        
                        for meta in ['start', 'end', 'freq']:
                            if meta in diffs:
                                error_msg += f"      - {meta.capitalize()}: actual={diffs[meta]['actual']}, expected={diffs[meta]['expected']}\n"
                    
                    elif item_diff['type'] == 'array':
                        if 'shape_difference' in item_diff:
                            shape_diff = item_diff['shape_difference']
                            error_msg += f"      - Shape: actual={shape_diff['actual']}, expected={shape_diff['expected']}\n"
                        
                        if 'value_differences' in item_diff:
                            vals = item_diff['value_differences']
                            error_msg += f"      - Values: max_diff={vals['max_diff']:.6f}, mean_diff={vals['mean_diff']:.6f}\n"
                            error_msg += f"        - Mean: actual={vals['actual_mean']:.6f}, expected={vals['expected_mean']:.6f}\n"
                            error_msg += f"        - Std: actual={vals['actual_std']:.6f}, expected={vals['expected_std']:.6f}\n"
                    
                    elif item_diff['type'] == 'scalar':
                        diffs = item_diff['differences']
                        # Safely access 'diff' key
                        if diffs.get('diff') is not None:
                            error_msg += f"      - Value: actual={diffs['actual']}, expected={diffs['expected']}, diff={diffs['diff']:.6f}\n"
                        else:
                            error_msg += f"      - Value: actual={diffs['actual']}, expected={diffs['expected']}\n"
                    
                    elif item_diff['type'] == 'dict':
                        if 'key_differences' in item_diff:
                            key_diffs = item_diff['key_differences']
                            if key_diffs['missing']:
                                error_msg += f"      - Missing keys: {key_diffs['missing']}\n"
                            if key_diffs['extra']:
                                error_msg += f"      - Extra keys: {key_diffs['extra']}\n"
                        
                        if 'value_differences' in item_diff:
                            val_diffs = item_diff['value_differences']
                            for k, diff in val_diffs.items():
                                if 'diff' in diff:
                                    error_msg += f"      - {k}: actual={diff['actual']}, expected={diff['expected']}, diff={diff['diff']:.6f}\n"
                                else:
                                    error_msg += f"      - {k}: actual={diff['actual']}, expected={diff['expected']}\n"
        
        pytest.fail(error_msg)


if __name__ == "__main__":
    # When run directly, this will test and print out differences
    test_dataset_values_full_comparison()