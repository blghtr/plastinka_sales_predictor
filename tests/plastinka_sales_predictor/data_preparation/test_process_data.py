import pandas as pd
import pytest
import json
import pickle
import numpy as np
from pathlib import Path
from plastinka_sales_predictor.data_preparation import (
    process_raw,
    filter_by_date,
    get_preprocessed_df,
    process_data,
    get_stock_features,
    get_monthly_sales_pivot,
    PlastinkaTrainingTSDataset,
    MultiColumnLabelBinarizer,
    GlobalLogMinMaxScaler,
    GROUP_KEYS, # Assuming GROUP_KEYS is used by get_preprocessed_df or other functions
    categorize_prices # Added categorize_prices
)
from darts import TimeSeries # For type hinting if PlastinkaTrainingTSDataset uses it directly in signature
from tests.plastinka_sales_predictor.data_preparation.test_utils import assert_dataframes_equal_with_details
from tests.plastinka_sales_predictor.data_preparation.test_dataset_comparison import compare_dataset_values
import os

# --- Configuration for Test Artifact Paths ---
ISOLATED_TESTS_BASE_DIR = Path("tests/example_data/isolated_tests")
GENERAL_EXAMPLES_DIR = Path("generated_general_examples")
SAMPLE_STOCK_PATH = Path("tests/example_data/sample_stocks.xlsx")
SAMPLE_SALES_PATH = Path("tests/example_data/sales")

# Common cutoff date used in examples by generate_pipeline_examples.py
CUTOFF_DATE = '30-09-2022' # As per generate_pipeline_examples.py


# --- Helper Functions for Loading Artifacts ---

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

def load_isolated_input_artifact(func_name: str, artifact_name: str):
    return _load_artifact(ISOLATED_TESTS_BASE_DIR / func_name / "inputs" / artifact_name)

def load_isolated_output_artifact(func_name: str, artifact_name: str):
    return _load_artifact(ISOLATED_TESTS_BASE_DIR / func_name / "outputs" / artifact_name)

def load_general_example_artifact(artifact_name: str):
    return _load_artifact(GENERAL_EXAMPLES_DIR / artifact_name)

# --- Reusable Test Executor Functions ---

def _run_dataframe_test(actual_df: pd.DataFrame, expected_artifact_path_or_df, stage_name: str):
    """Compares an actual DataFrame with an expected one (loaded from path or direct)."""
    if isinstance(expected_artifact_path_or_df, (str, Path)):
        expected_df = _load_artifact(Path(expected_artifact_path_or_df))
    else:
        expected_df = expected_artifact_path_or_df
    
    try:
        assert_dataframes_equal_with_details(actual_df, expected_df, check_dtype=False) # Consistent with previous tests
    except AssertionError as e:
        pytest.fail(f"DataFrame comparison failed for stage '{stage_name}':\\n{e}")

def _run_dataframe_tuple_test(actual_tuple: tuple, expected_artifact_path_or_tuple, stage_name: str):
    """Compares a tuple of (DataFrame, bins) with an expected one."""
    if isinstance(expected_artifact_path_or_tuple, (str, Path)):
        expected_tuple = _load_artifact(Path(expected_artifact_path_or_tuple))
    else:
        expected_tuple = expected_artifact_path_or_tuple

    actual_df, actual_bins = actual_tuple
    expected_df, expected_bins = expected_tuple

    # Compare DataFrame part
    try:
        assert_dataframes_equal_with_details(actual_df, expected_df, check_dtype=False)
    except AssertionError as e:
        pytest.fail(f"DataFrame comparison failed for stage '{stage_name}' (DataFrame part):\\n{e}")

    # Compare bins part (assuming bins are comparable with ==, e.g., numpy arrays or lists)
    if not isinstance(actual_bins, type(expected_bins)):
         pytest.fail(f"Bins type mismatch for stage '{stage_name}': Actual type {type(actual_bins)}, Expected type {type(expected_bins)}")

    if isinstance(actual_bins, np.ndarray) and isinstance(expected_bins, np.ndarray):
        if not np.array_equal(actual_bins, expected_bins):
            pytest.fail(f"Bins comparison failed for stage '{stage_name}':\\nActual: {actual_bins}\\nExpected: {expected_bins}")
    elif actual_bins != expected_bins:
        pytest.fail(f"Bins comparison failed for stage '{stage_name}':\\nActual: {actual_bins}\\nExpected: {expected_bins}")


def _run_dict_of_dataframes_test(actual_dict_df: dict, expected_artifacts_map: dict, stage_name: str):
    """Compares a dictionary of DataFrames with expected ones loaded from paths."""
    for key, expected_path in expected_artifacts_map.items():
        if key not in actual_dict_df:
            pytest.fail(f"Missing key '{key}' in result for stage '{stage_name}'")
        
        _run_dataframe_test(actual_dict_df[key], expected_path, f"{stage_name} - key: {key}")

def _run_dataset_test(actual_dataset_dict: dict, expected_artifact_path_or_dict, stage_name: str):
    """Compares a dataset's .to_dict() representation."""
    # In a real CI environment, you might want to validate exact matches
    # For now, we will use the custom compare_dataset_values for rich diff,
    # and then fail if there are any differences.

    if isinstance(expected_artifact_path_or_dict, (str, Path)):
        expected_dict = _load_artifact(Path(expected_artifact_path_or_dict))
    else:
        expected_dict = expected_artifact_path_or_dict
        
    comparison_results = compare_dataset_values(actual_dataset_dict, expected_dict)
    
    diff_messages = []
    if comparison_results.get('missing_keys'):
        diff_messages.append(f"Missing keys: {comparison_results['missing_keys']}")
    if comparison_results.get('extra_keys'):
        diff_messages.append(f"Extra keys: {comparison_results['extra_keys']}")
    if comparison_results.get('length_differences'):
        diff_messages.append(f"Length differences: {comparison_results['length_differences']}")
    if comparison_results.get('type_differences'): # Assuming compare_dataset_values can report this
        diff_messages.append(f"Type differences: {comparison_results['type_differences']}")
    if comparison_results.get('value_differences'): # Assuming compare_dataset_values can report this
        diff_messages.append(f"Value differences: {comparison_results['value_differences']}")
    if comparison_results.get('sample_differences'): # From original README
        diff_messages.append(f"Sample differences: {comparison_results['sample_differences']}")

    if diff_messages:
        pytest.fail(f"Dataset comparison failed for stage '{stage_name}':\\n" + "\\n".join(diff_messages) + f"\\nFull comparison: {comparison_results}")

# --- Isolated Tests ---

# Stage: process_raw
PROCESS_RAW_FUNC_NAME = "process_raw"

def test_isolated_process_raw_stock():
    stock_df_raw = load_isolated_input_artifact(PROCESS_RAW_FUNC_NAME, "stock_df_raw.pkl")
    actual_processed_stock = process_raw(stock_df_raw.copy()) # Ensure original is not modified
    expected_processed_stock_path = ISOLATED_TESTS_BASE_DIR / PROCESS_RAW_FUNC_NAME / "outputs" / "processed_stock.pkl"
    _run_dataframe_test(actual_processed_stock, expected_processed_stock_path, "process_raw (stock)")

def test_isolated_process_raw_sales():
    sales_df_raw = load_isolated_input_artifact(PROCESS_RAW_FUNC_NAME, "sales_df_raw.pkl")
    actual_processed_sales = process_raw(sales_df_raw.copy()) # Ensure original is not modified
    expected_processed_sales_path = ISOLATED_TESTS_BASE_DIR / PROCESS_RAW_FUNC_NAME / "outputs" / "processed_sales.pkl"
    _run_dataframe_test(actual_processed_sales, expected_processed_sales_path, "process_raw (sales)")

# Stage: filter_by_date
FILTER_BY_DATE_FUNC_NAME = "filter_by_date"

@pytest.fixture
def filter_by_date_stock_input_df():
    """Fixture to load the common stock input DataFrame for filter_by_date tests."""
    return load_isolated_input_artifact(FILTER_BY_DATE_FUNC_NAME, "processed_stock_input.pkl")

@pytest.fixture
def filter_by_date_sales_input_df():
    """Fixture to load the common sales input DataFrame for filter_by_date tests."""
    return load_isolated_input_artifact(FILTER_BY_DATE_FUNC_NAME, "processed_sales_input.pkl")

def test_isolated_filter_by_date_stock_before_cutoff(filter_by_date_stock_input_df):
    # processed_stock_input = load_isolated_input_artifact(FILTER_BY_DATE_FUNC_NAME, "processed_stock_input.pkl") # Replaced by fixture
    actual_filtered_stock = filter_by_date(filter_by_date_stock_input_df.copy(), CUTOFF_DATE, cut_before=False)
    expected_path = ISOLATED_TESTS_BASE_DIR / FILTER_BY_DATE_FUNC_NAME / "outputs" / "stock_filtered_before_cutoff.pkl"
    _run_dataframe_test(actual_filtered_stock, expected_path, "filter_by_date (stock, before cutoff)")

def test_isolated_filter_by_date_stock_after_cutoff(filter_by_date_stock_input_df):
    # processed_stock_input = load_isolated_input_artifact(FILTER_BY_DATE_FUNC_NAME, "processed_stock_input.pkl") # Replaced by fixture
    actual_filtered_stock = filter_by_date(filter_by_date_stock_input_df.copy(), CUTOFF_DATE, cut_before=True)
    expected_path = ISOLATED_TESTS_BASE_DIR / FILTER_BY_DATE_FUNC_NAME / "outputs" / "stock_filtered_after_cutoff.pkl"
    _run_dataframe_test(actual_filtered_stock, expected_path, "filter_by_date (stock, after cutoff)")

def test_isolated_filter_by_date_sales_before_cutoff(filter_by_date_sales_input_df):
    # processed_sales_input = load_isolated_input_artifact(FILTER_BY_DATE_FUNC_NAME, "processed_sales_input.pkl") # Replaced by fixture
    actual_filtered_sales = filter_by_date(filter_by_date_sales_input_df.copy(), CUTOFF_DATE, cut_before=False)
    expected_path = ISOLATED_TESTS_BASE_DIR / FILTER_BY_DATE_FUNC_NAME / "outputs" / "sales_filtered_before_cutoff.pkl"
    _run_dataframe_test(actual_filtered_sales, expected_path, "filter_by_date (sales, before cutoff)")

def test_isolated_filter_by_date_sales_after_cutoff(filter_by_date_sales_input_df):
    # processed_sales_input = load_isolated_input_artifact(FILTER_BY_DATE_FUNC_NAME, "processed_sales_input.pkl") # Replaced by fixture
    actual_filtered_sales = filter_by_date(filter_by_date_sales_input_df.copy(), CUTOFF_DATE, cut_before=True)
    expected_path = ISOLATED_TESTS_BASE_DIR / FILTER_BY_DATE_FUNC_NAME / "outputs" / "sales_filtered_after_cutoff.pkl"
    _run_dataframe_test(actual_filtered_sales, expected_path, "filter_by_date (sales, after cutoff)")

# Stage: get_preprocessed_df
GET_PREPROCESSED_DF_FUNC_NAME = "get_preprocessed_df"

def test_isolated_get_preprocessed_df_stock():
    stock_df_raw = load_isolated_input_artifact(GET_PREPROCESSED_DF_FUNC_NAME, "stock_df_raw.pkl")
    # Define processing function as used in generate_pipeline_examples.py
    all_keys = ['Дата создания', *GROUP_KEYS] # Assuming GROUP_KEYS is imported or defined
    def count_items_stock(group):
        return pd.Series({'count': group['Экземпляры'].astype('int64').sum()})
    
    actual_result_tuple = get_preprocessed_df(stock_df_raw.copy(), all_keys, count_items_stock)
    expected_path = ISOLATED_TESTS_BASE_DIR / GET_PREPROCESSED_DF_FUNC_NAME / "outputs" / "preprocessed_stock_with_bins.pkl"
    _run_dataframe_tuple_test(actual_result_tuple, expected_path, "get_preprocessed_df (stock)")

def test_isolated_get_preprocessed_df_sales_with_stock_bins():
    sales_df_raw = load_isolated_input_artifact(GET_PREPROCESSED_DF_FUNC_NAME, "sales_df_raw.pkl")
    # Let's load the stock preprocessed result first to extract the bins
    stock_preprocessed_tuple = load_isolated_output_artifact(GET_PREPROCESSED_DF_FUNC_NAME, "preprocessed_stock_with_bins.pkl")
    _, stock_bins = stock_preprocessed_tuple  # Extract the bins from the tuple

    # We need to include 'Дата создания' in the keys since it's in the expected result
    all_keys = ['Дата создания', 'Дата продажи', *GROUP_KEYS]
    def process_movements_sales(group):
        return pd.Series({
            'count': len(group),
            'mean_price': group['Цена, руб.'].astype('float64').mean()
        })
    actual_result_tuple = get_preprocessed_df(sales_df_raw.copy(), all_keys, process_movements_sales, bins=stock_bins)
    expected_path = ISOLATED_TESTS_BASE_DIR / GET_PREPROCESSED_DF_FUNC_NAME / "outputs" / "preprocessed_sales_with_bins.pkl"
    _run_dataframe_tuple_test(actual_result_tuple, expected_path, "get_preprocessed_df (sales with stock bins)")

# Stage: process_data
PROCESS_DATA_FUNC_NAME = "process_data"

def test_isolated_process_data():
    # process_data takes file paths. The generator script saves these paths as JSON strings in .json files.
    # The script saved parameters as a pickle instead of individual files
    input_parameters = load_isolated_input_artifact(PROCESS_DATA_FUNC_NAME, "input_parameters.pkl")
    
    # Extract parameters from the loaded object
    stock_path_str = input_parameters.get("stock_path", str(SAMPLE_STOCK_PATH))
    sales_path_str = input_parameters.get("sales_path", str(SAMPLE_SALES_PATH))
    cutoff_date = input_parameters.get("cutoff_date", CUTOFF_DATE)

    # The loaded paths might be relative to the project root where generate_pipeline_examples.py runs.
    # Ensure they are correctly resolved if needed, or assume they are usable as is.
    # For this test, we assume the paths saved by generate_pipeline_examples are directly usable.

    actual_features_dict = process_data(stock_path_str, sales_path_str, cutoff_date)
    
    expected_outputs_map = {
        key: ISOLATED_TESTS_BASE_DIR / PROCESS_DATA_FUNC_NAME / "outputs" / f"feature_{key}.pkl"
        for key in ['stock', 'prices', 'sales', 'change']
    }
    _run_dict_of_dataframes_test(actual_features_dict, expected_outputs_map, "process_data")

# Stage: get_stock_features
GET_STOCK_FEATURES_FUNC_NAME = "get_stock_features"

def test_isolated_get_stock_features():
    stock_data = load_isolated_input_artifact(GET_STOCK_FEATURES_FUNC_NAME, "stock_data.pkl")
    change_data = load_isolated_input_artifact(GET_STOCK_FEATURES_FUNC_NAME, "change_data.pkl")
    actual_stock_features = get_stock_features(stock_data.copy(), change_data.copy())
    expected_path = ISOLATED_TESTS_BASE_DIR / GET_STOCK_FEATURES_FUNC_NAME / "outputs" / "stock_features.pkl"
    _run_dataframe_test(actual_stock_features, expected_path, "get_stock_features")

# Stage: get_monthly_sales_pivot
GET_MONTHLY_SALES_PIVOT_FUNC_NAME = "get_monthly_sales_pivot"

def test_isolated_get_monthly_sales_pivot():
    sales_data = load_isolated_input_artifact(GET_MONTHLY_SALES_PIVOT_FUNC_NAME, "sales_data.pkl")
    actual_sales_pivot = get_monthly_sales_pivot(sales_data.copy())
    expected_path = ISOLATED_TESTS_BASE_DIR / GET_MONTHLY_SALES_PIVOT_FUNC_NAME / "outputs" / "monthly_sales_pivot.pkl"
    _run_dataframe_test(actual_sales_pivot, expected_path, "get_monthly_sales_pivot")

# Stage: PlastinkaTrainingTSDataset
TRAINING_DATASET_FUNC_NAME = "training_dataset"

def test_isolated_PlastinkaTrainingTSDataset():
    stock_features = load_isolated_input_artifact(TRAINING_DATASET_FUNC_NAME, "stock_features.pkl")
    
    monthly_sales_pivot = load_isolated_input_artifact(TRAINING_DATASET_FUNC_NAME, "monthly_sales_pivot.pkl")
    
    # Parameters as used in generate_pipeline_examples.py (or a simplified version for isolated test if applicable)
    # These might need to be loaded from a config or defined if they are complex
    static_transformer = MultiColumnLabelBinarizer() # Or load if pickled
    scaler = GlobalLogMinMaxScaler() # Or load if pickled
    # Load other params from a json if they were saved, or define directly
    
    # Directly define standard parameters since dataset_params.json doesn't exist in the artifacts
    
    dataset_params = {
        "static_features": ['Конверт','Тип','Ценовая категория','Стиль','Год записи','Год выпуска'], 
        "input_chunk_length": monthly_sales_pivot.shape[0] - 1,
        "output_chunk_length": 1,
        "past_covariates_span": 14,
        "past_covariates_fnames": ['Тип','Конверт','Стиль','Ценовая категория'],
        "minimum_sales_months": 2
    }
    
    dataset = PlastinkaTrainingTSDataset(
        stock_features=stock_features,
        monthly_sales=monthly_sales_pivot,
        static_transformer=static_transformer, # Assuming new instances are okay for testing structure
        scaler=scaler, # Assuming new instances are okay for testing structure
        **dataset_params
    )
    actual_dataset_dict = dataset.to_dict()
    expected_path = GENERAL_EXAMPLES_DIR / "PlastinkaTrainingTSDataset_values.json" # As per generate_pipeline_examples
    # Or if there is an isolated output: ISOLATED_TESTS_BASE_DIR / TRAINING_DATASET_FUNC_NAME / "outputs" / "dataset_dict.json"
    # Using general example path based on current understanding from generate_pipeline_examples.py for this specific full dataset output
    _run_dataset_test(actual_dataset_dict, expected_path, "PlastinkaTrainingTSDataset")
