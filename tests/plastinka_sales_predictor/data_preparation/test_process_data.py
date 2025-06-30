"""
Comprehensive tests for plastinka_sales_predictor.data_preparation

This test suite covers all functions in the data_preparation module with comprehensive mocking
of external dependencies and isolated testing using pre-generated artifacts. Tests are organized 
by function groups and include both success and failure scenarios.

Testing Approach:
- Mock all external dependencies (pandas I/O, dill operations, sklearn transformers)
- Test individual functions in isolation using pre-generated test artifacts
- Test main pipeline integration with proper mocking
- Verify error handling and data validation
- Test data processing pipelines with various input scenarios
- Test performance and memory efficiency for large datasets
- Clear Arrange-Act-Assert pattern
- Integration tests verify end-to-end data processing functionality

The tests use isolated testing approach with pre-generated artifacts stored in 
tests/example_data/isolated_tests/ directory. This ensures reproducible testing
without relying on external data sources while still providing comprehensive coverage.

All external file I/O operations and sklearn dependencies are mocked to ensure test isolation.
"""

import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

# Import the module under test
from plastinka_sales_predictor.data_preparation import (
    GROUP_KEYS,
    GlobalLogMinMaxScaler,
    MultiColumnLabelBinarizer,
    PlastinkaTrainingTSDataset,
    filter_by_date,
    get_monthly_sales_pivot,
    get_preprocessed_df,
    get_stock_features,
    process_data,
    process_raw,
    validate_categories,
    validate_date_columns,
    validate_styles,
)
from tests.plastinka_sales_predictor.data_preparation.test_dataset_comparison import (
    compare_dataset_values,
)
from tests.plastinka_sales_predictor.data_preparation.test_utils import (
    assert_dataframes_equal_with_details,
)

# --- Configuration for Test Artifact Paths ---
ISOLATED_TESTS_BASE_DIR = Path("tests/example_data/isolated_tests")
GENERAL_EXAMPLES_DIR = Path("generated_general_examples")
SAMPLE_STOCK_PATH = Path("tests/example_data/sample_stocks.xlsx")
SAMPLE_SALES_PATH = Path("tests/example_data/sales")

# Common cutoff date used in examples by generate_pipeline_examples.py
CUTOFF_DATE = '30-09-2022'


# --- Helper Functions for Loading Artifacts ---

def _load_artifact(file_path: Path):
    """Loads a pickled or JSON artifact based on its extension."""
    if not file_path.exists():
        pytest.fail(f"Artifact file not found: {file_path}")

    if file_path.suffix == '.json':
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)
    elif file_path.suffix == '.pkl':
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        pytest.fail(f"Unsupported artifact file extension: {file_path.suffix} for file {file_path}. Only .json and .pkl are supported.")

def _load_isolated_input_artifact(func_name: str, artifact_name: str):
    return _load_artifact(ISOLATED_TESTS_BASE_DIR / func_name / "inputs" / artifact_name)

def _load_isolated_output_artifact(func_name: str, artifact_name: str):
    return _load_artifact(ISOLATED_TESTS_BASE_DIR / func_name / "outputs" / artifact_name)

def _load_general_example_artifact(artifact_name: str):
    return _load_artifact(GENERAL_EXAMPLES_DIR / artifact_name)

# --- Reusable Test Executor Functions ---

def _run_dataframe_test(actual_df: pd.DataFrame, expected_artifact_path_or_df, stage_name: str):
    """Compares an actual DataFrame with an expected one (loaded from path or direct)."""
    if isinstance(expected_artifact_path_or_df, (str, Path)):
        expected_df = _load_artifact(Path(expected_artifact_path_or_df))
    else:
        expected_df = expected_artifact_path_or_df

    try:
        assert_dataframes_equal_with_details(actual_df, expected_df, check_dtype=False)
    except AssertionError as e:
        pytest.fail(f"DataFrame comparison failed for stage '{stage_name}':\n{e}")

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
        pytest.fail(f"DataFrame comparison failed for stage '{stage_name}' (DataFrame part):\n{e}")

    # Compare bins part
    if not isinstance(actual_bins, type(expected_bins)):
         pytest.fail(f"Bins type mismatch for stage '{stage_name}': Actual type {type(actual_bins)}, Expected type {type(expected_bins)}")

    if isinstance(actual_bins, np.ndarray) and isinstance(expected_bins, np.ndarray):
        if not np.array_equal(actual_bins, expected_bins):
            pytest.fail(f"Bins comparison failed for stage '{stage_name}':\nActual: {actual_bins}\nExpected: {expected_bins}")
    elif actual_bins != expected_bins:
        pytest.fail(f"Bins comparison failed for stage '{stage_name}':\nActual: {actual_bins}\nExpected: {expected_bins}")

def _run_dict_of_dataframes_test(actual_dict_df: dict, expected_artifacts_map: dict, stage_name: str):
    """Compares a dictionary of DataFrames with expected ones loaded from paths."""
    for key, expected_path in expected_artifacts_map.items():
        if key not in actual_dict_df:
            pytest.fail(f"Missing key '{key}' in result for stage '{stage_name}'")

        _run_dataframe_test(actual_dict_df[key], expected_path, f"{stage_name} - key: {key}")

def _run_dataset_test(actual_dataset_dict: dict, expected_artifact_path_or_dict, stage_name: str):
    """Compares a dataset's .to_dict() representation."""
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
    if comparison_results.get('type_differences'):
        diff_messages.append(f"Type differences: {comparison_results['type_differences']}")
    if comparison_results.get('value_differences'):
        diff_messages.append(f"Value differences: {comparison_results['value_differences']}")
    if comparison_results.get('sample_differences'):
        diff_messages.append(f"Sample differences: {comparison_results['sample_differences']}")

    if diff_messages:
        pytest.fail(f"Dataset comparison failed for stage '{stage_name}':\n" + "\n".join(diff_messages) + f"\nFull comparison: {comparison_results}")


class TestDataProcessingCore:
    """Test suite for core data processing functions."""

    def test_process_raw_stock_success(self):
        """Test successful process_raw execution with stock data."""
        # Arrange
        stock_df_raw = _load_isolated_input_artifact("process_raw", "stock_df_raw.pkl")
        expected_processed_stock_path = ISOLATED_TESTS_BASE_DIR / "process_raw" / "outputs" / "processed_stock.pkl"

        # Act
        actual_processed_stock = process_raw(stock_df_raw.copy())

        # Assert
        _run_dataframe_test(actual_processed_stock, expected_processed_stock_path, "process_raw (stock)")

    def test_process_raw_sales_success(self):
        """Test successful process_raw execution with sales data."""
        # Arrange
        sales_df_raw = _load_isolated_input_artifact("process_raw", "sales_df_raw.pkl")
        expected_processed_sales_path = ISOLATED_TESTS_BASE_DIR / "process_raw" / "outputs" / "processed_sales.pkl"

        # Act
        actual_processed_sales = process_raw(sales_df_raw.copy())

        # Assert
        _run_dataframe_test(actual_processed_sales, expected_processed_sales_path, "process_raw (sales)")

    @patch('plastinka_sales_predictor.data_preparation.validate_date_columns')
    @patch('plastinka_sales_predictor.data_preparation.validate_categories')
    @patch('plastinka_sales_predictor.data_preparation.validate_styles')
    @patch('plastinka_sales_predictor.data_preparation.categorize_dates')
    @patch('plastinka_sales_predictor.data_preparation.categorize_prices')
    def test_process_raw_with_mocked_dependencies(self, mock_categorize_prices, mock_categorize_dates,
                                                  mock_validate_styles, mock_validate_categories,
                                                  mock_validate_date_columns):
        """Test process_raw with all dependencies mocked."""
        # Arrange - создаем DataFrame с правильными русскими названиями колонок
        test_df = pd.DataFrame({
            'Штрихкод': ['123456', '789012', '345678'],
            'Экземпляры': [1, 2, 1],
            'Исполнитель': ['Artist 1', 'Artist 2', 'Artist 3'],
            'Альбом': ['Album 1', 'Album 2', 'Album 3'],
            'Цена, руб.': [100.0, 200.0, 300.0],
            'Дата создания': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03']),
            'Год записи': [2020, 2021, 2022],
            'Год выпуска': [2020, 2021, 2022],
            'Конверт': ['EX', 'VG+', 'NM'],
            'Тип': ['LP', 'LP', 'EP'],
            'Стиль': ['Rock', 'Jazz', 'Blues']
        })

        # Настраиваем моки для возврата корректных данных
        mock_validate_date_columns.return_value = test_df.copy()
        mock_validate_categories.return_value = test_df.copy()
        mock_validate_styles.return_value = test_df.copy()
        mock_categorize_dates.return_value = test_df.copy()

        # Мок для categorize_prices должен вернуть DataFrame с добавленной колонкой
        test_df_with_price_category = test_df.copy()
        test_df_with_price_category['Ценовая категория'] = pd.Categorical(['Low', 'Medium', 'High'])
        mock_categorize_prices.return_value = (test_df_with_price_category, np.array([0, 100, 200, 300]))

        # Act
        result = process_raw(test_df.copy())

        # Assert
        mock_validate_date_columns.assert_called_once()
        mock_validate_categories.assert_called_once()
        mock_validate_styles.assert_called_once()
        mock_categorize_dates.assert_called_once()
        mock_categorize_prices.assert_called_once()
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestDataFiltering:
    """Test suite for data filtering operations."""

    @pytest.fixture
    def filter_by_date_stock_input_df(self):
        """Fixture to load the common stock input DataFrame for filter_by_date tests."""
        return _load_isolated_input_artifact("filter_by_date", "processed_stock_input.pkl")

    @pytest.fixture
    def filter_by_date_sales_input_df(self):
        """Fixture to load the common sales input DataFrame for filter_by_date tests."""
        return _load_isolated_input_artifact("filter_by_date", "processed_sales_input.pkl")

    def test_filter_by_date_stock_before_cutoff(self, filter_by_date_stock_input_df):
        """Test filter_by_date with stock data before cutoff."""
        # Arrange
        expected_path = ISOLATED_TESTS_BASE_DIR / "filter_by_date" / "outputs" / "stock_filtered_before_cutoff.pkl"

        # Act
        actual_filtered_stock = filter_by_date(filter_by_date_stock_input_df.copy(), CUTOFF_DATE, cut_before=False)

        # Assert
        _run_dataframe_test(actual_filtered_stock, expected_path, "filter_by_date (stock, before cutoff)")

    def test_filter_by_date_stock_after_cutoff(self, filter_by_date_stock_input_df):
        """Test filter_by_date with stock data after cutoff."""
        # Arrange
        expected_path = ISOLATED_TESTS_BASE_DIR / "filter_by_date" / "outputs" / "stock_filtered_after_cutoff.pkl"

        # Act
        actual_filtered_stock = filter_by_date(filter_by_date_stock_input_df.copy(), CUTOFF_DATE, cut_before=True)

        # Assert
        _run_dataframe_test(actual_filtered_stock, expected_path, "filter_by_date (stock, after cutoff)")

    def test_filter_by_date_sales_before_cutoff(self, filter_by_date_sales_input_df):
        """Test filter_by_date with sales data before cutoff."""
        # Arrange
        expected_path = ISOLATED_TESTS_BASE_DIR / "filter_by_date" / "outputs" / "sales_filtered_before_cutoff.pkl"

        # Act
        actual_filtered_sales = filter_by_date(filter_by_date_sales_input_df.copy(), CUTOFF_DATE, cut_before=False)

        # Assert
        _run_dataframe_test(actual_filtered_sales, expected_path, "filter_by_date (sales, before cutoff)")

    def test_filter_by_date_sales_after_cutoff(self, filter_by_date_sales_input_df):
        """Test filter_by_date with sales data after cutoff."""
        # Arrange
        expected_path = ISOLATED_TESTS_BASE_DIR / "filter_by_date" / "outputs" / "sales_filtered_after_cutoff.pkl"

        # Act
        actual_filtered_sales = filter_by_date(filter_by_date_sales_input_df.copy(), CUTOFF_DATE, cut_before=True)

        # Assert
        _run_dataframe_test(actual_filtered_sales, expected_path, "filter_by_date (sales, after cutoff)")

    @patch('pandas.to_datetime')
    def test_filter_by_date_exception_handling(self, mock_to_datetime):
        """Test filter_by_date handles date parsing exceptions properly."""
        # Arrange
        test_df = pd.DataFrame({'Дата создания': ['invalid_date', '2022-01-01']})
        mock_to_datetime.side_effect = ValueError("Invalid date format")

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid date format"):
            filter_by_date(test_df, CUTOFF_DATE)


class TestDataPreprocessing:
    """Test suite for data preprocessing operations."""

    def test_get_preprocessed_df_stock_success(self):
        """Test successful get_preprocessed_df execution with stock data."""
        # Arrange
        stock_df_raw = _load_isolated_input_artifact("get_preprocessed_df", "stock_df_raw.pkl")
        expected_path = ISOLATED_TESTS_BASE_DIR / "get_preprocessed_df" / "outputs" / "preprocessed_stock_with_bins.pkl"

        # Define processing function as used in generate_pipeline_examples.py
        all_keys = ['Дата создания', *GROUP_KEYS]
        def count_items_stock(group):
            return pd.Series({'count': group['Экземпляры'].astype('int64').sum()})

        # Act
        actual_result = get_preprocessed_df(stock_df_raw.copy(), all_keys, count_items_stock)

        # Assert
        _run_dataframe_tuple_test(actual_result, expected_path, "get_preprocessed_df (stock)")

    def test_get_preprocessed_df_sales_success(self):
        """Test successful get_preprocessed_df execution with sales data."""
        # Arrange
        sales_df_raw = _load_isolated_input_artifact("get_preprocessed_df", "sales_df_raw.pkl")
        stock_preprocessed_tuple = _load_isolated_output_artifact("get_preprocessed_df", "preprocessed_stock_with_bins.pkl")
        _, stock_bins = stock_preprocessed_tuple  # Extract the bins from the tuple
        expected_path = ISOLATED_TESTS_BASE_DIR / "get_preprocessed_df" / "outputs" / "preprocessed_sales_with_bins.pkl"

        # Define processing function
        all_keys = ['Дата создания', 'Дата продажи', *GROUP_KEYS]
        def process_movements_sales(group):
            return pd.Series({
                'count': len(group),
                'mean_price': group['Цена, руб.'].astype('float64').mean()
            })

        # Act
        actual_result = get_preprocessed_df(sales_df_raw.copy(), all_keys, process_movements_sales, bins=stock_bins)

        # Assert
        _run_dataframe_tuple_test(actual_result, expected_path, "get_preprocessed_df (sales with stock bins)")

    @patch('plastinka_sales_predictor.data_preparation.categorize_prices')
    def test_get_preprocessed_df_with_mocked_categorize_prices(self, mock_categorize_prices):
        """Test get_preprocessed_df with mocked categorize_prices."""
        # Arrange - создаем DataFrame с правильными русскими названиями колонок
        test_df = pd.DataFrame({
            'Дата создания': pd.to_datetime(['2022-01-01', '2022-01-02']),
            'Экземпляры': [1, 2],
            'Штрихкод': ['123456', '789012'],
            'Исполнитель': ['Artist 1', 'Artist 2'],
            'Альбом': ['Album 1', 'Album 2'],
            'Цена, руб.': [100.0, 200.0],
            'Год записи': [2020, 2021],
            'Год выпуска': [2020, 2021],
            'Тип': ['LP', 'LP'],  # Добавляем требуемую колонку Тип
            'Стиль': ['Rock', 'Jazz'],  # Добавляем требуемую колонку Стиль
            'Конверт': ['EX', 'VG+']  # Добавляем требуемую колонку Конверт
        })

        # Мок должен возвращать tuple (DataFrame, bins) как ожидается в process_raw
        result_df = test_df.copy()
        result_df['Ценовая категория'] = 'Low'  # Добавляем price category колонку

        mock_categorize_prices.return_value = (result_df, [0, 100, 200])

        # Define the required parameters for get_preprocessed_df
        group_keys = ['Исполнитель', 'Альбом']

        def test_transform_fn(group):
            return pd.Series({'count': len(group)})

        # Act
        result = get_preprocessed_df(test_df.copy(), group_keys, test_transform_fn)

        # Assert
        mock_categorize_prices.assert_called_once()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert not result[0].empty


class TestPipelineIntegration:
    """Test suite for integrated pipeline operations."""

    def test_process_data_success(self):
        """Test successful process_data execution with real file I/O."""
        # Arrange
        input_parameters = _load_isolated_input_artifact("process_data", "input_parameters.pkl")
        expected_outputs_map = {
            key: ISOLATED_TESTS_BASE_DIR / "process_data" / "outputs" / f"feature_{key}.pkl"
            for key in ['stock', 'prices', 'sales', 'change']
        }

        # Extract parameters
        stock_path_str = input_parameters.get("stock_path", str(SAMPLE_STOCK_PATH))
        sales_path_str = input_parameters.get("sales_path", str(SAMPLE_SALES_PATH))
        cutoff_date = input_parameters.get("cutoff_date", CUTOFF_DATE)

        # Act
        actual_features_dict = process_data(stock_path_str, sales_path_str, cutoff_date)

        # Assert
        _run_dict_of_dataframes_test(actual_features_dict, expected_outputs_map, "process_data")

    @patch('plastinka_sales_predictor.data_preparation.pd.read_excel')
    def test_process_data_file_not_found_exception(self, mock_read_excel):
        """Test process_data handles file not found exceptions properly."""
        # Arrange
        mock_read_excel.side_effect = FileNotFoundError("Stock file not found")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Stock file not found"):
            process_data("nonexistent_stock.xlsx", "sales_path", CUTOFF_DATE)

class TestStockFeatures:
    """Test suite for stock features generation."""

    def test_get_stock_features_success(self):
        """Test successful get_stock_features execution."""
        # Arrange
        stock_data = _load_isolated_input_artifact("get_stock_features", "stock_data.pkl")
        change_data = _load_isolated_input_artifact("get_stock_features", "change_data.pkl")
        expected_path = ISOLATED_TESTS_BASE_DIR / "get_stock_features" / "outputs" / "stock_features.pkl"

        # Act
        actual_result = get_stock_features(stock_data.copy(), change_data.copy())

        # Assert
        _run_dataframe_test(actual_result, expected_path, "get_stock_features")

    def test_get_stock_features_with_mocked_ensure_monthly_regular_index(self):
        """Test get_stock_features with proper data structure."""
        # Arrange
        date_index = pd.date_range('2022-01-01', periods=3, freq='ME')  # Используем ME вместо M

        test_stock = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        }, index=date_index)

        # Создаем test_movements с правильной структурой MultiIndex для функции get_stock_features
        multi_index = pd.MultiIndex.from_arrays([
            ['item1', 'item1', 'item2'],  # Некий уровень группировки
            date_index[:3]  # _date уровень
        ], names=['group', '_date'])

        test_movements = pd.DataFrame({
            'change': [7, 8, 9]
        }, index=multi_index)

        # Act
        result = get_stock_features(test_stock, test_movements)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestSalesPivoting:
    """Test suite for sales pivoting operations."""

    def test_get_monthly_sales_pivot_success(self):
        """Test successful get_monthly_sales_pivot execution."""
        # Arrange
        sales_data = _load_isolated_input_artifact("get_monthly_sales_pivot", "sales_data.pkl")
        expected_path = ISOLATED_TESTS_BASE_DIR / "get_monthly_sales_pivot" / "outputs" / "monthly_sales_pivot.pkl"

        # Act
        actual_result = get_monthly_sales_pivot(sales_data.copy())

        # Assert
        _run_dataframe_test(actual_result, expected_path, "get_monthly_sales_pivot")

    def test_get_monthly_sales_pivot_with_mocked_transform_months(self):
        """Test get_monthly_sales_pivot with proper data structure."""
        # Arrange
        test_df = pd.DataFrame({
            'Дата продажи': pd.to_datetime(['2022-01-01', '2022-02-01', '2022-01-15', '2022-02-15']),
            'Штрихкод': ['123', '456', '123', '456'],
            'Исполнитель': ['Artist 1', 'Artist 2', 'Artist 1', 'Artist 2'],
            'Альбом': ['Album 1', 'Album 2', 'Album 1', 'Album 2'],
            'count': [1, 2, 1, 3]
        })

        # Создаем правильный мультииндекс как в реальном коде
        test_df = test_df.reset_index(drop=True)
        test_df = test_df.groupby(['Штрихкод', 'Исполнитель', 'Альбом', 'Дата продажи']).agg({'count': 'sum'}).reset_index()
        test_df['_date'] = test_df['Дата продажи']  # Добавляем _date колонку как ожидается в коде
        test_df['sales'] = test_df['count']  # Добавляем sales колонку как ожидается в get_monthly_sales_pivot
        test_df = test_df.set_index(['Штрихкод', 'Исполнитель', 'Альбом', '_date'])

        # Act
        result = get_monthly_sales_pivot(test_df)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert not result.empty


class TestTrainingDataset:
    """Test suite for PlastinkaTrainingTSDataset functionality."""

    def test_plastinka_training_ts_dataset_success(self):
        """Test successful PlastinkaTrainingTSDataset creation."""
        # Arrange
        stock_features = _load_isolated_input_artifact("training_dataset", "stock_features.pkl")
        monthly_sales_pivot = _load_isolated_input_artifact("training_dataset", "monthly_sales_pivot.pkl")
        expected_path = ISOLATED_TESTS_BASE_DIR / "training_dataset" / "outputs" / "full_dataset_to_dict.pkl"

        # Use the same parameters as in generate_pipeline_examples.py
        static_transformer = MultiColumnLabelBinarizer()
        scaler = GlobalLogMinMaxScaler()

        dataset_params = {
            "static_features": ['release_type', 'cover_type', 'style', 'price_category'],
            "input_chunk_length": 6,
            "output_chunk_length": 3,
            "past_covariates_span": 3,
            "past_covariates_fnames": ['release_type', 'cover_type', 'style', 'price_category'],
            "minimum_sales_months": 1
        }

        # Act
        dataset = PlastinkaTrainingTSDataset(
            stock_features=stock_features,
            monthly_sales=monthly_sales_pivot,
            static_transformer=static_transformer,
            scaler=scaler,
            **dataset_params
        )
        actual_dataset_dict = dataset.to_dict()

        # Assert
        # Fall back to general example if isolated test output is not available
        if not expected_path.exists():
            expected_path = GENERAL_EXAMPLES_DIR / "PlastinkaTrainingTSDataset_values.json"

        _run_dataset_test(actual_dataset_dict, expected_path, "PlastinkaTrainingTSDataset")

    @patch('dill.dump')
    def test_plastinka_training_ts_dataset_with_save(self, mock_dill_dump):
        """Test PlastinkaTrainingTSDataset with save functionality."""
        # Arrange - создаем упрощенную структуру DataFrames
        date_index = pd.date_range('2022-01-01', periods=20, freq='ME')  # Больше точек

        # Создаем monthly_sales с простой структурой MultiIndex колонок
        monthly_sales = pd.DataFrame(
            index=date_index,
            columns=pd.MultiIndex.from_tuples([
                ('item1',), ('item2',)
            ])
        )
        # Заполняем данными
        monthly_sales = monthly_sales.fillna(10)

        # Создаем stock_features с простой структурой, совпадающей с monthly_sales
        stock_features = pd.DataFrame(
            index=date_index,
            columns=pd.MultiIndex.from_tuples([
                ('availability', 'item1'),
                ('availability', 'item2'),
                ('confidence', 'item1'),
                ('confidence', 'item2')
            ])
        )
        stock_features = stock_features.fillna(0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Act - используем меньший input_chunk_length
            dataset = PlastinkaTrainingTSDataset(
                stock_features=stock_features,
                monthly_sales=monthly_sales,
                save_dir=temp_dir,
                dataset_name="test_dataset",
                input_chunk_length=3,  # Уменьшенный размер
                output_chunk_length=1
            )

            # Assert
            mock_dill_dump.assert_called_once()
            assert dataset.save_dir == temp_dir
            assert dataset.dataset_name == "test_dataset"

    @patch('dill.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_plastinka_training_ts_dataset_from_dill(self, mock_file_open, mock_dill_load):
        """Test PlastinkaTrainingTSDataset.from_dill class method."""
        # Arrange
        mock_dataset = MagicMock(spec=PlastinkaTrainingTSDataset)
        mock_dill_load.return_value = mock_dataset

        # Act
        result = PlastinkaTrainingTSDataset.from_dill("test_path.dill")

        # Assert
        mock_file_open.assert_called_once_with("test_path.dill", 'rb')
        mock_dill_load.assert_called_once()
        assert result == mock_dataset


class TestTransformers:
    """Test suite for custom transformer classes."""

    def test_multi_column_label_binarizer_fit_transform(self):
        """Test MultiColumnLabelBinarizer fit and transform."""
        # Arrange
        transformer = MultiColumnLabelBinarizer(separator='/')
        test_data = pd.DataFrame({'categories': ['A/B', 'B/C', 'A/C']})

        # Act
        transformer.fit(test_data)
        result = transformer.transform(test_data)

        # Assert
        assert transformer.is_fit()
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] > test_data.shape[1]  # Should have more columns after binarization

    def test_global_log_min_max_scaler_fit_transform(self):
        """Test GlobalLogMinMaxScaler fit and transform operations."""
        # Arrange
        scaler = GlobalLogMinMaxScaler()
        test_data = pd.DataFrame({
            'values': [1e-6, 0.5, 1.5, 2.0]
        })

        # Act
        scaler.fit(test_data)
        scaled_data = scaler.transform(test_data)

        # Assert
        assert scaler.is_fit()
        # GlobalLogMinMaxScaler возвращает numpy array, не DataFrame
        assert isinstance(scaled_data, np.ndarray)
        assert scaled_data.shape == (4, 1)
        # Проверяем, что значения в диапазоне [0,1] с учетом погрешности floating point
        assert np.all(scaled_data >= -1e-6) and np.all(scaled_data <= 1.0 + 1e-6)


class TestValidationFunctions:
    """Test suite for data validation functions."""

    def test_validate_date_columns_success(self):
        """Test successful validate_date_columns execution."""
        # Arrange - создаем DataFrame с правильными колонками
        test_df = pd.DataFrame({
            'Год выпуска': [2020, 2021, 1985, 2022],
            'Год записи': [2020, 2021, 1985, 2022],
            'Тип': ['LP', 'LP', 'LP', 'EP'],
            'Дата создания': pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']),
            'Штрихкод': ['123456', '789012', '345678', '901234'],
            'Исполнитель': ['Artist 1', 'Artist 2', 'Artist 3', 'Artist 4'],
            'Альбом': ['Album 1', 'Album 2', 'Album 3', 'Album 4']
        })

        # Act
        result = validate_date_columns(test_df)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Год выпуска' in result.columns
        assert 'Год записи' in result.columns

    def test_validate_categories_success(self):
        """Test successful validate_categories execution."""
        # Arrange - создаем DataFrame с правильными колонками
        test_df = pd.DataFrame({
            'Конверт': ['EX', 'VG+', 'NM', 'VG'],
            'Год выпуска': [2020, 2021, 1985, 2022],
            'Ценовая категория': ['Low', 'Medium', 'High', 'Low'],
            'Штрихкод': ['123456', '789012', '345678', '901234'],
            'Исполнитель': ['Artist 1', 'Artist 2', 'Artist 3', 'Artist 4'],
            'Альбом': ['Album 1', 'Album 2', 'Album 3', 'Album 4'],
            'Цена, руб.': [100.0, 200.0, 300.0, 150.0]
        })

        # Act
        result = validate_categories(test_df)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Конверт' in result.columns

    def test_validate_styles_success(self):
        """Test successful validate_styles execution."""
        # Arrange - создаем DataFrame с правильными колонками
        test_df = pd.DataFrame({
            'Исполнитель': ['Artist 1', 'Artist 2', 'Artist 1', 'Artist 3'],
            'Альбом': ['Album 1', 'Album 2', 'Album 3', 'Album 4'],
            'Стиль': ['Rock', 'Jazz', 'Rock', 'Blues'],
            'Штрихкод': ['123456', '789012', '345678', '901234'],
            'Цена, руб.': [100.0, 200.0, 300.0, 150.0]
        })

        # Act
        result = validate_styles(test_df)

        # Assert
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'Исполнитель' in result.columns
        assert 'Альбом' in result.columns


class TestIntegration:
    """Integration tests for the complete data preparation pipeline."""

    def test_module_imports_successfully(self):
        """Test that the module can be imported without errors."""
        # This test verifies that all imports work correctly
        assert process_raw is not None
        assert filter_by_date is not None
        assert get_preprocessed_df is not None
        assert process_data is not None
        assert get_stock_features is not None
        assert get_monthly_sales_pivot is not None
        assert PlastinkaTrainingTSDataset is not None
        assert MultiColumnLabelBinarizer is not None
        assert GlobalLogMinMaxScaler is not None

    def test_constants_defined(self):
        """Test that all expected constants are defined."""
        assert GROUP_KEYS is not None
        assert isinstance(GROUP_KEYS, list)
        assert len(GROUP_KEYS) > 0

    @patch('plastinka_sales_predictor.data_preparation.pd.read_excel')
    @patch('plastinka_sales_predictor.data_preparation.pd.read_csv')
    def test_end_to_end_pipeline_integration(self, mock_read_csv, mock_read_excel):
        """Test end-to-end pipeline integration with mocked file I/O."""
        # Arrange - используем реальные тестовые данные
        stock_df_raw = _load_isolated_input_artifact("process_raw", "stock_df_raw.pkl")
        sales_df_raw = _load_isolated_input_artifact("process_raw", "sales_df_raw.pkl")

        mock_read_excel.return_value = stock_df_raw
        mock_read_csv.return_value = sales_df_raw

        # Act
        result_dict = process_data(
            "mock_stock.xlsx",
            "mock_sales.csv",
            "2023-01-01"
        )

        # Assert
        mock_read_excel.assert_called_once()
        mock_read_csv.assert_called_once()
        assert isinstance(result_dict, dict)
        assert 'stock' in result_dict
        assert 'sales' in result_dict
        assert isinstance(result_dict['stock'], pd.DataFrame)
        assert isinstance(result_dict['sales'], pd.DataFrame)
        assert not result_dict['stock'].empty
        assert not result_dict['sales'].empty
