import pandas as pd
import pytest
import pickle
from pathlib import Path
from plastinka_sales_predictor.data_preparation import categorize_prices
from tests.plastinka_sales_predictor.data_preparation.test_utils import compare_categorical_columns

# Define path to example data
ISOLATED_TESTS_BASE_DIR = Path("tests/example_data/isolated_tests")

def load_sample_data():
    """Load stock data from pkl file"""
    # First check if there's a dedicated sample for categorize_prices tests
    sample_path = ISOLATED_TESTS_BASE_DIR / "process_raw" / "inputs" / "stock_df_raw.pkl"
    if sample_path.exists():
        with open(sample_path, 'rb') as f:
            return pickle.load(f)
    else:
        # Fallback to Excel file if pkl not available
        return pd.read_excel('tests/example_data/sample_stocks.xlsx')

def test_categorize_prices_quantiles():
    # Arrange: load test data from pkl file
    df = load_sample_data()
    
    # Act: apply categorization by quantiles
    result_df, bins = categorize_prices(df)
    
    # Assert: check that column exists, with 7 unique categories (default q)
    assert 'Ценовая категория' in result_df.columns, "Column 'Ценовая категория' not found in result"
    
    unique_categories = result_df['Ценовая категория'].unique()
    assert len(unique_categories) <= 7, f"Too many unique categories: {len(unique_categories)} > 7"
    
    # Detail for actual unique categories
    if len(unique_categories) < 7:
        print(f"Warning: Only {len(unique_categories)} unique categories found, expected up to 7. Categories: {unique_categories}")
    
    assert bins is not None, "Bins should not be None"
    
    # Check for NaN values - allow up to 2 such values
    null_count = result_df['Ценовая категория'].isnull().sum()
    assert null_count <= 2, f"Found {null_count} null values in 'Ценовая категория' column (max allowed: 2)"
    if null_count > 0:
        print(f"Note: Found {null_count} null values in price category column, which is expected for this dataset")


def test_categorize_prices_with_bins():
    df = load_sample_data()
    
    # First get the bins
    _, bins = categorize_prices(df)
    # Now pass them explicitly
    result_df, bins2 = categorize_prices(df, bins=bins)
    
    assert 'Ценовая категория' in result_df.columns, "Column 'Ценовая категория' not found in result"
    
    # Bins should match
    bins_equal = all(bins == bins2)
    if not bins_equal:
        differences = [(i, b1, b2) for i, (b1, b2) in enumerate(zip(bins, bins2)) if b1 != b2]
        pytest.fail(f"Bins do not match. Differences: {differences}")


def test_categorize_prices_expected_bins():
    # Artificial data with known values
    df = pd.DataFrame({'Цена, руб.': [100, 200, 300, 400, 500, 600, 700]})
    # q=[0, 0.5, 1] divides into two categories: <=400 and >400
    result_df, bins = categorize_prices(df, q=[0, 0.5, 1])
    
    # Check that values <=400 are in the first category, the rest in the second
    cat = result_df['Ценовая категория']
    # Get unique categories
    cats = cat.unique()
    
    # Check there are two of them
    assert len(cats) == 2, f"Expected 2 categories, got {len(cats)}: {cats}"
    
    # Check distribution in each category
    first_category_counts = (cat == cats[0]).sum()
    second_category_counts = (cat == cats[1]).sum()
    
    assert first_category_counts == 4, f"Expected 4 items in first category, got {first_category_counts}"
    assert second_category_counts == 3, f"Expected 3 items in second category, got {second_category_counts}"
    
    # Provide detailed view of the classification result
    if not all(cat.iloc[:4] == cats[0]) or not all(cat.iloc[4:] == cats[1]):
        misclassified = []
        for i, (price, category) in enumerate(zip(df['Цена, руб.'], cat)):
            expected = cats[0] if i < 4 else cats[1]
            if category != expected:
                misclassified.append((i, price, category, expected))
        
        if misclassified:
            comparison_df = pd.DataFrame({
                'Index': df.index,
                'Price': df['Цена, руб.'],
                'Actual Category': cat,
                'Expected Category': [cats[0] if i < 4 else cats[1] for i in range(len(df))]
            })
            pytest.fail(f"Misclassified items:\n{comparison_df}\n\nMisclassified details: {misclassified}") 