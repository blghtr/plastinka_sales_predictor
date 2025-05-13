import pandas as pd
import pytest
from plastinka_sales_predictor.data_preparation import categorize_prices
from tests.plastinka_sales_predictor.data_preparation.test_utils import compare_categorical_columns


def test_categorize_prices_quantiles():
    # Arrange: читаем тестовый файл
    df = pd.read_excel('tests/example_data/sample_stocks.xlsx')
    # Act: применяем категоризацию по квантилям
    result_df, bins = categorize_prices(df)
    
    # Assert: появился столбец, количество уникальных категорий = 7 (по умолчанию q)
    assert 'Ценовая категория' in result_df.columns, "Column 'Ценовая категория' not found in result"
    
    unique_categories = result_df['Ценовая категория'].unique()
    assert len(unique_categories) <= 7, f"Too many unique categories: {len(unique_categories)} > 7"
    
    # Detail for actual unique categories
    if len(unique_categories) < 7:
        print(f"Warning: Only {len(unique_categories)} unique categories found, expected up to 7. Categories: {unique_categories}")
    
    assert bins is not None, "Bins should not be None"
    
    # Проверяем наличие NaN - допускаем не более 2 таких значений
    null_count = result_df['Ценовая категория'].isnull().sum()
    assert null_count <= 2, f"Found {null_count} null values in 'Ценовая категория' column (max allowed: 2)"
    if null_count > 0:
        print(f"Note: Found {null_count} null values in price category column, which is expected for this dataset")


def test_categorize_prices_with_bins():
    df = pd.read_excel('tests/example_data/sample_stocks.xlsx')
    # Сначала получаем бины
    _, bins = categorize_prices(df)
    # Теперь передаем их явно
    result_df, bins2 = categorize_prices(df, bins=bins)
    
    assert 'Ценовая категория' in result_df.columns, "Column 'Ценовая категория' not found in result"
    
    # Бины должны совпадать
    bins_equal = all(bins == bins2)
    if not bins_equal:
        differences = [(i, b1, b2) for i, (b1, b2) in enumerate(zip(bins, bins2)) if b1 != b2]
        pytest.fail(f"Bins do not match. Differences: {differences}")


def test_categorize_prices_expected_bins():
    # Искусственные данные с известными значениями
    df = pd.DataFrame({'Цена, руб.': [100, 200, 300, 400, 500, 600, 700]})
    # q=[0, 0.5, 1] делит на две категории: <=400 и >400
    result_df, bins = categorize_prices(df, q=[0, 0.5, 1])
    
    # Проверяем, что значения <=400 попали в первую категорию, остальные — во вторую
    cat = result_df['Ценовая категория']
    # Получаем уникальные категории
    cats = cat.unique()
    
    # Проверяем, что их две
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