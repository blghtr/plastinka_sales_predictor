import pytest
from unittest.mock import MagicMock, call
from plastinka_sales_predictor.data_preparation import run_data_processing_pipeline
import pandas as pd
from collections import OrderedDict

def test_run_data_processing_pipeline_happy_path():
    """Verify the pipeline runner executes all stages and handles artifacts correctly with more realistic mocks."""
    # Arrange
    mock_init_func = MagicMock()
    mock_step_func = MagicMock()
    mock_closure_func = MagicMock()

    initial_pool = {
        'base_features_names': ['sales', 'stock'],
        'report_features_names': ['availability']
    }
    mock_sales_df_jan = pd.DataFrame({'id': [1], 'sales_val': [10]}).set_index('id')
    mock_sales_df_feb = pd.DataFrame({'id': [1], 'sales_val': [20]}).set_index('id')
    mock_sorted_sales_dfs = OrderedDict([
        (pd.to_datetime('2024-02-01'), mock_sales_df_feb),
        (pd.to_datetime('2024-01-01'), mock_sales_df_jan),
    ])

    def init_side_effect(**kwargs):
        return (mock_sorted_sales_dfs,)

    def step_side_effect(**kwargs):
        month_date = kwargs['month_date']
        if month_date.month == 1:
            return pd.DataFrame([[10]], columns=[month_date], index=[1]), pd.DataFrame([[100]], columns=[month_date], index=[1]), pd.Series([0.9], name='availability', index=[1])
        else:
            return pd.DataFrame([[20]], columns=[month_date], index=[1]), pd.DataFrame([[90]], columns=[month_date], index=[1]), pd.Series([0.8], name='availability', index=[1])

    def closure_side_effect(**kwargs):
        base_features = kwargs['base_features_dict']
        report_features = kwargs['report_features_df']
        assert isinstance(base_features['sales'], pd.DataFrame)
        assert base_features['sales'].shape == (1, 2)
        # report_features может быть None, если нет report features
        if report_features is not None:
            assert isinstance(report_features, pd.DataFrame)
        return {'final_result': 'ok'}

    mock_init_func.side_effect = init_side_effect
    mock_step_func.side_effect = step_side_effect
    mock_closure_func.side_effect = closure_side_effect

    mock_config = {
        'init_steps': {
            'init': {'func': mock_init_func, 'inputs': [], 'outputs': ['sorted_sales_dfs']}
        },
        'steps': {
            'step': {'func': mock_step_func, 'inputs': ['processed_df', 'month_date'], 'outputs': ['sales', 'stock', 'availability']}
        },
        'closure': {
            'closure': {'func': mock_closure_func, 'inputs': ['base_features_dict', 'report_features_df'], 'outputs': ['pipeline_outputs']}
        }
    }

    # Act
    final_artifacts = run_data_processing_pipeline(mock_config, initial_pool)

    # Assert
    mock_init_func.assert_called_once()
    assert mock_step_func.call_count == len(mock_sorted_sales_dfs)
    mock_step_func.assert_has_calls([
        call(processed_df=mock_sales_df_feb, month_date=pd.to_datetime('2024-02-01')),
        call(processed_df=mock_sales_df_jan, month_date=pd.to_datetime('2024-01-01'))
    ])
    mock_closure_func.assert_called_once()
    assert final_artifacts == {'final_result': 'ok'}
