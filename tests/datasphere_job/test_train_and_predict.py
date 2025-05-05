import pytest
from unittest.mock import patch, MagicMock
import sys
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel

# Assume the main script will be in plastinka_sales_predictor.datasphere_job.train_and_predict
# We might need to adjust the import path later
from plastinka_sales_predictor.datasphere_job.train_and_predict import parse_arguments, load_data, prepare_features, train_model, predict_sales # Import train_model and predict_sales
# from plastinka_sales_predictor.datasphere_job.train_and_predict import train_model # Import later
# from darts import TimeSeries # Example imports for type hints / mocks if needed
# from darts.models import NBEATSModel # Example model

# Placeholder test - This will fail until parse_arguments is implemented
def test_parse_arguments_minimal():
    # Test minimal required args
    test_args = ['--db-host', 'localhost', '--db-user', 'test', '--db-password', 'pass', '--db-name', 'mydb']
    with patch.object(sys, 'argv', ['script_name'] + test_args):
        args = parse_arguments() # This line should now work
        # assert args.mode == 'train' # Removed
        assert args.db_host == 'localhost'
        assert args.db_user == 'test'
        assert args.db_password == 'pass'
        assert args.db_name == 'mydb'
        assert args.db_port == 5432 # Check default port
        assert args.model_output_ref == 'model.pkl' # Check default output ref
        assert args.prediction_output_ref == 'predictions.csv' # Check default output ref
        # assert False # Remove placeholder failure

def test_parse_arguments_custom_port_outputs():
    # Test custom port and output refs
    test_args = [
        '--db-host', 'db.example.com', 
        '--db-user', 'prod', 
        '--db-password', 'secret', 
        '--db-name', 'production', 
        '--db-port', '5433',
        '--model-output-ref', 'final_model.joblib',
        '--prediction-output-ref', 'forecast_next_month.parquet'
    ]
    with patch.object(sys, 'argv', ['script_name'] + test_args):
        args = parse_arguments()
        assert args.db_host == 'db.example.com'
        assert args.db_port == 5433
        assert args.model_output_ref == 'final_model.joblib'
        assert args.prediction_output_ref == 'forecast_next_month.parquet'

# Add more tests for other arguments like dates as they become relevant 

# Test data loading
def test_load_data_success():
    """Tests the data loading function succeeds, mocking feature_storage.load_features."""
    db_config = {
        'host': 'mock_host',
        'port': 5432,
        'user': 'mock_user',
        'password': 'mock_pass',
        'dbname': 'mock_db'
    }
    # Example expected data structure 
    expected_sales = pd.DataFrame({'date': pd.to_datetime(['2023-01-01']), 'sales': [10]})
    expected_stock = pd.DataFrame({'date': pd.to_datetime(['2023-01-01']), 'stock': [100]})
    expected_features = {'sales': expected_sales, 'stock': expected_stock}

    start_date_arg = None # Test with default dates first
    end_date_arg = None

    # Patch the target function used by load_data
    with patch('deployment.app.db.feature_storage.load_features') as mock_load_features:
        
        # Configure mock return value
        mock_load_features.return_value = expected_features

        # Call the function to be tested 
        features = load_data(db_config, start_date=start_date_arg, end_date=end_date_arg)

        # Assertions 
        # Check that load_features was called correctly
        mock_load_features.assert_called_once_with(
            store_type='sql',
            cutoff_date=end_date_arg, # Should be None in this case
            **db_config # Check that db_config was passed as kwargs
        )
        
        # Check the returned data matches expected
        assert features is not None
        assert 'sales' in features
        assert 'stock' in features
        pd.testing.assert_frame_equal(features['sales'], expected_features['sales'])
        pd.testing.assert_frame_equal(features['stock'], expected_features['stock'])
        # assert False # Remove placeholder failure

def test_load_data_with_dates():
    """Tests the data loading function with specific dates."""
    db_config = {'host': 'mock_host', 'user': 'mock_user', 'password': 'mock_pass', 'dbname': 'mock_db'}
    expected_features = {'sales': pd.DataFrame(), 'stock': pd.DataFrame()} # Dummy data for this test
    start_date_arg = '2023-01-01'
    end_date_arg = '2023-12-31' # This should map to cutoff_date

    with patch('deployment.app.db.feature_storage.load_features') as mock_load_features:
        mock_load_features.return_value = expected_features
        features = load_data(db_config, start_date=start_date_arg, end_date=end_date_arg)

        mock_load_features.assert_called_once_with(
            store_type='sql',
            cutoff_date=end_date_arg, # Check end_date passed as cutoff_date
            **db_config
        )
        assert features is not None

def test_load_data_exception():
    """Tests that an exception from load_features is propagated."""
    db_config = {'host': 'mock_host', 'user': 'mock_user', 'password': 'mock_pass', 'dbname': 'mock_db'}
    
    with patch('deployment.app.db.feature_storage.load_features') as mock_load_features:
        # Configure mock to raise an exception
        mock_load_features.side_effect = ValueError("DB connection failed")

        # Assert that calling load_data raises the expected exception
        with pytest.raises(ValueError, match="DB connection failed"):
            load_data(db_config, start_date=None, end_date=None) 

# Test Data Preparation
def test_prepare_features():
    """Tests the data preparation and transformation logic."""
    # Sample raw data - structure reflects loading 'stock' first
    raw_sales = pd.DataFrame({
        '_date': pd.to_datetime(['2023-01-15', '2023-01-20', '2023-02-10']), 
        'store_id': [1, 1, 1],
        'product_id': [101, 101, 101],
        'sales': [5, 3, 7]
    }).set_index(['_date', 'store_id', 'product_id'])
    
    raw_stock = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']), # Added Mar 1 for cutoff
        'store_id': [1, 1, 1],
        'product_id': [101, 101, 101],
        'stock': [100, 90, 85],
        'change': [-10, -5, -3] # Example change column 
    }).set_index('date') # Assuming index is just date for stock table
    
    # Input dict for prepare_features
    # Assumes load_features returns keys 'sales' and 'stock'
    raw_features = {'sales': raw_sales, 'stock': raw_stock}

    params = {'cutoff_date_upper': '01-03-2023'} # Cutoff March 1st

    # ---- Mock external dependencies used within prepare_features ----
    # Mock get_stock_features - assume it returns a DataFrame with date index
    mock_stock_features_df = pd.DataFrame({
        'stock': [100, 90], # Jan, Feb data 
        'change': [-10, -5] 
    }, index=pd.to_datetime(['2023-01-01', '2023-02-01']))
    mock_stock_features_df.index.name = 'date' # Ensure index name matches expectation

    # Mock get_monthly_sales_pivot - assume it aggregates Jan & Feb sales
    # Example: Pivot might have stores/products as columns, months as index
    mock_sales_pivot_df = pd.DataFrame({
        (1, 101): [8.0, 7.0] # Sum of sales for (store 1, prod 101) in Jan, Feb
    }, index=pd.to_datetime(['2023-01-01', '2023-02-01'])) # Monthly index
    mock_sales_pivot_df.index.name = '_date' # Or appropriate index name

    # --- Expected output --- 
    # Based on the implemented logic and mocks
    expected_output = {
        'sales_pivot': mock_sales_pivot_df, # Output from mocked get_monthly_sales_pivot
        'rounded_stocks': mock_stock_features_df # Output from mocked get_stock_features, filtered by cutoff
        # Add expected transformers/params if they were initialized
    }

    # Patch the imported functions from scripts.prepare_datasets
    with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.get_stock_features') as mock_get_stock:
        with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.get_monthly_sales_pivot') as mock_get_pivot:

            # Configure mock return values
            mock_get_stock.return_value = mock_stock_features_df
            mock_get_pivot.return_value = mock_sales_pivot_df

            # Call the function to be tested
            transformed_features = prepare_features(raw_features, params)

            # --- Assertions --- 
            # Check mocked functions were called correctly
            mock_get_stock.assert_called_once() # Check it was called
            # Extract call args and compare DataFrames/Series explicitly
            call_args_stock, _ = mock_get_stock.call_args
            assert len(call_args_stock) == 2 # Ensure correct number of positional args
            pd.testing.assert_frame_equal(call_args_stock[0], raw_features['stock'])
            pd.testing.assert_series_equal(call_args_stock[1], raw_features['stock']['change'])

            # Check call to get_monthly_sales_pivot (needs expected input dataframe)
            # The input to get_monthly_sales_pivot is filtered sales, check this:
            expected_rounded_sales = raw_features['sales'][raw_features['sales'].index.get_level_values('_date') < pd.to_datetime(params['cutoff_date_upper'], dayfirst=True)]
            mock_get_pivot.assert_called_once()
            # More robust: check the first argument passed to the mock
            call_args_pivot, _ = mock_get_pivot.call_args
            assert len(call_args_pivot) == 1
            pd.testing.assert_frame_equal(call_args_pivot[0], expected_rounded_sales)
            
            # Check the structure and content of the returned dictionary
            assert 'sales_pivot' in transformed_features
            assert 'rounded_stocks' in transformed_features
            pd.testing.assert_frame_equal(transformed_features['sales_pivot'], expected_output['sales_pivot'])
            pd.testing.assert_frame_equal(transformed_features['rounded_stocks'], expected_output['rounded_stocks'])
            # assert False # Remove placeholder failure

def test_prepare_features_missing_data():
    """Tests that prepare_features raises ValueError if input data is missing."""
    with pytest.raises(ValueError, match="Missing required raw feature data."):
        prepare_features({'sales': pd.DataFrame()}, {}) # Missing 'stock'
    with pytest.raises(ValueError, match="Missing required raw feature data."):
        prepare_features({'stock': pd.DataFrame()}, {}) # Missing 'sales'

def test_prepare_features_auto_cutoff():
    """Tests automatic cutoff date determination."""
    # Sample raw data
    raw_sales = pd.DataFrame({
        '_date': pd.to_datetime(['2023-01-15', '2023-01-20', '2023-02-10', '2023-03-05']), # Includes March data
        'store_id': [1]*4, 'product_id': [101]*4, 'sales': [5, 3, 7, 2]
    }).set_index(['_date', 'store_id', 'product_id'])
    raw_stock = pd.DataFrame({'date': pd.to_datetime(['2023-01-01']), 'stock': [100], 'change': [-10]}).set_index('date')
    raw_features = {'sales': raw_sales, 'stock': raw_stock}
    params = {'cutoff_date_upper': None} # No cutoff provided

    # Mock dependencies
    mock_stock_features_df = pd.DataFrame({'stock': [100]}, index=pd.to_datetime(['2023-01-01']))
    mock_stock_features_df.index.name='date'
    # Auto cutoff should be 2023-04-01 (start of next month after Mar 5)
    # So, pivot should include Jan, Feb, Mar data
    mock_sales_pivot_df = pd.DataFrame({(1, 101): [8.0, 7.0, 2.0]}, index=pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']))
    mock_sales_pivot_df.index.name = '_date'

    # Nested with statements for patching (Corrected)
    with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.get_stock_features') as mock_get_stock:
        with patch('plastinka_sales_predictor.datasphere_job.train_and_predict.get_monthly_sales_pivot') as mock_get_pivot:
            # Indentation fixed for lines below
            mock_get_stock.return_value = mock_stock_features_df
            mock_get_pivot.return_value = mock_sales_pivot_df
            
            transformed_features = prepare_features(raw_features, params)
            
            # Assert pivot includes March data due to auto cutoff being April 1st
            assert transformed_features['sales_pivot'].shape[0] == 3 
            # Assert rounded_stocks filtered correctly (up to April 1st)
            # In this case, the mock only has Jan data, which is before Apr 1st
            assert transformed_features['rounded_stocks'].shape[0] == 1 

# Test Model Training
# Use autospec=True for better mock signature matching if needed
@patch('plastinka_sales_predictor.datasphere_job.train_and_predict.NBEATSModel', autospec=True) 
def test_train_model(MockNBEATSModel):
    """Tests the model training function."""
    # Sample prepared data (output from prepare_features)
    # Simplify pivot table to have a single, named value column for NBEATS
    sales_pivot_data = pd.DataFrame({
        'sales': [8.0, 7.0, 9.0, 6.0] # Example monthly sales with simple column name
    }, index=pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01']))
    # Ensure frequency is set for TimeSeries conversion
    sales_pivot_data = sales_pivot_data.asfreq('MS') 
    sales_pivot_data.index.name = '_date' # Keep datetime index name if needed
    prepared_features = {'sales_pivot': sales_pivot_data}

    # Sample training parameters used inside train_model (defaults + passed)
    training_params = {
        'input_chunk_length': 12, 
        'output_chunk_length': 1, 
        'n_epochs': 50 
    }
    # Expected TimeSeries for fit call (now uses simplified DataFrame)
    expected_ts = TimeSeries.from_dataframe(sales_pivot_data, value_cols=['sales'], freq='MS') # Explicitly state value col

    # Configure the mock model instance 
    mock_model_instance = MockNBEATSModel.return_value

    # Call the function to be tested
    trained_model = train_model(prepared_features, training_params)

    # --- Assertions ---
    # Check that the model was instantiated correctly
    MockNBEATSModel.assert_called_once_with(
        input_chunk_length=training_params['input_chunk_length'], 
        output_chunk_length=training_params['output_chunk_length'],
        n_epochs=training_params['n_epochs'],
        random_state=42 # Check fixed random state
    )

    # Check that fit was called correctly on the mock instance
    mock_model_instance.fit.assert_called_once()
    call_args_fit, _ = mock_model_instance.fit.call_args
    assert len(call_args_fit) == 1
    # Compare underlying dataframe/values of the TimeSeries object
    pd.testing.assert_frame_equal(call_args_fit[0].pd_dataframe(), expected_ts.pd_dataframe())
    
    # Check that the returned object is the mock model instance
    assert trained_model is mock_model_instance

def test_train_model_missing_pivot():
    """Tests that train_model raises ValueError if sales_pivot is missing."""
    with pytest.raises(ValueError, match="Missing sales pivot data for training."):
        train_model({}, {}) # Empty prepared_features
    with pytest.raises(ValueError, match="Missing sales pivot data for training."):
        train_model({'sales_pivot': pd.DataFrame()}, {}) # Empty DataFrame

# Test Model Prediction
@patch('plastinka_sales_predictor.datasphere_job.train_and_predict.NBEATSModel', autospec=True)
def test_predict_sales(MockNBEATSModel):
    """Tests the sales prediction function."""
    # Mock a trained model instance
    mock_trained_model = MockNBEATSModel.return_value 

    # Define prediction parameters
    prediction_params = {
        'n': 3 # Predict 3 steps ahead (e.g., 3 months)
    }

    # Define the expected output from model.predict()
    expected_prediction_index = pd.date_range(start='2023-05-01', periods=prediction_params['n'], freq='MS')
    expected_prediction_values = [[10.0], [11.0], [10.5]]
    expected_predictions_ts = TimeSeries.from_times_and_values(expected_prediction_index, expected_prediction_values, columns=['sales'])

    # Configure the mock predict method
    mock_trained_model.predict.return_value = expected_predictions_ts

    # Call the function to be tested
    predictions = predict_sales(mock_trained_model, prediction_params)

    # --- Assertions ---
    # Check that the model's predict method was called correctly
    mock_trained_model.predict.assert_called_once_with(n=prediction_params['n'])
    
    # Check that the returned predictions match the expected TimeSeries
    # Compare underlying DataFrames for robustness
    assert predictions is not None
    pd.testing.assert_frame_equal(predictions.pd_dataframe(), expected_predictions_ts.pd_dataframe())
    # assert False # Remove placeholder failure

def test_predict_sales_invalid_n():
    """Tests prediction with invalid n defaults to 1."""
    mock_trained_model = MagicMock(spec=NBEATSModel)
    expected_ts = TimeSeries.from_values(pd.Series([1.0])) # Dummy output for n=1
    mock_trained_model.predict.return_value = expected_ts

    # Test with n=0
    predict_sales(mock_trained_model, {'n': 0})
    mock_trained_model.predict.assert_called_with(n=1) # Should default to 1

    # Reset mock for next call
    mock_trained_model.reset_mock()
    mock_trained_model.predict.return_value = expected_ts

    # Test with n=-5
    predict_sales(mock_trained_model, {'n': -5})
    mock_trained_model.predict.assert_called_with(n=1) # Should default to 1

    # Reset mock for next call
    mock_trained_model.reset_mock()
    mock_trained_model.predict.return_value = expected_ts

    # Test with n missing (should use default n=1 from .get)
    predict_sales(mock_trained_model, {})
    mock_trained_model.predict.assert_called_with(n=1) # Should default to 1

# Add more tests for other edge cases or additional functionality as needed 