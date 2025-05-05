import argparse
import logging
import sys
import psycopg2 # For database connection
import pandas as pd # Added for data manipulation
from datetime import timedelta # Added for date logic

# Add project root to path to allow importing deployment/scripts modules
# This might be fragile; consider better packaging/path handling later
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from deployment.app.db import feature_storage
# Import necessary components from prepare_datasets.py (adjust path if needed)
from scripts.prepare_datasets import (
    get_stock_features, 
    get_monthly_sales_pivot, 
    MultiColumnLabelBinarizer, # Assuming these are defined there
    GlobalLogMinMaxScaler      # Assuming these are defined there
)
# Add Darts imports
from darts import TimeSeries
from darts.models import NBEATSModel # Assuming NBEATS, adjust if train.py uses another model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_db(db_config):
    """Placeholder for establishing database connection."""
    # In real implementation, use db_config to connect
    # For testing with mocks, this might not even be called if patched correctly.
    logging.info(f"Attempting to connect to DB: {db_config.get('dbname')} on {db_config.get('host')}")
    # Replace with actual psycopg2 connection logic later
    # For now, return a dummy object or None if the mock handles it
    # return psycopg2.connect(**db_config)
    return "mock_connection_object" # Dummy return for structure

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Run training and prediction in DataSphere Job.')
    # parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='Operation mode: train or predict') # Removed
    
    # Database arguments
    parser.add_argument('--db-host', required=True, help='Database host')
    parser.add_argument('--db-port', type=int, default=5432, help='Database port') # Assuming PostgreSQL default
    parser.add_argument('--db-user', required=True, help='Database user')
    parser.add_argument('--db-password', required=True, help='Database password')
    parser.add_argument('--db-name', required=True, help='Database name')

    # Data parameters (add more as needed based on prepare_datasets.py/train.py)
    parser.add_argument('--start-date', help='Start date for data loading (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for data loading (YYYY-MM-DD)')
    parser.add_argument('--cutoff-date-upper', help='Upper cutoff date for sales data (DD-MM-YYYY) - used in prepare_datasets logic')

    # Model parameters (add more as needed)
    parser.add_argument('--model-output-ref', default='model.pkl', help='Reference/name for saving the trained model in DB')
    parser.add_argument('--prediction-output-ref', default='predictions.csv', help='Reference/name for saving predictions in DB')
    # Add arguments needed specifically for prediction period?
    # parser.add_argument('--prediction-months', type=int, default=1, help='Number of months to predict ahead')

    return parser.parse_args()

def load_data(db_config, start_date=None, end_date=None):
    """Loads features using feature_storage.load_features (factory pattern)."""
    # conn = None # Connection likely handled by factory/store
    features = None
    try:
        # conn = connect_db(db_config) # May not need explicit connection here
        logging.info("Loading features using feature_storage.load_features...")
        # Call the function using the factory
        # Pass DB config directly as kwargs for the factory/SQL store
        # Dates might be handled differently by load_features (cutoff_date?)
        # Need to verify how start/end dates map to load_features arguments
        # For now, assume start_date/end_date might be used internally or via kwargs
        # Adapt based on feature_storage.py logic
        features = feature_storage.load_features(
            store_type='sql', 
            cutoff_date=end_date, # Assuming end_date maps to cutoff_date?
            # start_date might be used internally by the store or needs specific kwarg
            **db_config # Pass DB connection params
        )
        logging.info("Features loaded successfully via factory.")
    except Exception as e:
        logging.error(f"Error loading data via factory: {e}", exc_info=True)
        raise
    # finally:
        # Connection closing likely handled by the store context manager if used
    return features

def prepare_features(raw_features: dict, params: dict) -> dict:
    """Prepares and transforms raw features based on logic from prepare_datasets.py."""
    logging.info("Starting feature preparation...")
    
    # Extract cutoff date from params (passed via args)
    cutoff_date_upper_str = params.get('cutoff_date_upper')

    # Validate required raw features exist
    if 'sales' not in raw_features or 'stock' not in raw_features: # Assuming load_features returns 'stock' initially
        logging.error("Missing 'sales' or 'stock' data in raw_features.")
        raise ValueError("Missing required raw feature data.")

    prepared_data = {}
    features = raw_features.copy() # Work on a copy

    try:
        # 1. Stock Features (using get_stock_features from prepare_datasets.py)
        # Assuming 'stock' and 'change' columns are present in raw_features['stock']
        logging.info("Calculating stock features...")
        # Correctly access 'change' column from the stock DataFrame
        change_series = features['stock'].get('change') # Use .get for safety if column might be missing
        features['stock_features'] = get_stock_features(
            features['stock'], change_series 
        )
        logging.info("Stock features calculated.")
        # We might not need the original 'stock' DF after this
        # del features['stock'] 

        # 2. Determine Cutoff Date if not provided
        if cutoff_date_upper_str is None:
            logging.info("Cutoff date not provided, determining from latest sales date...")
            latest_date = features['sales'].index.get_level_values('_date').max()
            # Logic from prepare_datasets.py to determine end of month
            if (latest_date + timedelta(days=1)).month == latest_date.month:
                cutoff_date_upper = latest_date.replace(day=1)
            else:
                # Go to the first day of the *next* month
                cutoff_date_upper = (latest_date.replace(day=1) + timedelta(days=32)).replace(day=1)
            cutoff_date_upper_str = cutoff_date_upper.strftime('%d-%m-%Y')
            logging.info(f"Determined cutoff date: {cutoff_date_upper_str}")
        else:
             cutoff_date_upper = pd.to_datetime(cutoff_date_upper_str, dayfirst=True)

        # 3. Filter data based on cutoff date
        logging.info(f"Filtering data with cutoff date: {cutoff_date_upper_str}")
        rounded_sales = features['sales'][
            features['sales'].index.get_level_values('_date') < cutoff_date_upper
        ]
        rounded_stocks = features['stock_features'][
            features['stock_features'].index < cutoff_date_upper # Assuming stock_features index is date
        ]
        logging.info("Data filtered.")

        # 4. Create Sales Pivot Table
        logging.info("Creating monthly sales pivot table...")
        sales_pivot = get_monthly_sales_pivot(rounded_sales)
        prepared_data['sales_pivot'] = sales_pivot
        logging.info("Sales pivot created.")

        # 5. Initialize Transformers (as per prepare_datasets.py example)
        # static_transformer = MultiColumnLabelBinarizer()
        # scaler = GlobalLogMinMaxScaler()
        # input_chunk_length = sales_pivot.shape[0] - 1 # Example calculation
        # output_chunk_length = 1                   # Example calculation
        # Store transformers/params if needed for prediction step
        # prepared_data['static_transformer'] = static_transformer
        # prepared_data['scaler'] = scaler
        # prepared_data['input_chunk_length'] = input_chunk_length
        # prepared_data['output_chunk_length'] = output_chunk_length
        logging.info("Placeholder for transformer initialization.")
        
        # Add other prepared data as needed
        prepared_data['rounded_stocks'] = rounded_stocks # Keep potentially needed stock data
        
        logging.info("Feature preparation finished.")

    except Exception as e:
        logging.error(f"Error during feature preparation: {e}", exc_info=True)
        raise

    return prepared_data

def train_model(prepared_features: dict, params: dict) -> NBEATSModel:
    """Trains an NBEATS model using prepared features."""
    logging.info("Starting model training...")

    # Extract necessary data and params
    sales_pivot = prepared_features.get('sales_pivot')
    if sales_pivot is None or sales_pivot.empty:
        logging.error("Sales pivot table is missing or empty in prepared_features.")
        raise ValueError("Missing sales pivot data for training.")

    # Convert pandas DataFrame to Darts TimeSeries
    # Ensure the DataFrame index is a DatetimeIndex and has a frequency
    # sales_pivot.index = pd.to_datetime(sales_pivot.index)
    # sales_pivot = sales_pivot.asfreq('MS') # Example: Set frequency to Month Start
    # The exact conversion might depend on get_monthly_sales_pivot output
    try:
        logging.info("Converting sales pivot to Darts TimeSeries...")
        # Simple conversion assuming correct index and single value column (or handle multi-column)
        # This might need refinement based on actual sales_pivot structure and Darts requirements
        target_series = TimeSeries.from_dataframe(sales_pivot, freq='MS') # Assuming monthly frequency 
        logging.info("Conversion to TimeSeries successful.")
    except Exception as e:
        logging.error(f"Failed to convert sales pivot to TimeSeries: {e}", exc_info=True)
        raise

    # Extract model parameters (use .get with defaults or raise errors)
    input_chunk_length = params.get('input_chunk_length', 12) # Example default
    output_chunk_length = params.get('output_chunk_length', 1) # Example default
    n_epochs = params.get('n_epochs', 50) # Example default
    # Add other hyperparameters from params dict as needed
    model_params = {
        'input_chunk_length': input_chunk_length,
        'output_chunk_length': output_chunk_length,
        'n_epochs': n_epochs,
        'random_state': 42 # For reproducibility
        # Add other relevant NBEATSModel parameters
    }
    logging.info(f"Initializing NBEATSModel with params: {model_params}")

    # Instantiate the model
    model = NBEATSModel(**model_params)

    # Train the model
    try:
        logging.info(f"Fitting model for {n_epochs} epochs...")
        model.fit(target_series) # Pass the TimeSeries object
        logging.info("Model fitting complete.")
    except Exception as e:
        logging.error(f"Error during model fitting: {e}", exc_info=True)
        raise

    return model

def predict_sales(model: NBEATSModel, params: dict) -> TimeSeries:
    """Generates sales predictions using the trained model."""
    logging.info("Starting sales prediction...")
    
    # Extract prediction parameters
    n_steps = params.get('n', 1) # Default to predict 1 step if not specified
    if not isinstance(n_steps, int) or n_steps <= 0:
        logging.warning(f"Invalid number of prediction steps '{n_steps}'. Defaulting to 1.")
        n_steps = 1
        
    logging.info(f"Predicting {n_steps} steps ahead.")

    predictions = None
    try:
        # Generate predictions
        predictions = model.predict(n=n_steps)
        logging.info("Prediction generation complete.")
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        raise
        
    return predictions

def main():
    args = parse_arguments()
    logging.info("Starting train and predict job.")
    logging.info(f"Database Host: {args.db_host}")
    
    db_connection_config = {
        "host": args.db_host,
        "port": args.db_port,
        "user": args.db_user,
        "password": args.db_password,
        "dbname": args.db_name
    }

    logging.info("Executing training and prediction pipeline...")
    # 1. Load data
    try:
        features_raw = load_data(db_connection_config, args.start_date, args.end_date)
    except Exception as e:
        logging.error(f"Pipeline failed during data loading: {e}")
        sys.exit(1)

    if not features_raw:
        logging.error("Failed to load features (returned None/empty). Exiting pipeline.")
        sys.exit(1)

    # 2. Prepare features
    try:
        params = {
            'cutoff_date_upper': args.cutoff_date_upper 
            # Add any other relevant args needed by prepare_features
        }
        features_prepared = prepare_features(features_raw, params)
    except Exception as e:
        logging.error(f"Pipeline failed during feature preparation: {e}")
        sys.exit(1)

    if not features_prepared:
        logging.error("Feature preparation returned empty results. Exiting pipeline.")
        sys.exit(1)

    # Extract parameters relevant for training (from args or config file potentially)
    training_params = {
        'input_chunk_length': 12, # Example - get from args eventually
        'output_chunk_length': 1, # Example - get from args eventually
        'n_epochs': args.n_epochs if hasattr(args, 'n_epochs') else 50 # Example - get from args if exists
        # Add other model hyperparameters from args
    }

    # 3. Train model
    try:
        model = train_model(features_prepared, training_params)
    except Exception as e:
        logging.error(f"Pipeline failed during model training: {e}")
        sys.exit(1)

    if not model:
        logging.error("Model training failed to return a model object. Exiting.")
        sys.exit(1)

    # 4. Predict using the trained model
    prediction_params = {
        'n': 3 # Example: Predict 3 months ahead - get from args eventually
    }
    try:
        predictions = predict_sales(model, prediction_params)
    except Exception as e:
        logging.error(f"Pipeline failed during prediction: {e}")
        sys.exit(1)

    if predictions is None:
        logging.error("Prediction failed to return results. Exiting.")
        sys.exit(1)

    # 5. Save model (placeholder)
    # save_model(db_connection_config, model, args.model_output_ref)
    logging.info(f"Model trained (placeholder for saving). Model ref: {args.model_output_ref}")

    # 6. Save predictions (placeholder)
    # save_predictions(db_connection_config, predictions, args.prediction_output_ref)
    logging.info(f"Predictions generated (placeholder for saving). Prediction ref: {args.prediction_output_ref}")

    logging.info("Train and predict pipeline finished.")

if __name__ == '__main__':
    main() 