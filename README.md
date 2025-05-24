# Plastinka Sales Predictor

A machine learning system for predicting vinyl record sales using time-series forecasting and quantile regression.

## Overview

Plastinka Sales Predictor uses a TiDE (Time-series Dense Encoder) deep learning model to generate probabilistic sales forecasts for vinyl records. It specializes in handling the unique characteristics of vinyl sales data, including intermittent demand patterns and zero-inflation.

## Key Features

- Time series forecasting with TiDE neural network architecture
- Quantile regression for probabilistic forecasts (predictive intervals)
- Custom metrics for evaluating zero/non-zero sales predictions (MIWS, MIC)
- Hyperparameter optimization with Ray Tune
- Cloud deployment support

## Project Structure

```
plastinka_sales_predictor/
├── __init__.py                 # Module exports
├── callbacks.py                # Training callbacks
├── data_preparation.py         # Data transformations and dataset creation
├── logger_setup.py             # Logging configuration
├── losses.py                   # Custom loss functions
├── metrics.py                  # Custom evaluation metrics
├── training_utils.py           # Model training utilities
├── tuning_utils.py             # Hyperparameter tuning
└── datasphere_job/             # Files specific to DataSphere job execution
    └── requirements.txt        # Python dependencies for the DataSphere job
```

## Installation

```bash
pip install -e .
```

## Usage

### Data Preparation

```python
from plastinka_sales_predictor.data_preparation import (
    process_data,
    get_stock_features,
    get_monthly_sales_pivot,
    PlastinkaTrainingTSDataset
)

# Process raw data
# NOTE: The following paths point to placeholder/sample files in the 'examples' directory.
# Replace 'examples/data/stocks_placeholder.txt' with your actual Excel file (e.g., stocks.xlsx).
# Populate 'examples/data/sales_data/' with your actual sales data files.
features = process_data(
    stocks_path="examples/data/stocks_placeholder.txt", # Or your actual 'stocks.xlsx'
    sales_path="examples/data/sales_data/",
    cutoff_date="30-09-2022" # Adjust as needed for your data
)

# Generate features and create dataset
stock_features = get_stock_features(features['stock'], features['change'])
sales_pivot = get_monthly_sales_pivot(features['sales'])

# Create training dataset
dataset = PlastinkaTrainingTSDataset(
    stock_features=stock_features,
    monthly_sales=sales_pivot,
    static_features=["Конверт", "Тип", "Ценовая категория", "Стиль"],
    input_chunk_length=12,
    output_chunk_length=1
)
```

### Model Training

```python
from plastinka_sales_predictor import (
    prepare_for_training,
    train_model
)
import json

# Load configuration
# NOTE: This example uses a sample configuration file.
with open("examples/configs/model_config.json", "r") as f:
    config = json.load(f)

# Prepare model and train
# Ensure the 'dataset' object is created as shown in the Data Preparation section.
model = train_model(
    *prepare_for_training(
        config=config,
        ds=dataset # Assumes 'dataset' is available from the Data Preparation step
    )
)

# Save model
# NOTE: Ensure the 'examples/models/' directory exists or is created before running.
model.save("examples/models/my_model.pt")
```

### Prediction

```python
# Load model and make predictions
predictions = model.predict(
    n=3,  # Forecast horizon
    series=dataset[0][0],  # Input series
    past_covariates=dataset[0][1]  # Past covariates
)
```

## Configuration Options

The model is configured through JSON files with the following structure:

```json
{
  "model_config": {
    "num_encoder_layers": 3,
    "num_decoder_layers": 2,
    "temporal_width_past": 2,
    "dropout": 0.5
  },
  "optimizer_config": { "lr": 7.7e-05, "weight_decay": 0.0028 },
  "lr_shed_config": { "T_0": 160, "T_mult": 1 },
  "train_ds_config": { "alpha": 1.48, "span": 4 },
  "weights_config": { "sigma_left": 0.92, "sigma_right": 1.81 },
  "lags": 12,
  "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]
}
```

## Dependencies

- torch
- darts
- pytorch_lightning
- ray
- pandas
- numpy

## License

[MIT](LICENSE)
