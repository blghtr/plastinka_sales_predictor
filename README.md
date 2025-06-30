# Plastinka Sales Predictor

A comprehensive machine learning system for predicting vinyl record sales using advanced time-series forecasting with cloud-based deployment infrastructure.

## 🎯 Overview

Plastinka Sales Predictor is a production-ready ML system that combines:

- **🧠 Advanced ML Models**: Time-series forecasting using TiDE (Time-series Dense Encoder) architecture
- **📊 Specialized Metrics**: Custom MIWS/MIC metrics for evaluating zero/non-zero sales predictions
- **🔮 Probabilistic Forecasting**: Quantile regression for predictive intervals and uncertainty estimation
- **☁️ Cloud Integration**: Seamless integration with Yandex DataSphere for scalable model training
- **🚀 Production API**: FastAPI-based REST API with asynchronous job processing
- **🏗️ Infrastructure as Code**: Terraform automation for cloud resource management

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Plastinka Sales Predictor System                         │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   ML Module     │    │   FastAPI App   │    │  Infrastructure │         │
│  │                 │    │                 │    │                 │         │
│  │ • TiDE Model    │    │ • REST API      │    │ • Terraform     │         │
│  │ • Data Prep     │    │ • Job Queue     │    │ • DataSphere    │         │
│  │ • Metrics       │    │ • Database      │    │ • Service Acc   │         │
│  │ • Training      │    │ • File Storage  │    │ • Monitoring     │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│           │                       │                       │                 │
│           └───────────────────────┼───────────────────────┘                 │
│                                   │                                         │
└───────────────────────────────────┼─────────────────────────────────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │   Yandex DataSphere   │
                        │                       │
                        │ • ML Compute          │
                        │ • Model Training      │
                        │ • Prediction Gen      │
                        │ • Resource Scaling    │
                        └───────────────────────┘
```

## 🔧 Key Components

### 1. ML Module (`plastinka_sales_predictor/`)
- **TiDE Architecture**: State-of-the-art time-series dense encoder for forecasting
- **Custom Metrics**: MIWS (Mean Interval Width Score) and MIC (Mean Interval Coverage) for specialized evaluation
- **Data Processing**: Advanced preprocessing for vinyl sales data with intermittent demand patterns
- **Quantile Regression**: Probabilistic forecasts with confidence intervals
- **Hyperparameter Tuning**: Ray Tune integration for automated optimization

### 2. FastAPI Application (`deployment/`)
- **Asynchronous API**: Non-blocking job processing with status tracking
- **Database Integration**: SQLite with star schema for efficient data storage
- **File Management**: Excel upload processing and model artifact storage
- **Cloud Integration**: Seamless DataSphere job submission and monitoring
- **Error Handling**: Comprehensive retry logic and failure recovery

### 3. Infrastructure (`deployment/infrastructure/`)
- **Terraform Modules**: Reusable infrastructure components
- **DataSphere Projects**: Automated ML compute environment setup
- **Service Accounts**: Proper IAM configuration for cloud access
- **Resource Management**: Cost-effective resource allocation and cleanup

## 📦 Installation

### Prerequisites
- Python 3.10+
- SQLite3
- 2GB+ RAM
- 1GB+ disk space
- Yandex Cloud account (for production deployment)

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd plastinka_sales_predictor
```

2. **Install dependencies**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e ".[dev,deployment]"
```

3. **Environment configuration**
```bash
# Check environment and generate template
python deployment/scripts/check_environment.py .env.template

# Copy and configure environment variables
cp .env.template .env
# Edit .env with your values
```

## 🚀 Usage

### Local Development

**Start the API server:**
```bash
cd deployment
python run.py --reload
```

The API will be available at `http://localhost:8000` with documentation at `/docs`.

### Core ML Functionality

**Data Processing:**
```python
from plastinka_sales_predictor.data_preparation import (
    process_data,
    PlastinkaTrainingTSDataset
)

# Process raw Excel files
features = process_data(
    stocks_path="data/stocks.xlsx",
    sales_path="data/sales/",
    cutoff_date="30-09-2022"
)

# Create training dataset
dataset = PlastinkaTrainingTSDataset(
    stock_features=features['stock_features'],
    monthly_sales=features['sales_pivot'],
    static_features=["Конверт", "Тип", "Ценовая категория", "Стиль"],
    input_chunk_length=12,
    output_chunk_length=1
)
```

**Model Training:**
```python
from plastinka_sales_predictor import train_model, prepare_for_training
import json

# Load configuration
with open("configs/model_config.json") as f:
    config = json.load(f)

# Train model
model = train_model(
    *prepare_for_training(config=config, ds=dataset)
)

# Save trained model
model.save("models/my_model.pt")
```

**Predictions:**
```python
# Generate probabilistic forecasts
predictions = model.predict(
    n=3,  # 3-month forecast
    num_samples=500,  # Monte Carlo samples
    series=dataset[0][0],
    past_covariates=dataset[0][1]
)
```

### API Usage

**Upload Data:**
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/data-upload" \
  -F "stock_file=@stocks.xlsx" \
  -F "sales_files=@sales_jan.xlsx"
```

**Train Model & Generate Predictions:**
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/training"
```

**Check Job Status:**
```bash
curl "http://localhost:8000/api/v1/jobs/{job_id}"
```

**Generate Reports:**
```bash
curl -X POST "http://localhost:8000/api/v1/jobs/reports"
```

## 🔧 Configuration

### Model Configuration
Models are configured via JSON files with the following structure:

```json
{
  "model_config": {
    "num_encoder_layers": 3,
    "num_decoder_layers": 2,
    "temporal_width_past": 2,
    "dropout": 0.5,
    "batch_size": 128
  },
  "optimizer_config": {
    "lr": 7.7e-05,
    "weight_decay": 0.0028
  },
  "lr_shed_config": {
    "T_0": 160,
    "T_mult": 1
  },
  "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95]
}
```

### Environment Variables
Key environment variables for deployment:

```bash
# Yandex Cloud Configuration
DATASPHERE_PROJECT_ID="your-project-id"
DATASPHERE_FOLDER_ID="your-folder-id"
DATASPHERE_OAUTH_TOKEN="your-oauth-token"

# API Configuration
API_HOST="0.0.0.0"
API_PORT="8000"
API_DEBUG="false"

# Data Storage
DATA_ROOT_DIR="~/.plastinka_sales_predictor"
MAX_UPLOAD_SIZE="52428800"  # 50MB
```

## 🏗️ Project Structure

```
plastinka_sales_predictor/
├── plastinka_sales_predictor/          # Core ML module
│   ├── __init__.py                     # Module exports
│   ├── data_preparation.py             # Data processing & dataset creation
│   ├── model.py                        # TiDE model implementation
│   ├── metrics.py                      # Custom MIWS/MIC metrics
│   ├── training_utils.py               # Training orchestration
│   ├── tuning_utils.py                 # Hyperparameter optimization
│   └── datasphere_job/                 # DataSphere execution scripts
│       ├── train_and_predict.py        # Main training script
│       └── config.yaml                 # DataSphere job configuration
├── deployment/                         # FastAPI application
│   ├── app/                           # Application core
│   │   ├── api/                       # REST API endpoints
│   │   ├── db/                        # Database schema & operations
│   │   ├── services/                  # Business logic services
│   │   └── utils/                     # Utility functions
│   ├── datasphere/                    # DataSphere integration
│   ├── infrastructure/                # Terraform IaC
│   └── scripts/                       # Deployment utilities
├── tests/                             # Comprehensive test suite


```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# All tests
pytest

# Specific test modules
pytest tests/plastinka_sales_predictor/  # ML module tests
pytest tests/deployment/app/             # API tests
pytest tests/deployment/app/db/          # Database tests

# With coverage
pytest --cov=plastinka_sales_predictor --cov=deployment
```

## 🚀 Cloud Deployment

### 1. Setup Infrastructure
```bash
cd deployment/infrastructure/envs/prod
cp terraform.tfvars.example terraform.tfvars
# Configure your Yandex Cloud credentials
terraform init && terraform apply
```

### 2. Deploy Application
```bash
# Get infrastructure outputs
export DATASPHERE_PROJECT_ID=$(terraform output -raw datasphere_project_id)
export YC_SERVICE_ACCOUNT_KEY_ID=$(terraform output -raw static_access_key_id)
export YC_SERVICE_ACCOUNT_KEY=$(terraform output -raw static_secret_key)

# Start the application
cd deployment
python run.py
```

## 📊 Key Features

### Advanced Metrics
- **MIWS (Mean Interval Width Score)**: Measures prediction interval efficiency
- **MIC (Mean Interval Coverage)**: Evaluates interval coverage accuracy
- **Zero-Inflation Handling**: Specialized metrics for intermittent demand patterns

### Model Architecture
- **TiDE (Time-series Dense Encoder)**: Cutting-edge architecture for time-series forecasting
- **Quantile Regression**: Full distribution predictions with uncertainty quantification
- **Multi-variate Features**: Integration of stock levels, pricing, and categorical features

### Production Features
- **Asynchronous Processing**: Non-blocking job execution with status tracking
- **Automatic Scaling**: Cloud-based compute with on-demand resource allocation
- **Data Retention**: Configurable cleanup policies for production data management
- **Monitoring**: Comprehensive logging and health monitoring


## 🛠️ Dependencies

### Core ML Dependencies
- `torch>=2.6.0` - PyTorch for deep learning
- `darts>=0.34.0` - Time series forecasting library
- `numpy>=1.26.4`, `pandas>=2.2.3` - Data manipulation
- `ray[tune]>=2.44.1` - Hyperparameter optimization
- `scikit-learn>=1.6.1` - Machine learning utilities

### Deployment Dependencies
- `fastapi>=0.115.12` - Modern web framework
- `uvicorn>=0.34.2` - ASGI server
- `datasphere>=0.10.0` - Yandex DataSphere SDK
- `pydantic-settings>=2.9.1` - Configuration management
- `aiofiles>=24.1.0` - Async file operations

### Development Dependencies
- `pytest>=8.3.5` - Testing framework
- `ruff>=0.8.0` - Fast Python linter
- `mypy>=1.13.0` - Static type checking

## 📄 License

[MIT License](LICENSE)


