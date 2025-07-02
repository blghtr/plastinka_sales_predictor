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

2. **Install dependencies for your use case**

**For ML Development & Training (CPU):**
```bash
# ML environment with CPU PyTorch (рекомендуется для разработки)
uv sync --extra ml --extra cpu --extra dev
```

**For ML Development & Training (GPU):**
```bash
# ML environment with CUDA PyTorch (для GPU обучения)
uv sync --extra ml --extra cu118 --extra dev
```

**For Deployment Only:**
```bash
# Deployment with basic data processing (no PyTorch)
uv sync --extra deployment
```

**For Notebook Development:**
```bash
# ML + Jupyter environment (CPU)
uv sync --extra ml --extra cpu --extra notebook
```

**For Full Development:**
```bash
# All dependencies except PyTorch (choose cpu or cu118 separately)
uv sync --extra ml --extra dev --extra notebook --extra deployment --extra cpu
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

**For ML Development:**
```bash
# Install ML dependencies with CPU PyTorch
uv sync --extra ml --extra cpu --extra dev

# Run training or model development
python -m plastinka_sales_predictor.datasphere_job.train_and_predict
```

**For Deployment/API Development:**
```bash
# Install deployment dependencies (no PyTorch, but includes darts for data processing)
uv sync --extra deployment --extra dev

# Start the API server
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

#### Required Variables (must be set in .env file):

Create a `.env` file with these **REQUIRED** variables (the `check_environment.py` script can generate a template):
```bash
# DataSphere Configuration (REQUIRED)
DATASPHERE_PROJECT_ID=your-project-id
DATASPHERE_FOLDER_ID=your-folder-id
DATASPHERE_YC_PROFILE=datasphere-prod

# API Security (REQUIRED)
API_ADMIN_API_KEY=admin-bearer-token  # Bearer <token> for admin-level endpoints
API_X_API_KEY=public-api-key          # X-API-Key for regular endpoints
```

Alternatively, you can export these as environment variables:
```bash
export DATASPHERE_PROJECT_ID="your-project-id"
export DATASPHERE_FOLDER_ID="your-folder-id"
export DATASPHERE_YC_PROFILE="datasphere-prod"
export API_ADMIN_API_KEY="admin-bearer-token"
export API_X_API_KEY="public-api-key"
```

#### Optional Variables (have defaults):
```bash
# Authentication Method (defaults to "auto")
DATASPHERE_AUTH_METHOD=auto

# For legacy/development OAuth authentication (optional)
DATASPHERE_OAUTH_TOKEN=your-oauth-token

# API Configuration (optional, defaults shown)
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Data Storage (optional, defaults shown)
DATA_ROOT_DIR=~/.plastinka_sales_predictor
MAX_UPLOAD_SIZE=52428800
```

#### Terraform Variables:
```bash
# For terraform operations, set as environment variable:
export TF_VAR_yc_token="your-oauth-token"

# Or pass directly to terraform:
terraform apply -var="yc_token=your-oauth-token"
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

# Edit terraform.tfvars with your Yandex Cloud IDs:
# yc_cloud_id        = "your-cloud-id-here"
# yc_folder_id       = "your-folder-id-here"  
# yc_organization_id = "your-organization-id-here"

# Set OAuth token for terraform (choose one method):
export TF_VAR_yc_token="your-oauth-token"
# OR pass directly: terraform apply -var="yc_token=your-oauth-token"

terraform init && terraform apply
```

### 2. Deploy Application
```bash
# Get infrastructure outputs and set required .env variables
export DATASPHERE_PROJECT_ID=$(terraform output -raw datasphere_project_id)
export DATASPHERE_FOLDER_ID="your-folder-id"
export DATASPHERE_YC_PROFILE="datasphere-prod"
export API_ADMIN_API_KEY="admin-bearer-token"
export API_X_API_KEY="public-api-key"

# Start the application
cd deployment
uv run --extra deployment python run.py
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

Проект использует модульную систему зависимостей с 4 окружениями:

### 1. Base Dependencies (всегда устанавливаются)
- `click>=8.1.8` - CLI interface
- `PyYAML>=6.0.1` - Configuration files
- `dill>=0.3.9` - Serialization (needed for data processing)
- `darts>=0.34.0` - Time series library (core functionality)
- `build>=1.2.2.post1`, `setuptools>=78.0.2`, `wheel>=0.45.1` - Build tools

### 2. ML Environment (`--extra ml`)
**Для машинного обучения (plastinka_sales_predictor/)**
- `configspace<=0.7.1` - Configuration space for hyperparameter tuning
- `hpbandster>=0.7.4` - Hyperparameter optimization backend
- `numpy>=1.26.4`, `pandas>=2.2.3` - Data manipulation
- `ray[tune]>=2.44.1` - Hyperparameter optimization
- `scikit-learn>=1.6.1` - Machine learning utilities
- `scipy>=1.15.2` - Scientific computing
- `tensorboard>=2.19.0` - Training visualization
- `torchmetrics>=1.7.0` - ML metrics
- `onnx>=1.18.0` - Model export format

### 3. Deployment Environment (`--extra deployment`)
**Для веб-сервиса с базовой обработкой данных (deployment/)**
- `fastapi>=0.115.12` - Modern web framework
- `uvicorn>=0.34.2`, `gunicorn>=23.0.0` - ASGI/WSGI servers
- `pandas>=2.2.3`, `numpy>=1.26.4` - Data processing
- `datasphere>=0.10.0` - Yandex DataSphere SDK
- `pydantic-settings>=2.9.1` - Configuration management
- `aiofiles>=24.1.0` - Async file operations
- `openpyxl>=3.1.5` - Excel file processing
- `boto3>=1.38.2`, `botocore>=1.38.2` - AWS SDK
- `psutil>=7.0.0` - System monitoring
- `psycopg2-binary>=2.9.10` - PostgreSQL adapter

### 4. Development Environment (`--extra dev`)
- `pytest>=8.3.5` - Testing framework
- `ruff>=0.8.0` - Fast Python linter
- `fastapi>=0.115.12` - For API development
- `httpx>=0.28.1` - HTTP client for testing

### 5. Notebook Environment (`--extra notebook`)
- `ipykernel>=6.29.5` - Jupyter kernel
- `ipywidgets>=8.1.5` - Interactive widgets

### Installation Commands:
```bash
# ML разработка (CPU)
uv sync --extra ml --extra cpu --extra dev

# ML разработка (GPU)
uv sync --extra ml --extra cu118 --extra dev

# Deployment (production, no PyTorch)
uv sync --extra deployment

# Notebook разработка (CPU)
uv sync --extra ml --extra cpu --extra notebook

# Полная разработка (CPU)
uv sync --extra ml --extra dev --extra notebook --extra deployment --extra cpu
```

### 6. PyTorch Backend Selection:
Выберите версию PyTorch в зависимости от ваших потребностей:

**CPU версия (рекомендуется для deployment и разработки):**
```bash
# ML с CPU PyTorch
uv sync --extra ml --extra cpu

# Deployment не требует PyTorch (использует darts для базовой обработки)
uv sync --extra deployment
```

**CUDA версия (для GPU обучения):**
```bash
# ML с CUDA PyTorch
uv sync --extra ml --extra cu118
```

**⚠️ Важно**: Нельзя одновременно устанавливать `cpu` и `cu118` - система предотвратит конфликты.

## 📄 License

[MIT License](LICENSE)


