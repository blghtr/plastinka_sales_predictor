import io
import json
from unittest.mock import MagicMock
from pathlib import Path

def create_mock_open_function(metrics_data=None, predictions_data=None, model_content="dummy model content"):
    """
    Создает мок для функции open, который возвращает разное содержимое в зависимости от пути файла.
    
    Args:
        metrics_data: Словарь с метриками, который будет возвращен для metrics.json, или None если файл не существует
        predictions_data: Строка с данными предсказаний для predictions.csv, или None если файл не существует
        model_content: Содержимое файла модели для model.onnx
    
    Returns:
        Функция-мок для patching функции open
    """
    if metrics_data is None:
        metrics_data = {"val_MIC": 0.85, "train_loss": 0.12, "val_loss": 0.15}
    
    if predictions_data is None:
        predictions_data = "date,store_id,sku_id,prediction\n2023-01-01,1,1,10.5\n2023-01-02,1,1,11.2"
    
    # Создаем имитацию объекта файла с методом write
    class MockFile:
        def __init__(self, content):
            self.content = content
            self.write_buffer = io.StringIO()
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def read(self):
            return self.content
        
        def write(self, data):
            self.write_buffer.write(data)
            return len(data)
        
        def close(self):
            pass
    
    def mock_open(file, mode="r", *args, **kwargs):
        file_path = str(file)
        
        # Для разных типов файлов возвращаем разное содержимое
        if "metrics.json" in file_path:
            if metrics_data is not None:
                content = json.dumps(metrics_data)
                return MockFile(content)
            else:
                raise FileNotFoundError(f"Simulated FileNotFoundError: {file_path}")
        
        elif "predictions.csv" in file_path:
            if predictions_data is not None:
                return MockFile(predictions_data)
            else:
                raise FileNotFoundError(f"Simulated FileNotFoundError: {file_path}")
        
        elif "model.onnx" in file_path or ".pt" in file_path:
            return MockFile(model_content)
        
        elif "config.json" in file_path:
            # Для режима записи возвращаем объект с методом write
            if 'w' in mode:
                return MockFile("")
            else:
                content = json.dumps({
                    "config_id": "test_config_id",
                    "config": {
                        "model_id": "test_model",
                        "lags": 5,
                        "model_config": {"num_encoder_layers": 2}
                    }
                })
                return MockFile(content)
        
        elif "job_config" in file_path:
            content = """
            apiVersion: datasphere.yandex-cloud/v1
            kind: TrainingJob
            metadata:
              name: plasnika-predict-job
            spec:
              inputs:
                - name: ds_input_dir
                  resourceUri: dir://test-dir
            """
            return MockFile(content)
        
        else:
            # Для любого другого файла возвращаем объект с методом write
            return MockFile("")
    
    return mock_open

def prepare_success_results_dir(fs, output_dir, job_id, metrics_data=None, predictions_data=None):
    """
    Подготавливает директорию с результатами успешного запуска DataSphere job.
    
    Args:
        fs: Фейковая файловая система (pyfakefs)
        output_dir: Директория для выходных данных
        job_id: ID задания
        metrics_data: Словарь с метриками или None для стандартных значений
        predictions_data: Строка с данными предсказаний или None для стандартных значений
    """
    job_results_dir = Path(output_dir) / job_id
    fs.create_dir(job_results_dir)
    
    # Создаем метрики
    if metrics_data is not None:
        metrics_path = job_results_dir / "metrics.json"
        fs.create_file(metrics_path, contents=json.dumps(metrics_data))
    
    # Создаем предсказания
    if predictions_data is not None:
        predictions_path = job_results_dir / "predictions.csv"
        fs.create_file(predictions_path, contents=predictions_data)
    
    # Создаем модель
    model_path = job_results_dir / "model.onnx"
    fs.create_file(model_path, contents="dummy model content")
    
    return job_results_dir

def prepare_failed_results_dir(fs, output_dir, job_id, error_log=None):
    """
    Подготавливает директорию с результатами неудачного запуска DataSphere job.
    
    Args:
        fs: Фейковая файловая система (pyfakefs)
        output_dir: Директория для выходных данных
        job_id: ID задания
        error_log: Строка с содержимым лога ошибок или None для стандартного лога
    """
    job_results_dir = Path(output_dir) / job_id
    fs.create_dir(job_results_dir)
    
    # Создаем лог ошибок
    if error_log is None:
        error_log = "Error: Training failed due to out of memory error\nTraceback: ...\n"
    
    error_log_path = job_results_dir / "error.log"
    fs.create_file(error_log_path, contents=error_log)
    
    return job_results_dir

def prepare_incomplete_results_dir(fs, output_dir, job_id, include_metrics=False, include_model=False):
    """
    Подготавливает директорию с неполными результатами запуска DataSphere job.
    
    Args:
        fs: Фейковая файловая система (pyfakefs)
        output_dir: Директория для выходных данных
        job_id: ID задания
        include_metrics: Включить файл метрик
        include_model: Включить файл модели
    """
    job_results_dir = Path(output_dir) / job_id
    fs.create_dir(job_results_dir)
    
    # Создаем некоторые файлы по запросу
    if include_metrics:
        metrics_data = {"val_MIC": 0.65, "train_loss": 0.22, "val_loss": 0.25}
        metrics_path = job_results_dir / "metrics.json"
        fs.create_file(metrics_path, contents=json.dumps(metrics_data))
    
    if include_model:
        model_path = job_results_dir / "model.onnx"
        fs.create_file(model_path, contents="dummy incomplete model content")
    
    # Добавляем файл лога, но не ошибки
    log_path = job_results_dir / "training.log"
    fs.create_file(log_path, contents="Training started...\nEpoch 1/10 completed\n...")
    
    return job_results_dir

def prepare_config_files(fs):
    """
    Создает необходимые конфигурационные файлы в файковой файловой системе.
    
    Args:
        fs: Объект файковой файловой системы
    """
    # Создаем директорию configs/datasphere
    configs_dir = Path("configs/datasphere")
    
    # Проверяем существование директории перед созданием
    if not fs.exists(configs_dir):
        fs.create_dir(configs_dir)
    
    # Создаем конфигурационный файл для задания
    job_config_path = configs_dir / "job_config_test.yaml"
    job_config_content = """
    apiVersion: datasphere.yandex-cloud/v1
    kind: TrainingJob
    metadata:
      name: plastinka-sales-predictor-job
    spec:
      inputs:
        - name: ds_input_dir
          resourceUri: dir://test-dir
      resources:
        cpu: 4
        memory: 16Gi
        accelerator:
          type: gpu-b100
          count: 1
      outputs:
        - name: ds_output_dir
          resourceUri: dir://test-output-dir
    """
    fs.create_file(job_config_path, contents=job_config_content)
    
    # Создаем конфигурационный файл для статического ресурса
    static_config_path = configs_dir / "static_resource_config.yaml"
    static_config_content = """
    apiVersion: datasphere.yandex-cloud/v1
    kind: StaticResource
    metadata:
      name: plastinka-sales-predictor-resources
    spec:
      uri: dir://test-static-dir
    """
    fs.create_file(static_config_path, contents=static_config_content)
    
    return {
        "job_config_path": str(job_config_path),
        "static_config_path": str(static_config_path)
    } 