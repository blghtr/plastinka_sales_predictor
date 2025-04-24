"""
Performance benchmarks for the API and cloud functions.
Measures response times, throughput, and resource utilization.
"""
import os
import time
import json
import statistics
import datetime
import asyncio
import aiohttp
import pytest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock
from pathlib import Path

from deployment.app.cloud_integration.client.function_client import CloudFunctionClient
from deployment.app.cloud_integration.client.storage_client import CloudStorageClient
from deployment.app.config import settings


class TestPerformance:
    """Performance benchmarks for the API and cloud functions."""

    @pytest.fixture
    def mock_env_variables(self, monkeypatch):
        """Set up environment variables for tests."""
        monkeypatch.setenv("YANDEX_CLOUD_ACCESS_KEY", "test_access_key")
        monkeypatch.setenv("YANDEX_CLOUD_SECRET_KEY", "test_secret_key")
        monkeypatch.setenv("YANDEX_CLOUD_API_KEY", "test_api_key")
        monkeypatch.setenv("YANDEX_CLOUD_FOLDER_ID", "test_folder_id")
        monkeypatch.setenv("FASTAPI_CLOUD_CALLBACK_AUTH_TOKEN", "test_callback_token")
        monkeypatch.setenv("YANDEX_CLOUD_STORAGE_ENDPOINT", "https://test-storage.example.com")
        monkeypatch.setenv("DATABASE_PATH", "deployment/data/test_db.sqlite")
        monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000,https://app.example.com")
    
    @pytest.fixture
    def output_dir(self):
        """Get output directory for benchmark results."""
        output_dir = os.environ.get("BENCHMARK_OUTPUT_DIR", "benchmark_results")
        os.makedirs(output_dir, exist_ok=True)
        return Path(output_dir)
        
    @pytest.fixture
    def report_name(self):
        """Get report name for benchmark results."""
        return os.environ.get("BENCHMARK_REPORT_NAME", f"benchmark_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    @pytest.fixture
    def server_url(self):
        """Get server URL from environment."""
        return os.environ.get("BENCHMARK_SERVER_URL", "http://localhost:8000")
    
    async def benchmark_api_endpoint(self, endpoint, server_url, method="GET", data=None, headers=None, 
                                    num_requests=100, concurrent_requests=10):
        """
        Benchmark an API endpoint by sending multiple requests concurrently.
        
        Args:
            endpoint: API endpoint to benchmark
            server_url: Base URL of the server
            method: HTTP method to use
            data: Request payload
            headers: HTTP headers
            num_requests: Total number of requests to send
            concurrent_requests: Number of concurrent requests
            
        Returns:
            Dictionary with benchmark results
        """
        url = f"{server_url}{endpoint}"
        
        # Default headers
        if headers is None:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
        # Results storage
        response_times = []
        status_codes = []
        errors = []
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request():
            async with semaphore:
                try:
                    start_time = time.time()
                    async with aiohttp.ClientSession() as session:
                        if method.upper() == "GET":
                            async with session.get(url, headers=headers) as response:
                                await response.text()
                                elapsed = time.time() - start_time
                                response_times.append(elapsed)
                                status_codes.append(response.status)
                        elif method.upper() == "POST":
                            async with session.post(url, headers=headers, json=data) as response:
                                await response.text()
                                elapsed = time.time() - start_time
                                response_times.append(elapsed)
                                status_codes.append(response.status)
                except Exception as e:
                    errors.append(str(e))
        
        # Create and run tasks
        tasks = [make_request() for _ in range(num_requests)]
        await asyncio.gather(*tasks)
        
        # Calculate statistics
        if response_times:
            results = {
                "endpoint": endpoint,
                "method": method,
                "total_requests": num_requests,
                "successful_requests": len(response_times),
                "failed_requests": len(errors),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "p95_response_time": np.percentile(response_times, 95),
                "p99_response_time": np.percentile(response_times, 99),
                "throughput": len(response_times) / sum(response_times),
                "status_code_counts": {code: status_codes.count(code) for code in set(status_codes)},
                "errors": errors[:10] if errors else []  # Show first 10 errors only
            }
        else:
            results = {
                "endpoint": endpoint,
                "method": method,
                "total_requests": num_requests,
                "successful_requests": 0,
                "failed_requests": len(errors),
                "errors": errors[:10]
            }
            
        return results
    
    def benchmark_cloud_function(self, function_name, params, input_data, num_requests=10):
        """
        Benchmark a cloud function by measuring execution times.
        
        Args:
            function_name: Name of the cloud function to benchmark
            params: Function parameters
            input_data: Input data for the function
            num_requests: Number of requests to make
            
        Returns:
            Dictionary with benchmark results
        """
        function_client = CloudFunctionClient()
        response_times = []
        results = []
        errors = []
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                result = function_client.invoke_function(
                    function_name=function_name,
                    job_id=f"benchmark_test_{i}",
                    params=params,
                    input_data=input_data
                )
                elapsed = time.time() - start_time
                response_times.append(elapsed)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        if response_times:
            return {
                "function_name": function_name,
                "total_requests": num_requests,
                "successful_requests": len(response_times),
                "failed_requests": len(errors),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "avg_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "errors": errors[:10] if errors else []
            }
        else:
            return {
                "function_name": function_name,
                "total_requests": num_requests,
                "successful_requests": 0,
                "failed_requests": len(errors),
                "errors": errors[:10]
            }
    
    @patch('deployment.app.cloud_integration.client.function_client.CloudFunctionClient.invoke_function')
    @patch('deployment.app.cloud_integration.client.storage_client.CloudStorageClient.upload_file')
    @pytest.mark.asyncio
    async def test_api_endpoints_performance(self, mock_upload, mock_invoke, mock_env_variables, 
                                            output_dir, report_name, server_url):
        """Test the performance of key API endpoints."""
        # Mock the cloud function response
        mock_invoke.return_value = {
            "job_id": "test_job_id",
            "status": "success",
            "result": {"message": "Operation completed successfully"}
        }
        
        # Mock the storage upload
        mock_upload.return_value = "test_upload_path"
        
        # Test endpoints to benchmark
        endpoints = [
            {
                "endpoint": "/api/health",
                "method": "GET",
                "data": None,
                "num_requests": 500,
                "concurrent_requests": 50
            },
            {
                "endpoint": "/api/predictions",
                "method": "POST",
                "data": {
                    "model_id": "test_model",
                    "start_date": "2023-01-01",
                    "end_date": "2023-03-01",
                    "items": ["item1", "item2", "item3"]
                },
                "num_requests": 100,
                "concurrent_requests": 10
            },
            {
                "endpoint": "/api/training/models",
                "method": "POST",
                "data": {
                    "model_name": "test_benchmark_model",
                    "training_params": {
                        "epochs": 10,
                        "batch_size": 64,
                        "learning_rate": 0.001
                    },
                    "dataset_id": "test_dataset"
                },
                "num_requests": 50,
                "concurrent_requests": 5
            }
        ]
        
        # Run benchmarks
        benchmark_results = []
        for endpoint_config in endpoints:
            result = await self.benchmark_api_endpoint(
                endpoint=endpoint_config["endpoint"],
                server_url=server_url,
                method=endpoint_config["method"],
                data=endpoint_config["data"],
                num_requests=endpoint_config["num_requests"],
                concurrent_requests=endpoint_config["concurrent_requests"]
            )
            benchmark_results.append(result)
            print(f"\nBenchmark results for {endpoint_config['method']} {endpoint_config['endpoint']}:")
            print(f"  Average response time: {result['avg_response_time']:.4f} seconds")
            print(f"  95th percentile: {result['p95_response_time']:.4f} seconds")
            print(f"  Throughput: {result['throughput']:.2f} requests/second")
            
        # Generate performance report
        self.generate_performance_report(
            benchmark_results, 
            output_path=os.path.join(output_dir, f"{report_name}_api.html")
        )
        
        # Save results to JSON file
        self.save_results_json(
            benchmark_results,
            output_path=os.path.join(output_dir, f"{report_name}_api.json")
        )
        
        # Generate performance charts
        self._generate_performance_charts(
            pd.DataFrame([
                {
                    "Endpoint": result["endpoint"],
                    "Method": result["method"],
                    "Avg Response Time": result["avg_response_time"],
                    "Median Response Time": result.get("median_response_time", 0),
                    "95th Percentile": result.get("p95_response_time", 0),
                    "Throughput": result.get("throughput", 0)
                }
                for result in benchmark_results
            ]),
            output_path=os.path.join(output_dir, f"{report_name}_api.png")
        )
        
        # Verify that performance meets requirements
        for result in benchmark_results:
            if result["endpoint"] == "/api/health":
                assert result["avg_response_time"] < 0.05, "Health endpoint response time too slow"
            elif "predictions" in result["endpoint"]:
                assert result["avg_response_time"] < 0.5, "Predictions endpoint response time too slow"
            elif "training" in result["endpoint"]:
                assert result["avg_response_time"] < 1.0, "Training endpoint response time too slow"
    
    @patch('deployment.app.cloud_integration.client.function_client.CloudFunctionClient.invoke_function')
    def test_cloud_function_performance(self, mock_invoke, mock_env_variables, output_dir, report_name):
        """Test the performance of cloud functions."""
        # Configure the mock to return different results based on the function name
        def mock_invoke_function(function_name, job_id, params, input_data):
            time.sleep(0.1)  # Simulate network latency
            return {
                "job_id": job_id,
                "status": "success",
                "result": {"message": f"{function_name} completed successfully"}
            }
            
        mock_invoke.side_effect = mock_invoke_function
        
        # Functions to benchmark
        functions = [
            {
                "name": "training",
                "params": {
                    "model_type": "lstm",
                    "epochs": 10,
                    "batch_size": 64
                },
                "input_data": "test_input_data",
                "num_requests": 20
            },
            {
                "name": "prediction",
                "params": {
                    "model_id": "test_model",
                    "prediction_horizon": 30
                },
                "input_data": "test_input_data",
                "num_requests": 30
            }
        ]
        
        # Run benchmarks
        benchmark_results = []
        for function_config in functions:
            result = self.benchmark_cloud_function(
                function_name=function_config["name"],
                params=function_config["params"],
                input_data=function_config["input_data"],
                num_requests=function_config["num_requests"]
            )
            benchmark_results.append(result)
            print(f"\nBenchmark results for {function_config['name']} function:")
            print(f"  Average response time: {result['avg_response_time']:.4f} seconds")
            print(f"  Median response time: {result['median_response_time']:.4f} seconds")
        
        # Generate performance report
        self.generate_cloud_performance_report(
            benchmark_results,
            output_path=os.path.join(output_dir, f"{report_name}_cloud.html")
        )
        
        # Save results to JSON file
        self.save_results_json(
            benchmark_results,
            output_path=os.path.join(output_dir, f"{report_name}_cloud.json")
        )
        
        # Generate performance charts
        self._generate_cloud_performance_charts(
            pd.DataFrame([
                {
                    "Function": result["function_name"],
                    "Avg Response Time": result["avg_response_time"],
                    "Median Response Time": result.get("median_response_time", 0),
                    "Min Response Time": result.get("min_response_time", 0),
                    "Max Response Time": result.get("max_response_time", 0)
                }
                for result in benchmark_results
            ]),
            output_path=os.path.join(output_dir, f"{report_name}_cloud.png")
        )
        
        # Verify that performance meets requirements
        for result in benchmark_results:
            assert result["avg_response_time"] < 5.0, f"{result['function_name']} function response time too slow"
    
    def save_results_json(self, benchmark_results, output_path):
        """
        Save benchmark results to a JSON file.
        
        Args:
            benchmark_results: Benchmark results to save
            output_path: Path to save the JSON file
        """
        # Convert non-serializable objects to strings
        serializable_results = []
        for result in benchmark_results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                    serializable_result[key] = value
                else:
                    serializable_result[key] = str(value)
            serializable_results.append(serializable_result)
            
        # Save to file
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to {output_path}")
    
    def generate_performance_report(self, benchmark_results, output_path="benchmark_report.html"):
        """
        Generate a performance report with visualizations.
        
        Args:
            benchmark_results: Benchmark results to include in the report
            output_path: Path to save the report
        """
        # Create dataframe for visualization
        df = pd.DataFrame([
            {
                "Endpoint": result["endpoint"],
                "Method": result["method"],
                "Avg Response Time": result["avg_response_time"],
                "Median Response Time": result.get("median_response_time", 0),
                "95th Percentile": result.get("p95_response_time", 0),
                "99th Percentile": result.get("p99_response_time", 0),
                "Throughput": result.get("throughput", 0),
                "Success Rate": result["successful_requests"] / result["total_requests"] * 100
            }
            for result in benchmark_results
        ])
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>API Performance Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .danger {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>API Performance Benchmark Report</h1>
            <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Endpoint Performance Summary</h2>
            <table>
                <tr>
                    <th>Endpoint</th>
                    <th>Method</th>
                    <th>Avg Response Time (s)</th>
                    <th>Median Response Time (s)</th>
                    <th>95th Percentile (s)</th>
                    <th>Throughput (req/s)</th>
                    <th>Success Rate (%)</th>
                </tr>
                {"".join([f"<tr><td>{row['Endpoint']}</td><td>{row['Method']}</td><td>{row['Avg Response Time']:.4f}</td><td>{row['Median Response Time']:.4f}</td><td>{row['95th Percentile']:.4f}</td><td>{row['Throughput']:.2f}</td><td>{row['Success Rate']:.1f}</td></tr>" for _, row in df.iterrows()])}
            </table>
            
            <h2>Performance Insights</h2>
            <ul>
                <li>Fastest endpoint: {df.loc[df['Avg Response Time'].idxmin()]['Endpoint']} ({df['Avg Response Time'].min():.4f}s)</li>
                <li>Slowest endpoint: {df.loc[df['Avg Response Time'].idxmax()]['Endpoint']} ({df['Avg Response Time'].max():.4f}s)</li>
                <li>Highest throughput: {df.loc[df['Throughput'].idxmax()]['Endpoint']} ({df['Throughput'].max():.2f} req/s)</li>
            </ul>
            
            <h2>Recommendations</h2>
            <ul>
                {"".join([f"<li>{self._get_recommendation(row)}</li>" for _, row in df.iterrows()])}
            </ul>
        </body>
        </html>
        """
        
        # Save report to file
        with open(output_path, "w") as f:
            f.write(html_content)
            
        print(f"Performance report generated: {output_path}")
        
        return df
    
    def generate_cloud_performance_report(self, benchmark_results, output_path="cloud_benchmark_report.html"):
        """
        Generate a performance report for cloud functions.
        
        Args:
            benchmark_results: Benchmark results to include in the report
            output_path: Path to save the report
        """
        # Create dataframe for visualization
        df = pd.DataFrame([
            {
                "Function": result["function_name"],
                "Avg Response Time": result["avg_response_time"],
                "Median Response Time": result.get("median_response_time", 0),
                "Min Response Time": result.get("min_response_time", 0),
                "Max Response Time": result.get("max_response_time", 0),
                "Success Rate": result["successful_requests"] / result["total_requests"] * 100
            }
            for result in benchmark_results
        ])
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Cloud Function Performance Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .danger {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Cloud Function Performance Benchmark Report</h1>
            <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Function Performance Summary</h2>
            <table>
                <tr>
                    <th>Function</th>
                    <th>Avg Response Time (s)</th>
                    <th>Median Response Time (s)</th>
                    <th>Min Response Time (s)</th>
                    <th>Max Response Time (s)</th>
                    <th>Success Rate (%)</th>
                </tr>
                {"".join([f"<tr><td>{row['Function']}</td><td>{row['Avg Response Time']:.4f}</td><td>{row['Median Response Time']:.4f}</td><td>{row['Min Response Time']:.4f}</td><td>{row['Max Response Time']:.4f}</td><td>{row['Success Rate']:.1f}</td></tr>" for _, row in df.iterrows()])}
            </table>
            
            <h2>Performance Insights</h2>
            <ul>
                <li>Fastest function: {df.loc[df['Avg Response Time'].idxmin()]['Function']} ({df['Avg Response Time'].min():.4f}s)</li>
                <li>Slowest function: {df.loc[df['Avg Response Time'].idxmax()]['Function']} ({df['Avg Response Time'].max():.4f}s)</li>
            </ul>
            
            <h2>Recommendations</h2>
            <ul>
                {"".join([f"<li>{self._get_cloud_recommendation(row)}</li>" for _, row in df.iterrows()])}
            </ul>
        </body>
        </html>
        """
        
        # Save report to file
        with open(output_path, "w") as f:
            f.write(html_content)
            
        print(f"Cloud performance report generated: {output_path}")
        
        return df
    
    def _get_recommendation(self, row):
        """Generate performance recommendation for an endpoint."""
        if row["Avg Response Time"] < 0.1:
            return f"<span class='success'>{row['Endpoint']}: Performance is excellent, no action needed.</span>"
        elif row["Avg Response Time"] < 0.5:
            return f"<span class='success'>{row['Endpoint']}: Performance is good, monitor for changes.</span>"
        elif row["Avg Response Time"] < 1.0:
            return f"<span class='warning'>{row['Endpoint']}: Performance could be improved, consider optimization.</span>"
        else:
            return f"<span class='danger'>{row['Endpoint']}: Performance is poor, immediate optimization required.</span>"
    
    def _get_cloud_recommendation(self, row):
        """Generate performance recommendation for a cloud function."""
        if row["Avg Response Time"] < 1.0:
            return f"<span class='success'>{row['Function']}: Performance is excellent, no action needed.</span>"
        elif row["Avg Response Time"] < 3.0:
            return f"<span class='success'>{row['Function']}: Performance is good, monitor for changes.</span>"
        elif row["Avg Response Time"] < 5.0:
            return f"<span class='warning'>{row['Function']}: Performance could be improved, consider optimization.</span>"
        else:
            return f"<span class='danger'>{row['Function']}: Performance is poor, immediate optimization required.</span>"
    
    def _generate_performance_charts(self, df, output_path):
        """Generate performance visualization charts for API endpoints."""
        plt.figure(figsize=(12, 8))
        
        # Response time comparison
        plt.subplot(2, 1, 1)
        df.plot(x="Endpoint", y=["Avg Response Time", "Median Response Time", "95th Percentile"], 
                kind="bar", ax=plt.gca())
        plt.title("Response Time Comparison")
        plt.ylabel("Time (seconds)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Throughput comparison
        plt.subplot(2, 1, 2)
        df.plot(x="Endpoint", y="Throughput", kind="bar", ax=plt.gca(), color="green")
        plt.title("Throughput Comparison")
        plt.ylabel("Requests per Second")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Performance chart generated: {output_path}")
    
    def _generate_cloud_performance_charts(self, df, output_path):
        """Generate performance visualization charts for cloud functions."""
        plt.figure(figsize=(12, 6))
        
        # Response time comparison
        df.plot(x="Function", y=["Avg Response Time", "Median Response Time", "Min Response Time", "Max Response Time"], 
                kind="bar", ax=plt.gca())
        plt.title("Cloud Function Response Time Comparison")
        plt.ylabel("Time (seconds)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"Cloud performance chart generated: {output_path}") 