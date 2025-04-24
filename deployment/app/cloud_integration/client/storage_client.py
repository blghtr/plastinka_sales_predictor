"""
Client for interacting with Yandex Cloud Storage.
Handles uploading, downloading, and managing cloud storage objects.
"""
import os
import uuid
import json
import logging
from typing import Dict, Any, Optional, BinaryIO, List, Tuple, Union
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError

from deployment.app.cloud_integration.config.cloud_config import cloud_settings
from deployment.app.db.database import get_db_connection
from app.utils.retry import retry_cloud_operation, is_retryable_cloud_error, RetryContext, RetryableError

logger = logging.getLogger(__name__)


class CloudStorageClient:
    """Client for interacting with Yandex Cloud Storage using boto3."""
    
    def __init__(self):
        """Initialize S3 client for Yandex Cloud Storage."""
        self.bucket_name = cloud_settings.storage.bucket_name
        self.s3_client = boto3.client(
            's3',
            endpoint_url=cloud_settings.storage.endpoint_url,
            aws_access_key_id=cloud_settings.storage.access_key,
            aws_secret_access_key=cloud_settings.storage.secret_key,
            region_name=cloud_settings.storage.region
        )
    
    def upload_file(self, file_path: str, object_key: Optional[str] = None,
                    metadata: Optional[Dict[str, str]] = None,
                    job_id: Optional[str] = None,
                    object_type: str = 'dataset') -> str:
        """
        Upload a file to cloud storage.
        
        Args:
            file_path: Local path to file
            object_key: Key for the object in cloud storage (generated if None)
            metadata: Optional metadata for the object
            job_id: Optional job ID to associate with the object
            object_type: Type of object ('dataset', 'model', 'result')
            
        Returns:
            Cloud storage object key
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if object_key is None:
            # Generate a unique object key if not provided
            filename = os.path.basename(file_path)
            prefix = self._get_prefix_for_type(object_type)
            object_key = f"{prefix}{uuid.uuid4().hex}_{filename}"
        
        # Set content type based on file extension
        content_type = self._get_content_type(file_path)
        
        # Use retry context for robust error handling
        with RetryContext(max_tries=5, base_delay=2.0, max_delay=60.0,
                         exceptions=(ClientError,), 
                         giveup_func=lambda e: not is_retryable_cloud_error(e)) as retry:
            
            while retry.attempts():
                try:
                    # Prepare upload parameters
                    extra_args = {
                        'ContentType': content_type
                    }
                    
                    if metadata:
                        extra_args['Metadata'] = metadata
                    
                    # Upload file
                    self.s3_client.upload_file(
                        file_path, 
                        self.bucket_name, 
                        object_key,
                        ExtraArgs=extra_args
                    )
                    
                    # Get file size and md5 hash
                    file_stats = os.stat(file_path)
                    file_size = file_stats.st_size
                    
                    # Store object metadata in database
                    self._store_object_metadata(
                        object_key=object_key,
                        content_type=content_type,
                        size_bytes=file_size,
                        job_id=job_id,
                        object_type=object_type
                    )
                    
                    logger.info(f"Uploaded file {file_path} to {object_key}")
                    retry.success()
                    return object_key
                    
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    error_msg = e.response.get('Error', {}).get('Message', str(e))
                    
                    # Log the error
                    logger.warning(
                        f"Error uploading file {file_path} to {object_key}: "
                        f"{error_code} - {error_msg}. Attempt {retry._attempt}"
                    )
                    
                    # Handle the error through retry mechanism
                    retry.failed(e)
        
        # If we exit the retry context without returning, it means all retries failed
        raise RuntimeError(f"Failed to upload file {file_path} after multiple attempts")
    
    def upload_data(self, data: Union[str, bytes, BinaryIO], 
                    object_key: Optional[str] = None,
                    content_type: str = 'application/octet-stream',
                    metadata: Optional[Dict[str, str]] = None,
                    job_id: Optional[str] = None,
                    object_type: str = 'result') -> str:
        """
        Upload data directly to cloud storage.
        
        Args:
            data: Data to upload (string, bytes, or file-like object)
            object_key: Key for the object in cloud storage (generated if None)
            content_type: Content type of the data
            metadata: Optional metadata for the object
            job_id: Optional job ID to associate with the object
            object_type: Type of object ('dataset', 'model', 'result')
            
        Returns:
            Cloud storage object key
        """
        if object_key is None:
            # Generate a unique object key if not provided
            prefix = self._get_prefix_for_type(object_type)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            object_key = f"{prefix}{uuid.uuid4().hex}_{timestamp}"
        
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode('utf-8')
            if content_type == 'application/octet-stream':
                content_type = 'text/plain'
        
        # Get size of data
        if isinstance(data, bytes):
            size_bytes = len(data)
        elif hasattr(data, 'seek') and hasattr(data, 'tell'):
            current_pos = data.tell()
            data.seek(0, os.SEEK_END)
            size_bytes = data.tell()
            data.seek(current_pos)
        else:
            size_bytes = 0
            
        # Use retry context for robust error handling
        with RetryContext(max_tries=5, base_delay=2.0, max_delay=60.0,
                         exceptions=(ClientError,), 
                         giveup_func=lambda e: not is_retryable_cloud_error(e)) as retry:
            
            while retry.attempts():
                try:
                    # Prepare upload parameters
                    extra_args = {
                        'ContentType': content_type
                    }
                    
                    if metadata:
                        extra_args['Metadata'] = metadata
                    
                    # Upload data
                    response = self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=object_key,
                        Body=data,
                        **extra_args
                    )
                    
                    # Store object metadata in database
                    self._store_object_metadata(
                        object_key=object_key,
                        content_type=content_type,
                        size_bytes=size_bytes,
                        job_id=job_id,
                        object_type=object_type,
                        md5_hash=response.get('ETag', '').strip('"')
                    )
                    
                    logger.info(f"Uploaded data to {object_key}")
                    retry.success()
                    return object_key
                    
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    error_msg = e.response.get('Error', {}).get('Message', str(e))
                    
                    # Log the error
                    logger.warning(
                        f"Error uploading data to {object_key}: "
                        f"{error_code} - {error_msg}. Attempt {retry._attempt}"
                    )
                    
                    # Handle the error through retry mechanism
                    retry.failed(e)
        
        # If we exit the retry context without returning, it means all retries failed
        raise RuntimeError(f"Failed to upload data to {object_key} after multiple attempts")
    
    @retry_cloud_operation(max_tries=5, base_delay=2.0, max_delay=30.0)
    def download_file(self, object_key: str, file_path: str) -> None:
        """
        Download a file from cloud storage.
        
        Args:
            object_key: Cloud storage object key
            file_path: Local path to save the file
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, object_key, file_path)
            logger.info(f"Downloaded {object_key} to {file_path}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_msg = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"Object {object_key} not found in bucket {self.bucket_name}")
            
            logger.error(f"Error downloading file from cloud storage: {error_code} - {error_msg}")
            raise
        except OSError as e:
            logger.error(f"OS error while saving file to {file_path}: {str(e)}")
            raise
    
    @retry_cloud_operation(max_tries=5, base_delay=1.0, max_delay=20.0)
    def download_data(self, object_key: str) -> bytes:
        """
        Download data from cloud storage.
        
        Args:
            object_key: Cloud storage object key
            
        Returns:
            Object data as bytes
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=object_key)
            data = response['Body'].read()
            logger.info(f"Downloaded data from {object_key}, size: {len(data)} bytes")
            return data
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            if error_code == 'NoSuchKey':
                raise FileNotFoundError(f"Object {object_key} not found in bucket {self.bucket_name}")
            
            logger.error(f"Error downloading data from {object_key}: {str(e)}")
            raise
    
    @retry_cloud_operation(max_tries=3, base_delay=1.0, max_delay=10.0)
    def get_presigned_url(self, object_key: str, expiration: int = None, 
                          operation: str = 'get_object') -> str:
        """
        Generate a presigned URL for an object.
        
        Args:
            object_key: Cloud storage object key
            expiration: URL expiration time in seconds (default: 1 hour)
            operation: S3 operation ('get_object' or 'put_object')
            
        Returns:
            Presigned URL
        """
        if expiration is None:
            expiration = 3600  # Default: 1 hour
        
        try:
            url = self.s3_client.generate_presigned_url(
                ClientMethod=operation,
                Params={
                    'Bucket': self.bucket_name,
                    'Key': object_key
                },
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for {object_key}, expiration: {expiration}s")
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {object_key}: {str(e)}")
            raise
    
    @retry_cloud_operation(max_tries=3, base_delay=1.0, max_delay=10.0)
    def delete_object(self, object_key: str) -> None:
        """
        Delete an object from cloud storage.
        
        Args:
            object_key: Cloud storage object key
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            
            # Delete from database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM cloud_storage_objects WHERE object_key = ?",
                (object_key,)
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Deleted object {object_key}")
        except ClientError as e:
            logger.error(f"Error deleting object {object_key}: {str(e)}")
            raise
    
    @retry_cloud_operation(max_tries=3, base_delay=1.0, max_delay=10.0)
    def list_objects(self, prefix: str = '', max_keys: int = 1000) -> List[Dict[str, Any]]:
        """
        List objects in cloud storage with a given prefix.
        
        Args:
            prefix: Object key prefix
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object metadata
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj.get('Key'),
                    'size': obj.get('Size'),
                    'last_modified': obj.get('LastModified'),
                    'etag': obj.get('ETag', '').strip('"')
                })
            
            logger.info(f"Listed {len(objects)} objects with prefix '{prefix}'")
            return objects
        except ClientError as e:
            logger.error(f"Error listing objects with prefix '{prefix}': {str(e)}")
            raise
    
    def _get_content_type(self, file_path: str) -> str:
        """
        Get content type based on file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Content type string
        """
        content_types = {
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.txt': 'text/plain',
            '.pkl': 'application/octet-stream',
            '.pt': 'application/octet-stream',  # PyTorch model
            '.pth': 'application/octet-stream',  # PyTorch model
            '.h5': 'application/octet-stream',  # HDF5 model
            '.zip': 'application/zip',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.pdf': 'application/pdf',
            '.parquet': 'application/octet-stream',
        }
        
        _, ext = os.path.splitext(file_path.lower())
        return content_types.get(ext, 'application/octet-stream')
    
    def _get_prefix_for_type(self, object_type: str) -> str:
        """
        Get storage prefix for the object type.
        
        Args:
            object_type: Type of object ('dataset', 'model', 'result', 'temp')
            
        Returns:
            Storage prefix string
        """
        prefixes = {
            'dataset': cloud_settings.storage.dataset_prefix,
            'model': cloud_settings.storage.model_prefix,
            'result': cloud_settings.storage.result_prefix,
            'temp': cloud_settings.storage.temp_prefix
        }
        
        return prefixes.get(object_type, 'other/')
    
    def _store_object_metadata(self, object_key: str, content_type: str, 
                              size_bytes: int, job_id: Optional[str] = None,
                              object_type: str = 'dataset', 
                              md5_hash: Optional[str] = None) -> None:
        """
        Store object metadata in the database.
        
        Args:
            object_key: Cloud storage object key
            content_type: Content type of the object
            size_bytes: Size of the object in bytes
            job_id: Job ID associated with the object
            object_type: Type of object ('dataset', 'model', 'result')
            md5_hash: MD5 hash of the object
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        
        object_id = uuid.uuid4().hex
        created_at = datetime.now().isoformat()
        
        # Calculate expiration for temporary objects
        expiration_time = None
        if object_type == 'temp':
            expiration_time = (datetime.now() + timedelta(days=1)).isoformat()
        
        cursor.execute(
            """
            INSERT INTO cloud_storage_objects 
            (object_id, bucket_name, object_path, content_type, size_bytes, 
             created_at, md5_hash, related_job_id, object_type, expiration_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (object_id, self.bucket_name, object_key, content_type, size_bytes,
             created_at, md5_hash, job_id, object_type, expiration_time)
        )
        
        conn.commit()
        conn.close()


# Create a global instance of the storage client
storage_client = CloudStorageClient() 