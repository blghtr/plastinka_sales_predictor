#!/usr/bin/env python3
"""
Test script to verify YAML configuration loading.
Run this script to test if your configuration files are valid.
"""

import os
import sys
from pathlib import Path

# Add the deployment directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "deployment"))

def test_config_loading():
    """Test loading different configuration files."""
    
    # Import after adding to path
    from app.config import get_settings, AppSettings
    
    config_files = [
        "config.yaml",
        "config.development.yaml", 
        "config.production.yaml"
    ]
    
    print("Testing YAML configuration loading...\n")
    
    for config_file in config_files:
        if not os.path.exists(config_file):
            print(f"❌ {config_file} not found, skipping...")
            continue
            
        print(f"🧪 Testing {config_file}...")
        
        try:
            # Set the config file path
            os.environ["CONFIG_FILE_PATH"] = config_file
            
            # Clear the cache to force reload
            get_settings.cache_clear()
            
            # Load settings
            settings = get_settings()
            
            # Basic validation
            assert isinstance(settings, AppSettings)
            assert hasattr(settings, 'api')
            assert hasattr(settings, 'db')
            assert hasattr(settings, 'datasphere')
            assert hasattr(settings, 'data_retention')
            
            # Test computed properties
            db_url = settings.database_url
            models_dir = settings.models_dir
            
            print("  ✅ Configuration loaded successfully")
            print(f"  📊 Environment: {settings.env}")
            print(f"  🌐 API Host: {settings.api.host}:{settings.api.port}")
            print(f"  🗄️  Database: {settings.db.filename}")
            print(f"  📁 Data Root: {settings.data_root_dir}")
            print(f"  🔗 Database URL: {db_url}")
            print(f"  📦 Models Dir: {models_dir}")
            
            # Test DataSphere config
            if settings.datasphere.project_id:
                print(f"  🚀 DataSphere Project: {settings.datasphere.project_id}")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error loading {config_file}: {str(e)}")
            print()
            
        finally:
            # Clean up environment
            if "CONFIG_FILE_PATH" in os.environ:
                del os.environ["CONFIG_FILE_PATH"]
    
    print("Testing complete!")

def test_environment_variables():
    """Test environment variable override functionality."""
    
    print("\n🔧 Testing environment variable overrides...")
    
    # Import after adding to path
    from app.config import get_settings
    
    try:
        # Set some environment variables
        os.environ["API_HOST"] = "test.example.com"
        os.environ["API_PORT"] = "9000"
        os.environ["API_DEBUG"] = "true"
        os.environ["DB_FILENAME"] = "test.db"
        
        # Clear cache and load settings
        get_settings.cache_clear()
        settings = get_settings()
        
        # Check if environment variables were applied
        print(f"  🔍 Checking: API Host = {settings.api.host} (expected: test.example.com)")
        print(f"  🔍 Checking: API Port = {settings.api.port} (expected: 9000)")
        print(f"  🔍 Checking: API Debug = {settings.api.debug} (expected: True)")
        print(f"  🔍 Checking: DB Filename = {settings.db.filename} (expected: test.db)")
        
        assert settings.api.host == "test.example.com"
        assert settings.api.port == 9000
        assert settings.api.debug == True
        assert settings.db.filename == "test.db"
        
        print("  ✅ Environment variables override working correctly")
        print(f"  🌐 API Host from env: {settings.api.host}")
        print(f"  🔢 API Port from env: {settings.api.port}")
        print(f"  🐛 Debug mode from env: {settings.api.debug}")
        print(f"  🗄️  DB filename from env: {settings.db.filename}")
        
    except Exception as e:
        print(f"  ❌ Error testing environment variables: {str(e)}")
        
    finally:
        # Clean up environment variables
        for var in ["API_HOST", "API_PORT", "API_DEBUG", "DB_FILENAME"]:
            if var in os.environ:
                del os.environ[var]
        get_settings.cache_clear()

if __name__ == "__main__":
    test_config_loading()
    test_environment_variables() 