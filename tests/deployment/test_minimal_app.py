"""
Minimal test file that imports modules from the app directory to check if that causes hanging.
"""

import sys
print(f"Python version: {sys.version}")
print(f"Current working directory: {sys.path[0]}")

def test_minimal_with_app_imports():
    """A minimal test that imports various app modules."""
    print("Running minimal test with app imports")
    
    # Import app modules one by one to isolate the import effect
    print("Importing config...")
    from deployment.app import config
    print(f"Successfully imported config: {config}")
    
    print("Importing db...")
    from deployment.app import db
    print(f"Successfully imported db: {db}")
    
    print("Importing models...")
    from deployment.app import models
    print(f"Successfully imported models: {models}")
    
    print("Importing services...")
    from deployment.app import services
    print(f"Successfully imported services: {services}")
    
    print("Importing api...")
    from deployment.app import api
    print(f"Successfully imported api: {api}")
    
    assert True 