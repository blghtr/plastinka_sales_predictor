import hashlib
import os
from pathlib import Path
from typing import List, Optional

def _get_directory_hash(path: Path, exclude_patterns: Optional[List[str]] = None, include_patterns: Optional[List[str]] = None) -> str:
    """Calculates a hash of the directory contents (names and hashes of files).
    Excludes specified patterns and includes only specified patterns.
    
    Args:
        path: The path to the directory to hash.
        exclude_patterns: Optional list of glob-style patterns to exclude from hashing.
                          If None, a default set of common ignore patterns will be used.
        include_patterns: Optional list of glob-style patterns to include for hashing.
                          If None, all non-excluded files are considered. Example: ['*.py']
    """
    hasher = hashlib.sha256()
    
    # Default ignore patterns if none are provided
    default_ignore_patterns = [
        '.venv', '.pytest_cache', '.git', '__pycache__', 'logs', '.temp', '.idea',
        '*.pyc', '*.log', '*.swp', '*.tmp', '*.bak', '*.DS_Store',
        'datasets', 'models', 'notebooks', 'ray', 'temp_uploads', '.benchmarks',
        '.kilocode', '.memory-bank', 'deployment/data', 'deployment/logs', # Common project-level ignores
        '__pycache__' # Explicitly ignore pycache at any level
    ]
    effective_exclude_patterns = exclude_patterns if exclude_patterns is not None else default_ignore_patterns

    # Ensure consistent iteration order for consistent hash
    for root, dirnames, filenames in os.walk(path, followlinks=True):
        # Filter out excluded directories early
        dirnames[:] = [d for d in dirnames if not any(p in str(Path(root) / d) for p in effective_exclude_patterns)]
        
        # Sort for consistency
        dirnames.sort()
        filenames.sort()

        for fname in filenames:
            file_path = Path(root) / fname
            
            # Check if file should be excluded
            if any(p in str(file_path) for p in effective_exclude_patterns):
                continue

            # Check if file should be included (if include_patterns are specified)
            if include_patterns and not any(file_path.match(p) for p in include_patterns):
                continue

            try:
                # Update hash with relative path to ensure changes in structure are caught
                relative_path = file_path.relative_to(path)
                hasher.update(str(relative_path).encode('utf-8'))
                
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        while chunk := f.read(4096):
                            hasher.update(chunk)
            except Exception as e:
                # Log the warning but continue hashing if possible.
                # This ensures the process doesn't halt for minor file access issues,
                # but acknowledges potential hash inconsistency if the problem persists.
                print(f"WARNING: Error hashing file {file_path}: {e}") # Using print here for direct output as logger is not available in file_utilities.py before it's imported and configured.
                
    return hasher.hexdigest() 