"""
Security and Performance Enhancements for Plotiva
This module provides additional security and performance optimizations
"""

import os
import logging
import hashlib
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SecurityManager:
    """Enhanced security manager for the application"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix='plotiva_')
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}
        
    def validate_file_upload(self, file) -> bool:
        """Validate uploaded files for security"""
        try:
            # Check file size
            if hasattr(file, 'size') and file.size > self.max_file_size:
                logger.warning(f"File too large: {file.size} bytes")
                return False
            
            # Check file extension
            if hasattr(file, 'name'):
                file_ext = Path(file.name).suffix.lower()
                if file_ext not in self.allowed_extensions:
                    logger.warning(f"Invalid file extension: {file_ext}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename
    
    def create_secure_temp_file(self, data: bytes, suffix: str = '.tmp') -> str:
        """Create a secure temporary file"""
        try:
            fd, temp_path = tempfile.mkstemp(suffix=suffix, dir=self.temp_dir)
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            
            # Set secure permissions
            os.chmod(temp_path, 0o600)
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create secure temp file: {e}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        try:
            original_memory = df.memory_usage(deep=True).sum()
            
            # Optimize numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
            
            # Optimize object columns
            for col in df.select_dtypes(include=['object']).columns:
                if df[col].nunique() / len(df) < 0.5:  # If less than 50% unique
                    df[col] = df[col].astype('category')
            
            optimized_memory = df.memory_usage(deep=True).sum()
            reduction = (original_memory - optimized_memory) / original_memory * 100
            
            logger.info(f"Memory usage reduced by {reduction:.1f}%")
            return df
            
        except Exception as e:
            logger.error(f"DataFrame optimization failed: {e}")
            return df
    
    @staticmethod
    def sample_large_dataset(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
        """Sample large datasets for better performance"""
        if len(df) > max_rows:
            logger.info(f"Sampling dataset from {len(df)} to {max_rows} rows")
            return df.sample(n=max_rows, random_state=42)
        return df

class DataValidator:
    """Enhanced data validation"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive DataFrame validation"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Basic checks
            if df.empty:
                validation_results['is_valid'] = False
                validation_results['issues'].append("DataFrame is empty")
                return validation_results
            
            # Memory check
            memory_usage = df.memory_usage(deep=True).sum()
            if memory_usage > 500 * 1024 * 1024:  # 500MB
                validation_results['warnings'].append(f"Large memory usage: {memory_usage / 1024 / 1024:.1f}MB")
            
            # Column checks
            if len(df.columns) > 1000:
                validation_results['warnings'].append(f"Many columns: {len(df.columns)}")
            
            # Data quality checks
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            if missing_pct > 50:
                validation_results['warnings'].append(f"High missing data: {missing_pct:.1f}%")
            
            # Duplicate check
            duplicates = df.duplicated().sum()
            if duplicates > len(df) * 0.1:  # More than 10% duplicates
                validation_results['warnings'].append(f"Many duplicates: {duplicates}")
            
            # Statistics
            validation_results['stats'] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': memory_usage / 1024 / 1024,
                'missing_pct': missing_pct,
                'duplicates': duplicates
            }
            
            logger.info(f"DataFrame validation completed: {validation_results['stats']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"DataFrame validation failed: {e}")
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Validation error: {str(e)}")
            return validation_results

class ConfigManager:
    """Secure configuration management"""
    
    def __init__(self):
        self.config = {
            'max_upload_size': 100 * 1024 * 1024,  # 100MB
            'max_rows_display': 1000,
            'max_columns_display': 50,
            'cache_ttl': 3600,  # 1 hour
            'log_level': 'INFO',
            'enable_debug': False,
            'secure_mode': True
        }
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration (with validation)"""
        for key, value in updates.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Config updated: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")

# Global instances
security_manager = SecurityManager()
performance_optimizer = PerformanceOptimizer()
data_validator = DataValidator()
config_manager = ConfigManager()

def cleanup_on_exit():
    """Cleanup function to be called on application exit"""
    try:
        security_manager.cleanup_temp_files()
        logger.info("Application cleanup completed")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

# Register cleanup function
import atexit
atexit.register(cleanup_on_exit)