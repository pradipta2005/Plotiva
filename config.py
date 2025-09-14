"""
Configuration file for the Advanced Data Analysis Platform
"""

# Application settings
APP_CONFIG = {
    'title': 'Advanced Data Analysis Platform',
    'icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Supported file formats
SUPPORTED_FORMATS = {
    'csv': 'Comma Separated Values',
    'xlsx': 'Excel Spreadsheet',
    'xls': 'Excel Spreadsheet (Legacy)',
    'json': 'JSON Data',
    'parquet': 'Parquet Data'
}

# Plot types and their configurations
PLOT_TYPES = {
    'histogram': {
        'name': 'Histogram',
        'description': 'Distribution of a single numeric variable',
        'required_columns': ['numeric'],
        'min_columns': 1
    },
    'scatter': {
        'name': 'Scatter Plot',
        'description': 'Relationship between two numeric variables',
        'required_columns': ['numeric', 'numeric'],
        'min_columns': 2
    },
    'line': {
        'name': 'Line Plot',
        'description': 'Trend of a numeric variable over index',
        'required_columns': ['numeric'],
        'min_columns': 1
    },
    'bar': {
        'name': 'Bar Chart',
        'description': 'Comparison across categories',
        'required_columns': ['categorical', 'numeric'],
        'min_columns': 2
    },
    'box': {
        'name': 'Box Plot',
        'description': 'Distribution comparison across categories',
        'required_columns': ['categorical', 'numeric'],
        'min_columns': 2
    },
    'correlation': {
        'name': 'Correlation Matrix',
        'description': 'Correlation between numeric variables',
        'required_columns': ['numeric'],
        'min_columns': 2
    }
}

# Machine learning algorithms
ML_ALGORITHMS = {
    'linear_regression': {
        'name': 'Linear Regression',
        'type': 'regression',
        'description': 'Predict continuous values using linear relationships',
        'min_samples': 10
    },
    'clustering': {
        'name': 'K-Means Clustering',
        'type': 'unsupervised',
        'description': 'Group similar data points together',
        'min_samples': 5
    }
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'excellent': 0.9,
    'good': 0.7,
    'fair': 0.5,
    'poor': 0.3
}

# Color schemes
COLOR_SCHEMES = {
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'pastel': ['#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5'],
    'dark': ['#1f1f1f', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
    'professional': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
}

# Default settings
DEFAULT_SETTINGS = {
    'max_rows_display': 1000,
    'max_file_size_mb': 200,
    'cache_ttl_seconds': 3600,
    'plot_height': 500,
    'plot_width': 700,
    'decimal_places': 3
}

# Error messages
ERROR_MESSAGES = {
    'file_too_large': 'File size exceeds maximum limit of {max_size}MB',
    'unsupported_format': 'File format not supported. Supported formats: {formats}',
    'insufficient_data': 'Insufficient data for analysis. Need at least {min_rows} rows',
    'no_numeric_columns': 'No numeric columns found for analysis',
    'no_categorical_columns': 'No categorical columns found for analysis',
    'ml_not_available': 'Machine learning libraries not available. Please install required packages.'
}

# Success messages
SUCCESS_MESSAGES = {
    'data_loaded': 'Successfully loaded {rows} rows and {columns} columns',
    'data_processed': 'Data processing completed successfully',
    'plot_created': 'Plot created successfully',
    'analysis_complete': 'Analysis completed successfully'
}

# Feature flags
FEATURES = {
    'advanced_ml': True,
    'export_pdf': True,
    'real_time_updates': False,
    'collaborative_features': False,
    'cloud_storage': False
}