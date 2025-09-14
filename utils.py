"""
Utility functions for the Advanced Data Analysis Platform
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
import io
import base64
from config import QUALITY_THRESHOLDS, ERROR_MESSAGES, SUCCESS_MESSAGES

def validate_dataframe(df: pd.DataFrame, min_rows: int = 1) -> Tuple[bool, str]:
    """
    Validate if dataframe meets minimum requirements
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
    
    Returns:
        Tuple of (is_valid, message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"Insufficient data. Need at least {min_rows} rows, got {len(df)}"
    
    return True, "DataFrame is valid"

def get_column_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about DataFrame columns
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with column information
    """
    if df.empty:
        return {}
    
    info = {
        'total_columns': len(df.columns),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'boolean_columns': df.select_dtypes(include=['bool']).columns.tolist(),
        'column_types': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns}
    }
    
    return info

def calculate_data_quality_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive data quality metrics
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with quality metrics
    """
    if df.empty:
        return {'overall_score': 0.0, 'completeness': 0.0, 'consistency': 0.0, 'validity': 0.0}
    
    # Completeness: percentage of non-null values
    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = total_cells - df.isnull().sum().sum()
    completeness = non_null_cells / total_cells if total_cells > 0 else 0
    
    # Consistency: percentage of non-duplicate rows
    total_rows = len(df)
    unique_rows = len(df.drop_duplicates())
    consistency = unique_rows / total_rows if total_rows > 0 else 0
    
    # Validity: check for reasonable data ranges and types
    validity_score = 1.0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        try:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                validity_score *= (1 - inf_count / len(df))
            
            # Check for extreme outliers (beyond 4 standard deviations)
            if len(df[col].dropna()) > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-8))
                extreme_outliers = (z_scores > 4).sum()
                if extreme_outliers > 0:
                    validity_score *= (1 - extreme_outliers / len(df))
        except:
            pass
    
    # Overall score (weighted average)
    overall_score = (completeness * 0.4 + consistency * 0.3 + validity_score * 0.3)
    
    return {
        'overall_score': overall_score,
        'completeness': completeness,
        'consistency': consistency,
        'validity': validity_score
    }

def get_quality_label(score: float) -> str:
    """
    Get quality label based on score
    
    Args:
        score: Quality score between 0 and 1
    
    Returns:
        Quality label string
    """
    if score >= QUALITY_THRESHOLDS['excellent']:
        return "Excellent"
    elif score >= QUALITY_THRESHOLDS['good']:
        return "Good"
    elif score >= QUALITY_THRESHOLDS['fair']:
        return "Fair"
    else:
        return "Poor"

def safe_convert_numeric(series: pd.Series) -> pd.Series:
    """
    Safely convert series to numeric, handling errors gracefully
    
    Args:
        series: Pandas series to convert
    
    Returns:
        Converted series
    """
    try:
        return pd.to_numeric(series, errors='coerce')
    except:
        return series

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Detect outliers in a numeric column
    
    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        method: Method to use ('iqr', 'zscore', 'isolation')
    
    Returns:
        DataFrame containing outlier rows
    """
    if column not in df.columns or df[column].dtype not in [np.number]:
        return pd.DataFrame()
    
    try:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / (df[column].std() + 1e-8))
            outliers = df[z_scores > 3]
        
        else:
            return pd.DataFrame()
        
        return outliers
    
    except Exception:
        return pd.DataFrame()

def generate_sample_data(data_type: str = 'sales') -> pd.DataFrame:
    """
    Generate sample data for demonstration
    
    Args:
        data_type: Type of sample data to generate
    
    Returns:
        Sample DataFrame
    """
    np.random.seed(42)
    
    if data_type == 'sales':
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        data = {
            'Date': dates,
            'Sales': np.random.normal(1000, 200, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 100,
            'Customers': np.random.poisson(50, 365),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 365),
            'Product': np.random.choice(['Product A', 'Product B', 'Product C'], 365),
            'Revenue': np.random.normal(5000, 1000, 365),
            'Profit_Margin': np.random.uniform(0.1, 0.3, 365)
        }
    
    elif data_type == 'customer':
        data = {
            'Customer_ID': range(1, 1001),
            'Age': np.random.normal(35, 12, 1000).astype(int),
            'Income': np.random.normal(50000, 15000, 1000),
            'Spending_Score': np.random.randint(1, 101, 1000),
            'Gender': np.random.choice(['Male', 'Female'], 1000),
            'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 1000),
            'Satisfaction': np.random.randint(1, 6, 1000)
        }
    
    elif data_type == 'financial':
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        price = 100
        prices = [price]
        
        for _ in range(999):
            change = np.random.normal(0, 2)
            price = max(price + change, 1)  # Ensure price doesn't go negative
            prices.append(price)
        
        data = {
            'Date': dates,
            'Price': prices,
            'Volume': np.random.lognormal(10, 1, 1000).astype(int),
            'High': [p + np.random.uniform(0, 5) for p in prices],
            'Low': [p - np.random.uniform(0, 5) for p in prices],
            'Sector': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Energy'], 1000)
        }
    
    else:  # Default to simple numeric data
        data = {
            'X': np.random.normal(0, 1, 100),
            'Y': np.random.normal(0, 1, 100),
            'Z': np.random.normal(0, 1, 100),
            'Category': np.random.choice(['A', 'B', 'C'], 100)
        }
    
    return pd.DataFrame(data)

def format_number(value: float, decimal_places: int = 2) -> str:
    """
    Format number for display
    
    Args:
        value: Number to format
        decimal_places: Number of decimal places
    
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1e6:
        return f"{value/1e6:.{decimal_places}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimal_places}f}K"
    else:
        return f"{value:.{decimal_places}f}"

def create_download_link(df: pd.DataFrame, filename: str, file_format: str = 'csv') -> str:
    """
    Create a download link for DataFrame
    
    Args:
        df: DataFrame to download
        filename: Name of the file
        file_format: Format of the file ('csv', 'excel')
    
    Returns:
        Download link HTML
    """
    if file_format == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    
    elif file_format == 'excel':
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
        finally:
            output.close()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel</a>'
    
    else:
        return "Unsupported format"
    
    return href

def log_user_action(action: str, details: Dict[str, Any] = None):
    """
    Log user actions for analytics
    
    Args:
        action: Action performed
        details: Additional details about the action
    """
    if 'user_actions' not in st.session_state:
        st.session_state.user_actions = []
    
    log_entry = {
        'timestamp': pd.Timestamp.now(),
        'action': action,
        'details': details or {}
    }
    
    st.session_state.user_actions.append(log_entry)
    
    # Keep only last 100 actions to prevent memory issues
    if len(st.session_state.user_actions) > 100:
        st.session_state.user_actions = st.session_state.user_actions[-100:]

def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get memory usage information for DataFrame
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with memory usage info
    """
    if df.empty:
        return {'total': '0 B', 'per_column': {}}
    
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} TB"
    
    return {
        'total': format_bytes(total_memory),
        'per_column': {col: format_bytes(usage) for col, usage in memory_usage.items()}
    }

def suggest_plot_types(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Suggest appropriate plot types based on data characteristics
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        List of plot suggestions with reasons
    """
    suggestions = []
    
    if df.empty:
        return suggestions
    
    col_info = get_column_info(df)
    numeric_cols = col_info['numeric_columns']
    categorical_cols = col_info['categorical_columns']
    datetime_cols = col_info['datetime_columns']
    
    # Single numeric column
    if len(numeric_cols) == 1:
        suggestions.append({
            'type': 'histogram',
            'reason': f'Single numeric column ({numeric_cols[0]}) - histogram shows distribution'
        })
    
    # Two numeric columns
    if len(numeric_cols) >= 2:
        suggestions.append({
            'type': 'scatter',
            'reason': 'Multiple numeric columns - scatter plot shows relationships'
        })
        suggestions.append({
            'type': 'correlation',
            'reason': 'Multiple numeric columns - correlation matrix shows all relationships'
        })
    
    # Categorical + numeric
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        suggestions.append({
            'type': 'bar',
            'reason': 'Categorical and numeric columns - bar chart compares values across categories'
        })
        suggestions.append({
            'type': 'box',
            'reason': 'Categorical and numeric columns - box plot shows distribution by category'
        })
    
    # Time series data
    if len(datetime_cols) >= 1 and len(numeric_cols) >= 1:
        suggestions.append({
            'type': 'line',
            'reason': 'DateTime column detected - line plot shows trends over time'
        })
    
    return suggestions

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names to be more user-friendly
    
    Args:
        df: DataFrame with potentially messy column names
    
    Returns:
        DataFrame with cleaned column names
    """
    df_clean = df.copy()
    
    # Clean column names
    df_clean.columns = (
        df_clean.columns
        .str.strip()  # Remove leading/trailing whitespace
        .str.replace(r'[^\w\s]', '_', regex=True)  # Replace special chars with underscore
        .str.replace(r'\s+', '_', regex=True)  # Replace spaces with underscore
        .str.replace(r'_+', '_', regex=True)  # Replace multiple underscores with single
        .str.strip('_')  # Remove leading/trailing underscores
        .str.title()  # Title case
    )
    
    return df_clean

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive data summary
    
    Args:
        df: DataFrame to summarize
    
    Returns:
        Dictionary with summary information
    """
    if df.empty:
        return {}
    
    col_info = get_column_info(df)
    quality_metrics = calculate_data_quality_metrics(df)
    memory_info = get_memory_usage(df)
    
    summary = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': memory_info['total'],
            'file_size_estimate': memory_info['total']
        },
        'column_info': col_info,
        'quality_metrics': quality_metrics,
        'data_types': {
            'numeric': len(col_info['numeric_columns']),
            'categorical': len(col_info['categorical_columns']),
            'datetime': len(col_info['datetime_columns']),
            'boolean': len(col_info['boolean_columns'])
        },
        'missing_data': {
            'total_missing': df.isnull().sum().sum(),
            'columns_with_missing': len([col for col in df.columns if df[col].isnull().any()]),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        },
        'duplicates': {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
        }
    }
    
    return summary