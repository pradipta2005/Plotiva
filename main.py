import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import warnings
import time
import datetime
warnings.filterwarnings('ignore')

# Import premium modules
try:
    from premium_config import PREMIUM_COLOR_PALETTES, PREMIUM_THEMES, PREMIUM_CSS
    from premium_plots import PremiumPlotGenerator
    from premium_analytics import create_analytics_dashboard
    from premium_visualization_tab import render_premium_visualization_tab
    from dashboard_tab import render_dashboard_tab
    from utils import get_column_info, calculate_data_quality_metrics, generate_sample_data, get_data_summary
    PREMIUM_AVAILABLE = True
except ImportError:
    PREMIUM_AVAILABLE = False
    
    def render_premium_visualization_tab(df):
        st.header("📈 Basic Visualizations")
        st.info("Premium visualization features not available")
    
    def render_dashboard_tab():
        st.header("📋 Dashboard")
        st.info("Dashboard features not available")

# ML imports with error handling
try:
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
    from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier
    from sklearn.metrics import (
        r2_score, mean_squared_error, mean_absolute_error, silhouette_score,
        accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
    )
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.decomposition import PCA
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    try:
        from xgboost import XGBRegressor, XGBClassifier
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    XGBOOST_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Premium Data Analysis Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium CSS if available
if PREMIUM_AVAILABLE:
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    defaults = {
        'uploaded_file_name': None,
        'original_data': pd.DataFrame(),
        'working_data': pd.DataFrame(),
        'selected_theme': 'executive' if PREMIUM_AVAILABLE else 'default',
        'selected_palette': 'aurora' if PREMIUM_AVAILABLE else 'default',
        'plot_history': [],
        'analytics_results': {},
        'data_quality_score': 0.0,
        'gallery_plots': {},
        'ml_results': {},
        'filter_values': {},
        'report_elements': [],
        'report_content': {},
        'dashboard_selections': [],
        'performance_log': [],
        'dashboard_charts': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Helper functions
def get_working_data():
    return st.session_state.get('working_data', pd.DataFrame())

def update_working_data(df):
    # Only copy when necessary to preserve original data integrity
    st.session_state.working_data = df.copy() if 'original_data' in st.session_state else df

def get_column_types(df):
    if df.empty:
        return [], [], []
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return all_columns, numeric_columns, categorical_columns

def get_premium_colors(palette='aurora'):
    if PREMIUM_AVAILABLE and palette in PREMIUM_COLOR_PALETTES:
        return PREMIUM_COLOR_PALETTES[palette]
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# AI Insights Engine
def generate_ai_insights(df):
    # Always use current working data (filtered data)
    current_df = st.session_state.get('working_data', df)
    if current_df.empty:
        return ["No data available for analysis"]
    df = current_df
    
    insights = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    insights.append(f"📊 Dataset contains {len(df):,} rows and {len(df.columns)} columns")
    
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 10:
        insights.append(f"⚠️ High missing data detected ({missing_pct:.1f}%) - consider data cleaning")
    elif missing_pct > 0:
        insights.append(f"✅ Low missing data ({missing_pct:.1f}%) - dataset is relatively clean")
    else:
        insights.append("✅ No missing data - perfect dataset quality")
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        high_corr = np.where(np.abs(corr_matrix) > 0.8)
        high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr) if x != y and x < y]
        if high_corr_pairs:
            insights.append(f"🔗 Found {len(high_corr_pairs)} highly correlated feature pairs")
    
    if len(numeric_cols) >= 2:
        insights.append("🤖 Dataset is suitable for regression analysis")
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 2:
        insights.append("🎯 Dataset is suitable for classification analysis")
    if len(numeric_cols) >= 3:
        insights.append("🔍 Dataset is suitable for clustering analysis")
    
    return insights

def suggest_best_model(df, target_col):
    # Always use current working data (filtered data)
    current_df = st.session_state.get('working_data', df)
    if current_df.empty:
        return "No data available for model recommendation"
    df = current_df
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if target_col in numeric_cols:
        n_samples = len(df)
        if n_samples < 1000:
            return "Random Forest - Good for small datasets with interpretability"
        else:
            return "Gradient Boosting - Best overall performance for medium datasets"
    else:
        n_classes = df[target_col].nunique()
        if n_classes == 2:
            return "Logistic Regression - Excellent for binary classification"
        else:
            return "Random Forest - Handles multi-class problems well"

def detect_anomalies(df, column):
    # Always use current working data (filtered data)
    current_df = st.session_state.get('working_data', df)
    if current_df.empty:
        return []
    df = current_df
    
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return []
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return anomalies.index.tolist()

# Advanced Visualization Functions
def create_3d_scatter(df, x_col, y_col, z_col, color_col=None):
    try:
        clean_df = df[[x_col, y_col, z_col]].dropna()
        if color_col and color_col in df.columns:
            clean_df[color_col] = df[color_col]
            clean_df = clean_df.dropna()
        
        if len(clean_df) > 0:
            fig = px.scatter_3d(
                clean_df, x=x_col, y=y_col, z=z_col,
                color=color_col if color_col else None,
                title=f'3D Scatter: {x_col} vs {y_col} vs {z_col}'
            )
            return fig
    except Exception as e:
        st.error(f"Error creating 3D plot: {str(e)}")
    return None

def create_animated_plot(df, x_col, y_col, animation_col):
    try:
        clean_df = df[[x_col, y_col, animation_col]].dropna()
        if len(clean_df) > 0:
            fig = px.scatter(
                clean_df, x=x_col, y=y_col,
                animation_frame=animation_col,
                title=f'Animated Plot: {x_col} vs {y_col} over {animation_col}'
            )
            return fig
    except Exception as e:
        st.error(f"Error creating animated plot: {str(e)}")
    return None

def create_enhanced_heatmap(df, method='pearson'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr(method=method)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect='auto',
            title=f'{method.title()} Correlation Matrix',
            color_continuous_scale='RdBu_r'
        )
        return fig
    return None

# Data loading
@st.cache_data(ttl=3600)
def load_data(file):
    try:
        file_ext = file.name.split('.')[-1].lower()
        if file_ext == 'csv':
            df = pd.read_csv(file)
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        elif file_ext == 'json':
            df = pd.read_json(file)
        elif file_ext == 'parquet':
            df = pd.read_parquet(file)
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None
        
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except Exception as e:
                    # Continue processing other columns if datetime conversion fails
                    continue
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Data quality assessment
def calculate_data_quality_score(df):
    # Always use current working data (filtered data) for quality assessment
    current_df = st.session_state.get('working_data', df)
    if current_df.empty:
        return {'overall_score': 0, 'completeness': 0, 'consistency': 0, 'validity': 0}
    df = current_df
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
    
    duplicates = df.duplicated().sum()
    consistency = (df.shape[0] - duplicates) / df.shape[0] if df.shape[0] > 0 else 0
    
    validity_score = 1.0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        try:
            if SCIPY_AVAILABLE:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = (z_scores > 3).sum()
                col_validity = 1 - (outliers / len(df[col].dropna()))
                validity_score = min(validity_score, col_validity)
        except Exception as e:
            st.error(f"Error in silhouette score calculation: {str(e)}")
    
    overall_score = (completeness * 0.4 + consistency * 0.3 + validity_score * 0.3)
    
    return {
        'overall_score': overall_score,
        'completeness': completeness,
        'consistency': consistency,
        'validity': validity_score
    }

# Plot creation functions
def create_plot(plot_type, df, **kwargs):
    try:
        # Always use current working data (filtered data)
        current_df = st.session_state.get('working_data', df)
        if current_df.empty:
            return None
        df = current_df
        
        palette = st.session_state.get('selected_palette', 'aurora')
        colors = get_premium_colors(palette)
        
        if plot_type == 'histogram':
            column = kwargs.get('column')
            if column and column in df.columns:
                clean_data = df[column].dropna()
                if len(clean_data) > 0:
                    fig = px.histogram(df, x=column, title=f'✨ {column} Distribution', 
                                     color_discrete_sequence=colors,
                                     template='plotly_white')
                    fig.update_traces(marker_line_width=2, marker_line_color='white')
                    fig.update_layout(
                        title_font_size=20, title_font_color=colors[0],
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1]),
                        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1])
                    )
                    return fig
            return None
        
        elif plot_type == 'scatter':
            x_col = kwargs.get('x_column')
            y_col = kwargs.get('y_column')
            color_col = kwargs.get('color_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                cols_to_use = [x_col, y_col]
                if color_col and color_col in df.columns:
                    cols_to_use.append(color_col)
                clean_df = df[cols_to_use].dropna()
                if len(clean_df) > 0:
                    fig = px.scatter(clean_df, x=x_col, y=y_col, 
                                   color=color_col if color_col in clean_df.columns else None, 
                                   title=f'✨ {y_col} vs {x_col}',
                                   color_discrete_sequence=colors,
                                   template='plotly_white')
                    fig.update_traces(marker_size=8, marker_line_width=2, marker_line_color='white')
                    fig.update_layout(
                        title_font_size=20, title_font_color=colors[0],
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1]),
                        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1])
                    )
                    return fig
            return None
        
        elif plot_type == 'line':
            y_col = kwargs.get('y_column')
            x_col = kwargs.get('x_column')
            if y_col and y_col in df.columns:
                clean_df = df[[y_col]].dropna() if not x_col else df[[x_col, y_col]].dropna()
                if len(clean_df) > 0:
                    if x_col and x_col in df.columns:
                        fig = px.line(clean_df, x=x_col, y=y_col, title=f'📈 {y_col} Trend',
                                    color_discrete_sequence=colors, template='plotly_white')
                    else:
                        clean_df = clean_df.reset_index()
                        fig = px.line(clean_df, x=clean_df.index, y=y_col, title=f'📈 {y_col} Trend',
                                    color_discrete_sequence=colors, template='plotly_white')
                    fig.update_traces(line_width=4, marker_size=8)
                    fig.update_layout(
                        title_font_size=20, title_font_color=colors[0],
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1]),
                        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1])
                    )
                    return fig
            return None
        
        elif plot_type == 'bar':
            x_col = kwargs.get('x_column')
            y_col = kwargs.get('y_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                clean_df = df[[x_col, y_col]].dropna()
                if len(clean_df) > 0:
                    agg_df = clean_df.groupby(x_col)[y_col].mean().reset_index()
                    fig = px.bar(agg_df, x=x_col, y=y_col, title=f'📊 {y_col} by {x_col}',
                               color=x_col, color_discrete_sequence=colors, template='plotly_white')
                    fig.update_traces(marker_line_width=2, marker_line_color='white')
                    fig.update_layout(
                        title_font_size=20, title_font_color=colors[0],
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1]),
                        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1]),
                        showlegend=False
                    )
                    return fig
            return None
        
        elif plot_type == 'box':
            x_col = kwargs.get('x_column')
            y_col = kwargs.get('y_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                clean_df = df[[x_col, y_col]].dropna()
                if len(clean_df) > 0:
                    fig = px.box(clean_df, x=x_col, y=y_col, title=f'📦 {y_col} Distribution',
                               color=x_col, color_discrete_sequence=colors, template='plotly_white')
                    fig.update_traces(marker_size=6, line_width=3)
                    fig.update_layout(
                        title_font_size=20, title_font_color=colors[0],
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1]),
                        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1])
                    )
                    return fig
            return None
        
        elif plot_type == 'violin':
            x_col = kwargs.get('x_column')
            y_col = kwargs.get('y_column')
            if x_col and y_col and x_col in df.columns and y_col in df.columns:
                clean_df = df[[x_col, y_col]].dropna()
                if len(clean_df) > 0 and clean_df[x_col].nunique() > 1:
                    fig = px.violin(clean_df, x=x_col, y=y_col, title=f'🎻 {y_col} Violin Plot',
                                  color=x_col, color_discrete_sequence=colors, template='plotly_white',
                                  box=True)
                    fig.update_traces(meanline_visible=True, line_width=3)
                    fig.update_layout(
                        title_font_size=20, title_font_color=colors[0],
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1]),
                        yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title_font_color=colors[1])
                    )
                    return fig
            return None
        
        elif plot_type == 'pie':
            column = kwargs.get('column')
            if column and column in df.columns:
                clean_data = df[column].dropna()
                if len(clean_data) > 0:
                    value_counts = clean_data.value_counts().head(10)
                    if len(value_counts) > 0:
                        fig = px.pie(values=value_counts.values, names=value_counts.index, 
                                   title=f'🍰 {column} Distribution',
                                   color_discrete_sequence=colors, template='plotly_white')
                        fig.update_traces(textposition='inside', textinfo='percent+label',
                                        marker_line_width=3, marker_line_color='white')
                        fig.update_layout(
                            title_font_size=20, title_font_color=colors[0],
                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                        )
                        return fig
            return None
        
        elif plot_type == 'correlation':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                              title="🔥 Correlation Heatmap",
                              color_continuous_scale='RdBu_r', template='plotly_white')
                fig.update_layout(
                    title_font_size=20, title_font_color=colors[0],
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                )
                return fig
        
        return None
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

# ML functions
def run_linear_regression(df, features, target):
    if not ML_AVAILABLE:
        st.error("Machine learning libraries not available")
        return None
    
    try:
        # Always use current working data (filtered data)
        current_df = st.session_state.get('working_data', df)
        if current_df.empty:
            st.error("No data available for analysis")
            return None
        df = current_df
        
        # Validate inputs
        if not features or target not in df.columns:
            st.error("Invalid features or target column")
            return None
        
        # Select only numeric features that exist in df
        valid_features = [f for f in features if f in df.columns]
        if not valid_features:
            st.error("No valid features found")
            return None
        
        X = df[valid_features].select_dtypes(include=[np.number])
        y = df[target]
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            st.error(f"Not enough data points for regression. Need at least 10, got {len(X)}")
            return None
        
        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(y):
            st.error("Target variable must be numeric for regression")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        results = {
            'model': model,
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'feature_importance': dict(zip(X.columns, model.coef_)),
            'predictions': y_pred,
            'actual': y_test
        }
        
        return results
    except Exception as e:
        st.error(f"Error in regression analysis: {str(e)}")
        return None

def run_advanced_model(df, features, target, model_type, task_type='regression', tune_hyperparams=False):
    if not ML_AVAILABLE:
        return None
    
    try:
        # Always use current working data (filtered data)
        current_df = st.session_state.get('working_data', df)
        if current_df.empty:
            return None
        df = current_df
        X = df[features].select_dtypes(include=[np.number])
        y = df[target]
        
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model selection
        if task_type == 'regression':
            models = {
                'Random Forest': RandomForestRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(random_state=42),
                'Extra Trees': ExtraTreesRegressor(random_state=42),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'SVR': SVR(),
                'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBRegressor(random_state=42)
        else:
            models = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'Extra Trees': ExtraTreesClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42),
                'SVC': SVC(random_state=42),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBClassifier(random_state=42)
        
        model = models[model_type]
        
        # Hyperparameter tuning
        if tune_hyperparams:
            param_grids = {
                'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]},
                'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]},
                'Ridge': {'alpha': [0.1, 1.0, 10.0]},
                'Lasso': {'alpha': [0.1, 1.0, 10.0]},
                'KNN': {'n_neighbors': [3, 5, 7, 9]}
            }
            if XGBOOST_AVAILABLE:
                param_grids['XGBoost'] = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
            
            if model_type in param_grids:
                grid_search = GridSearchCV(model, param_grids[model_type], cv=3, scoring='r2' if task_type == 'regression' else 'accuracy')
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task_type == 'regression':
            results = {
                'model': model,
                'model_type': model_type,
                'r2_score': r2_score(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'predictions': y_pred,
                'actual': y_test
            }
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = dict(zip(X.columns, model.feature_importances_))
        else:
            results = {
                'model': model,
                'model_type': model_type,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'predictions': y_pred,
                'actual': y_test
            }
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = dict(zip(X.columns, model.feature_importances_))
        
        return results
    except Exception as e:
        st.error(f"Error in {model_type}: {str(e)}")
        return None

def run_automl(df, features, target, task_type='regression'):
    """Run multiple models and return best performing one"""
    if not ML_AVAILABLE:
        return None
    
    # Always use current working data (filtered data)
    current_df = st.session_state.get('working_data', df)
    if current_df.empty:
        return None
    df = current_df
    
    models_to_try = ['Random Forest', 'Gradient Boosting', 'Extra Trees']
    if XGBOOST_AVAILABLE:
        models_to_try.append('XGBoost')
    
    best_model = None
    best_score = -float('inf')
    results_list = []
    
    for model_type in models_to_try:
        result = run_advanced_model(df, features, target, model_type, task_type, tune_hyperparams=True)
        if result:
            results_list.append(result)
            score = result['r2_score'] if task_type == 'regression' else result['accuracy']
            if (task_type == 'regression' and score > best_score) or (task_type == 'classification' and score > best_score):
                best_score = score
                best_model = result
    
    return {'best_model': best_model, 'all_results': results_list}

def run_advanced_clustering(df, features, algorithm='kmeans', n_clusters=3):
    if not ML_AVAILABLE:
        return None
    
    try:
        # Always use current working data (filtered data)
        current_df = st.session_state.get('working_data', df)
        if current_df.empty:
            return None
        df = current_df
        valid_features = [f for f in features if f in df.columns]
        X = df[valid_features].select_dtypes(include=[np.number]).dropna()
        
        if len(X) < n_clusters:
            return None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'dbscan':
            model = DBSCAN(eps=0.5, min_samples=5)
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        
        clusters = model.fit_predict(X_scaled)
        
        sil_score = 0
        if len(set(clusters)) > 1:
            try:
                sil_score = silhouette_score(X_scaled, clusters)
            except Exception as e:
                st.warning(f"Could not calculate silhouette score: {str(e)}")
                sil_score = 0
        
        return {
            'clusters': clusters,
            'algorithm': algorithm,
            'features': valid_features,
            'silhouette_score': sil_score,
            'n_clusters': len(set(clusters))
        }
    except Exception as e:
        st.error(f"Error in clustering: {str(e)}")
        return None

def run_clustering(df, features, n_clusters=3):
    if not ML_AVAILABLE:
        st.error("Machine learning libraries not available")
        return None
    
    try:
        # Always use current working data (filtered data)
        current_df = st.session_state.get('working_data', df)
        if current_df.empty:
            st.error("No data available for clustering")
            return None
        df = current_df
        
        # Validate inputs
        if not features:
            st.error("No features selected for clustering")
            return None
        
        # Select only valid numeric features
        valid_features = [f for f in features if f in df.columns]
        if not valid_features:
            st.error("No valid features found")
            return None
        
        X = df[valid_features].select_dtypes(include=[np.number])
        X = X.dropna()
        
        if len(X) < n_clusters:
            st.error(f"Not enough data points for {n_clusters} clusters. Need at least {n_clusters}, got {len(X)}")
            return None
        
        if X.shape[1] == 0:
            st.error("No numeric features available for clustering")
            return None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score safely
        sil_score = 0
        if len(set(clusters)) > 1 and len(X_scaled) > n_clusters:
            try:
                sil_score = silhouette_score(X_scaled, clusters)
            except Exception as e:
                st.error(f"Error calculating silhouette score: {str(e)}")
                sil_score = 0
        
        results = {
            'clusters': clusters,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'features': valid_features,
            'scaled_data': X_scaled,
            'silhouette_score': sil_score
        }
        
        return results
    except Exception as e:
        st.error(f"Error in clustering: {str(e)}")
        return None

# Main application
def regenerate_all_charts():
    """Smart chart regeneration with premium animations"""
    if 'gallery_plots' in st.session_state and st.session_state.gallery_plots:
        current_data = st.session_state.working_data
        updated_charts = {}
        
        for key, plot_data in st.session_state.gallery_plots.items():
            try:
                # Regenerate chart with current filtered data
                new_fig = create_plot(
                    plot_data['type'], 
                    current_data, 
                    **plot_data['params']
                )
                if new_fig:
                    updated_charts[key] = {
                        **plot_data,
                        'fig': new_fig,
                        'last_updated': pd.Timestamp.now(),
                        'data_rows': len(current_data)
                    }
            except:
                # Keep original if regeneration fails
                updated_charts[key] = plot_data
        
        st.session_state.gallery_plots = updated_charts
    
    # Update dashboard charts
    if 'dashboard_charts' in st.session_state and st.session_state.dashboard_charts:
        current_data = st.session_state.working_data
        updated_dashboard = {}
        
        for key, chart_data in st.session_state.dashboard_charts.items():
            try:
                # Regenerate with premium visualization function
                from premium_visualization_tab import create_premium_chart
                new_fig = create_premium_chart(
                    current_data,
                    chart_data['type'],
                    chart_data['config'],
                    chart_data['colors'],
                    'modern'
                )
                if new_fig:
                    updated_dashboard[key] = {
                        **chart_data,
                        'fig': new_fig,
                        'data_rows': len(current_data),
                        'filter_applied': bool(st.session_state.get('filter_values')),
                        'last_updated': pd.Timestamp.now()
                    }
            except:
                updated_dashboard[key] = chart_data
        
        st.session_state.dashboard_charts = updated_dashboard

def main():
    initialize_session_state()
    
    if PREMIUM_AVAILABLE:
        st.markdown('<h1 class="main-header">🚀 Premium Data Analysis Platform</h1>', unsafe_allow_html=True)
    else:
        st.markdown('<h1 class="main-header">📊 Advanced Data Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Premium theme selection
        if PREMIUM_AVAILABLE:
            st.markdown("### 🎨 Customization")
            col1, col2 = st.columns(2)
            
            with col1:
                st.session_state.selected_theme = st.selectbox(
                    "Theme",
                    options=list(PREMIUM_THEMES.keys()),
                    index=list(PREMIUM_THEMES.keys()).index(st.session_state.selected_theme)
                )
            
            with col2:
                st.session_state.selected_palette = st.selectbox(
                    "Palette",
                    options=list(PREMIUM_COLOR_PALETTES.keys()),
                    index=list(PREMIUM_COLOR_PALETTES.keys()).index(st.session_state.selected_palette)
                )
            
            # Color preview
            colors = PREMIUM_COLOR_PALETTES[st.session_state.selected_palette]
            color_preview = "".join([f'<div style="display:inline-block;width:20px;height:20px;background:{color};margin:2px;border-radius:3px;"></div>' for color in colors[:7]])
            st.markdown(f'<div style="margin-bottom:1rem;">{color_preview}</div>', unsafe_allow_html=True)
            st.markdown("---")
        
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
            help="Upload your data file to get started"
        )
        
        # Sample data options
        if PREMIUM_AVAILABLE:
            st.markdown("### 📊 Sample Data")
            sample_options = {
                'sales': '💰 Sales Data',
                'customer': '👥 Customer Data', 
                'financial': '📈 Financial Data'
            }
            
            selected_sample = st.selectbox(
                "Or try sample data:",
                options=['None'] + list(sample_options.keys()),
                format_func=lambda x: sample_options.get(x, x)
            )
            
            if st.button("Load Sample Data", disabled=(selected_sample == 'None')):
                if selected_sample != 'None':
                    df = generate_sample_data(selected_sample)
                    st.session_state.working_data = df
                    st.session_state.original_data = df.copy()
                    st.session_state.uploaded_file_name = f"sample_{selected_sample}.csv"
                    st.success(f"✨ Loaded {selected_sample} sample data!")
                    st.rerun()
        
        if uploaded_file is not None:
            if st.session_state.uploaded_file_name != uploaded_file.name:
                with st.spinner("Loading data..."):
                    df = load_data(uploaded_file)
                    if df is not None:
                        st.session_state.uploaded_file_name = uploaded_file.name
                        st.session_state.original_data = df.copy()
                        # Optimize memory usage by avoiding unnecessary copying
                        st.session_state.working_data = df
                        st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Data info and filters
        if not get_working_data().empty:
            st.header("📊 Data Info")
            df = get_working_data()
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", len(df.columns))
            
            quality = calculate_data_quality_score(df)
            st.session_state.data_quality_score = quality['overall_score']
            
            quality_color = "green" if quality['overall_score'] > 0.8 else "orange" if quality['overall_score'] > 0.6 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Data Quality</h3>
                <h2 style="color: {quality_color}">{quality['overall_score']:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Filters
            st.header("🔍 Filters")
            # Use original data for filter options to show all available values
            original_df = st.session_state.original_data
            all_columns, numeric_columns, categorical_columns = get_column_types(original_df)
            
            filter_columns = st.multiselect("Select columns to filter", all_columns, key="filter_columns_select")
            
            filter_values = {}
            for col in filter_columns:
                if col in categorical_columns:
                    unique_vals = sorted(original_df[col].dropna().unique())
                    # Preserve existing selections if they exist
                    current_selection = st.session_state.filter_values.get(col, [])
                    if isinstance(current_selection, list):
                        default_vals = [v for v in current_selection if v in unique_vals]
                    else:
                        default_vals = []
                    selected_vals = st.multiselect(f"Filter {col}", unique_vals, default=default_vals, key=f"filter_{col}")
                    if selected_vals:
                        filter_values[col] = selected_vals
                elif col in numeric_columns:
                    min_val, max_val = float(original_df[col].min()), float(original_df[col].max())
                    # Preserve existing range if it exists
                    current_range = st.session_state.filter_values.get(col, (min_val, max_val))
                    if isinstance(current_range, tuple) and len(current_range) == 2:
                        default_range = current_range
                    else:
                        default_range = (min_val, max_val)
                    range_vals = st.slider(f"Filter {col}", min_val, max_val, default_range, key=f"slider_{col}")
                    if range_vals != (min_val, max_val):
                        filter_values[col] = range_vals
            
            # Smart auto-update system with premium animations
            if filter_values != st.session_state.filter_values:
                st.session_state.filter_values = filter_values
                
                # Apply filters with loading animation
                with st.spinner("🎨 Updating visualizations..."):
                    if filter_values:
                        filtered_df = st.session_state.original_data.copy()
                        for col, values in filter_values.items():
                            if isinstance(values, list) and values:
                                filtered_df = filtered_df[filtered_df[col].isin(values)]
                            elif isinstance(values, tuple) and len(values) == 2:
                                filtered_df = filtered_df[
                                    (filtered_df[col] >= values[0]) & 
                                    (filtered_df[col] <= values[1])
                                ]
                        st.session_state.working_data = filtered_df
                    else:
                        st.session_state.working_data = st.session_state.original_data.copy()
                    
                    # Auto-regenerate all existing charts with new data
                    regenerate_all_charts()
                    
                # Premium success notification
                if filter_values:
                    st.success(f"✨ {len(st.session_state.working_data):,} rows • Filters applied • Charts updated")
                    st.balloons()
                else:
                    st.info(f"🔄 {len(st.session_state.working_data):,} rows • All data shown • Charts refreshed")
            else:
                # Ensure working data is current
                if filter_values:
                    filtered_df = st.session_state.original_data.copy()
                    for col, values in filter_values.items():
                        if isinstance(values, list) and values:
                            filtered_df = filtered_df[filtered_df[col].isin(values)]
                        elif isinstance(values, tuple) and len(values) == 2:
                            filtered_df = filtered_df[
                                (filtered_df[col] >= values[0]) & 
                                (filtered_df[col] <= values[1])
                            ]
                    st.session_state.working_data = filtered_df
                else:
                    st.session_state.working_data = st.session_state.original_data.copy()
    
    # Main content
    if get_working_data().empty:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 2rem; color: white; box-shadow: 0 15px 40px rgba(0,0,0,0.3); position: relative; overflow: hidden;">
            <h1 style="font-size: 3.5rem; margin-bottom: 1rem; text-shadow: 3px 3px 6px rgba(0,0,0,0.4); font-weight: 800;">🚀 PLOTIVA</h1>
            <h2 style="font-size: 1.8rem; margin-bottom: 1rem; opacity: 0.95; font-weight: 300;">Advanced Data Analysis Platform</h2>
            <p style="font-size: 1.3rem; opacity: 0.9; margin-bottom: 0; max-width: 800px; margin: 0 auto;">Transform raw data into powerful insights with our comprehensive suite of visualization, machine learning, and data processing tools</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main feature cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a24); padding: 2rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15); margin-bottom: 1rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
                <h3 style="margin-bottom: 1.5rem; font-weight: bold; font-size: 1.4rem;">Data Visualization</h3>
                <div style="text-align: left; font-size: 0.95rem; line-height: 1.8;">
                    📈 Histogram & Distribution plots<br>
                    🔍 Scatter & Correlation analysis<br>
                    📊 Bar charts & Pie charts<br>
                    📉 Line plots & Time series<br>
                    🎻 Box & Violin plots<br>
                    🌡️ Heatmaps & Matrix plots<br>
                    🎨 Custom styling & themes
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4834d4, #686de0); padding: 2rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15); margin-bottom: 1rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🤖</div>
                <h3 style="margin-bottom: 1.5rem; font-weight: bold; font-size: 1.4rem;">Machine Learning</h3>
                <div style="text-align: left; font-size: 0.95rem; line-height: 1.8;">
                    📈 Linear Regression analysis<br>
                    🎯 Logistic Classification<br>
                    🌳 Random Forest (Reg & Class)<br>
                    🔍 K-Means Clustering<br>
                    📊 PCA Dimensionality reduction<br>
                    ⚡ Feature importance ranking<br>
                    📋 Model evaluation metrics
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #00d2d3, #54a0ff); padding: 2rem; border-radius: 15px; text-align: center; color: white; box-shadow: 0 8px 25px rgba(0,0,0,0.15); margin-bottom: 1rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🔧</div>
                <h3 style="margin-bottom: 1.5rem; font-weight: bold; font-size: 1.4rem;">Data Processing</h3>
                <div style="text-align: left; font-size: 0.95rem; line-height: 1.8;">
                    🧹 Smart data cleaning<br>
                    🔍 Missing value imputation<br>
                    📉 Outlier detection & removal<br>
                    ⚙️ Feature engineering tools<br>
                    🔄 Data transformation<br>
                    📊 Quality score calculation<br>
                    🎛️ Advanced filtering system
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional features row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8edea, #fed6e3); padding: 1.5rem; border-radius: 12px; text-align: center; color: #2c3e50; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">📋</div>
                <h4 style="margin-bottom: 1rem; font-weight: bold;">Interactive Dashboard</h4>
                <p style="font-size: 0.9rem; line-height: 1.5;">Create custom dashboards with multiple visualizations and real-time data filtering</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ffecd2, #fcb69f); padding: 1.5rem; border-radius: 12px; text-align: center; color: #2c3e50; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧠</div>
                <h4 style="margin-bottom: 1rem; font-weight: bold;">AI Insights</h4>
                <p style="font-size: 0.9rem; line-height: 1.5;">Get automated insights and recommendations powered by artificial intelligence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #d299c2, #fef9d7); padding: 1.5rem; border-radius: 12px; text-align: center; color: #2c3e50; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">🎨</div>
                <h4 style="margin-bottom: 1rem; font-weight: bold;">Advanced Visualization</h4>
                <p style="font-size: 0.9rem; line-height: 1.5;">Create stunning 3D plots, animated charts, and interactive visualizations</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("## 🚀 Ready to Transform Your Data?")
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("✨ **AI-Powered** - Smart insights")
        with col2:
            st.info("🚀 **Lightning Fast** - 3 second setup")
        with col3:
            st.info("🔒 **100% Secure** - Local processing")
        
        
        
        st.markdown("### 🎆 Unleash Your Data's Potential")
        st.markdown("*Experience the future of data analysis with our comprehensive suite of tools*")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("📁 **Universal File Support**\n\nCSV • Excel • JSON • Parquet\n\nDrag & drop any format")
        
        with col2:
            st.success("⚡ **Zero-Setup Analytics**\n\n✓ Ready in 3 seconds\n\nNo installation required")
        
        with col3:
            st.success("🔒 **Privacy First**\n\n🏠 Local Processing\n\nYour data stays secure")
        
        st.markdown("#### 🎯 What You'll Get Instantly:")
        st.markdown("""
        - ✓ Interactive visualizations
        - ✓ AI-powered insights  
        - ✓ Advanced ML models
        - ✓ Real-time data processing
        - ✓ Custom dashboards
        """)
        
        st.markdown("---")
        
        if st.button("🎯 Load Sample Dataset", type="primary"):
                np.random.seed(42)
                sample_data = pd.DataFrame({
                    'Date': pd.date_range('2023-01-01', periods=200),
                    'Sales': np.random.normal(1000, 200, 200) + np.sin(np.arange(200) * 2 * np.pi / 50) * 100,
                    'Customers': np.random.poisson(50, 200),
                    'Region': np.random.choice(['North', 'South', 'East', 'West'], 200),
                    'Product': np.random.choice(['A', 'B', 'C'], 200),
                    'Revenue': np.random.normal(5000, 1000, 200),
                    'Satisfaction': np.random.uniform(1, 5, 200),
                    'Age_Group': np.random.choice(['18-25', '26-35', '36-45', '46+'], 200)
                })
                
                st.session_state.uploaded_file_name = "sample_data.csv"
                st.session_state.original_data = sample_data.copy()
                st.session_state.working_data = sample_data.copy()
                st.rerun()
    
    else:
        df = get_working_data()
        
        # Ensure working_data is always up to date with current filters
        if st.session_state.filter_values:
            if 'working_data' not in st.session_state or st.session_state.working_data.empty:
                filtered_df = st.session_state.original_data.copy()
                for col, values in st.session_state.filter_values.items():
                    if isinstance(values, list):
                        filtered_df = filtered_df[filtered_df[col].isin(values)]
                    elif isinstance(values, tuple):
                        filtered_df = filtered_df[
                            (filtered_df[col] >= values[0]) & 
                            (filtered_df[col] <= values[1])
                        ]
                st.session_state.working_data = filtered_df
        else:
            # No filters applied, use original data
            if 'working_data' not in st.session_state or len(st.session_state.working_data) != len(st.session_state.original_data):
                st.session_state.working_data = st.session_state.original_data.copy()
        
        # Use filtered data for all operations
        df = st.session_state.working_data
        
        # Tabs with User Guide
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Data Overview", "📈 Visualizations", "🤖 Machine Learning", 
            "🔧 Data Processing", "📋 Dashboard", "🧠 AI Insights", "🎨 Advanced Viz", "📚 User Guide"
        ])
        
        with tab1:
            st.header("📊 Data Overview")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            with col4:
                duplicates = df.duplicated().sum()
                st.metric("Duplicates", duplicates)
            
            # Enhanced data preview with new column highlighting
            st.subheader("📋 Data Preview")
            
            # Enhanced column info with refresh capability
            col_info_display = st.columns(4)
            with col_info_display[0]:
                st.metric("Total Columns", len(df.columns))
            with col_info_display[1]:
                new_cols_count = len(st.session_state.get('new_columns', []))
                st.metric("New Columns", new_cols_count)
            with col_info_display[2]:
                st.metric("Data Rows", len(df))
            with col_info_display[3]:
                if st.button("🔄 Refresh", help="Refresh data preview"):
                    st.rerun()
            
            search_term = st.text_input("🔍 Search in data", "")
            
            if search_term:
                mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                preview_df = df[mask].head(20)
                st.write(f"Found {mask.sum()} rows containing '{search_term}'")
            else:
                preview_df = df.head(20)
            
            # Highlight new columns if any
            if 'new_columns' in st.session_state and st.session_state.new_columns:
                new_col_names = [col['name'] for col in st.session_state.new_columns]
                existing_new_cols = [col for col in new_col_names if col in preview_df.columns]
                
                if existing_new_cols:
                    st.markdown(f"✨ **New columns highlighted:** {', '.join(existing_new_cols)}")
            
            # Force refresh the dataframe display
            st.dataframe(preview_df, use_container_width=True, key=f"preview_{len(df.columns)}_{pd.Timestamp.now().timestamp()}")
            
            # Enhanced statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Summary Statistics")
                st.dataframe(df.describe(), use_container_width=True)
            
            with col2:
                st.subheader("🏷️ Data Types & Info")
                dtype_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': [df[col].nunique() for col in df.columns]
                })
                st.dataframe(dtype_df, use_container_width=True)
        
        with tab2:
            render_premium_visualization_tab(st.session_state.working_data)
            
            all_columns, numeric_columns, categorical_columns = get_column_types(df)
            
            # Plot gallery with multiple charts
            st.subheader("🎨 Plot Gallery")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("### Chart Controls")
                
                # Multiple chart creation
                chart_types = ["Histogram", "Scatter Plot", "Line Plot", "Bar Chart", "Box Plot", "Violin Plot", "Pie Chart", "Correlation Matrix"]
                selected_charts = st.multiselect("Select Chart Types", chart_types, default=["Histogram"])
                
                # Quick create multiple charts
                if st.button("🚀 Create All Selected Charts"):
                    for chart_type in selected_charts:
                        chart_key = f"{chart_type}_{int(time.time())}"
                        
                        if chart_type == "Histogram" and numeric_columns:
                            for col in numeric_columns[:3]:  # Limit to first 3
                                fig = create_plot('histogram', df, column=col)
                                if fig:
                                    st.session_state.gallery_plots[f"Histogram_{col}"] = {
                                        'fig': fig,
                                        'type': 'histogram',
                                        'params': {'column': col}
                                    }
                        
                        elif chart_type == "Bar Chart" and categorical_columns and numeric_columns:
                            for cat_col in categorical_columns[:2]:
                                for num_col in numeric_columns[:2]:
                                    fig = create_plot('bar', df, x_column=cat_col, y_column=num_col)
                                    if fig:
                                        st.session_state.gallery_plots[f"Bar_{cat_col}_{num_col}"] = {
                                            'fig': fig,
                                            'type': 'bar',
                                            'params': {'x_column': cat_col, 'y_column': num_col}
                                        }
                    
                    st.success(f"Created {len(selected_charts)} chart types!")
                
                # Individual chart creation
                st.markdown("### Individual Charts")
                plot_type = st.selectbox("Chart Type", chart_types)
                
                if plot_type == "Histogram":
                    column = st.selectbox("Column", numeric_columns)
                    if st.button("Create Histogram"):
                        fig = create_plot('histogram', df, column=column)
                        if fig:
                            st.session_state.gallery_plots[f"Histogram_{column}_{int(time.time())}"] = {
                                'fig': fig,
                                'type': 'histogram',
                                'params': {'column': column}
                            }
                
                elif plot_type == "Scatter Plot":
                    x_col = st.selectbox("X-axis", numeric_columns)
                    y_col = st.selectbox("Y-axis", [col for col in numeric_columns if col != x_col])
                    color_col = st.selectbox("Color by (optional)", [None] + categorical_columns)
                    if st.button("Create Scatter Plot"):
                        fig = create_plot('scatter', df, x_column=x_col, y_column=y_col, color_column=color_col)
                        if fig:
                            st.session_state.gallery_plots[f"Scatter_{x_col}_{y_col}_{int(time.time())}"] = {
                                'fig': fig,
                                'type': 'scatter',
                                'params': {'x_column': x_col, 'y_column': y_col, 'color_column': color_col}
                            }
                
                elif plot_type == "Bar Chart":
                    if categorical_columns:
                        x_col = st.selectbox("X-axis (Category)", categorical_columns)
                        y_col = st.selectbox("Y-axis (Numeric)", numeric_columns)
                        if st.button("Create Bar Chart"):
                            fig = create_plot('bar', df, x_column=x_col, y_column=y_col)
                            if fig:
                                st.session_state.gallery_plots[f"Bar_{x_col}_{y_col}_{int(time.time())}"] = {
                                    'fig': fig,
                                    'type': 'bar',
                                    'params': {'x_column': x_col, 'y_column': y_col}
                                }
                
                elif plot_type == "Box Plot":
                    if categorical_columns and numeric_columns:
                        x_col = st.selectbox("X-axis (Category)", categorical_columns)
                        y_col = st.selectbox("Y-axis (Numeric)", numeric_columns)
                        if st.button("Create Box Plot"):
                            fig = create_plot('box', df, x_column=x_col, y_column=y_col)
                            if fig:
                                st.session_state.gallery_plots[f"Box_{x_col}_{y_col}_{int(time.time())}"] = {
                                    'fig': fig,
                                    'type': 'box',
                                    'params': {'x_column': x_col, 'y_column': y_col}
                                }
                    else:
                        st.warning("Need both categorical and numeric columns for box plots")
                
                elif plot_type == "Line Plot":
                    y_col = st.selectbox("Y-axis", numeric_columns)
                    x_col = st.selectbox("X-axis (optional)", [None] + all_columns)
                    if st.button("Create Line Plot"):
                        fig = create_plot('line', df, y_column=y_col, x_column=x_col)
                        if fig:
                            st.session_state.gallery_plots[f"Line_{y_col}_{int(time.time())}"] = {
                                'fig': fig,
                                'type': 'line',
                                'params': {'y_column': y_col, 'x_column': x_col}
                            }
                
                elif plot_type == "Violin Plot":
                    if categorical_columns:
                        x_col = st.selectbox("X-axis (Category)", categorical_columns)
                        y_col = st.selectbox("Y-axis (Numeric)", numeric_columns)
                        if st.button("Create Violin Plot"):
                            fig = create_plot('violin', df, x_column=x_col, y_column=y_col)
                            if fig:
                                st.session_state.gallery_plots[f"Violin_{x_col}_{y_col}_{int(time.time())}"] = {
                                    'fig': fig,
                                    'type': 'violin',
                                    'params': {'x_column': x_col, 'y_column': y_col}
                                }
                
                elif plot_type == "Pie Chart":
                    if categorical_columns:
                        column = st.selectbox("Column", categorical_columns)
                        if st.button("Create Pie Chart"):
                            fig = create_plot('pie', df, column=column)
                            if fig:
                                st.session_state.gallery_plots[f"Pie_{column}_{int(time.time())}"] = {
                                    'fig': fig,
                                    'type': 'pie',
                                    'params': {'column': column}
                                }
                
                elif plot_type == "Correlation Matrix":
                    if len(numeric_columns) > 1:
                        if st.button("Create Correlation Matrix"):
                            fig = create_plot('correlation', df)
                            if fig:
                                st.session_state.gallery_plots[f"Correlation_{int(time.time())}"] = {
                                    'fig': fig,
                                    'type': 'correlation',
                                    'params': {}
                                }
            
            with col2:
                st.markdown("### Generated Charts")
                
                if st.session_state.gallery_plots:
                    # Display charts in grid
                    chart_keys = list(st.session_state.gallery_plots.keys())
                    
                    for i in range(0, len(chart_keys), 2):
                        cols = st.columns(2)
                        
                        for j, col in enumerate(cols):
                            if i + j < len(chart_keys):
                                key = chart_keys[i + j]
                                plot_data = st.session_state.gallery_plots[key]
                                
                                with col:
                                    st.plotly_chart(plot_data['fig'], use_container_width=True, key=f"gallery_plot_{key}")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        if st.button(f"📊 Add to Dashboard", key=f"add_{key}"):
                                            if key not in st.session_state.dashboard_selections:
                                                st.session_state.dashboard_selections.append(key)
                                                st.success("Added to dashboard!")
                                    
                                    with col_b:
                                        if st.button(f"🗑️ Remove", key=f"remove_{key}"):
                                            del st.session_state.gallery_plots[key]
                                            st.rerun()
                else:
                    st.info("No charts created yet. Use the controls on the left to create charts.")
        
        with tab3:
            st.header("🤖 Advanced Machine Learning")
            
            if not ML_AVAILABLE:
                st.error("Machine learning libraries not available. Please install scikit-learn.")
            else:
                all_columns, numeric_columns, categorical_columns = get_column_types(df)
                
                if len(numeric_columns) < 2:
                    st.warning("Need at least 2 numeric columns for machine learning analysis")
                else:
                    ml_tabs = st.tabs(["📈 Regression", "🎯 Classification", "🔍 Clustering", "⚡ AutoML", "📊 Model Comparison"])
                    
                    with ml_tabs[0]:  # Regression
                        st.subheader("📈 Regression Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Linear Regression")
                            target = st.selectbox("Target Variable", numeric_columns, key="lr_target")
                            features = st.multiselect(
                                "Feature Variables", 
                                [col for col in numeric_columns if col != target],
                                default=[col for col in numeric_columns if col != target][:3],
                                key="lr_features"
                            )
                            
                            if features and st.button("Run Linear Regression"):
                                with st.spinner("Running analysis..."):
                                    results = run_linear_regression(df, features, target)
                                    
                                    if results:
                                        st.session_state.ml_results['linear_regression'] = results
                                        
                                        st.metric("R² Score", f"{results['r2_score']:.3f}")
                                        st.metric("MSE", f"{results['mse']:.3f}")
                                        st.metric("MAE", f"{results['mae']:.3f}")
                                        
                                        # Feature importance
                                        importance_df = pd.DataFrame(
                                            list(results['feature_importance'].items()),
                                            columns=['Feature', 'Coefficient']
                                        )
                                        st.dataframe(importance_df)
                        
                        with col2:
                            st.markdown("#### Advanced Models")
                            adv_target = st.selectbox("Target Variable", numeric_columns, key="adv_target")
                            adv_features = st.multiselect(
                                "Feature Variables", 
                                [col for col in numeric_columns if col != adv_target],
                                default=[col for col in numeric_columns if col != adv_target][:3],
                                key="adv_features"
                            )
                            
                            model_options = ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'Ridge', 'Lasso']
                            if XGBOOST_AVAILABLE:
                                model_options.append('XGBoost')
                            
                            selected_model = st.selectbox("Model Type", model_options)
                            tune_params = st.checkbox("Enable Hyperparameter Tuning", value=False)
                            
                            if adv_features and st.button("Run Advanced Model"):
                                with st.spinner(f"Running {selected_model}..."):
                                    results = run_advanced_model(df, adv_features, adv_target, selected_model, 'regression', tune_params)
                                    
                                    if results:
                                        st.session_state.ml_results[f'{selected_model.lower()}_reg'] = results
                                        
                                        st.metric("R² Score", f"{results['r2_score']:.3f}")
                                        st.metric("MSE", f"{results['mse']:.3f}")
                                        st.metric("MAE", f"{results['mae']:.3f}")
                                        
                                        if 'feature_importance' in results:
                                            importance_df = pd.DataFrame(
                                                list(results['feature_importance'].items()),
                                                columns=['Feature', 'Importance']
                                            ).sort_values('Importance', ascending=False)
                                            st.dataframe(importance_df)
                                            
                                            fig = px.bar(
                                                importance_df.head(10),
                                                x='Importance',
                                                y='Feature',
                                                orientation='h',
                                                title=f'{selected_model} Feature Importance'
                                            )
                                            st.plotly_chart(fig, use_container_width=True, key=f"{selected_model.lower()}_importance_plot")
                        
                        # Prediction plots
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'linear_regression' in st.session_state.ml_results:
                                results = st.session_state.ml_results['linear_regression']
                                fig = px.scatter(
                                    x=results['actual'], 
                                    y=results['predictions'],
                                    labels={'x': 'Actual', 'y': 'Predicted'},
                                    title='Linear Regression: Predictions vs Actual'
                                )
                                fig.add_shape(
                                    type="line",
                                    x0=results['actual'].min(),
                                    y0=results['actual'].min(),
                                    x1=results['actual'].max(),
                                    y1=results['actual'].max(),
                                    line=dict(dash="dash", color="red")
                                )
                                st.plotly_chart(fig, use_container_width=True, key="lr_prediction_plot")
                        
                        with col2:
                            # Show latest advanced model results
                            adv_model_keys = [k for k in st.session_state.ml_results.keys() if k.endswith('_reg') and k != 'linear_regression']
                            if adv_model_keys:
                                latest_key = adv_model_keys[-1]
                                results = st.session_state.ml_results[latest_key]
                                fig = px.scatter(
                                    x=results['actual'], 
                                    y=results['predictions'],
                                    labels={'x': 'Actual', 'y': 'Predicted'},
                                    title=f'{results["model_type"]}: Predictions vs Actual'
                                )
                                fig.add_shape(
                                    type="line",
                                    x0=results['actual'].min(),
                                    y0=results['actual'].min(),
                                    x1=results['actual'].max(),
                                    y1=results['actual'].max(),
                                    line=dict(dash="dash", color="red")
                                )
                                st.plotly_chart(fig, use_container_width=True, key="adv_prediction_plot")
                    
                    with ml_tabs[1]:  # Classification
                        st.subheader("🎯 Classification Analysis")
                        
                        if categorical_columns:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### Basic Classification")
                                target = st.selectbox("Target Variable", categorical_columns, key="clf_target")
                                features = st.multiselect(
                                    "Feature Variables", 
                                    numeric_columns,
                                    default=numeric_columns[:3],
                                    key="clf_features"
                                )
                                
                                if features and st.button("Run Random Forest Classification"):
                                    with st.spinner("Running classification..."):
                                        results = run_advanced_model(df, features, target, 'Random Forest', 'classification')
                                        
                                        if results:
                                            st.session_state.ml_results['rf_classification'] = results
                                            
                                            st.metric("Accuracy", f"{results['accuracy']:.3f}")
                                            st.metric("F1 Score", f"{results['f1_score']:.3f}")
                            
                            with col2:
                                st.markdown("#### Advanced Classification")
                                adv_clf_target = st.selectbox("Target Variable", categorical_columns, key="adv_clf_target")
                                adv_clf_features = st.multiselect(
                                    "Feature Variables", 
                                    numeric_columns,
                                    default=numeric_columns[:3],
                                    key="adv_clf_features"
                                )
                                
                                clf_models = ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'Logistic Regression']
                                if XGBOOST_AVAILABLE:
                                    clf_models.append('XGBoost')
                                
                                selected_clf_model = st.selectbox("Model Type", clf_models, key="clf_model_select")
                                tune_clf_params = st.checkbox("Enable Hyperparameter Tuning", key="clf_tune")
                                
                                if adv_clf_features and st.button("Run Advanced Classification"):
                                    with st.spinner(f"Running {selected_clf_model}..."):
                                        results = run_advanced_model(df, adv_clf_features, adv_clf_target, selected_clf_model, 'classification', tune_clf_params)
                                        
                                        if results:
                                            st.session_state.ml_results[f'{selected_clf_model.lower()}_clf'] = results
                                            
                                            st.metric("Accuracy", f"{results['accuracy']:.3f}")
                                            st.metric("F1 Score", f"{results['f1_score']:.3f}")
                            
                            # Show classification results
                            clf_results = [v for k, v in st.session_state.ml_results.items() if 'clf' in k or 'classification' in k]
                            if clf_results:
                                latest_clf = clf_results[-1]
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    try:
                                        cm = confusion_matrix(latest_clf['actual'], latest_clf['predictions'])
                                        fig = px.imshow(cm, text_auto=True, aspect="auto", title="Confusion Matrix")
                                        st.plotly_chart(fig, use_container_width=True, key="confusion_matrix_plot")
                                    except:
                                        pass
                                
                                with col2:
                                    if 'feature_importance' in latest_clf:
                                        importance_df = pd.DataFrame(
                                            list(latest_clf['feature_importance'].items()),
                                            columns=['Feature', 'Importance']
                                        ).sort_values('Importance', ascending=False)
                                        
                                        fig = px.bar(
                                            importance_df.head(10),
                                            x='Importance',
                                            y='Feature',
                                            orientation='h',
                                            title='Feature Importance'
                                        )
                                        st.plotly_chart(fig, use_container_width=True, key="clf_importance_plot")
                        else:
                            st.warning("No categorical columns found for classification")
                    
                    with ml_tabs[2]:  # Clustering
                        st.subheader("🔍 K-Means Clustering")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### K-Means Clustering")
                            features = st.multiselect(
                                "Features for Clustering", 
                                numeric_columns,
                                default=numeric_columns[:3],
                                key="cluster_features"
                            )
                            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                            
                            if features and st.button("Run K-Means"):
                                with st.spinner("Running K-Means..."):
                                    results = run_advanced_clustering(df, features, 'kmeans', n_clusters)
                                    
                                    if results:
                                        st.session_state.ml_results['kmeans'] = results
                                        st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
                                        st.metric("Clusters Found", results['n_clusters'])
                        
                        with col2:
                            st.markdown("#### Advanced Clustering")
                            adv_features = st.multiselect(
                                "Features for Advanced Clustering", 
                                numeric_columns,
                                default=numeric_columns[:3],
                                key="adv_cluster_features"
                            )
                            
                            algorithm = st.selectbox("Algorithm", ['kmeans', 'dbscan', 'hierarchical'])
                            
                            if algorithm == 'kmeans':
                                adv_n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="adv_clusters")
                            else:
                                adv_n_clusters = 3
                            
                            if adv_features and st.button("Run Advanced Clustering"):
                                with st.spinner(f"Running {algorithm.upper()}..."):
                                    results = run_advanced_clustering(df, adv_features, algorithm, adv_n_clusters)
                                    
                                    if results:
                                        st.session_state.ml_results[f'{algorithm}_clustering'] = results
                                        st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
                                        st.metric("Clusters Found", results['n_clusters'])
                        
                        # Show clustering results
                        cluster_results = [v for k, v in st.session_state.ml_results.items() if 'cluster' in k or 'kmeans' in k]
                        if cluster_results:
                            latest_cluster = cluster_results[-1]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                cluster_counts = pd.Series(latest_cluster['clusters']).value_counts().sort_index()
                                fig = px.bar(
                                    x=cluster_counts.index,
                                    y=cluster_counts.values,
                                    labels={'x': 'Cluster', 'y': 'Count'},
                                    title=f'{latest_cluster["algorithm"].upper()} Cluster Distribution'
                                )
                                st.plotly_chart(fig, use_container_width=True, key="cluster_distribution")
                            
                            with col2:
                                if len(latest_cluster['features']) >= 2:
                                    X = df[latest_cluster['features']].select_dtypes(include=[np.number]).dropna()
                                    if len(X) == len(latest_cluster['clusters']):
                                        df_with_clusters = X.copy()
                                        df_with_clusters['Cluster'] = latest_cluster['clusters']
                                        
                                        fig = px.scatter(
                                            df_with_clusters,
                                            x=latest_cluster['features'][0],
                                            y=latest_cluster['features'][1],
                                            color='Cluster',
                                            title=f'Clusters: {latest_cluster["features"][0]} vs {latest_cluster["features"][1]}'
                                        )
                                        st.plotly_chart(fig, use_container_width=True, key="cluster_scatter_plot")
                    
                    with ml_tabs[3]:  # AutoML
                        st.subheader("⚡ Automated Machine Learning")
                        st.markdown("Let AutoML find the best model and hyperparameters for your data!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### AutoML Regression")
                            automl_reg_target = st.selectbox("Target Variable", numeric_columns, key="automl_reg_target")
                            automl_reg_features = st.multiselect(
                                "Feature Variables", 
                                [col for col in numeric_columns if col != automl_reg_target],
                                default=[col for col in numeric_columns if col != automl_reg_target][:5],
                                key="automl_reg_features"
                            )
                            
                            if automl_reg_features and st.button("🚀 Run AutoML Regression"):
                                with st.spinner("Running AutoML (this may take a while)..."):
                                    automl_results = run_automl(df, automl_reg_features, automl_reg_target, 'regression')
                                    
                                    if automl_results and automl_results['best_model']:
                                        st.session_state.ml_results['automl_regression'] = automl_results
                                        best = automl_results['best_model']
                                        
                                        st.success(f"🏆 Best Model: {best['model_type']}")
                                        st.metric("Best R² Score", f"{best['r2_score']:.3f}")
                                        st.metric("MSE", f"{best['mse']:.3f}")
                                        
                                        # Show all model comparison
                                        comparison_data = []
                                        for result in automl_results['all_results']:
                                            comparison_data.append({
                                                'Model': result['model_type'],
                                                'R² Score': result['r2_score'],
                                                'MSE': result['mse'],
                                                'MAE': result['mae']
                                            })
                                        
                                        comparison_df = pd.DataFrame(comparison_data).sort_values('R² Score', ascending=False)
                                        st.dataframe(comparison_df, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### AutoML Classification")
                            if categorical_columns:
                                automl_clf_target = st.selectbox("Target Variable", categorical_columns, key="automl_clf_target")
                                automl_clf_features = st.multiselect(
                                    "Feature Variables", 
                                    numeric_columns,
                                    default=numeric_columns[:5],
                                    key="automl_clf_features"
                                )
                                
                                if automl_clf_features and st.button("🚀 Run AutoML Classification"):
                                    with st.spinner("Running AutoML (this may take a while)..."):
                                        automl_results = run_automl(df, automl_clf_features, automl_clf_target, 'classification')
                                        
                                        if automl_results and automl_results['best_model']:
                                            st.session_state.ml_results['automl_classification'] = automl_results
                                            best = automl_results['best_model']
                                            
                                            st.success(f"🏆 Best Model: {best['model_type']}")
                                            st.metric("Best Accuracy", f"{best['accuracy']:.3f}")
                                            st.metric("F1 Score", f"{best['f1_score']:.3f}")
                                            
                                            # Show all model comparison
                                            comparison_data = []
                                            for result in automl_results['all_results']:
                                                comparison_data.append({
                                                    'Model': result['model_type'],
                                                    'Accuracy': result['accuracy'],
                                                    'F1 Score': result['f1_score']
                                                })
                                            
                                            comparison_df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
                                            st.dataframe(comparison_df, use_container_width=True)
                            else:
                                st.info("No categorical columns available for classification")
                        
                        # AutoML Results Visualization
                        if 'automl_regression' in st.session_state.ml_results:
                            st.markdown("#### AutoML Regression Results")
                            automl_reg = st.session_state.ml_results['automl_regression']
                            best_reg = automl_reg['best_model']
                            
                            fig = px.scatter(
                                x=best_reg['actual'], 
                                y=best_reg['predictions'],
                                labels={'x': 'Actual', 'y': 'Predicted'},
                                title=f'Best Model ({best_reg["model_type"]}): Predictions vs Actual'
                            )
                            fig.add_shape(
                                type="line",
                                x0=best_reg['actual'].min(),
                                y0=best_reg['actual'].min(),
                                x1=best_reg['actual'].max(),
                                y1=best_reg['actual'].max(),
                                line=dict(dash="dash", color="red")
                            )
                            st.plotly_chart(fig, use_container_width=True, key="automl_reg_plot")
                    
                    with ml_tabs[4]:  # Model Comparison
                        st.subheader("📊 Model Performance Comparison")
                        
                        if st.session_state.ml_results:
                            comparison_data = []
                            
                            for model_name, results in st.session_state.ml_results.items():
                                if 'r2_score' in results:
                                    comparison_data.append({
                                        'Model': model_name,
                                        'R² Score': results['r2_score'],
                                        'MSE': results['mse'],
                                        'MAE': results['mae']
                                    })
                                elif 'accuracy' in results:
                                    comparison_data.append({
                                        'Model': model_name,
                                        'Accuracy': results['accuracy'],
                                        'F1 Score': results['f1_score']
                                    })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True)
                        else:
                            st.info("Run some models first to see comparison")
        
        with tab4:
            st.header("🔧 Advanced Data Processing")
            
            processing_tabs = st.tabs(["🔍 Data Quality", "🧹 Data Cleaning", "🔧 Feature Engineering"])
            
            with processing_tabs[0]:  # Data Quality
                st.subheader("🔍 Data Quality Assessment")
                
                quality = calculate_data_quality_score(df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Overall Score</h4>
                        <h2>{quality['overall_score']:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Completeness</h4>
                        <h2>{quality['completeness']:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Consistency</h4>
                        <h2>{quality['consistency']:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Validity</h4>
                        <h2>{quality['validity']:.1%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Missing values analysis
                st.subheader("📊 Missing Values Analysis")
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                
                if len(missing_data) > 0:
                    missing_df = pd.DataFrame({
                        'Column': missing_data.index,
                        'Missing Count': missing_data.values,
                        'Missing %': (missing_data.values / len(df)) * 100
                    })
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Visualize missing data
                    fig = px.bar(missing_df, x='Column', y='Missing %', title='Missing Data by Column')
                    st.plotly_chart(fig, use_container_width=True, key="missing_data_viz")
                else:
                    st.success("✅ No missing values found!")
                
                # Outlier analysis
                st.subheader("📈 Outlier Analysis")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    outlier_col = st.selectbox("Select column for outlier analysis", numeric_cols)
                    
                    if SCIPY_AVAILABLE:
                        Q1 = df[outlier_col].quantile(0.25)
                        Q3 = df[outlier_col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Outliers Found", len(outliers))
                        with col2:
                            st.metric("Outlier %", f"{(len(outliers)/len(df))*100:.1f}%")
                        
                        # Box plot for outliers
                        fig = px.box(df, y=outlier_col, title=f'Box Plot: {outlier_col}')
                        st.plotly_chart(fig, use_container_width=True, key=f"outlier_box_{outlier_col}")
            
            with processing_tabs[1]:  # Data Cleaning
                st.subheader("🧹 Data Cleaning Operations")
                
                # Missing values handling
                st.markdown("#### Missing Values")
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                
                if len(missing_data) > 0:
                    column_to_fix = st.selectbox("Select column to fix", missing_data.index)
                    method = st.selectbox("Select method", [
                        "Drop rows", "Fill with mean", "Fill with median", 
                        "Fill with mode", "Forward fill", "Backward fill"
                    ])
                    
                    if st.button("Apply Missing Value Fix"):
                        df_fixed = df.copy()
                        
                        if method == "Drop rows":
                            df_fixed = df_fixed.dropna(subset=[column_to_fix])
                        elif method == "Fill with mean":
                            if column_to_fix in df.select_dtypes(include=[np.number]).columns:
                                df_fixed[column_to_fix].fillna(df[column_to_fix].mean(), inplace=True)
                        elif method == "Fill with median":
                            if column_to_fix in df.select_dtypes(include=[np.number]).columns:
                                df_fixed[column_to_fix].fillna(df[column_to_fix].median(), inplace=True)
                        elif method == "Fill with mode":
                            mode_value = df[column_to_fix].mode()[0] if not df[column_to_fix].mode().empty else 0
                            df_fixed[column_to_fix].fillna(mode_value, inplace=True)
                        elif method == "Forward fill":
                            df_fixed[column_to_fix].fillna(method='ffill', inplace=True)
                        elif method == "Backward fill":
                            df_fixed[column_to_fix].fillna(method='bfill', inplace=True)
                        
                        update_working_data(df_fixed)
                        st.success(f"Applied {method} to {column_to_fix}")
                        st.rerun()
                
                # Duplicate handling
                st.markdown("#### Duplicate Rows")
                duplicates = df.duplicated().sum()
                
                if duplicates > 0:
                    st.warning(f"Found {duplicates} duplicate rows")
                    if st.button("Remove Duplicates"):
                        df_no_dups = df.drop_duplicates()
                        update_working_data(df_no_dups)
                        st.success(f"Removed {duplicates} duplicate rows")
                        st.rerun()
                else:
                    st.success("✅ No duplicate rows found!")
                
                # Outlier removal
                st.markdown("#### Outlier Removal")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    outlier_col = st.selectbox("Select column for outlier removal", numeric_cols, key="outlier_removal")
                    outlier_method = st.selectbox("Outlier detection method", ["IQR", "Z-score"])
                    
                    if outlier_method == "IQR":
                        Q1 = df[outlier_col].quantile(0.25)
                        Q3 = df[outlier_col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                    else:  # Z-score
                        if SCIPY_AVAILABLE:
                            z_scores = np.abs(stats.zscore(df[outlier_col].dropna()))
                            outliers = df[z_scores > 3]
                        else:
                            outliers = pd.DataFrame()
                    
                    st.write(f"Found {len(outliers)} outliers")
                    
                    if len(outliers) > 0 and st.button("Remove Outliers"):
                        if outlier_method == "IQR":
                            df_no_outliers = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                        else:
                            if SCIPY_AVAILABLE:
                                z_scores = np.abs(stats.zscore(df[outlier_col].dropna()))
                                df_no_outliers = df[z_scores <= 3]
                            else:
                                df_no_outliers = df
                        
                        update_working_data(df_no_outliers)
                        st.success(f"Removed {len(outliers)} outliers")
                        st.rerun()
            
            with processing_tabs[2]:  # Feature Engineering
                from premium_feature_engineering import render_premium_feature_engineering
                render_premium_feature_engineering()
        
        with tab5:
            render_dashboard_tab()
            
            # Dashboard controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Available Charts")
                if st.session_state.gallery_plots:
                    for key in st.session_state.gallery_plots.keys():
                        if st.button(f"➕ {key}", key=f"dash_add_{key}"):
                            if key not in st.session_state.dashboard_selections:
                                st.session_state.dashboard_selections.append(key)
                                st.success(f"Added {key} to dashboard!")
                else:
                    st.info("Create charts in the Visualizations tab first")
            
            with col2:
                st.subheader("🎛️ Dashboard Settings")
                dashboard_title = st.text_input("Dashboard Title", "My Data Dashboard")
                num_columns = st.slider("Columns per row", 1, 3, 2)
                show_metrics = st.checkbox("Show data quality metrics", True)
            
            with col3:
                st.subheader("📈 Quick Actions")
                if st.button("🔄 Refresh Dashboard"):
                    st.rerun()
                
                if st.button("🗑️ Clear Dashboard"):
                    st.session_state.dashboard_selections = []
                    st.rerun()
                
                if st.button("📊 Auto-populate Dashboard"):
                    # Add first few charts automatically
                    available_charts = list(st.session_state.gallery_plots.keys())[:6]
                    st.session_state.dashboard_selections = available_charts
                    st.rerun()
            
            st.markdown("---")
            
            # Dashboard display
            st.markdown(f"## {dashboard_title}")
            
            if show_metrics:
                quality = calculate_data_quality_score(df)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Data Quality", f"{quality['overall_score']:.1%}")
                with col2:
                    st.metric("Total Rows", f"{len(df):,}")
                with col3:
                    st.metric("Total Columns", len(df.columns))
                with col4:
                    st.metric("Missing Data", f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%")
            
            # Display selected charts
            if st.session_state.dashboard_selections:
                selected_charts = [key for key in st.session_state.dashboard_selections if key in st.session_state.gallery_plots]
                
                # Arrange charts in grid
                for i in range(0, len(selected_charts), num_columns):
                    cols = st.columns(num_columns)
                    
                    for j in range(num_columns):
                        if i + j < len(selected_charts):
                            chart_key = selected_charts[i + j]
                            plot_data = st.session_state.gallery_plots[chart_key]
                            
                            with cols[j]:
                                st.markdown(f"<div class='plot-container'>", unsafe_allow_html=True)
                                st.plotly_chart(plot_data['fig'], use_container_width=True, key=f"dashboard_plot_{chart_key}_{i}_{j}")
                                
                                # Chart controls
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    if st.button(f"📤 Export", key=f"export_{chart_key}"):
                                        # Export functionality can be added here
                                        st.info("Export feature coming soon!")
                                
                                with col_b:
                                    if st.button(f"❌ Remove", key=f"dash_remove_{chart_key}"):
                                        st.session_state.dashboard_selections.remove(chart_key)
                                        st.rerun()
                                
                                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("📊 No charts selected for dashboard. Add charts from the 'Available Charts' section above or create new charts in the Visualizations tab.")
            
            # Dashboard export
            st.markdown("---")
            st.subheader("📤 Export Dashboard")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Download Dashboard Data"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"dashboard_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📈 Generate Dashboard Report"):
                    quality = calculate_data_quality_score(df)
                    report = f"""
# Dashboard Report
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Overview
- Rows: {len(df):,}
- Columns: {len(df.columns)}
- Data Quality: {quality['overall_score']:.1%}

## Charts in Dashboard
{chr(10).join([f"- {chart}" for chart in st.session_state.dashboard_selections])}

## Data Quality Metrics
- Completeness: {quality['completeness']:.1%}
- Consistency: {quality['consistency']:.1%}
- Validity: {quality['validity']:.1%}
                    """
                    
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"dashboard_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        
        with tab6:  # AI Insights
            st.header("🧠 AI-Powered Data Insights")
            
            with st.spinner("Analyzing your data with AI..."):
                insights = generate_ai_insights(df)
            
            st.subheader("📋 Smart Analysis Summary")
            for i, insight in enumerate(insights, 1):
                st.markdown(f"**{i}.** {insight}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Model Recommendations")
                all_columns, numeric_columns, categorical_columns = get_column_types(df)
                
                if numeric_columns or categorical_columns:
                    target_col = st.selectbox("Select target for recommendation", numeric_columns + categorical_columns)
                    recommendation = suggest_best_model(df, target_col)
                    st.success(f"💡 **Recommended Model:** {recommendation}")
                
                st.subheader("🔍 Anomaly Detection")
                if numeric_columns:
                    anomaly_col = st.selectbox("Select column for anomaly detection", numeric_columns)
                    anomalies = detect_anomalies(df, anomaly_col)
                    
                    if anomalies:
                        st.warning(f"⚠️ Found {len(anomalies)} anomalies in '{anomaly_col}'")
                        if st.checkbox("Show anomalous rows"):
                            st.dataframe(df.iloc[anomalies[:10]])
                    else:
                        st.success(f"✅ No anomalies detected in '{anomaly_col}'")
            
            with col2:
                st.subheader("📊 Advanced Statistics")
                
                if len(numeric_columns) > 1:
                    corr_method = st.selectbox("Correlation method", ['pearson', 'spearman', 'kendall'])
                    fig = create_enhanced_heatmap(df, corr_method)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"ai_corr_{corr_method}")
                
                st.subheader("📈 Distribution Insights")
                if numeric_columns:
                    dist_col = st.selectbox("Select column for distribution analysis", numeric_columns, key="dist_col")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        skewness = df[dist_col].skew()
                        st.metric("Skewness", f"{skewness:.3f}")
                    with col_b:
                        kurtosis = df[dist_col].kurtosis()
                        st.metric("Kurtosis", f"{kurtosis:.3f}")
                    
                    fig = px.histogram(df, x=dist_col, marginal='box', title=f'Distribution: {dist_col}')
                    st.plotly_chart(fig, use_container_width=True, key="ai_dist_plot")
        
        with tab7:  # Advanced Visualizations
            st.header("🎨 Advanced Visualizations")
            
            all_columns, numeric_columns, categorical_columns = get_column_types(df)
            
            viz_tabs = st.tabs(["🌐 3D Plots", "🎬 Animated Plots", "🔥 Heatmaps"])
            
            with viz_tabs[0]:  # 3D Plots
                st.subheader("🌐 3D Visualizations")
                
                if len(numeric_columns) >= 3:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        x_3d = st.selectbox("X-axis", numeric_columns, key="3d_x")
                        y_3d = st.selectbox("Y-axis", [col for col in numeric_columns if col != x_3d], key="3d_y")
                        z_3d = st.selectbox("Z-axis", [col for col in numeric_columns if col not in [x_3d, y_3d]], key="3d_z")
                        color_3d = st.selectbox("Color by (optional)", [None] + categorical_columns, key="3d_color")
                        
                        if st.button("Create 3D Scatter Plot"):
                            fig = create_3d_scatter(df, x_3d, y_3d, z_3d, color_3d)
                            if fig:
                                st.session_state.gallery_plots[f"3D_{x_3d}_{y_3d}_{z_3d}_{int(time.time())}"] = {
                                    'fig': fig,
                                    'type': '3d_scatter',
                                    'params': {'x': x_3d, 'y': y_3d, 'z': z_3d, 'color': color_3d}
                                }
                    
                    with col2:
                        plot_3d_keys = [k for k in st.session_state.gallery_plots.keys() if '3D_' in k]
                        if plot_3d_keys:
                            latest_3d = plot_3d_keys[-1]
                            plot_data = st.session_state.gallery_plots[latest_3d]
                            st.plotly_chart(plot_data['fig'], use_container_width=True, key="display_3d_plot")
                        else:
                            st.info("Create a 3D plot to see it here")
                else:
                    st.warning("Need at least 3 numeric columns for 3D visualization")
            
            with viz_tabs[1]:  # Animated Plots
                st.subheader("🎬 Animated Visualizations")
                
                if len(numeric_columns) >= 2:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        x_anim = st.selectbox("X-axis", numeric_columns, key="anim_x")
                        y_anim = st.selectbox("Y-axis", [col for col in numeric_columns if col != x_anim], key="anim_y")
                        
                        anim_options = categorical_columns + [col for col in numeric_columns if col not in [x_anim, y_anim]]
                        if anim_options:
                            anim_frame = st.selectbox("Animation frame", anim_options, key="anim_frame")
                            
                            if st.button("Create Animated Plot"):
                                fig = create_animated_plot(df, x_anim, y_anim, anim_frame)
                                if fig:
                                    st.session_state.gallery_plots[f"Anim_{x_anim}_{y_anim}_{int(time.time())}"] = {
                                        'fig': fig,
                                        'type': 'animated',
                                        'params': {'x': x_anim, 'y': y_anim, 'frame': anim_frame}
                                    }
                        else:
                            st.warning("Need categorical columns for animation")
                    
                    with col2:
                        plot_anim_keys = [k for k in st.session_state.gallery_plots.keys() if 'Anim_' in k]
                        if plot_anim_keys:
                            latest_anim = plot_anim_keys[-1]
                            plot_data = st.session_state.gallery_plots[latest_anim]
                            st.plotly_chart(plot_data['fig'], use_container_width=True, key="display_anim_plot")
                        else:
                            st.info("Create an animated plot to see it here")
                else:
                    st.warning("Need at least 2 numeric columns for animated visualization")
            
            with viz_tabs[2]:  # Heatmaps
                st.subheader("🔥 Advanced Heatmaps")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if len(numeric_columns) > 1:
                        corr_method = st.selectbox("Correlation method", ['pearson', 'spearman', 'kendall'], key="heatmap_method")
                        
                        if st.button("Create Enhanced Heatmap"):
                            fig = create_enhanced_heatmap(df, corr_method)
                            if fig:
                                st.session_state.gallery_plots[f"Heatmap_{corr_method}_{int(time.time())}"] = {
                                    'fig': fig,
                                    'type': 'heatmap',
                                    'params': {'method': corr_method}
                                }
                
                with col2:
                    heatmap_keys = [k for k in st.session_state.gallery_plots.keys() if 'Heatmap_' in k]
                    if heatmap_keys:
                        latest_heatmap = heatmap_keys[-1]
                        plot_data = st.session_state.gallery_plots[latest_heatmap]
                        st.plotly_chart(plot_data['fig'], use_container_width=True, key="display_heatmap")
                    else:
                        st.info("Create a heatmap to see it here")
        
        with tab8:
            from user_guide import render_user_guide
            render_user_guide()

if __name__ == "__main__":
    main()