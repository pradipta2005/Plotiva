"""
Premium Analytics Module
Advanced Statistical Analysis and Machine Learning
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

class PremiumAnalytics:
    """Advanced analytics and statistical testing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        summary = {
            'numeric_summary': {},
            'categorical_summary': {},
            'distribution_tests': {},
            'outlier_analysis': {}
        }
        
        # Numeric summary with advanced statistics
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                summary['numeric_summary'][col] = {
                    'count': len(data),
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75),
                    'iqr': data.quantile(0.75) - data.quantile(0.25),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'cv': data.std() / data.mean() if data.mean() != 0 else 0
                }
                
                # Normality test
                if len(data) >= 8:
                    shapiro_stat, shapiro_p = stats.shapiro(data[:5000])  # Limit for performance
                    summary['distribution_tests'][col] = {
                        'shapiro_stat': shapiro_stat,
                        'shapiro_p': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    }
                
                # Outlier detection
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
                summary['outlier_analysis'][col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(data) * 100,
                    'outlier_values': outliers.tolist()[:10]  # Limit to first 10
                }
        
        # Categorical summary
        for col in categorical_cols:
            data = df[col].dropna()
            if len(data) > 0:
                value_counts = data.value_counts()
                summary['categorical_summary'][col] = {
                    'unique_count': data.nunique(),
                    'most_frequent': value_counts.index[0],
                    'most_frequent_count': value_counts.iloc[0],
                    'least_frequent': value_counts.index[-1],
                    'least_frequent_count': value_counts.iloc[-1],
                    'entropy': stats.entropy(value_counts.values),
                    'value_counts': value_counts.head(10).to_dict()
                }
        
        return summary
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Advanced correlation analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        # Pearson correlation
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Spearman correlation (rank-based)
        spearman_corr = numeric_df.corr(method='spearman')
        
        # Kendall correlation
        kendall_corr = numeric_df.corr(method='kendall')
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                corr_val = pearson_corr.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        'var1': pearson_corr.columns[i],
                        'var2': pearson_corr.columns[j],
                        'correlation': corr_val,
                        'strength': 'Very Strong' if abs(corr_val) > 0.9 else 'Strong'
                    })
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'kendall_correlation': kendall_corr,
            'strong_correlations': strong_correlations
        }
    
    def hypothesis_testing(self, df: pd.DataFrame, group_col: str, 
                          value_col: str, test_type: str = 'auto') -> Dict[str, Any]:
        """Perform hypothesis testing"""
        
        groups = df[group_col].unique()
        group_data = [df[df[group_col] == group][value_col].dropna() for group in groups]
        
        # Remove empty groups
        group_data = [data for data in group_data if len(data) > 0]
        groups = [group for i, group in enumerate(groups) if len(group_data[i]) > 0]
        
        if len(group_data) < 2:
            return {'error': 'Need at least 2 groups with data for hypothesis testing'}
        
        results = {}
        
        if test_type == 'auto':
            # Determine appropriate test based on data
            if len(group_data) == 2:
                # Two groups - t-test or Mann-Whitney U
                # Check normality
                normal_p_values = []
                for data in group_data:
                    if len(data) >= 8:
                        _, p_val = stats.shapiro(data[:5000])
                        normal_p_values.append(p_val)
                
                if all(p > 0.05 for p in normal_p_values):
                    # Both groups are normal - use t-test
                    stat, p_val = stats.ttest_ind(group_data[0], group_data[1])
                    results['test_used'] = 'Independent t-test'
                    results['assumption'] = 'Normal distribution'
                else:
                    # Non-normal - use Mann-Whitney U
                    stat, p_val = stats.mannwhitneyu(group_data[0], group_data[1])
                    results['test_used'] = 'Mann-Whitney U test'
                    results['assumption'] = 'Non-normal distribution'
            
            else:
                # Multiple groups - ANOVA or Kruskal-Wallis
                # Check normality for all groups
                normal_p_values = []
                for data in group_data:
                    if len(data) >= 8:
                        _, p_val = stats.shapiro(data[:5000])
                        normal_p_values.append(p_val)
                
                if all(p > 0.05 for p in normal_p_values):
                    # All groups are normal - use ANOVA
                    stat, p_val = stats.f_oneway(*group_data)
                    results['test_used'] = 'One-way ANOVA'
                    results['assumption'] = 'Normal distribution'
                else:
                    # Non-normal - use Kruskal-Wallis
                    stat, p_val = stats.kruskal(*group_data)
                    results['test_used'] = 'Kruskal-Wallis test'
                    results['assumption'] = 'Non-normal distribution'
        
        results.update({
            'statistic': stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'groups': groups,
            'group_means': [data.mean() for data in group_data],
            'group_stds': [data.std() for data in group_data],
            'effect_size': self._calculate_effect_size(group_data)
        })
        
        return results
    
    def _calculate_effect_size(self, group_data: List[np.ndarray]) -> float:
        """Calculate Cohen's d for effect size"""
        if len(group_data) == 2:
            # Cohen's d for two groups
            mean1, mean2 = group_data[0].mean(), group_data[1].mean()
            std1, std2 = group_data[0].std(), group_data[1].std()
            n1, n2 = len(group_data[0]), len(group_data[1])
            
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            # Prevent division by zero
            if pooled_std == 0:
                return 0.0
            cohens_d = (mean1 - mean2) / pooled_std
            return abs(cohens_d)
        else:
            # Eta-squared for multiple groups
            all_data = np.concatenate(group_data)
            grand_mean = all_data.mean()
            
            ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in group_data)
            ss_total = sum((x - grand_mean)**2 for x in all_data)
            
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            return eta_squared
    
    def advanced_clustering(self, df: pd.DataFrame, features: List[str],
                          algorithm: str = 'kmeans', n_clusters: int = 3,
                          **kwargs) -> Dict[str, Any]:
        """Perform advanced clustering analysis"""
        
        # Prepare data
        X = df[features].select_dtypes(include=[np.number]).dropna()
        
        if X.empty:
            return {'error': 'No numeric data available for clustering'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            return {'error': f'Unknown clustering algorithm: {algorithm}'}
        
        clusters = model.fit_predict(X_scaled)
        
        # Calculate metrics
        n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
        
        metrics = {}
        if n_clusters_found > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(X_scaled, clusters)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, clusters)
            except Exception as e:
                # Clustering metrics calculation failed, continue without them
                pass
        
        # Cluster analysis
        cluster_analysis = {}
        for cluster_id in set(clusters):
            if cluster_id != -1:  # Exclude noise points in DBSCAN
                cluster_mask = clusters == cluster_id
                cluster_data = X[cluster_mask]
                
                cluster_analysis[f'Cluster_{cluster_id}'] = {
                    'size': cluster_mask.sum(),
                    'percentage': cluster_mask.sum() / len(clusters) * 100,
                    'means': cluster_data.mean().to_dict(),
                    'stds': cluster_data.std().to_dict()
                }
        
        return {
            'clusters': clusters,
            'n_clusters': n_clusters_found,
            'algorithm': algorithm,
            'features': features,
            'metrics': metrics,
            'cluster_analysis': cluster_analysis,
            'scaled_data': X_scaled,
            'original_data': X
        }
    
    def dimensionality_reduction(self, df: pd.DataFrame, features: List[str],
                               method: str = 'pca', n_components: int = 2) -> Dict[str, Any]:
        """Perform dimensionality reduction"""
        
        X = df[features].select_dtypes(include=[np.number]).dropna()
        
        if X.empty:
            return {'error': 'No numeric data available for dimensionality reduction'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X_scaled)
            
            # Calculate explained variance
            explained_variance = reducer.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Get component loadings
            components = pd.DataFrame(
                reducer.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=features
            )
            
            results = {
                'method': 'PCA',
                'reduced_data': X_reduced,
                'explained_variance': explained_variance,
                'cumulative_variance': cumulative_variance,
                'components': components,
                'feature_names': [f'PC{i+1}' for i in range(n_components)]
            }
        
        elif method == 'tsne':
            perplexity = min(30, len(X_scaled) - 1)
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
            X_reduced = reducer.fit_transform(X_scaled)
            
            results = {
                'method': 't-SNE',
                'reduced_data': X_reduced,
                'feature_names': [f'tSNE{i+1}' for i in range(n_components)]
            }
        
        else:
            return {'error': f'Unknown dimensionality reduction method: {method}'}
        
        return results
    
    def anomaly_detection(self, df: pd.DataFrame, features: List[str],
                         method: str = 'isolation_forest', **kwargs) -> Dict[str, Any]:
        """Detect anomalies in the data"""
        
        X = df[features].select_dtypes(include=[np.number]).dropna()
        
        if X.empty:
            return {'error': 'No numeric data available for anomaly detection'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'isolation_forest':
            contamination = kwargs.get('contamination', 0.1)
            model = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = model.fit_predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)
        
        elif method == 'statistical':
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(X_scaled, axis=0))
            threshold = kwargs.get('threshold', 3)
            anomaly_labels = np.where((z_scores > threshold).any(axis=1), -1, 1)
            anomaly_scores = np.max(z_scores, axis=1)
        
        else:
            return {'error': f'Unknown anomaly detection method: {method}'}
        
        # Analyze anomalies
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        normal_indices = np.where(anomaly_labels == 1)[0]
        
        anomaly_analysis = {
            'total_anomalies': len(anomaly_indices),
            'anomaly_percentage': len(anomaly_indices) / len(X) * 100,
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_scores': anomaly_scores[anomaly_indices].tolist(),
            'top_anomalies': X.iloc[anomaly_indices].head(10).to_dict('records')
        }
        
        return {
            'method': method,
            'anomaly_labels': anomaly_labels,
            'anomaly_scores': anomaly_scores,
            'analysis': anomaly_analysis,
            'features': features
        }
    
    def time_series_analysis(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Perform time series analysis"""
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        ts_data = df.set_index(date_col)[value_col].sort_index()
        
        # Remove missing values
        ts_data = ts_data.dropna()
        
        if len(ts_data) < 10:
            return {'error': 'Need at least 10 data points for time series analysis'}
        
        results = {}
        
        # Basic statistics
        results['basic_stats'] = {
            'count': len(ts_data),
            'mean': ts_data.mean(),
            'std': ts_data.std(),
            'min': ts_data.min(),
            'max': ts_data.max(),
            'trend': 'increasing' if ts_data.iloc[-1] > ts_data.iloc[0] else 'decreasing'
        }
        
        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(ts_data)
            results['stationarity'] = {
                'adf_statistic': adf_result[0],
                'p_value': adf_result[1],
                'is_stationary': adf_result[1] < 0.05,
                'critical_values': adf_result[4]
            }
        except ImportError:
            results['stationarity'] = {'error': 'statsmodels not available'}
        
        # Seasonality detection
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            if len(ts_data) >= 24:  # Need enough data for decomposition
                decomposition = seasonal_decompose(ts_data, model='additive', period=min(12, len(ts_data)//2))
                results['seasonality'] = {
                    'trend': decomposition.trend.dropna().tolist(),
                    'seasonal': decomposition.seasonal.dropna().tolist(),
                    'residual': decomposition.resid.dropna().tolist()
                }
        except ImportError:
            results['seasonality'] = {'error': 'statsmodels not available'}
        
        # Autocorrelation
        autocorr = [ts_data.autocorr(lag=i) for i in range(1, min(21, len(ts_data)//2))]
        results['autocorrelation'] = {
            'lags': list(range(1, len(autocorr) + 1)),
            'values': autocorr
        }
        
        return results
    
    def feature_importance_analysis(self, df: pd.DataFrame, target_col: str,
                                  feature_cols: List[str], method: str = 'random_forest') -> Dict[str, Any]:
        """Analyze feature importance"""
        
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.inspection import permutation_importance
        
        X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = df[target_col].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10:
            return {'error': 'Need at least 10 samples for feature importance analysis'}
        
        # Determine if regression or classification
        is_classification = y.dtype == 'object' or y.nunique() < 10
        
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Fit model
        model.fit(X, y)
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Permutation importance
        try:
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            perm_df = pd.DataFrame({
                'feature': X.columns,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
        except Exception as e:
            # Permutation importance calculation failed
            perm_df = None
        
        return {
            'method': method,
            'task_type': 'classification' if is_classification else 'regression',
            'feature_importance': feature_importance,
            'permutation_importance': perm_df,
            'model_score': model.score(X, y)
        }

def create_analytics_dashboard(df: pd.DataFrame, analytics_type: str, **kwargs) -> Dict[str, Any]:
    """Create analytics dashboard with results and visualizations"""
    
    analytics = PremiumAnalytics()
    
    if analytics_type == 'statistical_summary':
        return analytics.statistical_summary(df)
    elif analytics_type == 'correlation_analysis':
        return analytics.correlation_analysis(df)
    elif analytics_type == 'hypothesis_testing':
        return analytics.hypothesis_testing(df, **kwargs)
    elif analytics_type == 'clustering':
        return analytics.advanced_clustering(df, **kwargs)
    elif analytics_type == 'dimensionality_reduction':
        return analytics.dimensionality_reduction(df, **kwargs)
    elif analytics_type == 'anomaly_detection':
        return analytics.anomaly_detection(df, **kwargs)
    elif analytics_type == 'time_series':
        return analytics.time_series_analysis(df, **kwargs)
    elif analytics_type == 'feature_importance':
        return analytics.feature_importance_analysis(df, **kwargs)
    else:
        return {'error': f'Unknown analytics type: {analytics_type}'}