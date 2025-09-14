"""
Premium Data Analysis Platform - Top 1% Developer Quality
Advanced Interactive Dashboard with Premium Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import premium modules
from premium_config import (
    PREMIUM_COLOR_PALETTES, PREMIUM_THEMES, PREMIUM_CHART_TYPES, 
    PREMIUM_UI_COMPONENTS, PREMIUM_CSS
)
from premium_plots import PremiumPlotGenerator, create_premium_plot
from premium_analytics import PremiumAnalytics, create_analytics_dashboard
from utils import (
    validate_dataframe, get_column_info, calculate_data_quality_metrics,
    get_quality_label, generate_sample_data, get_data_summary
)

# Configure Streamlit page
st.set_page_config(
    page_title="Premium Data Analysis Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium CSS
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

class PremiumDataApp:
    """Premium Data Analysis Application"""
    
    def __init__(self):
        self.initialize_session_state()
        self.plot_generator = PremiumPlotGenerator()
        self.analytics = PremiumAnalytics()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'uploaded_file_name' not in st.session_state:
            st.session_state.uploaded_file_name = None
        if 'working_data' not in st.session_state:
            st.session_state.working_data = pd.DataFrame()
        if 'original_data' not in st.session_state:
            st.session_state.original_data = pd.DataFrame()
        if 'selected_theme' not in st.session_state:
            st.session_state.selected_theme = 'executive'
        if 'selected_palette' not in st.session_state:
            st.session_state.selected_palette = 'aurora'
        if 'plot_history' not in st.session_state:
            st.session_state.plot_history = []
        if 'analytics_results' not in st.session_state:
            st.session_state.analytics_results = {}
    
    def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            else:
                st.error("Unsupported file format")
                return None
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace(' ', '_')
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def render_header(self):
        """Render premium header"""
        st.markdown("""
        <div class="main-header">
            🚀 Premium Data Analysis Platform
        </div>
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #6B7280; font-family: 'Inter', sans-serif;">
                Professional-grade analytics with stunning visualizations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render premium sidebar"""
        with st.sidebar:
            st.markdown('<div class="sidebar-premium">', unsafe_allow_html=True)
            
            # Theme Selection
            st.markdown("### 🎨 Customization")
            theme_col, palette_col = st.columns(2)
            
            with theme_col:
                st.session_state.selected_theme = st.selectbox(
                    "Theme",
                    options=list(PREMIUM_THEMES.keys()),
                    index=list(PREMIUM_THEMES.keys()).index(st.session_state.selected_theme)
                )
            
            with palette_col:
                st.session_state.selected_palette = st.selectbox(
                    "Palette",
                    options=list(PREMIUM_COLOR_PALETTES.keys()),
                    index=list(PREMIUM_COLOR_PALETTES.keys()).index(st.session_state.selected_palette)
                )
            
            # Color Preview
            colors = PREMIUM_COLOR_PALETTES[st.session_state.selected_palette]
            color_preview = "".join([f'<div style="display:inline-block;width:20px;height:20px;background:{color};margin:2px;border-radius:3px;"></div>' for color in colors[:7]])
            st.markdown(f'<div style="margin-bottom:1rem;">{color_preview}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Data Upload
            st.markdown("### 📁 Data Upload")
            uploaded_file = st.file_uploader(
                "Choose your data file",
                type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
                help="Upload CSV, Excel, JSON, or Parquet files"
            )
            
            # Sample Data Options
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
            
            # Handle file upload
            if uploaded_file is not None:
                if st.session_state.uploaded_file_name != uploaded_file.name:
                    with st.spinner("🔄 Loading your data..."):
                        df = self.load_data(uploaded_file)
                        if df is not None:
                            st.session_state.uploaded_file_name = uploaded_file.name
                            st.session_state.original_data = df.copy()
                            st.session_state.working_data = df.copy()
                            st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
            
            # Data Info Panel
            if not st.session_state.working_data.empty:
                st.markdown("---")
                st.markdown("### 📊 Data Overview")
                
                df = st.session_state.working_data
                col_info = get_column_info(df)
                quality_metrics = calculate_data_quality_metrics(df)
                
                # Metrics cards
                metrics_html = f"""
                <div class="metric-card">
                    <h3>Rows</h3>
                    <h2 style="color: #667eea">{len(df):,}</h2>
                </div>
                <div class="metric-card" style="margin-top: 1rem;">
                    <h3>Columns</h3>
                    <h2 style="color: #764ba2">{len(df.columns)}</h2>
                </div>
                <div class="metric-card" style="margin-top: 1rem;">
                    <h3>Data Quality</h3>
                    <h2 style="color: {'#10B981' if quality_metrics['overall_score'] > 0.8 else '#F59E0B' if quality_metrics['overall_score'] > 0.6 else '#EF4444'}">{quality_metrics['overall_score']:.1%}</h2>
                </div>
                """
                st.markdown(metrics_html, unsafe_allow_html=True)
                
                # Column type breakdown
                st.markdown("**Column Types:**")
                st.write(f"📊 Numeric: {len(col_info['numeric_columns'])}")
                st.write(f"📝 Categorical: {len(col_info['categorical_columns'])}")
                st.write(f"📅 DateTime: {len(col_info['datetime_columns'])}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def render_main_content(self):
        """Render main content area"""
        if st.session_state.working_data.empty:
            self.render_welcome_screen()
        else:
            self.render_analysis_interface()
    
    def render_welcome_screen(self):
        """Render welcome screen when no data is loaded"""
        st.markdown("""
        <div class="premium-card" style="text-align: center; padding: 4rem 2rem;">
            <h2 style="color: #667eea; margin-bottom: 2rem;">Welcome to Premium Analytics</h2>
            <p style="font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem;">
                Upload your data or try our sample datasets to get started with professional-grade analysis
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div class="feature-highlight">✨ Interactive Visualizations</div>
                <div class="feature-highlight">🔬 Advanced Analytics</div>
                <div class="feature-highlight">🎨 Customizable Themes</div>
                <div class="feature-highlight">📊 Statistical Testing</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="premium-card">
                <h3>🎯 Smart Analytics</h3>
                <p>Automated statistical analysis, hypothesis testing, and machine learning insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="premium-card">
                <h3>🎨 Premium Visuals</h3>
                <p>Beautiful, interactive charts with professional color schemes and animations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="premium-card">
                <h3>⚡ High Performance</h3>
                <p>Optimized for large datasets with real-time updates and smooth interactions</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_analysis_interface(self):
        """Render main analysis interface"""
        df = st.session_state.working_data
        
        # Create tabs
        tab_names = ["📊 Visualizations", "🔬 Analytics", "📈 Advanced Plots", "🎯 Insights"]
        tabs = st.tabs(tab_names)
        
        with tabs[0]:
            self.render_visualization_tab(df)
        
        with tabs[1]:
            self.render_analytics_tab(df)
        
        with tabs[2]:
            self.render_advanced_plots_tab(df)
        
        with tabs[3]:
            self.render_insights_tab(df)
    
    def render_visualization_tab(self, df: pd.DataFrame):
        """Render visualization tab"""
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        col_info = get_column_info(df)
        numeric_cols = col_info['numeric_columns']
        categorical_cols = col_info['categorical_columns']
        all_cols = df.columns.tolist()
        
        # Plot type selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🎨 Chart Configuration")
            
            chart_type = st.selectbox(
                "Chart Type",
                options=['scatter', 'line', 'bar', 'histogram', 'box', 'heatmap'],
                format_func=lambda x: {
                    'scatter': '📊 Scatter Plot',
                    'line': '📈 Line Chart', 
                    'bar': '📊 Bar Chart',
                    'histogram': '📊 Histogram',
                    'box': '📦 Box Plot',
                    'heatmap': '🔥 Heatmap'
                }[x]
            )
            
            # Dynamic column selection based on chart type
            if chart_type == 'scatter':
                x_col = st.selectbox("X-axis", numeric_cols)
                y_col = st.selectbox("Y-axis", [col for col in numeric_cols if col != x_col])
                color_col = st.selectbox("Color by", ['None'] + categorical_cols)
                size_col = st.selectbox("Size by", ['None'] + numeric_cols)
                
            elif chart_type == 'line':
                x_col = st.selectbox("X-axis", all_cols)
                y_col = st.selectbox("Y-axis", numeric_cols)
                color_col = st.selectbox("Group by", ['None'] + categorical_cols)
                
            elif chart_type == 'bar':
                x_col = st.selectbox("Category", categorical_cols)
                y_col = st.selectbox("Value", numeric_cols)
                
            elif chart_type == 'histogram':
                x_col = st.selectbox("Column", numeric_cols)
                
            elif chart_type == 'box':
                x_col = st.selectbox("Category", categorical_cols)
                y_col = st.selectbox("Value", numeric_cols)
                
            elif chart_type == 'heatmap':
                st.info("Correlation heatmap of all numeric columns")
            
            # Advanced options
            st.markdown("### ⚙️ Advanced Options")
            add_trendline = st.checkbox("Add Trendline", value=False)
            add_marginals = st.checkbox("Add Marginal Plots", value=False)
            smooth_line = st.checkbox("Smooth Lines", value=True)
        
        with col2:
            st.markdown("### 📊 Visualization")
            
            try:
                # Create plot based on selection
                if chart_type == 'scatter' and len(numeric_cols) >= 2:
                    fig = self.plot_generator.advanced_scatter_plot(
                        df, x_col, y_col,
                        color_col=color_col if color_col != 'None' else None,
                        size_col=size_col if size_col != 'None' else None,
                        palette=st.session_state.selected_palette,
                        add_regression=add_trendline,
                        add_marginals=add_marginals
                    )
                
                elif chart_type == 'line':
                    fig = self.plot_generator.animated_line_chart(
                        df, x_col, y_col,
                        color_col=color_col if color_col != 'None' else None,
                        palette=st.session_state.selected_palette,
                        add_trend=add_trendline,
                        smooth=smooth_line
                    )
                
                elif chart_type == 'heatmap':
                    fig = self.plot_generator.advanced_heatmap(
                        df, palette=st.session_state.selected_palette,
                        cluster=True, annotate=True
                    )
                
                else:
                    # Fallback to basic plotly express charts
                    colors = PREMIUM_COLOR_PALETTES[st.session_state.selected_palette]
                    
                    if chart_type == 'bar':
                        fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=colors)
                    elif chart_type == 'histogram':
                        fig = px.histogram(df, x=x_col, color_discrete_sequence=colors)
                    elif chart_type == 'box':
                        fig = px.box(df, x=x_col, y=y_col, color_discrete_sequence=colors)
                    else:
                        fig = go.Figure()
                        fig.add_annotation(text="Chart configuration not supported", 
                                         xref="paper", yref="paper", x=0.5, y=0.5)
                
                # Apply premium styling
                fig.update_layout(
                    font_family="Inter, sans-serif",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save to history
                if st.button("💾 Save to History"):
                    st.session_state.plot_history.append({
                        'type': chart_type,
                        'config': locals(),
                        'timestamp': pd.Timestamp.now()
                    })
                    st.success("Plot saved to history!")
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_analytics_tab(self, df: pd.DataFrame):
        """Render analytics tab"""
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🔬 Analytics Suite")
            
            analysis_type = st.selectbox(
                "Analysis Type",
                options=[
                    'statistical_summary',
                    'correlation_analysis', 
                    'hypothesis_testing',
                    'clustering',
                    'anomaly_detection',
                    'feature_importance'
                ],
                format_func=lambda x: {
                    'statistical_summary': '📊 Statistical Summary',
                    'correlation_analysis': '🔗 Correlation Analysis',
                    'hypothesis_testing': '🧪 Hypothesis Testing',
                    'clustering': '🎯 Clustering Analysis',
                    'anomaly_detection': '🚨 Anomaly Detection',
                    'feature_importance': '⭐ Feature Importance'
                }[x]
            )
            
            # Dynamic parameter selection
            col_info = get_column_info(df)
            numeric_cols = col_info['numeric_columns']
            categorical_cols = col_info['categorical_columns']
            
            params = {}
            
            if analysis_type == 'hypothesis_testing':
                params['group_col'] = st.selectbox("Group Column", categorical_cols)
                params['value_col'] = st.selectbox("Value Column", numeric_cols)
                
            elif analysis_type == 'clustering':
                params['features'] = st.multiselect("Features", numeric_cols, default=numeric_cols[:3])
                params['n_clusters'] = st.slider("Number of Clusters", 2, 10, 3)
                params['algorithm'] = st.selectbox("Algorithm", ['kmeans', 'dbscan', 'hierarchical'])
                
            elif analysis_type == 'anomaly_detection':
                params['features'] = st.multiselect("Features", numeric_cols, default=numeric_cols[:3])
                params['method'] = st.selectbox("Method", ['isolation_forest', 'statistical'])
                
            elif analysis_type == 'feature_importance':
                params['target_col'] = st.selectbox("Target Column", numeric_cols + categorical_cols)
                params['feature_cols'] = st.multiselect("Feature Columns", 
                                                       [col for col in numeric_cols if col != params.get('target_col')])
            
            if st.button("🚀 Run Analysis"):
                with st.spinner("Running analysis..."):
                    try:
                        results = create_analytics_dashboard(df, analysis_type, **params)
                        st.session_state.analytics_results[analysis_type] = results
                        st.success("Analysis completed!")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
        
        with col2:
            st.markdown("### 📊 Results")
            
            if analysis_type in st.session_state.analytics_results:
                results = st.session_state.analytics_results[analysis_type]
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    self.display_analytics_results(results, analysis_type)
            else:
                st.info("Run an analysis to see results here")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_advanced_plots_tab(self, df: pd.DataFrame):
        """Render advanced plots tab"""
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        st.markdown("### 🎨 Premium Chart Gallery")
        
        # Chart gallery
        chart_options = {
            'violin_plot': '🎻 Violin Plot',
            'sunburst': '☀️ Sunburst Chart',
            'treemap': '🌳 Treemap',
            'radar_chart': '📡 Radar Chart',
            'waterfall': '💧 Waterfall Chart'
        }
        
        selected_chart = st.selectbox("Select Premium Chart", 
                                    options=list(chart_options.keys()),
                                    format_func=lambda x: chart_options[x])
        
        col_info = get_column_info(df)
        numeric_cols = col_info['numeric_columns']
        categorical_cols = col_info['categorical_columns']
        
        # Chart-specific configuration
        if selected_chart == 'violin_plot' and len(categorical_cols) > 0 and len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Category Column", categorical_cols)
            with col2:
                y_col = st.selectbox("Value Column", numeric_cols)
            
            if st.button("Create Violin Plot"):
                fig = self.plot_generator.violin_plot(df, x_col, y_col, 
                                                    palette=st.session_state.selected_palette)
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_chart == 'radar_chart' and len(numeric_cols) >= 3:
            categories = st.multiselect("Select Metrics", numeric_cols, default=numeric_cols[:5])
            group_col = st.selectbox("Group By", ['None'] + categorical_cols)
            
            if st.button("Create Radar Chart") and len(categories) >= 3:
                fig = self.plot_generator.radar_chart(df, categories,
                                                    group_col=group_col if group_col != 'None' else None,
                                                    palette=st.session_state.selected_palette)
                st.plotly_chart(fig, use_container_width=True)
        
        elif selected_chart == 'waterfall' and len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Category Column", df.columns.tolist())
            with col2:
                y_col = st.selectbox("Value Column", numeric_cols)
            
            if st.button("Create Waterfall Chart"):
                fig = self.plot_generator.waterfall_chart(df, x_col, y_col,
                                                        palette=st.session_state.selected_palette)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info(f"Configure the {chart_options[selected_chart]} above and click create to generate the visualization.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_insights_tab(self, df: pd.DataFrame):
        """Render insights and recommendations tab"""
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        
        st.markdown("### 🎯 Smart Insights")
        
        # Generate automatic insights
        summary = get_data_summary(df)
        
        if summary:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Data Overview")
                st.metric("Total Records", f"{summary['basic_info']['rows']:,}")
                st.metric("Features", summary['basic_info']['columns'])
                st.metric("Data Quality", f"{summary['quality_metrics']['overall_score']:.1%}")
                
                # Missing data insights
                if summary['missing_data']['total_missing'] > 0:
                    st.warning(f"⚠️ {summary['missing_data']['total_missing']:,} missing values detected "
                             f"({summary['missing_data']['missing_percentage']:.1f}%)")
                else:
                    st.success("✅ No missing values detected")
                
                # Duplicate insights
                if summary['duplicates']['duplicate_rows'] > 0:
                    st.warning(f"⚠️ {summary['duplicates']['duplicate_rows']:,} duplicate rows found "
                             f"({summary['duplicates']['duplicate_percentage']:.1f}%)")
                else:
                    st.success("✅ No duplicate rows detected")
            
            with col2:
                st.markdown("#### 🔍 Column Analysis")
                
                # Column type breakdown
                type_data = summary['data_types']
                fig = px.pie(
                    values=list(type_data.values()),
                    names=list(type_data.keys()),
                    title="Column Types Distribution",
                    color_discrete_sequence=PREMIUM_COLOR_PALETTES[st.session_state.selected_palette]
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("#### 💡 Recommendations")
        
        recommendations = []
        
        if summary and summary['quality_metrics']['overall_score'] < 0.8:
            recommendations.append("🔧 Consider data cleaning to improve quality score")
        
        if summary and summary['missing_data']['missing_percentage'] > 10:
            recommendations.append("📝 Address missing values through imputation or removal")
        
        if summary and summary['duplicates']['duplicate_percentage'] > 5:
            recommendations.append("🗑️ Remove duplicate rows to improve data integrity")
        
        col_info = get_column_info(df)
        if len(col_info['numeric_columns']) >= 2:
            recommendations.append("📊 Explore correlations between numeric variables")
        
        if len(col_info['categorical_columns']) > 0 and len(col_info['numeric_columns']) > 0:
            recommendations.append("📈 Create group comparisons using categorical variables")
        
        if len(df) > 1000:
            recommendations.append("🎯 Consider clustering analysis to find data patterns")
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        if not recommendations:
            st.success("🎉 Your data looks great! Ready for advanced analysis.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def display_analytics_results(self, results: Dict[str, Any], analysis_type: str):
        """Display analytics results"""
        
        if analysis_type == 'statistical_summary':
            if 'numeric_summary' in results:
                st.markdown("#### 📊 Numeric Variables")
                for col, stats in results['numeric_summary'].items():
                    with st.expander(f"📈 {col}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean", f"{stats['mean']:.3f}")
                            st.metric("Std Dev", f"{stats['std']:.3f}")
                        with col2:
                            st.metric("Min", f"{stats['min']:.3f}")
                            st.metric("Max", f"{stats['max']:.3f}")
                        with col3:
                            st.metric("Skewness", f"{stats['skewness']:.3f}")
                            st.metric("Kurtosis", f"{stats['kurtosis']:.3f}")
        
        elif analysis_type == 'correlation_analysis':
            if 'strong_correlations' in results:
                st.markdown("#### 🔗 Strong Correlations")
                for corr in results['strong_correlations']:
                    strength_color = "#EF4444" if abs(corr['correlation']) > 0.9 else "#F59E0B"
                    st.markdown(f"**{corr['var1']}** ↔ **{corr['var2']}**: "
                              f"<span style='color:{strength_color}'>{corr['correlation']:.3f}</span> "
                              f"({corr['strength']})", unsafe_allow_html=True)
        
        elif analysis_type == 'hypothesis_testing':
            st.markdown("#### 🧪 Hypothesis Test Results")
            st.markdown(f"**Test Used:** {results['test_used']}")
            st.markdown(f"**Test Statistic:** {results['statistic']:.4f}")
            st.markdown(f"**P-value:** {results['p_value']:.4f}")
            
            if results['significant']:
                st.success("✅ Statistically significant difference detected (p < 0.05)")
            else:
                st.info("ℹ️ No statistically significant difference found (p ≥ 0.05)")
            
            st.markdown(f"**Effect Size:** {results['effect_size']:.4f}")
        
        elif analysis_type == 'clustering':
            st.markdown("#### 🎯 Clustering Results")
            st.markdown(f"**Algorithm:** {results['algorithm']}")
            st.markdown(f"**Clusters Found:** {results['n_clusters']}")
            
            if 'metrics' in results and results['metrics']:
                if 'silhouette_score' in results['metrics']:
                    score = results['metrics']['silhouette_score']
                    color = "#10B981" if score > 0.5 else "#F59E0B" if score > 0.25 else "#EF4444"
                    st.markdown(f"**Silhouette Score:** <span style='color:{color}'>{score:.3f}</span>", 
                              unsafe_allow_html=True)
        
        else:
            st.json(results)
    
    def run(self):
        """Run the premium application"""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()

# Run the application
if __name__ == "__main__":
    app = PremiumDataApp()
    app.run()