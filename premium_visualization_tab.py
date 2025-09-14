"""
Premium Visualization Tab - Colorful and Interactive Charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

try:
    from premium_config import PREMIUM_COLOR_PALETTES
    from premium_plots import PremiumPlotGenerator
    PREMIUM_AVAILABLE = True
except ImportError:
    PREMIUM_AVAILABLE = False

def get_premium_colors(palette='aurora'):
    if PREMIUM_AVAILABLE and palette in PREMIUM_COLOR_PALETTES:
        return PREMIUM_COLOR_PALETTES[palette]
    return ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']

def render_premium_visualization_tab(df):
    """Premium visualization tab with colorful plots"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem; color: white; text-align: center;">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">🎨 Premium Visualizations</h1>
        <p style="font-size: 1.2rem;">Create stunning, interactive charts with professional styling</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Always use current working data (filtered data)
    current_df = st.session_state.get('working_data', df)
    if current_df.empty:
        st.warning("No data available for visualization")
        return
    
    # Premium filter status with live updates
    if st.session_state.get('filter_values'):
        filter_count = len(st.session_state.filter_values)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10B981, #059669); padding: 1rem; border-radius: 10px; 
                    color: white; margin-bottom: 1rem; text-align: center; animation: pulse 2s infinite;">
            🎯 <strong>Live Filtered View</strong> • {len(current_df):,} rows • {filter_count} filter{'s' if filter_count != 1 else ''} active
            <br><small>Charts auto-update in real-time</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #6366F1, #4F46E5); padding: 1rem; border-radius: 10px; 
                    color: white; margin-bottom: 1rem; text-align: center;">
            📊 <strong>Complete Dataset</strong> • {len(current_df):,} total rows • No filters applied
        </div>
        """, unsafe_allow_html=True)
    
    df = current_df
    
    # Get column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Color palette selector
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        palette = st.selectbox(
            "🎨 Color Palette",
            options=['aurora', 'sunset', 'ocean', 'galaxy', 'neon', 'vibrant', 'pastel_dream'],
            index=0
        )
    
    with col2:
        chart_style = st.selectbox(
            "📊 Chart Style",
            options=['modern', 'classic', 'minimal'],
            index=0
        )
    
    with col3:
        # Color preview
        colors = get_premium_colors(palette)
        color_preview = "".join([
            f'<div style="display:inline-block;width:25px;height:25px;background:{color};'
            f'margin:3px;border-radius:50%;border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.2);"></div>' 
            for color in colors[:8]
        ])
        st.markdown(f'<div style="margin-top:1rem;">{color_preview}</div>', unsafe_allow_html=True)
    
    # Chart type selection with visual icons
    st.markdown("### 📊 Select Chart Type")
    
    chart_options = {
        'scatter': {'name': '📊 Scatter Plot', 'desc': 'Explore relationships between variables'},
        'line': {'name': '📈 Line Chart', 'desc': 'Show trends over time or sequence'},
        'bar': {'name': '📊 Bar Chart', 'desc': 'Compare values across categories'},
        'histogram': {'name': '📊 Histogram', 'desc': 'Show distribution of values'},
        'box': {'name': '📦 Box Plot', 'desc': 'Compare distributions across groups'},
        'heatmap': {'name': '🔥 Heatmap', 'desc': 'Show correlation matrix'},
        'violin': {'name': '🎻 Violin Plot', 'desc': 'Distribution with density curves'},
        'sunburst': {'name': '☀️ Sunburst', 'desc': 'Hierarchical data visualization'},
        'treemap': {'name': '🌳 Treemap', 'desc': 'Hierarchical proportions'},
        'radar': {'name': '📡 Radar Chart', 'desc': 'Multi-dimensional comparison'}
    }
    
    # Create chart type selector with descriptions
    chart_cols = st.columns(5)
    
    # Initialize selected chart from session state or default to scatter
    if 'selected_chart_type' not in st.session_state:
        st.session_state.selected_chart_type = 'scatter'
    
    for i, (key, info) in enumerate(chart_options.items()):
        with chart_cols[i % 5]:
            if st.button(f"{info['name']}", key=f"chart_{key}", help=info['desc']):
                st.session_state.selected_chart_type = key
    
    selected_chart = st.session_state.selected_chart_type
    
    st.markdown("---")
    
    # Chart configuration and creation
    col_config, col_chart = st.columns([1, 2])
    
    with col_config:
        st.markdown(f"### ⚙️ {chart_options[selected_chart]['name']} Settings")
        
        # Dynamic configuration based on chart type
        config = {}
        
        if selected_chart == 'scatter':
            if len(numeric_cols) >= 2:
                config['x_col'] = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                config['y_col'] = st.selectbox("Y-axis", [c for c in numeric_cols if c != config['x_col']], key="scatter_y")
                config['color_col'] = st.selectbox("Color by", ['None'] + categorical_cols + numeric_cols, key="scatter_color")
                config['size_col'] = st.selectbox("Size by", ['None'] + numeric_cols, key="scatter_size")
                config['add_trendline'] = st.checkbox("Add trendline", value=True)
                config['add_marginals'] = st.checkbox("Add marginal plots", value=False)
        
        elif selected_chart == 'line':
            if numeric_cols:
                config['x_col'] = st.selectbox("X-axis", all_cols, key="line_x")
                config['y_col'] = st.selectbox("Y-axis", numeric_cols, key="line_y")
                config['color_col'] = st.selectbox("Group by", ['None'] + categorical_cols, key="line_color")
                config['smooth'] = st.checkbox("Smooth line", value=True)
                config['add_markers'] = st.checkbox("Show markers", value=True)
            else:
                st.warning("Need numeric columns for line chart")
        
        elif selected_chart == 'bar':
            if categorical_cols and numeric_cols:
                config['x_col'] = st.selectbox("Category", categorical_cols, key="bar_x")
                config['y_col'] = st.selectbox("Value", numeric_cols, key="bar_y")
                config['color_col'] = st.selectbox("Color by", ['None'] + categorical_cols, key="bar_color")
                config['orientation'] = st.selectbox("Orientation", ['vertical', 'horizontal'])
            else:
                st.warning("Need categorical and numeric columns for bar chart")
        
        elif selected_chart == 'histogram':
            if numeric_cols:
                config['column'] = st.selectbox("Column", numeric_cols, key="hist_col")
                config['bins'] = st.slider("Number of bins", 10, 100, 30)
                config['show_kde'] = st.checkbox("Show density curve", value=True)
            else:
                st.warning("Need numeric columns for histogram")
        
        elif selected_chart == 'box':
            if categorical_cols and numeric_cols:
                config['x_col'] = st.selectbox("Category", categorical_cols, key="box_x")
                config['y_col'] = st.selectbox("Value", numeric_cols, key="box_y")
                config['show_points'] = st.selectbox("Show points", ['outliers', 'all', 'none'])
            else:
                st.warning("Need categorical and numeric columns for box plot")
        
        elif selected_chart == 'heatmap':
            if len(numeric_cols) > 1:
                st.info("Correlation heatmap of numeric columns")
                config['cluster'] = st.checkbox("Cluster rows/columns", value=True)
                config['annotate'] = st.checkbox("Show values", value=True)
            else:
                st.warning("Need at least 2 numeric columns for heatmap")
        
        elif selected_chart == 'violin':
            if categorical_cols and numeric_cols:
                config['x_col'] = st.selectbox("Category", categorical_cols, key="violin_x")
                config['y_col'] = st.selectbox("Value", numeric_cols, key="violin_y")
            else:
                st.warning("Need categorical and numeric columns for violin plot")
        
        elif selected_chart == 'treemap':
            if categorical_cols and numeric_cols:
                config['path_cols'] = st.multiselect("Hierarchy (path)", categorical_cols, default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols[:1], key="treemap_path")
                config['value_col'] = st.selectbox("Value", numeric_cols, key="treemap_value")
            else:
                st.warning("Need categorical and numeric columns for treemap")
        
        elif selected_chart == 'radar':
            if len(numeric_cols) >= 3:
                config['metrics'] = st.multiselect("Metrics", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))], key="radar_metrics")
                config['group_col'] = st.selectbox("Group by", ['None'] + categorical_cols, key="radar_group")
            else:
                st.warning("Need at least 3 numeric columns for radar chart")
        
        elif selected_chart == 'sunburst':
            if categorical_cols and numeric_cols:
                config['path_cols'] = st.multiselect("Hierarchy (path)", categorical_cols, default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols[:1], key="sunburst_path")
                config['value_col'] = st.selectbox("Value", numeric_cols, key="sunburst_value")
            else:
                st.warning("Need categorical and numeric columns for sunburst")
    
    with col_chart:
        st.markdown("### 📊 Interactive Chart")
        
        try:
            # Always use the most current filtered data
            current_data = st.session_state.get('working_data', df)
            fig = create_premium_chart(current_data, selected_chart, config, colors, chart_style)
            
            if fig:
                # Apply premium styling with vibrant colors
                fig.update_layout(
                    font_family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(255,255,255,0.95)',
                    height=600,
                    title_font_size=22,
                    title_font_color=colors[0],
                    title_x=0.5,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        bgcolor='rgba(255,255,255,0.9)',
                        bordercolor=colors[0],
                        borderwidth=1
                    ),
                    xaxis=dict(
                        gridcolor='rgba(102, 126, 234, 0.2)',
                        title_font_color=colors[1]
                    ),
                    yaxis=dict(
                        gridcolor='rgba(118, 75, 162, 0.2)',
                        title_font_color=colors[1]
                    )
                )
                
                # Add vibrant hover effects and animations
                fig.update_traces(
                    hovertemplate='<b>%{hovertext}</b><extra></extra>',
                    hoverlabel=dict(
                        bgcolor=colors[0],
                        font_color="white",
                        font_size=12,
                        font_family="Inter",
                        bordercolor="white"
                    )
                )
                
                # Add colorful annotations for enhanced visual appeal
                if selected_chart in ['scatter', 'line', 'bar']:
                    fig.add_annotation(
                        text=f"✨ {palette.title()} Palette",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=10, color=colors[2]),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor=colors[2],
                        borderwidth=1
                    )
                
                st.plotly_chart(fig, use_container_width=True, config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                })
                
                # Chart actions
                col_save, col_export, col_share = st.columns(3)
                
                with col_save:
                    if st.button("💾 Add to Dashboard", key=f"save_chart_{selected_chart}"):
                        chart_key = f"{selected_chart}_{int(time.time())}"
                        if 'dashboard_charts' not in st.session_state:
                            st.session_state.dashboard_charts = {}
                        
                        # Store complete chart data with all required fields
                        st.session_state.dashboard_charts[chart_key] = {
                            'fig': fig,
                            'type': selected_chart,
                            'config': config,
                            'palette': palette,
                            'title': f"{chart_options[selected_chart]['name']}",
                            'colors': colors,
                            'last_updated': pd.Timestamp.now(),
                            'data_rows': len(current_data)
                        }
                        st.success(f"✨ {chart_options[selected_chart]['name']} added to Dashboard!")
                        st.balloons()
                
                with col_export:
                    if st.button("📤 Export PNG", key="export_chart"):
                        st.info("Export functionality - Premium feature")
                
                with col_share:
                    if st.button("🔗 Share Chart", key="share_chart"):
                        st.info("Share functionality - Premium feature")
            
            else:
                # Show Add to Dashboard button even if chart creation failed
                st.warning(f"Unable to create {chart_options[selected_chart]['name']} with current configuration")
                
                # Chart actions (still show for failed charts)
                col_save, col_export, col_share = st.columns(3)
                
                with col_save:
                    if st.button("💾 Add to Dashboard", key=f"save_chart_{selected_chart}_fallback"):
                        # Create a placeholder chart for dashboard
                        placeholder_fig = go.Figure()
                        placeholder_fig.add_annotation(
                            text=f"{chart_options[selected_chart]['name']}\nConfiguration needed",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(size=16, color=colors[0])
                        )
                        placeholder_fig.update_layout(
                            title=f"{chart_options[selected_chart]['name']}",
                            showlegend=False
                        )
                        
                        chart_key = f"{selected_chart}_{int(time.time())}"
                        if 'dashboard_charts' not in st.session_state:
                            st.session_state.dashboard_charts = {}
                        
                        st.session_state.dashboard_charts[chart_key] = {
                            'fig': placeholder_fig,
                            'type': selected_chart,
                            'config': config,
                            'palette': palette,
                            'title': f"{chart_options[selected_chart]['name']}",
                            'colors': colors,
                            'last_updated': pd.Timestamp.now(),
                            'data_rows': len(current_data)
                        }
                        st.success(f"✨ {chart_options[selected_chart]['name']} placeholder added to Dashboard!")
                
                with col_export:
                    st.button("📤 Export PNG", key="export_chart_disabled", disabled=True)
                
                with col_share:
                    st.button("🔗 Share Chart", key="share_chart_disabled", disabled=True)
                
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
    
    # Premium dashboard status
    if hasattr(st.session_state, 'dashboard_charts') and st.session_state.dashboard_charts:
        st.markdown("---")
        chart_count = len(st.session_state.dashboard_charts)
        filter_status = "Live Filtered" if st.session_state.get('filter_values') else "Complete Dataset"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #8B5CF6, #7C3AED); padding: 1.5rem; 
                    border-radius: 15px; color: white; text-align: center; margin-top: 2rem;">
            📊 <strong>{chart_count} Premium Chart{'s' if chart_count != 1 else ''}</strong> in Dashboard<br>
            <small>🔄 Auto-synced • {filter_status} • Real-time updates</small>
        </div>
        """, unsafe_allow_html=True)

def create_premium_chart(df, chart_type, config, colors, style):
    """Create premium styled charts with current filtered data"""
    
    try:
        # Ensure we're using the most current data
        if df.empty:
            return None
            
        if chart_type == 'scatter':
            if 'x_col' in config and 'y_col' in config:
                fig = px.scatter(
                    df, 
                    x=config['x_col'], 
                    y=config['y_col'],
                    color=config.get('color_col') if config.get('color_col') != 'None' else None,
                    size=config.get('size_col') if config.get('size_col') != 'None' else None,
                    color_discrete_sequence=colors,
                    title=f"{config['y_col']} vs {config['x_col']}",
                    marginal_x="histogram" if config.get('add_marginals') else None,
                    marginal_y="histogram" if config.get('add_marginals') else None,
                    trendline="ols" if config.get('add_trendline') else None
                )
                return fig
        
        elif chart_type == 'line':
            if 'x_col' in config and 'y_col' in config:
                try:
                    fig = px.line(
                        df,
                        x=config['x_col'],
                        y=config['y_col'],
                        color=config.get('color_col') if config.get('color_col') != 'None' else None,
                        color_discrete_sequence=colors,
                        title=f"{config['y_col']} over {config['x_col']}",
                        line_shape='spline' if config.get('smooth') else 'linear',
                        markers=config.get('add_markers', True)
                    )
                    return fig
                except Exception:
                    return None
        
        elif chart_type == 'bar':
            if 'x_col' in config and 'y_col' in config:
                try:
                    orientation = config.get('orientation', 'vertical')
                    if orientation == 'horizontal':
                        fig = px.bar(
                            df, y=config['x_col'], x=config['y_col'],
                            color=config.get('color_col') if config.get('color_col') != 'None' else None,
                            color_discrete_sequence=colors,
                            title=f"{config['y_col']} by {config['x_col']}",
                            orientation='h'
                        )
                    else:
                        fig = px.bar(
                            df, x=config['x_col'], y=config['y_col'],
                            color=config.get('color_col') if config.get('color_col') != 'None' else None,
                            color_discrete_sequence=colors,
                            title=f"{config['y_col']} by {config['x_col']}"
                        )
                    return fig
                except Exception:
                    return None
        
        elif chart_type == 'histogram':
            if 'column' in config:
                try:
                    fig = px.histogram(
                        df,
                        x=config['column'],
                        nbins=config.get('bins', 30),
                        color_discrete_sequence=colors,
                        title=f"Distribution of {config['column']}"
                    )
                    
                    if config.get('show_kde'):
                        try:
                            # Add KDE curve
                            from scipy import stats
                            data = df[config['column']].dropna()
                            if len(data) > 0:
                                kde = stats.gaussian_kde(data)
                                x_range = np.linspace(data.min(), data.max(), 100)
                                kde_values = kde(x_range)
                                
                                # Scale KDE to match histogram
                                kde_values = kde_values * len(data) * (data.max() - data.min()) / config.get('bins', 30)
                                
                                fig.add_trace(go.Scatter(
                                    x=x_range, y=kde_values,
                                    mode='lines',
                                    name='Density',
                                    line=dict(color=colors[1], width=3)
                                ))
                        except Exception:
                            pass  # Skip KDE if it fails
                    
                    return fig
                except Exception:
                    return None
        
        elif chart_type == 'box':
            if 'x_col' in config and 'y_col' in config:
                try:
                    fig = px.box(
                        df,
                        x=config['x_col'],
                        y=config['y_col'],
                        color_discrete_sequence=colors,
                        title=f"{config['y_col']} distribution by {config['x_col']}",
                        points=config.get('show_points', 'outliers')
                    )
                    return fig
                except Exception:
                    return None
        
        elif chart_type == 'heatmap':
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=config.get('annotate', True),
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Heatmap"
                    )
                    return fig
            except Exception:
                return None
        
        elif chart_type == 'violin':
            if 'x_col' in config and 'y_col' in config:
                try:
                    fig = px.violin(
                        df,
                        x=config['x_col'],
                        y=config['y_col'],
                        color_discrete_sequence=colors,
                        title=f"{config['y_col']} distribution by {config['x_col']}",
                        box=True
                    )
                    return fig
                except Exception:
                    return None
        
        elif chart_type == 'treemap':
            if 'path_cols' in config and 'value_col' in config and config['path_cols']:
                try:
                    fig = px.treemap(
                        df,
                        path=config['path_cols'],
                        values=config['value_col'],
                        color_discrete_sequence=colors,
                        title=f"Treemap: {' → '.join(config['path_cols'])}"
                    )
                    return fig
                except Exception:
                    # Fallback for treemap issues
                    return None
        
        elif chart_type == 'radar':
            if 'metrics' in config and config['metrics'] and len(config['metrics']) >= 3:
                try:
                    metrics = config['metrics']
                    group_col = config.get('group_col')
                    
                    fig = go.Figure()
                    
                    if group_col and group_col != 'None':
                        groups = df[group_col].unique()[:5]
                        for i, group in enumerate(groups):
                            group_data = df[df[group_col] == group]
                            values = [group_data[metric].mean() for metric in metrics]
                            
                            fig.add_trace(go.Scatterpolar(
                                r=values + [values[0]],
                                theta=metrics + [metrics[0]],
                                fill='toself',
                                name=str(group),
                                line_color=colors[i % len(colors)]
                            ))
                    else:
                        values = [df[metric].mean() for metric in metrics]
                        fig.add_trace(go.Scatterpolar(
                            r=values + [values[0]],
                            theta=metrics + [metrics[0]],
                            fill='toself',
                            name='Average',
                            line_color=colors[0]
                        ))
                    
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True)),
                        title="Radar Chart"
                    )
                    return fig
                except Exception:
                    return None
        
        elif chart_type == 'sunburst':
            if 'path_cols' in config and 'value_col' in config and config['path_cols']:
                try:
                    fig = px.sunburst(
                        df,
                        path=config['path_cols'],
                        values=config['value_col'],
                        color_discrete_sequence=colors,
                        title=f"Sunburst: {' → '.join(config['path_cols'])}"
                    )
                    return fig
                except Exception:
                    # Fallback for sunburst issues
                    return None
        
        return None
        
    except Exception as e:
        st.error(f"Error in chart creation: {str(e)}")
        return None