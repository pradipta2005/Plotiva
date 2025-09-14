"""
Dashboard Tab - Interactive Charts Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_dashboard_tab():
    """Render the main dashboard tab with saved charts"""
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem; color: white; text-align: center;">
        <h1 style="font-size: 2.5rem; margin-bottom: 1rem;">📊 Interactive Dashboard</h1>
        <p style="font-size: 1.2rem;">Your premium charts collection - Create charts in Visualizations tab</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Premium data status with auto-update indicator
    current_df = st.session_state.get('working_data', None)
    if current_df is not None and not current_df.empty:
        if st.session_state.get('filter_values'):
            filter_count = len(st.session_state.filter_values)
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #F59E0B, #D97706); padding: 1rem; border-radius: 10px; 
                        color: white; margin-bottom: 1rem; text-align: center; animation: pulse 2s infinite;">
                🔄 <strong>Auto-Synced Dashboard</strong> • {len(current_df):,} filtered rows • {filter_count} active filter{'s' if filter_count != 1 else ''}
                <br><small>All charts update automatically when filters change</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #059669, #047857); padding: 1rem; border-radius: 10px; 
                        color: white; margin-bottom: 1rem; text-align: center;">
                📊 <strong>Complete Dashboard</strong> • {len(current_df):,} total rows • No filters applied
                <br><small>Showing all available data</small>
            </div>
            """, unsafe_allow_html=True)
    
    if not hasattr(st.session_state, 'dashboard_charts') or not st.session_state.dashboard_charts:
        # Empty dashboard state
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(102, 126, 234, 0.1); 
                    border-radius: 15px; border: 2px dashed #667eea;">
            <h2 style="color: #667eea; margin-bottom: 1rem;">🎨 Your Dashboard is Empty</h2>
            <p style="font-size: 1.1rem; color: #6B7280; margin-bottom: 2rem;">
                Create beautiful charts in the <strong>Visualizations</strong> tab and add them here!
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                    <div style="color: #374151;">Scatter Plots</div>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌳</div>
                    <div style="color: #374151;">Treemaps</div>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📡</div>
                    <div style="color: #374151;">Radar Charts</div>
                </div>
                <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">☀️</div>
                    <div style="color: #374151;">Sunburst</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Dashboard controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### 📊 **{len(st.session_state.dashboard_charts)}** Charts in Dashboard")
    
    with col2:
        layout_style = st.selectbox("Layout", ["Grid", "Single Column"], index=0)
    
    with col3:
        if st.button("🗑️ Clear All", help="Remove all charts from dashboard"):
            st.session_state.dashboard_charts = {}
            st.rerun()
    
    st.markdown("---")
    
    # Display charts based on layout
    if layout_style == "Grid":
        # Grid layout (2 columns)
        dashboard_cols = st.columns(2)
        for i, (key, plot_data) in enumerate(st.session_state.dashboard_charts.items()):
            with dashboard_cols[i % 2]:
                render_dashboard_chart(key, plot_data, i)
    else:
        # Single column layout
        for i, (key, plot_data) in enumerate(st.session_state.dashboard_charts.items()):
            render_dashboard_chart(key, plot_data, i)

def render_dashboard_chart(key, plot_data, index):
    """Render individual chart in dashboard with current filtered data"""
    
    # Get colors for styling
    colors = plot_data.get('colors', ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    # Premium auto-update status
    current_df = st.session_state.get('working_data', None)
    current_filter_status = bool(st.session_state.get('filter_values'))
    
    # Update chart with current filtered data if available
    if current_df is not None and not current_df.empty:
        # Recreate chart with current data
        updated_fig = recreate_chart_with_current_data(current_df, plot_data)
        if updated_fig:
            plot_data['fig'] = updated_fig
            plot_data['data_rows'] = len(current_df)
            plot_data['last_updated'] = pd.Timestamp.now()
    
    # Show premium update status
    if 'last_updated' in plot_data:
        update_time = plot_data['last_updated'].strftime('%H:%M:%S')
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10B981, #059669); padding: 0.5rem; 
                    border-radius: 8px; color: white; text-align: center; margin-bottom: 1rem; font-size: 0.9rem;">
            ✨ Auto-Updated at {update_time} • {plot_data.get('data_rows', 'N/A')} rows
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced chart container with gradient border
    st.markdown(f"""
    <div style="background: linear-gradient(45deg, {colors[index%len(colors)]}, {colors[(index+1)%len(colors)]}); 
                padding: 3px; border-radius: 15px; margin-bottom: 2rem;">
        <div style="background: white; border-radius: 12px; padding: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="color: {colors[index%len(colors)]}; margin: 0; font-size: 1.3rem;">
                    {plot_data['title']}
                </h3>
                <div style="background: {colors[index%len(colors)]}20; padding: 0.3rem 0.8rem; 
                            border-radius: 20px; font-size: 0.8rem; color: {colors[index%len(colors)]};">
                    {plot_data['palette'].title()} Palette
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced chart with colorful styling and data info
    enhanced_fig = plot_data['fig']
    
    # Premium title with live data info
    original_title = enhanced_fig.layout.title.text if enhanced_fig.layout.title else plot_data['title']
    filter_indicator = "🎯 Filtered" if st.session_state.get('filter_values') else "📊 Complete"
    data_info = f" • {filter_indicator} • {plot_data.get('data_rows', 'N/A')} rows"
    if not data_info in original_title:
        new_title = f"{original_title}{data_info}"
        enhanced_fig.update_layout(title=new_title)
    
    enhanced_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=colors[index%len(colors)],
        title_font_color=colors[index%len(colors)],
        title_font_size=18,
        showlegend=True,
        margin=dict(t=50, b=50, l=50, r=50),
        height=500
    )
    
    st.plotly_chart(enhanced_fig, use_container_width=True, key=f"dashboard_{key}")
    
    # Chart actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(f"📤 Export", key=f"export_{key}", help="Export chart"):
            st.info("Export feature - Premium functionality")
    
    with col2:
        if st.button(f"🔗 Share", key=f"share_{key}", help="Share chart"):
            st.info("Share feature - Premium functionality")
    
    with col3:
        if st.button(f"🗑️ Remove", key=f"remove_{key}", help="Remove from dashboard"):
            del st.session_state.dashboard_charts[key]
            st.rerun()
    
    st.markdown("---")

def recreate_chart_with_current_data(df, plot_data):
    """Recreate chart with current filtered data"""
    try:
        from premium_visualization_tab import create_premium_chart
        
        chart_type = plot_data.get('type', 'scatter')
        config = plot_data.get('config', {})
        colors = plot_data.get('colors', ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        
        # Recreate the chart with current data
        updated_fig = create_premium_chart(df, chart_type, config, colors, 'modern')
        
        if updated_fig:
            # Apply the same styling as the original
            updated_fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color=colors[0],
                title_font_color=colors[0],
                title_font_size=18,
                showlegend=True,
                margin=dict(t=50, b=50, l=50, r=50),
                height=500
            )
            
            return updated_fig
    except Exception as e:
        pass
    
    # Return original figure if recreation fails
    return plot_data.get('fig')