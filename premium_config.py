"""
Premium Configuration for Advanced Data Analysis Platform
Top 1% Developer Quality - Premium Features & Styling
"""

# Premium Color Palettes
PREMIUM_COLOR_PALETTES = {
    'aurora': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'],
    'sunset': ['#FF9A8B', '#A8E6CF', '#FFD93D', '#6BCF7F', '#4D96FF', '#9B59B6', '#F39C12'],
    'ocean': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b'],
    'galaxy': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b'],
    'neon': ['#FF073A', '#39FF14', '#FF1493', '#00FFFF', '#FFFF00', '#FF4500', '#9400D3'],
    'pastel_dream': ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD1DC', '#E0BBE4', '#C7CEEA'],
    'corporate': ['#2C3E50', '#3498DB', '#E74C3C', '#F39C12', '#27AE60', '#9B59B6', '#1ABC9C'],
    'vibrant': ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5', '#118AB2', '#073B4C', '#EF476F'],
    'dark_mode': ['#BB86FC', '#03DAC6', '#CF6679', '#FF0266', '#00E676', '#FFAB00', '#FF5722'],
    'gradient_blue': ['#667eea', '#764ba2', '#667eea', '#764ba2', '#667eea', '#764ba2', '#667eea'],
    'rainbow': ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3']
}

# Premium Themes
PREMIUM_THEMES = {
    'executive': {
        'primary_color': '#1E3A8A',
        'secondary_color': '#3B82F6',
        'accent_color': '#F59E0B',
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'text_color': '#1F2937',
        'card_background': 'rgba(255, 255, 255, 0.95)',
        'shadow': '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
    },
    'dark_premium': {
        'primary_color': '#BB86FC',
        'secondary_color': '#03DAC6',
        'accent_color': '#CF6679',
        'background': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        'text_color': '#E1E5E9',
        'card_background': 'rgba(30, 30, 46, 0.95)',
        'shadow': '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
    },
    'luxury': {
        'primary_color': '#D4AF37',
        'secondary_color': '#FFD700',
        'accent_color': '#B8860B',
        'background': 'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)',
        'text_color': '#ECF0F1',
        'card_background': 'rgba(52, 73, 94, 0.95)',
        'shadow': '0 25px 50px -12px rgba(212, 175, 55, 0.25)'
    }
}

# Advanced Plot Configurations
PREMIUM_PLOT_CONFIG = {
    'default_height': 600,
    'default_width': 900,
    'high_dpi': True,
    'animation_duration': 800,
    'hover_effects': True,
    'interactive_legends': True,
    'zoom_enabled': True,
    'pan_enabled': True,
    'export_formats': ['png', 'svg', 'pdf', 'html'],
    'font_family': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    'title_font_size': 24,
    'axis_font_size': 14,
    'legend_font_size': 12
}

# Premium Chart Types with Advanced Features
PREMIUM_CHART_TYPES = {
    'advanced_scatter': {
        'name': '✨ Premium Scatter Plot',
        'description': 'Interactive scatter with animations, regression lines, and clustering',
        'features': ['regression_line', 'confidence_intervals', 'clustering_overlay', 'size_mapping', 'color_mapping']
    },
    'animated_line': {
        'name': '🎬 Animated Line Chart',
        'description': 'Time-series with smooth animations and trend analysis',
        'features': ['smooth_animations', 'trend_lines', 'forecasting', 'interactive_tooltips']
    },
    'heatmap_advanced': {
        'name': '🔥 Advanced Heatmap',
        'description': 'Interactive heatmap with clustering and annotations',
        'features': ['hierarchical_clustering', 'dendrograms', 'custom_annotations', 'zoom_pan']
    },
    'violin_plot': {
        'name': '🎻 Violin Plot',
        'description': 'Distribution visualization with kernel density estimation',
        'features': ['kde_overlay', 'quartile_lines', 'mean_markers', 'split_violins']
    },
    'sunburst': {
        'name': '☀️ Sunburst Chart',
        'description': 'Hierarchical data visualization with interactive drilling',
        'features': ['drill_down', 'breadcrumbs', 'custom_colors', 'animations']
    },
    'treemap': {
        'name': '🌳 Treemap',
        'description': 'Hierarchical data with proportional rectangles',
        'features': ['hover_effects', 'custom_colors', 'labels', 'drill_down']
    },
    'radar_chart': {
        'name': '📡 Radar Chart',
        'description': 'Multi-dimensional data comparison',
        'features': ['multiple_series', 'fill_areas', 'custom_scales', 'animations']
    },
    'waterfall': {
        'name': '💧 Waterfall Chart',
        'description': 'Sequential value changes visualization',
        'features': ['cumulative_totals', 'color_coding', 'annotations', 'hover_details']
    },
    'candlestick': {
        'name': '📈 Candlestick Chart',
        'description': 'Financial data visualization with OHLC',
        'features': ['volume_overlay', 'technical_indicators', 'zoom_pan', 'crossfilter']
    },
    'sankey': {
        'name': '🌊 Sankey Diagram',
        'description': 'Flow visualization between categories',
        'features': ['interactive_flows', 'custom_colors', 'hover_details', 'animations']
    }
}

# Premium UI Components
PREMIUM_UI_COMPONENTS = {
    'glassmorphism_cards': True,
    'floating_action_buttons': True,
    'animated_transitions': True,
    'gradient_backgrounds': True,
    'custom_scrollbars': True,
    'loading_animations': True,
    'toast_notifications': True,
    'modal_dialogs': True,
    'drag_drop_interface': True,
    'keyboard_shortcuts': True
}

# Advanced Analytics Features
PREMIUM_ANALYTICS = {
    'statistical_tests': ['t_test', 'chi_square', 'anova', 'correlation_test'],
    'time_series_analysis': ['seasonality', 'trend_decomposition', 'forecasting', 'anomaly_detection'],
    'clustering_algorithms': ['kmeans', 'dbscan', 'hierarchical', 'gaussian_mixture'],
    'dimensionality_reduction': ['pca', 'tsne', 'umap', 'factor_analysis'],
    'feature_engineering': ['polynomial_features', 'interaction_terms', 'binning', 'scaling'],
    'model_interpretability': ['feature_importance', 'shap_values', 'lime', 'permutation_importance']
}

# Export and Sharing Options
PREMIUM_EXPORT_OPTIONS = {
    'formats': ['png', 'svg', 'pdf', 'html', 'json', 'csv', 'excel'],
    'quality_settings': ['standard', 'high', 'ultra'],
    'custom_branding': True,
    'watermark_options': True,
    'batch_export': True,
    'cloud_sharing': True,
    'embed_codes': True,
    'api_endpoints': True
}

# Performance Optimizations
PREMIUM_PERFORMANCE = {
    'lazy_loading': True,
    'data_sampling': True,
    'caching_strategy': 'advanced',
    'parallel_processing': True,
    'memory_optimization': True,
    'progressive_rendering': True,
    'virtual_scrolling': True,
    'debounced_updates': True
}

# Accessibility Features
PREMIUM_ACCESSIBILITY = {
    'high_contrast_mode': True,
    'screen_reader_support': True,
    'keyboard_navigation': True,
    'color_blind_friendly': True,
    'font_size_scaling': True,
    'focus_indicators': True,
    'alt_text_generation': True,
    'voice_commands': False  # Future feature
}

# Premium CSS Styles
PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --glass-bg: rgba(255, 255, 255, 0.1);
    --glass-border: rgba(255, 255, 255, 0.2);
    --shadow-premium: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    --shadow-card: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

.main-header {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 3.5rem;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    animation: glow 2s ease-in-out infinite alternate;
}

@keyframes glow {
    from { filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.5)); }
    to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.8)); }
}

.premium-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-card);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.premium-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--primary-gradient);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.premium-card:hover {
    transform: translateY(-5px);
    box-shadow: var(--shadow-premium);
}

.premium-card:hover::before {
    opacity: 1;
}

.metric-card {
    background: var(--glass-bg);
    backdrop-filter: blur(15px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(from 0deg, transparent, rgba(102, 126, 234, 0.1), transparent);
    animation: rotate 4s linear infinite;
    opacity: 0;
    transition: opacity 0.3s ease;
}

.metric-card:hover::after {
    opacity: 1;
}

@keyframes rotate {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.metric-card h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #4A5568;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
}

.metric-card h2 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 2.5rem;
    margin: 0;
    position: relative;
    z-index: 1;
}

.premium-button {
    background: var(--primary-gradient);
    border: none;
    border-radius: 12px;
    color: white;
    padding: 0.75rem 1.5rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.premium-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s ease;
}

.premium-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.premium-button:hover::before {
    left: 100%;
}

.sidebar-premium {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--glass-border);
}

.plot-container {
    background: var(--glass-bg);
    backdrop-filter: blur(15px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-card);
    transition: all 0.3s ease;
}

.plot-container:hover {
    box-shadow: var(--shadow-premium);
}

.loading-spinner {
    display: inline-block;
    width: 40px;
    height: 40px;
    border: 3px solid rgba(102, 126, 234, 0.3);
    border-radius: 50%;
    border-top-color: #667eea;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.success-toast {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    box-shadow: 0 10px 25px rgba(16, 185, 129, 0.3);
    animation: slideIn 0.3s ease;
}

.error-toast {
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    color: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    box-shadow: 0 10px 25px rgba(239, 68, 68, 0.3);
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

.feature-highlight {
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600;
    display: inline-block;
}

.premium-tabs {
    display: flex;
    background: var(--glass-bg);
    backdrop-filter: blur(15px);
    border-radius: 16px;
    padding: 0.5rem;
    margin-bottom: 2rem;
    border: 1px solid var(--glass-border);
}

.premium-tab {
    flex: 1;
    padding: 0.75rem 1.5rem;
    border: none;
    background: transparent;
    border-radius: 12px;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    color: #6B7280;
}

.premium-tab.active {
    background: var(--primary-gradient);
    color: white;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.premium-tab:hover:not(.active) {
    background: rgba(102, 126, 234, 0.1);
    color: #667eea;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary-gradient);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-gradient);
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header {
        font-size: 2.5rem;
    }
    
    .premium-card {
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        padding: 1rem;
    }
}

/* Dark Mode Support */
@media (prefers-color-scheme: dark) {
    :root {
        --glass-bg: rgba(0, 0, 0, 0.2);
        --glass-border: rgba(255, 255, 255, 0.1);
    }
}

/* High Contrast Mode */
@media (prefers-contrast: high) {
    .premium-card {
        border: 2px solid #000;
        background: #fff;
    }
    
    .premium-button {
        background: #000;
        color: #fff;
        border: 2px solid #000;
    }
}

/* Reduced Motion */
@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""