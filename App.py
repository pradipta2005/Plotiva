import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import io
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import r2_score, mean_squared_error # type: ignore
from sklearn.cluster import KMeans # type: ignore
import plotly.express as px # type: ignore
import matplotlib
import warnings
from typing import Optional, Dict, List, Tuple
import datetime
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Data Analysis App",
    page_icon="📊",
    layout="wide"
)

# Set matplotlib backend
matplotlib.use('Agg')

# Initialize seaborn style
sns.set_style("whitegrid")
palette = sns.color_palette("husl", 8)

def rgb_to_hex(rgb_tuple):
    return '#%02x%02x%02x' % tuple(int(x * 255) for x in rgb_tuple)

def plot_gallery(num_columns, selected_plots, numeric_columns, categorical_columns, filtered_data, palette, columns):
    col_idx = 0
    cols = st.columns(num_columns)

    for plot_type in selected_plots:
        with cols[col_idx]:
            try:
                if plot_type == "Line Chart":
                    st.write("## Line Chart")
                    column_for_line_chart = st.selectbox("Select a column for line chart", numeric_columns, key=f"line_chart_{col_idx}")
                    if column_for_line_chart:
                        line_color = st.color_picker("Line Color", "#1f77b4", key=f"line_color_{col_idx}")
                        line_style = st.selectbox("Line Style", ['-', '--', '-.', ':'], key=f"line_style_{col_idx}")
                        marker_style = st.selectbox("Marker Style", ['None', 'o', 'X', '^', 's', 'D'], key=f"marker_style_{col_idx}")

                        fig_line = plt.figure(figsize=(8, 6))
                        plt.plot(filtered_data[column_for_line_chart].values, color=line_color, linestyle=line_style, marker=marker_style if marker_style != 'None' else None)
                        plt.title(f"Line Chart of {column_for_line_chart}")
                        plt.xlabel("Index")
                        plt.ylabel(column_for_line_chart)
                        plt.grid(True)
                        st.pyplot(fig_line)

                        # Add to report
                        title = f"Line Chart of {column_for_line_chart}"
                        element_key = f"Plot: {title}"
                        if element_key not in st.session_state.report_elements:
                            st.session_state.report_elements.append(element_key)
                        
                        buf = io.BytesIO()
                        fig_line.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": f"A line chart showing the trend of {column_for_line_chart}."
                        }
                        plt.close(fig_line)

                elif plot_type == "Bar Chart":
                    st.write("## Bar Chart")
                    x_column = st.selectbox("Select a column for x-axis (categorical)", categorical_columns, key=f"bar_x_{col_idx}")
                    y_column = st.selectbox("Select a column for y-axis (numeric)", numeric_columns, key=f"bar_y_{col_idx}")
                    if x_column and y_column:
                        default_color = rgb_to_hex(palette[0])
                        color = st.color_picker("Pick a color for the bars", default_color, key=f"bar_color_{col_idx}")
                        orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key=f"bar_orientation_{col_idx}")
                        
                        fig_bar = plt.figure(figsize=(10, 6))
                        if orientation == "Vertical":
                            sns.barplot(x=filtered_data[x_column], y=filtered_data[y_column], color=color)
                            plt.xticks(rotation=45, ha='right')
                        else:
                            sns.barplot(y=filtered_data[x_column], x=filtered_data[y_column], color=color)
                            plt.yticks(rotation=0)
                        plt.title(f"Bar Chart: {y_column} by {x_column}")
                        plt.tight_layout()
                        st.pyplot(fig_bar)

                        # Add to report
                        title = f"Bar Chart: {y_column} by {x_column}"
                        element_key = f"Plot: {title}"
                        if element_key not in st.session_state.report_elements:
                            st.session_state.report_elements.append(element_key)
                        
                        buf = io.BytesIO()
                        fig_bar.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": f"A bar chart showing {y_column} for each category in {x_column}."
                        }
                        plt.close(fig_bar)

                elif plot_type == "Scatter Plot":
                    st.write("## Scatter Plot")
                    x_column = st.selectbox("Select a column for x-axis", numeric_columns, key=f"scatter_x_{col_idx}")
                    y_column = st.selectbox("Select a column for y-axis", numeric_columns, key=f"scatter_y_{col_idx}")
                    
                    if x_column and y_column:
                        hue_column = st.selectbox("Select a column for color (optional)", [None] + categorical_columns + numeric_columns, key=f"scatter_hue_{col_idx}")
                        size_variable = st.selectbox("Select a column for size (optional)", [None] + numeric_columns, key=f"scatter_size_{col_idx}")
                        marker_style = st.selectbox("Marker Style", ['o', 'X', '^', 's', 'D'], key=f"scatter_marker_{col_idx}")
                        
                        fig_scatter = plt.figure(figsize=(8, 6))
                        sns.scatterplot(data=filtered_data, x=x_column, y=y_column, hue=hue_column, size=size_variable, marker=marker_style, palette='viridis')
                        plt.title(f"Scatter Plot: {y_column} vs {x_column}")
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                        plt.grid(True)
                        st.pyplot(fig_scatter)

                        # Add to report
                        title = f"Scatter Plot: {y_column} vs {x_column}"
                        element_key = f"Plot: {title}"
                        if element_key not in st.session_state.report_elements:
                            st.session_state.report_elements.append(element_key)

                        buf = io.BytesIO()
                        fig_scatter.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": f"A scatter plot comparing {y_column} and {x_column}."
                        }
                        plt.close(fig_scatter)

                elif plot_type == "Heatmap":
                    st.write("## Heatmap")
                    numeric_data = filtered_data.select_dtypes(include=[np.number])
                    
                    if numeric_data.empty:
                        st.warning("No numeric columns available for correlation.")
                    else:
                        heatmap_data = numeric_data.corr()
                        cmap_option = st.selectbox("Select Colormap", plt.colormaps(), key=f"heatmap_cmap_{col_idx}")
                        annotate = st.checkbox("Show Annotations", value=True, key=f"heatmap_annot_{col_idx}")

                        fig_heatmap = plt.figure(figsize=(10, 8))
                        sns.heatmap(heatmap_data, annot=annotate, cmap=cmap_option, fmt='.2f')
                        plt.title("Correlation Heatmap")
                        plt.tight_layout()
                        st.pyplot(fig_heatmap)

                        # Add to report
                        title = "Correlation Heatmap"
                        element_key = f"Plot: {title}"
                        if element_key not in st.session_state.report_elements:
                            st.session_state.report_elements.append(element_key)
                        
                        buf = io.BytesIO()
                        fig_heatmap.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": "A heatmap showing the correlation between numeric variables."
                        }
                        plt.close(fig_heatmap)
                
                elif plot_type == "Pair Plot":
                    st.write("## Pair Plot")
                    pair_plot_columns = st.multiselect("Select columns for Pair Plot", numeric_columns, key=f"pair_plot_{col_idx}")
                    if len(pair_plot_columns) > 1:
                        hue_column_pair = st.selectbox("Select a column for color (optional)", [None] + categorical_columns, key=f"pair_plot_hue_{col_idx}")
                        fig_pair = sns.pairplot(filtered_data[pair_plot_columns], hue=hue_column_pair)
                        st.pyplot(fig_pair)

                        # Add to report
                        title = "Pair Plot"
                        element_key = f"Plot: {title}"
                        if element_key not in st.session_state.report_elements:
                            st.session_state.report_elements.append(element_key)

                        buf = io.BytesIO()
                        fig_pair.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": "A pair plot to visualize relationships between multiple numeric variables."
                        }
                        plt.close(fig_pair)
                    else:
                        st.warning("Please select at least 2 numeric columns for a Pair Plot.")

                elif plot_type == "Hexbin Plot":
                    st.write("## Hexbin Plot")
                    x_column = st.selectbox("Select x-axis for Hexbin Plot", numeric_columns, key=f"hexbin_x_{col_idx}")
                    y_column = st.selectbox("Select y-axis for Hexbin Plot", numeric_columns, key=f"hexbin_y_{col_idx}")
                    if x_column and y_column:
                        gridsize = st.slider("Grid Size", 10, 50, 20, key=f"hexbin_gridsize_{col_idx}")
                        cmap_option = st.selectbox("Select Colormap", list(plt.colormaps()), key=f"hexbin_cmap_{col_idx}", index=list(plt.colormaps()).index('Blues'))

                        fig_hexbin = plt.figure(figsize=(8, 6))
                        plt.hexbin(filtered_data[x_column], filtered_data[y_column], gridsize=gridsize, cmap=cmap_option)
                        plt.colorbar(label='Count')
                        plt.title(f"Hexbin Plot: {y_column} vs {x_column}")
                        plt.xlabel(x_column)
                        plt.ylabel(y_column)
                        st.pyplot(fig_hexbin)

                        # Add to report
                        title = f"Hexbin Plot: {y_column} vs {x_column}"
                        element_key = f"Plot: {title}"
                        if element_key not in st.session_state.report_elements:
                            st.session_state.report_elements.append(element_key)

                        buf = io.BytesIO()
                        fig_hexbin.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": f"A hexbin plot for {y_column} vs {x_column} to handle large datasets."
                        }
                        plt.close(fig_hexbin)
                    else:
                        st.warning("Please select both x and y columns for a Hexbin Plot.")

                elif plot_type == "Box Plot":
                    st.write("## Box Plot")
                    x_column = st.selectbox("Select categorical column for Box Plot", categorical_columns, key=f"box_x_{col_idx}")
                    y_column = st.selectbox("Select numeric column for Box Plot", numeric_columns, key=f"box_y_{col_idx}")
                    if x_column and y_column:
                        notch = st.checkbox("Show Notch", key=f"box_notch_{col_idx}")
                        box_color_by = st.selectbox("Color by (optional)", [None] + categorical_columns, key=f"box_color_{col_idx}")
                        orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key=f"box_orientation_{col_idx}")

                        fig_box = plt.figure(figsize=(10, 6))
                        if orientation == "Vertical":
                            sns.boxplot(data=filtered_data, x=x_column, y=y_column, notch=notch, hue=box_color_by, palette='viridis')
                            plt.xticks(rotation=45, ha='right')
                        else:
                            sns.boxplot(data=filtered_data, y=x_column, x=y_column, notch=notch, hue=box_color_by, palette='viridis')
                            plt.yticks(rotation=0)
                        plt.title(f"Box Plot of {y_column} by {x_column}")
                        plt.tight_layout()
                        st.pyplot(fig_box)

                        # Add to report
                        title = f"Box Plot of {y_column} by {x_column}"
                        element_key = f"Plot: {title}"
                        if element_key not in st.session_state.report_elements:
                            st.session_state.report_elements.append(element_key)

                        buf = io.BytesIO()
                        fig_box.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": f"A box plot showing the distribution of {y_column} across {x_column}."
                        }
                        plt.close(fig_box)
                    else:
                        st.warning("Please select both categorical and numeric columns for a Box Plot.")

                elif plot_type == "Violin Plot":
                    st.write("## Violin Plot")
                    x_column = st.selectbox("Select categorical column for Violin Plot", categorical_columns, key=f"violin_x_{col_idx}")
                    y_column = st.selectbox("Select numeric column for Violin Plot", numeric_columns, key=f"violin_y_{col_idx}")
                    if x_column and y_column:
                        inner_option = st.selectbox("Inner display", ["box", "quartile", "point", "stick", None], key=f"violin_inner_{col_idx}")
                        violin_color_by = st.selectbox("Color by (optional)", [None] + categorical_columns, key=f"violin_color_{col_idx}")
                        orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key=f"violin_orientation_{col_idx}")

                        fig_violin = plt.figure(figsize=(10, 6))
                        if orientation == "Vertical":
                            sns.violinplot(data=filtered_data, x=x_column, y=y_column, inner=inner_option, hue=violin_color_by, palette='plasma')
                            plt.xticks(rotation=45, ha='right')
                        else:
                            sns.violinplot(data=filtered_data, y=x_column, x=y_column, inner=inner_option, hue=violin_color_by, palette='plasma')
                            plt.yticks(rotation=0)
                        plt.title(f"Violin Plot of {y_column} by {x_column}")
                        plt.tight_layout()
                        st.pyplot(fig_violin)

                        # Add to report
                        title = f"Violin Plot of {y_column} by {x_column}"
                        element_key = f"Plot: {title}"
                        if element_key not in st.session_state.report_elements:
                            st.session_state.report_elements.append(element_key)

                        buf = io.BytesIO()
                        fig_violin.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": f"A violin plot combining a box plot with the kernel density estimation of {y_column} across {x_column}."
                        }
                        plt.close(fig_violin)
                    else:
                        st.warning("Please select both categorical and numeric columns for a Violin Plot.")

                elif plot_type == "Count Plot":
                    st.write("## Count Plot")
                    column = st.selectbox("Select categorical column for Count Plot", categorical_columns, key=f"count_plot_{col_idx}")
                    if column:
                        hue_column_count = st.selectbox("Select a column for hue (optional)", [None] + categorical_columns, key=f"count_hue_{col_idx}")
                        orientation = st.radio("Orientation", ["Vertical", "Horizontal"], key=f"count_orientation_{col_idx}")

                        fig_count = plt.figure(figsize=(10, 6))
                        if orientation == "Vertical":
                            sns.countplot(data=filtered_data, x=column, hue=hue_column_count, palette='Set3')
                            plt.xticks(rotation=45, ha='right')
                        else:
                            sns.countplot(data=filtered_data, y=column, hue=hue_column_count, palette='Set3')
                            plt.yticks(rotation=0)
                        plt.title(f"Count Plot of {column}")
                        plt.tight_layout()
                        st.pyplot(fig_count)

                        # Add to report
                        title = f"Count Plot of {column}"
                        element_key = f"Plot: {title}"
                        if element_key not in st.session_state.report_elements:
                            st.session_state.report_elements.append(element_key)

                        buf = io.BytesIO()
                        fig_count.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": f"A count plot showing the frequency of categories in {column}."
                        }
                        plt.close(fig_count)
                    else:
                        st.warning("Please select a categorical column for a Count Plot.")

                elif plot_type == "Pie/Donut Chart":
                    st.write("## Pie/Donut Chart")
                    column = st.selectbox("Select categorical column for Pie/Donut Chart", categorical_columns, key=f"pie_donut_{col_idx}")
                    if column:
                        counts = filtered_data[column].value_counts()
                        if not counts.empty:
                            donut_width = st.slider("Donut Hole Size (0 for Pie)", 0.0, 0.8, 0.3, 0.05, key=f"donut_width_{col_idx}")
                            start_angle = st.slider("Start Angle", 0, 360, 90, key=f"pie_start_angle_{col_idx}")
                            show_shadow = st.checkbox("Show Shadow", key=f"pie_shadow_{col_idx}")

                            fig_pie = plt.figure(figsize=(8, 8))
                            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=start_angle, wedgeprops=dict(width=donut_width), shadow=show_shadow)
                            plt.title(f"Pie/Donut Chart of {column}")
                            plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                            st.pyplot(fig_pie)

                            # Add to report
                            title = f"Pie/Donut Chart of {column}"
                            element_key = f"Plot: {title}"
                            if element_key not in st.session_state.report_elements:
                                st.session_state.report_elements.append(element_key)
                            
                            buf = io.BytesIO()
                            fig_pie.savefig(buf, format="png")
                            buf.seek(0)
                            st.session_state.report_content[element_key] = {
                                "type": "plot", "image": buf, "title": title,
                                "caption": f"A pie/donut chart showing the proportion of categories in {column}."
                            }
                            plt.close(fig_pie)
                        else:
                            st.warning(f"No data to display for Pie/Donut Chart for column {column}.")
                    else:
                        st.warning("Please select a categorical column for a Pie/Donut Chart.")

                elif plot_type == "Bubble Chart":
                    st.write("## Bubble Chart")
                    x_column = st.selectbox("Select x-axis for Bubble Chart", numeric_columns, key=f"bubble_x_{col_idx}")
                    y_column = st.selectbox("Select y-axis for Bubble Chart", numeric_columns, key=f"bubble_y_{col_idx}")
                    size_column = st.selectbox("Select size column for Bubble Chart", numeric_columns, key=f"bubble_size_{col_idx}")
                    color_column = st.selectbox("Select color column for Bubble Chart", columns, key=f"bubble_color_{col_idx}")

                    if x_column and y_column and size_column:
                        opacity = st.slider("Opacity", 0.0, 1.0, 0.7, 0.05, key=f"bubble_opacity_{col_idx}")
                        symbol_option = st.selectbox("Symbol", [None, 'circle', 'square', 'diamond', 'cross', 'x'], key=f"bubble_symbol_{col_idx}")

                        fig_bubble = px.scatter(filtered_data, x=x_column, y=y_column, size=size_column, color=color_column, hover_name=filtered_data.index, opacity=opacity)
                        st.plotly_chart(fig_bubble)
                    else:
                        st.warning("Please select x, y, and size columns for a Bubble Chart.")

                # Increment column index for next plot
                col_idx += 1
                if col_idx >= num_columns:  # Reset after reaching the limit
                    col_idx = 0

            except Exception as e:
                st.error(f"Error creating {plot_type}: {str(e)}")
                continue

# Cache data loading and processing
@st.cache_data(ttl=3600, show_spinner=True)
def load_data(file) -> Optional[pd.DataFrame]:
    """Load and preprocess data with caching."""
    try:
        df = pd.read_csv(file)
        # Convert date columns automatically
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    continue
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=True)
def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Get column types with caching."""
    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return columns, numeric_columns, categorical_columns

@st.cache_data(ttl=3600, show_spinner=True)
def apply_filters(df: pd.DataFrame, filter_values: Dict) -> pd.DataFrame:
    if not filter_values:
        return df
    filtered_data = df.copy()
    for filter_column, values in filter_values.items():
        if filter_column in df.select_dtypes(include=['object', 'category']).columns and values:
            filtered_data = filtered_data[filtered_data[filter_column].isin(values)]
        elif filter_column in df.select_dtypes(include=['float64', 'int64']).columns and isinstance(values, tuple):
            filtered_data = filtered_data[(filtered_data[filter_column] >= values[0]) & 
                                        (filtered_data[filter_column] <= values[1])]
    return filtered_data

def run_kmeans_analysis(filtered_data, numeric_columns, n_init):
    import time
    with st.form("kmeans_form_kmeans"):
        cluster_columns = st.multiselect("Columns for clustering (at least 2)", numeric_columns, key="cluster_cols_kmeans")
        k = st.slider("Number of clusters (k)", 2, 10, 3, key="k_kmeans")
        if st.form_submit_button("Run KMeans"):
            if len(cluster_columns) >= 2:
                from sklearn.cluster import KMeans
                import matplotlib.pyplot as plt
                import seaborn as sns
                kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42)
                filtered_data['Cluster'] = kmeans.fit_predict(filtered_data[cluster_columns])
                st.write("Clusters assigned to each row:")
                st.dataframe(filtered_data[['Cluster'] + cluster_columns])
                st.write(f"**Inertia:** {kmeans.inertia_:.2f} (lower is better)")
                fig, ax = plt.subplots()
                sns.scatterplot(data=filtered_data, x=cluster_columns[0], y=cluster_columns[1], hue='Cluster', palette='Set1', ax=ax)
                ax.set_title("KMeans Clustering Results")
                st.pyplot(fig)
                plt.close(fig)
                st.info("If clusters overlap, try increasing k or n_init.")
                unique_id = str(time.time())
                title = f"KMeans Clustering Results (k={k})"
                element_key = f"Advanced Analysis: {title}"
                st.session_state.report_elements.append(element_key)
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.session_state.report_content[element_key] = {
                    "type": "plot", "image": buf, "title": title,
                    "caption": f"KMeans clustering with k={k} and columns {', '.join(cluster_columns)}."
                }
            else:
                st.warning("Please select at least 2 numeric columns for clustering.")

def run_random_forest_analysis(filtered_data, numeric_columns, n_estimators):
    import time
    with st.form("random_forest_form_rf"):
        target = st.selectbox("Target variable (y)", numeric_columns, key="target_rf")
        features = st.multiselect("Feature variables (X)", numeric_columns, default=numeric_columns[:-1], key="features_rf")
        if st.form_submit_button("Run Random Forest"):
            if len(features) > 0:
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import r2_score, mean_squared_error
                import matplotlib.pyplot as plt
                X = filtered_data[features]
                y = filtered_data[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                st.write("### Model Performance")
                st.write(f"R² Score: {r2:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                importances = rf.feature_importances_
                indices = np.argsort(importances)[::-1]
                st.write("#### Feature Importances")
                for i in indices:
                    st.write(f"{X.columns[i]}: {importances[i]:.3f}")
                fig, ax = plt.subplots()
                ax.bar(range(X.shape[1]), importances[indices], align="center")
                ax.set_xticks(range(X.shape[1]))
                ax.set_xticklabels(X.columns[indices], rotation=90)
                ax.set_title("Random Forest Feature Importances")
                ax.set_ylabel("Importance")
                ax.set_xlabel("Features")
                st.pyplot(fig)
                plt.close(fig)
                st.info("If R² is low, try increasing n_estimators.")
                unique_id = str(time.time())
                title = f"Random Forest Results (n_estimators={n_estimators})"
                element_key = f"Advanced Analysis: {title}"
                st.session_state.report_elements.append(element_key)
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.session_state.report_content[element_key] = {
                    "type": "plot", "image": buf, "title": title,
                    "caption": f"Random Forest with n_estimators={n_estimators} and features {', '.join(features)}."
                }
            else:
                st.warning("Please select at least one feature column.")

def run_regression_analysis(filtered_data, numeric_columns, normalize):
    import time
    with st.form("regression_form_reg"):
        target = st.selectbox("Target variable (y)", numeric_columns, key="target_reg")
        features = st.multiselect("Feature variables (X)", numeric_columns, default=numeric_columns[:-1], key="features_reg")
        if st.form_submit_button("Run Regression"):
            if len(features) > 0:
                from sklearn.model_selection import train_test_split
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score, mean_squared_error
                import matplotlib.pyplot as plt
                X = filtered_data[features]
                y = filtered_data[target]
                if normalize:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                st.write("### Model Performance")
                st.write(f"R² Score: {r2:.2f}")
                st.write(f"Mean Squared Error: {mse:.2f}")
                st.write("#### Coefficients")
                if normalize:
                    for i, coef in enumerate(lr.coef_):
                        st.write(f"Feature {i+1}: {coef:.3f}")
                else:
                    for feat, coef in zip(features, lr.coef_):
                        st.write(f"{feat}: {coef:.3f}")
                if len(features) == 1:
                    try:
                        fig, ax = plt.subplots()
                        ax.scatter(X_test, y_test, color='blue', label='Actual')
                        ax.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
                        ax.set_title("Regression Analysis")
                        ax.set_xlabel(features[0])
                        ax.set_ylabel(target)
                        ax.legend()
                        st.pyplot(fig)
                        plt.close(fig)
                        unique_id = str(time.time())
                        title = f"Regression Plot ({features[0]})"
                        element_key = f"Advanced Analysis: {title}"
                        st.session_state.report_elements.append(element_key)
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        buf.seek(0)
                        st.session_state.report_content[element_key] = {
                            "type": "plot", "image": buf, "title": title,
                            "caption": f"Regression plot for {target} vs {features[0]}."
                        }
                    except Exception as e:
                        st.warning(f"Could not plot regression line: {e}")
                st.info("If R² is low, try enabling normalization.")
            else:
                st.warning("Please select at least one feature column.")

def run_time_series_analysis(filtered_data, columns, numeric_columns):
    import time
    with st.form("time_series_form_ts"):
        date_column = st.selectbox("Date column", columns, key="date_ts")
        value_column = st.selectbox("Value column", numeric_columns, key="value_ts")
        if st.form_submit_button("Run Time Series Analysis"):
            if date_column and value_column:
                import matplotlib.pyplot as plt
                import pandas as pd
                try:
                    filtered_data[date_column] = pd.to_datetime(filtered_data[date_column], errors='coerce')
                    time_series_data = filtered_data.dropna(subset=[date_column]).groupby(date_column)[value_column].sum().reset_index()
                    fig, ax = plt.subplots()
                    ax.plot(time_series_data[date_column], time_series_data[value_column])
                    ax.set_title(f"Time Series Analysis of {value_column} over Time")
                    ax.set_xlabel(date_column)
                    ax.set_ylabel(value_column)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.info("If the plot is noisy, try aggregating by week or month.")
                    unique_id = str(time.time())
                    title = f"Time Series Plot ({value_column})"
                    element_key = f"Advanced Analysis: {title}"
                    st.session_state.report_elements.append(element_key)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.session_state.report_content[element_key] = {
                        "type": "plot", "image": buf, "title": title,
                        "caption": f"Time series plot for {value_column}."
                    }
                except Exception as e:
                    st.error(f"Error plotting time series: {e}")
            else:
                st.warning("Please select both a date column and a value column.")

def main():
    st.title("Plotiva – Plot smarter. Predict faster")
    st.subheader("Developed by [@Mr.P.K](https://pradipta2005.github.io/My_Portfolio/)")
    st.markdown("""
    <div style='text-align:center; margin-bottom:2em; margin-top:-1em;'>
        <span style='font-size:1.1em; color:#636e72;'>
            Welcome to <b>Plotiva</b>! Effortlessly visualize, analyze, and predict with your data.<br>
            <span style='color:#00b894;'>Features:</span> Fast CSV upload, interactive plot gallery, advanced ML analysis, and a beautiful dashboard.<br>
            <span style='color:#0984e3;'>Built for speed, clarity, and insight.</span>
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Initialize session state for dashboard and plot management only
    if 'report_elements' not in st.session_state:
        st.session_state.report_elements = []
    if 'report_content' not in st.session_state:
        st.session_state.report_content = {}

    # Sidebar for uploading the file
    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        st.header("Filters")
        if uploaded_file:
            data = load_data(uploaded_file)
            if data is not None:
                columns, numeric_columns, categorical_columns = get_column_types(data)
                
                filter_columns = st.multiselect("Select columns to filter", columns)
                filter_values = {}

                for filter_column in filter_columns:
                    if filter_column in categorical_columns:
                        unique_values = data[filter_column].unique()
                        values = st.multiselect(
                            f"Select values for {filter_column}", 
                            options=unique_values
                        )
                        filter_values[filter_column] = values
                    elif filter_column in numeric_columns:
                        min_val = float(data[filter_column].min())
                        max_val = float(data[filter_column].max())
                        range_values = st.slider(
                            f"Select range for {filter_column}", 
                            min_value=min_val, 
                            max_value=max_val, 
                            value=(min_val, max_val)
                        )
                        filter_values[filter_column] = range_values
            else:
                data = pd.DataFrame() # Empty dataframe
        else:
            data = pd.DataFrame() # Empty dataframe

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin analysis", icon="🔔")
        st.stop()
    
    if data.empty and uploaded_file:
        st.error("Failed to load the dataset. Please check the file format and try again.")
        st.stop()
    elif data.empty:
        st.stop()

    # Get column types for the potentially filtered data later on
    columns, numeric_columns, categorical_columns = get_column_types(data)

    # Apply filters
    if 'filter_values' in locals():
        filtered_data = apply_filters(data, filter_values)
    else:
        filtered_data = data

    # Data preview and basic information
    st.write("### About Dataset")
    
    with st.expander("Data Preview"):
        rows = st.slider("Number of rows", min_value=1, max_value=len(filtered_data), value=5)
        st.dataframe(filtered_data.head(rows))

    with st.expander("Dataset Information"):
        buffer = io.StringIO()
        filtered_data.info(buf=buffer)
        st.text(buffer.getvalue())

    with st.expander("Show Descriptive Statistics"):
        st.write(filtered_data.describe())

    # Display column information
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Numerical Columns:", numeric_columns)
    with col2:
        st.write("Categorical Columns:", categorical_columns)

    # Handle missing values
    if filtered_data.isna().any().any():
        st.info("Missing values detected. Filled missing values with the median for numeric columns.")
        filtered_data[numeric_columns] = filtered_data[numeric_columns].apply(
            lambda col: col.fillna(col.median()), axis=0
        )
    else:
        st.info("No null or missing values are found in the Dataset.")

    # Add Visualization Recommendation Feature
    st.info("AI-Powered Visualization Recommendation")

    # Extracting features from data to guide visualization selection
    def extract_features(df):
        num_numeric_cols = len(df.select_dtypes(include=['float64', 'int64']).columns)
        num_categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        num_rows = len(df)
        return [num_numeric_cols, num_categorical_cols, num_rows]

    # A simple rule-based model for demo purposes
    def recommend_visualization(df):
        features = extract_features(df)
        num_numeric_cols, num_categorical_cols, num_rows = features

        recommended_vis = []

        if num_numeric_cols > 1:
            recommended_vis.append("Scatter Plot")
            recommended_vis.append("Pair Plot")
            recommended_vis.append("Heatmap")
            recommended_vis.append("Hexbin Plot")

        if num_categorical_cols > 0 and num_numeric_cols > 0:
            recommended_vis.append("Bar Chart")
            recommended_vis.append("Box Plot")
            recommended_vis.append("Violin Plot")
            recommended_vis.append("Count Plot")

        if num_numeric_cols == 1:
            recommended_vis.append("Line Chart")
            recommended_vis.append("Histogram")

        if num_categorical_cols > 0:
            recommended_vis.append("Pie/Donut Chart")

        if num_rows > 100:
            recommended_vis.append("Bubble Chart")

        return recommended_vis

    # Get recommended visualizations
    recommended_visualizations = recommend_visualization(filtered_data)
    st.write("### Recommended Visualizations Based on Your Data:")
    st.write(recommended_visualizations)

    # Allow users to select from all available plots
    all_available_plots = [
        "Line Chart", "Bar Chart", "Scatter Plot", "Heatmap",
        "Pair Plot", "Hexbin Plot", "Box Plot", "Violin Plot",
        "Count Plot", "Pie/Donut Chart", "Bubble Chart", "Histogram"
    ]
    selected_plots = st.multiselect("Select plots to display", all_available_plots)

    # Visualization Section
    st.write("## Data Visualization Gallery")
    num_columns = 3
    plot_gallery(num_columns, selected_plots, numeric_columns, categorical_columns, filtered_data, palette, columns)

    st.write("## Advanced Analysis")
    st.info("""
    **Important:** Before performing any kind of advanced analysis, please perform all necessary data pre-processing steps (e.g., handling missing values, encoding categories, scaling, etc.) for best results.
    """)
    st.write("Expand the sections below to run advanced machine learning models. The results can be added to your final report.")

    if filtered_data.empty:
        st.warning("Upload and process data to run advanced analysis.")
    else:
        with st.expander("KMeans Clustering"):
            st.info(r'''
            **KMeans Clustering**
            - **Formula:** Minimizes the sum of squared distances from each point to its assigned cluster center:
              
              $$ \sum_{i=1}^k \sum_{x \in C_i} \|x - \mu_i\|^2 $$
            - **How it works:** Assigns each data point to the nearest of $k$ cluster centers, then updates centers to the mean of assigned points, repeating until convergence.
            - **Things to know:**
                - Choose $k$ carefully (try several values).
                - Sensitive to feature scaling (standardize features if scales differ).
                - Random initialization can affect results (use higher $n_{init}$ for stability).
            ''')
            n_init = st.slider("Number of initializations (n_init)", 1, 20, 10, key="n_init_kmeans")
            run_kmeans_analysis(filtered_data, numeric_columns, n_init)

        with st.expander("Random Forest Analysis"):
            st.info(r'''
            **Random Forest Regression**
            - **Formula:** An ensemble of decision trees:
              
              $$ \hat{y} = \frac{1}{N_{trees}} \sum_{i=1}^{N_{trees}} T_i(X) $$
            - **How it works:** Builds many decision trees on random subsets of data/features, averages their predictions for regression (or majority vote for classification).
            - **Things to know:**
                - Handles non-linear relationships and feature interactions well.
                - Not sensitive to feature scaling.
                - Use more trees ($n_{estimators}$) for stability, but increases computation.
                - Can estimate feature importance.
            ''')
            n_estimators = st.slider("Number of Trees (n_estimators)", 10, 500, 100, key="n_estimators_rf")
            run_random_forest_analysis(filtered_data, numeric_columns, n_estimators)

        with st.expander("Regression Analysis"):
            st.info(r'''
            **Linear Regression**
            - **Formula:**
              
              $$ \hat{y} = \beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p $$
            - **How it works:** Fits a straight line (or hyperplane) to minimize squared error between predictions and actual values.
            - **Things to know:**
                - Assumes linear relationship between features and target.
                - Sensitive to outliers and feature scaling.
                - Use normalization if features have different scales.
            - **Note:** To display the regression plot, you must select a single feature variable ($X$).
            ''')
            normalize = st.checkbox("Enable normalization", value=False, key="normalize_reg")
            run_regression_analysis(filtered_data, numeric_columns, normalize)

        with st.expander("Time Series Analysis"):
            st.info(r'''
            **Time Series Analysis**
            - **Goal:** Analyze trends and patterns over time for a selected value column.
            - **Key steps:**
                - Ensure your date column is in datetime format.
                - Aggregate or resample if needed (e.g., by week or month).
                - Visualize to detect seasonality, trends, or anomalies.
            - **Things to know:**
                - Missing or irregular dates can affect results.
                - Outliers or missing values may need to be handled.
            ''')
            run_time_series_analysis(filtered_data, columns, numeric_columns)

    # Sidebar for dashboard configuration
    with st.sidebar:
        st.header("Dashboard Builder")
        st.write("Select plots from the gallery and advanced analysis to add to your dashboard.")
        dashboard_title = st.text_input("Dashboard Title", value="My Data Dashboard", key="dashboard_title_input")
        st.write("**Select Gallery Plots:**")
        gallery_selections = st.multiselect(
            "Gallery Plots",
            options=[el for el in st.session_state.report_elements if el.startswith("Plot:")],
            key="gallery_selections_dashboard",
            label_visibility="collapsed"
        )
        st.write("**Select Advanced Analysis Plots:**")
        advanced_selections = st.multiselect(
            "Advanced Analysis Plots",
            options=[el for el in st.session_state.report_elements if el.startswith("Advanced Analysis:") and st.session_state.report_content[el]["type"] == "plot"],
            key="advanced_selections_dashboard",
            label_visibility="collapsed"
        )
        if st.button("Dashboard", key="dashboard_button"):
            st.session_state.dashboard_clicked = True

    # Main dashboard display as a beautiful plot gallery
    if st.session_state.get("dashboard_clicked", False):
        st.markdown(f"<h1 style='text-align:center; color:#2c3e50; margin-bottom:0.5em'>{dashboard_title}</h1>", unsafe_allow_html=True)
        st.write("")
        dashboard_plots = st.session_state.get('gallery_selections_dashboard', []) + st.session_state.get('advanced_selections_dashboard', [])
        if not dashboard_plots:
            st.info("No plots selected for the dashboard. Please select plots from the sidebar.")
        else:
            st.markdown("<h3 style='color:#34495e; margin-top:1em'>Your Dashboard Gallery</h3>", unsafe_allow_html=True)
            cols = st.columns(3)
            for idx, plot_key in enumerate(dashboard_plots):
                content = st.session_state.report_content.get(plot_key)
                if content and content["type"] == "plot":
                    with cols[idx % 3]:
                        st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:1.1em; margin-bottom:0.5em'>{content['title']}</div>", unsafe_allow_html=True)
                        st.image(content["image"].getvalue(), use_column_width=True, caption=content.get("caption", ""))
                        st.write("")
            st.success("Dashboard ready! You can edit the title or plots anytime from the sidebar.")

if __name__ == "__main__":
    main()