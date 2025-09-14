"""
Premium Feature Engineering Module
Advanced data transformation and feature creation tools
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go

def render_premium_feature_engineering():
    """Render premium feature engineering interface"""
    
    st.subheader("🔧 Advanced Feature Engineering")
    st.markdown("Create powerful new features from your existing data")
    
    # Get current data
    if 'working_data' not in st.session_state or st.session_state.working_data.empty:
        st.warning("No data available. Please upload data first.")
        return
    
    df = st.session_state.working_data
    
    # Feature engineering tabs
    fe_tabs = st.tabs([
        "➕ Create Features", 
        "🔢 Mathematical Operations", 
        "📅 Date/Time Features",
        "🏷️ Categorical Encoding",
        "📊 Statistical Features"
    ])
    
    with fe_tabs[0]:  # Create Features
        st.markdown("#### ➕ Create New Features")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("**Feature Creation Options**")
            
            # Column selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            feature_type = st.selectbox(
                "Feature Type",
                ["Mathematical", "Conditional", "Binning", "Interaction"]
            )
            
            if feature_type == "Mathematical":
                st.markdown("**Mathematical Operations**")
                col1_select = st.selectbox("Column 1", numeric_cols, key="math_col1")
                operation = st.selectbox("Operation", ["+", "-", "*", "/", "**", "log", "sqrt"])
                
                if operation in ["+", "-", "*", "/", "**"]:
                    col2_select = st.selectbox("Column 2", numeric_cols, key="math_col2")
                    new_col_name = st.text_input("New Column Name", f"{col1_select}_{operation}_{col2_select}")
                else:
                    new_col_name = st.text_input("New Column Name", f"{operation}_{col1_select}")
                
                if st.button("Create Mathematical Feature"):
                    try:
                        if operation == "+":
                            df[new_col_name] = df[col1_select] + df[col2_select]
                        elif operation == "-":
                            df[new_col_name] = df[col1_select] - df[col2_select]
                        elif operation == "*":
                            df[new_col_name] = df[col1_select] * df[col2_select]
                        elif operation == "/":
                            # Prevent division by zero
                            df[new_col_name] = np.where(df[col2_select] != 0, 
                                                       df[col1_select] / df[col2_select], 0)
                        elif operation == "**":
                            df[new_col_name] = df[col1_select] ** df[col2_select]
                        elif operation == "log":
                            df[new_col_name] = np.log(df[col1_select].clip(lower=1e-10))
                        elif operation == "sqrt":
                            df[new_col_name] = np.sqrt(df[col1_select].clip(lower=0))
                        
                        st.session_state.working_data = df
                        st.success(f"✅ Created feature: {new_col_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating feature: {str(e)}")
            
            elif feature_type == "Conditional":
                st.markdown("**Conditional Features**")
                condition_col = st.selectbox("Condition Column", df.columns.tolist())
                condition_type = st.selectbox("Condition", ["Greater than", "Less than", "Equal to", "Contains"])
                
                if condition_type in ["Greater than", "Less than"]:
                    threshold = st.number_input("Threshold Value")
                elif condition_type == "Equal to":
                    threshold = st.text_input("Equal to Value")
                else:  # Contains
                    threshold = st.text_input("Contains Text")
                
                true_value = st.text_input("Value if True", "Yes")
                false_value = st.text_input("Value if False", "No")
                new_col_name = st.text_input("New Column Name", f"{condition_col}_condition")
                
                if st.button("Create Conditional Feature"):
                    try:
                        if condition_type == "Greater than":
                            df[new_col_name] = np.where(df[condition_col] > threshold, true_value, false_value)
                        elif condition_type == "Less than":
                            df[new_col_name] = np.where(df[condition_col] < threshold, true_value, false_value)
                        elif condition_type == "Equal to":
                            df[new_col_name] = np.where(df[condition_col] == threshold, true_value, false_value)
                        else:  # Contains
                            df[new_col_name] = np.where(df[condition_col].astype(str).str.contains(str(threshold), na=False), 
                                                       true_value, false_value)
                        
                        st.session_state.working_data = df
                        st.success(f"✅ Created feature: {new_col_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating feature: {str(e)}")
        
        with col2:
            st.markdown("**Feature Preview**")
            if len(df.columns) > 0:
                # Show recent columns (likely new features)
                recent_cols = df.columns[-5:].tolist()
                preview_df = df[recent_cols].head(10)
                st.dataframe(preview_df, use_container_width=True)
                
                # Show basic statistics for new features
                if len(recent_cols) > 0:
                    st.markdown("**Recent Feature Statistics**")
                    stats_df = df[recent_cols].describe()
                    st.dataframe(stats_df, use_container_width=True)
    
    with fe_tabs[1]:  # Mathematical Operations
        st.markdown("#### 🔢 Advanced Mathematical Operations")
        
        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Normalization & Scaling**")
                norm_col = st.selectbox("Select Column", numeric_cols, key="norm_col")
                norm_method = st.selectbox("Method", ["Min-Max Scaling", "Z-Score", "Robust Scaling"])
                
                if st.button("Apply Normalization"):
                    try:
                        new_col_name = f"{norm_col}_{norm_method.lower().replace(' ', '_').replace('-', '_')}"
                        
                        if norm_method == "Min-Max Scaling":
                            min_val = df[norm_col].min()
                            max_val = df[norm_col].max()
                            df[new_col_name] = (df[norm_col] - min_val) / (max_val - min_val)
                        elif norm_method == "Z-Score":
                            mean_val = df[norm_col].mean()
                            std_val = df[norm_col].std()
                            df[new_col_name] = (df[norm_col] - mean_val) / std_val
                        else:  # Robust Scaling
                            median_val = df[norm_col].median()
                            q75 = df[norm_col].quantile(0.75)
                            q25 = df[norm_col].quantile(0.25)
                            iqr = q75 - q25
                            df[new_col_name] = (df[norm_col] - median_val) / iqr
                        
                        st.session_state.working_data = df
                        st.success(f"✅ Applied {norm_method} to {norm_col}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error applying normalization: {str(e)}")
            
            with col2:
                st.markdown("**Binning & Discretization**")
                bin_col = st.selectbox("Select Column", numeric_cols, key="bin_col")
                n_bins = st.slider("Number of Bins", 2, 10, 5)
                bin_method = st.selectbox("Binning Method", ["Equal Width", "Equal Frequency", "Custom"])
                
                if st.button("Create Bins"):
                    try:
                        new_col_name = f"{bin_col}_binned"
                        
                        if bin_method == "Equal Width":
                            df[new_col_name] = pd.cut(df[bin_col], bins=n_bins, labels=False)
                        elif bin_method == "Equal Frequency":
                            df[new_col_name] = pd.qcut(df[bin_col], q=n_bins, labels=False, duplicates='drop')
                        
                        st.session_state.working_data = df
                        st.success(f"✅ Created bins for {bin_col}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating bins: {str(e)}")
    
    with fe_tabs[2]:  # Date/Time Features
        st.markdown("#### 📅 Date/Time Feature Extraction")
        
        # Find datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Also check for columns that might be dates but stored as strings
        potential_date_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(5).astype(str)
                if any(len(str(val)) > 8 and ('-' in str(val) or '/' in str(val)) for val in sample_values):
                    potential_date_cols.append(col)
        
        all_date_cols = datetime_cols + potential_date_cols
        
        if len(all_date_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Extract Date Components**")
                date_col = st.selectbox("Select Date Column", all_date_cols)
                
                components = st.multiselect(
                    "Components to Extract",
                    ["Year", "Month", "Day", "Weekday", "Quarter", "Week of Year"],
                    default=["Year", "Month", "Day"]
                )
                
                if st.button("Extract Date Components"):
                    try:
                        # Convert to datetime if not already
                        if df[date_col].dtype != 'datetime64[ns]':
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        
                        for component in components:
                            if component == "Year":
                                df[f"{date_col}_year"] = df[date_col].dt.year
                            elif component == "Month":
                                df[f"{date_col}_month"] = df[date_col].dt.month
                            elif component == "Day":
                                df[f"{date_col}_day"] = df[date_col].dt.day
                            elif component == "Weekday":
                                df[f"{date_col}_weekday"] = df[date_col].dt.dayofweek
                            elif component == "Quarter":
                                df[f"{date_col}_quarter"] = df[date_col].dt.quarter
                            elif component == "Week of Year":
                                df[f"{date_col}_week"] = df[date_col].dt.isocalendar().week
                        
                        st.session_state.working_data = df
                        st.success(f"✅ Extracted {len(components)} date components")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error extracting date components: {str(e)}")
            
            with col2:
                st.markdown("**Date Calculations**")
                if len(all_date_cols) >= 2:
                    date_col1 = st.selectbox("Start Date", all_date_cols, key="date1")
                    date_col2 = st.selectbox("End Date", all_date_cols, key="date2")
                    unit = st.selectbox("Unit", ["Days", "Weeks", "Months", "Years"])
                    
                    if st.button("Calculate Date Difference"):
                        try:
                            # Convert to datetime if not already
                            if df[date_col1].dtype != 'datetime64[ns]':
                                df[date_col1] = pd.to_datetime(df[date_col1], errors='coerce')
                            if df[date_col2].dtype != 'datetime64[ns]':
                                df[date_col2] = pd.to_datetime(df[date_col2], errors='coerce')
                            
                            diff = df[date_col2] - df[date_col1]
                            
                            if unit == "Days":
                                df[f"{date_col1}_to_{date_col2}_days"] = diff.dt.days
                            elif unit == "Weeks":
                                df[f"{date_col1}_to_{date_col2}_weeks"] = diff.dt.days / 7
                            elif unit == "Months":
                                df[f"{date_col1}_to_{date_col2}_months"] = diff.dt.days / 30.44
                            elif unit == "Years":
                                df[f"{date_col1}_to_{date_col2}_years"] = diff.dt.days / 365.25
                            
                            st.session_state.working_data = df
                            st.success(f"✅ Calculated date difference in {unit.lower()}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error calculating date difference: {str(e)}")
                else:
                    st.info("Need at least 2 date columns for date calculations")
        else:
            st.info("No date/time columns detected in your data")
    
    with fe_tabs[3]:  # Categorical Encoding
        st.markdown("#### 🏷️ Categorical Encoding")
        
        if len(categorical_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Encoding Methods**")
                cat_col = st.selectbox("Select Categorical Column", categorical_cols)
                encoding_method = st.selectbox(
                    "Encoding Method",
                    ["One-Hot Encoding", "Label Encoding", "Frequency Encoding", "Target Encoding"]
                )
                
                if encoding_method == "Target Encoding" and len(numeric_cols) > 0:
                    target_col = st.selectbox("Target Column", numeric_cols)
                
                if st.button("Apply Encoding"):
                    try:
                        if encoding_method == "One-Hot Encoding":
                            # Create dummy variables
                            dummies = pd.get_dummies(df[cat_col], prefix=cat_col)
                            df = pd.concat([df, dummies], axis=1)
                            
                        elif encoding_method == "Label Encoding":
                            # Simple label encoding
                            unique_vals = df[cat_col].unique()
                            label_map = {val: i for i, val in enumerate(unique_vals)}
                            df[f"{cat_col}_encoded"] = df[cat_col].map(label_map)
                            
                        elif encoding_method == "Frequency Encoding":
                            # Encode by frequency
                            freq_map = df[cat_col].value_counts().to_dict()
                            df[f"{cat_col}_frequency"] = df[cat_col].map(freq_map)
                            
                        elif encoding_method == "Target Encoding":
                            # Encode by target mean
                            target_map = df.groupby(cat_col)[target_col].mean().to_dict()
                            df[f"{cat_col}_target_encoded"] = df[cat_col].map(target_map)
                        
                        st.session_state.working_data = df
                        st.success(f"✅ Applied {encoding_method} to {cat_col}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error applying encoding: {str(e)}")
            
            with col2:
                st.markdown("**Encoding Preview**")
                if cat_col in df.columns:
                    # Show value counts
                    value_counts = df[cat_col].value_counts().head(10)
                    st.markdown("**Value Counts:**")
                    st.dataframe(value_counts, use_container_width=True)
                    
                    # Show unique values count
                    st.metric("Unique Values", df[cat_col].nunique())
        else:
            st.info("No categorical columns found in your data")
    
    with fe_tabs[4]:  # Statistical Features
        st.markdown("#### 📊 Statistical Feature Creation")
        
        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Rolling Statistics**")
                stat_col = st.selectbox("Select Column", numeric_cols, key="stat_col")
                window_size = st.slider("Window Size", 2, 20, 5)
                stat_type = st.selectbox("Statistic", ["Mean", "Median", "Std", "Min", "Max"])
                
                if st.button("Create Rolling Statistic"):
                    try:
                        new_col_name = f"{stat_col}_rolling_{stat_type.lower()}_{window_size}"
                        
                        if stat_type == "Mean":
                            df[new_col_name] = df[stat_col].rolling(window=window_size).mean()
                        elif stat_type == "Median":
                            df[new_col_name] = df[stat_col].rolling(window=window_size).median()
                        elif stat_type == "Std":
                            df[new_col_name] = df[stat_col].rolling(window=window_size).std()
                        elif stat_type == "Min":
                            df[new_col_name] = df[stat_col].rolling(window=window_size).min()
                        elif stat_type == "Max":
                            df[new_col_name] = df[stat_col].rolling(window=window_size).max()
                        
                        st.session_state.working_data = df
                        st.success(f"✅ Created rolling {stat_type.lower()} feature")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating rolling statistic: {str(e)}")
            
            with col2:
                st.markdown("**Interaction Features**")
                if len(numeric_cols) >= 2:
                    interact_col1 = st.selectbox("Column 1", numeric_cols, key="interact1")
                    interact_col2 = st.selectbox("Column 2", [col for col in numeric_cols if col != interact_col1], key="interact2")
                    
                    if st.button("Create Interaction Feature"):
                        try:
                            new_col_name = f"{interact_col1}_x_{interact_col2}"
                            df[new_col_name] = df[interact_col1] * df[interact_col2]
                            
                            # Also create ratio if no zeros
                            if (df[interact_col2] != 0).all():
                                ratio_col_name = f"{interact_col1}_div_{interact_col2}"
                                df[ratio_col_name] = df[interact_col1] / df[interact_col2]
                            
                            st.session_state.working_data = df
                            st.success(f"✅ Created interaction features")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error creating interaction feature: {str(e)}")
                else:
                    st.info("Need at least 2 numeric columns for interactions")
    
    # Feature summary
    st.markdown("---")
    st.markdown("### 📊 Feature Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Features", len(df.columns))
    
    with col2:
        new_features = len(df.columns) - len(st.session_state.get('original_data', df).columns)
        st.metric("New Features", max(0, new_features))
    
    with col3:
        if st.button("🔄 Reset to Original Data"):
            if 'original_data' in st.session_state:
                st.session_state.working_data = st.session_state.original_data.copy()
                st.success("✅ Reset to original data")
                st.rerun()

def apply_ai_suggestion(df, suggestion_type, **kwargs):
    """Apply AI-suggested feature engineering"""
    try:
        if suggestion_type == "create_ratio":
            col1, col2 = kwargs['col1'], kwargs['col2']
            new_col = f"{col1}_to_{col2}_ratio"
            df[new_col] = df[col1] / df[col2].replace(0, np.nan)
            
        elif suggestion_type == "create_interaction":
            col1, col2 = kwargs['col1'], kwargs['col2']
            new_col = f"{col1}_x_{col2}"
            df[new_col] = df[col1] * df[col2]
            
        elif suggestion_type == "normalize":
            col = kwargs['col']
            new_col = f"{col}_normalized"
            df[new_col] = (df[col] - df[col].mean()) / df[col].std()
        
        return df, f"✅ Applied AI suggestion: {suggestion_type}"
        
    except Exception as e:
        return df, f"❌ Error applying suggestion: {str(e)}"