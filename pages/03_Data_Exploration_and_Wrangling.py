import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Data Exploration & Wrangling", layout="wide")

# Title and description
st.title("Data Exploration & Wrangling")
st.write("""
Use this section to inspect the raw data, clean it, engineer features, and visualize distributions and relationships.
""")

@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_excel(uploaded_file)
    return df

# Sidebar for data upload
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Basic data summary
    st.subheader("Data Summary")
    buffer = df.describe(include='all').T
    buffer["missing"] = df.isna().sum()
    buffer["dtype"] = df.dtypes
    st.write(buffer)

    # Select columns for analysis
    st.subheader("Select Columns for Analysis")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    category_cols = df.select_dtypes(include='object').columns.tolist()

    st.write("Numeric columns:", numeric_cols)
    st.write("Categorical columns:", category_cols)

    selected_num = st.multiselect("Choose numeric columns to visualize", numeric_cols)
    selected_cat = st.multiselect("Choose categorical columns to inspect", category_cols)

    # Visualizations for numeric
    if selected_num:
        st.subheader("Numeric Features Distribution")
        for col in selected_num:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

        st.subheader("Scatter Plot")
        if len(selected_num) >= 2:
            x_col, y_col = st.columns(2)
            x_sel = x_col.selectbox("X-axis", selected_num, key="xaxis")
            y_sel = y_col.selectbox("Y-axis", selected_num, key="yaxis")
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_sel, y=y_sel, ax=ax)
            ax.set_title(f"{x_sel} vs. {y_sel}")
            st.pyplot(fig)

    # Analysis for categorical
    if selected_cat:
        st.subheader("Categorical Feature Counts")
        for col in selected_cat:
            fig, ax = plt.subplots()
            df[col].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f"Count plot of {col}")
            st.pyplot(fig)

    # Missing value analysis
    st.subheader("Missing Value Analysis")
    missing_df = df.isnull().sum()
    fig, ax = plt.subplots()
    missing_df[missing_df > 0].sort_values(ascending=False).plot.bar(ax=ax)
    ax.set_title("Missing Values per Column")
    st.pyplot(fig)

    # Option to handle missing
    st.subheader("Handling Missing Values")
    method = st.radio("Choose method", ("Drop rows with missing values", "Fill with mean (numeric) / mode (categorical)"))
    if st.button("Apply Missing Value Strategy"):
        df_clean = df.copy()
        if method.startswith("Drop"):
            df_clean = df_clean.dropna()
        else:
            for col in df_clean.columns:
                if df_clean[col].dtype in [np.float64, np.int64]:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                else:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        st.success("Missing values processed.")
        st.dataframe(df_clean.head())

else:
    st.info("Please upload a dataset to explore.")

