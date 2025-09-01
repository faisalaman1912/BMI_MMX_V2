import os
import time
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Ingestion & Curation Desk", layout="wide")
st.title("📂 Ingestion & Curation Desk")

UPLOAD_DIR = "storage/uploads"
CURATED_DIR = "storage/curated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CURATED_DIR, exist_ok=True)

# ---------------------------------------------------
# File Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ File {uploaded_file.name} uploaded successfully!")

# ---------------------------------------------------
# File Manager
# ---------------------------------------------------
st.subheader("Available Files")

files = os.listdir(UPLOAD_DIR)
file_details = []

for f in files:
    path = os.path.join(UPLOAD_DIR, f)
    stats = os.stat(path)
    file_type = f.split(".")[-1].upper()
    size_kb = stats.st_size / 1024
    mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats.st_mtime))

    try:
        if f.endswith(".csv"):
            df = pd.read_csv(path, nrows=5)
        elif f.endswith(".xlsx"):
            df = pd.read_excel(path, nrows=5)
        else:
            df = pd.DataFrame()
        rows, cols = df.shape
    except Exception:
        rows, cols = "-", "-"

    file_details.append({
        "Name": f,
        "Type": file_type,
        "Size (KB)": round(size_kb, 2),
        "Modified": mod_time,
        "Rows (preview)": rows,
        "Cols (preview)": cols,
        "Delete": f
    })

if file_details:
    df_files = pd.DataFrame(file_details)
    st.dataframe(df_files, use_container_width=True, height=250 if len(files) > 5 else None)

    delete_file = st.selectbox("Select file to delete", ["-"] + files)
    if delete_file != "-":
        os.remove(os.path.join(UPLOAD_DIR, delete_file))
        st.warning(f"🗑 File {delete_file} deleted. Refresh to update.")

# ---------------------------------------------------
# File Analyzer
# ---------------------------------------------------
st.subheader("📊 File Analyzer")

if files:
    selected_file = st.selectbox("Select a file to analyze", files)

    file_path = os.path.join(UPLOAD_DIR, selected_file)
    if selected_file.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    st.write("### Preview (Top 100 rows)")
    st.dataframe(df.head(100), use_container_width=True, height=300)

    st.write("### Column Summary & Imputation Options")
    summary = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str),
        "Null Count": df.isnull().sum().values,
        "Null %": (df.isnull().mean().values * 100).round(2)
    })

    impute_methods = {}
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            impute_methods[col] = st.selectbox(
                f"Imputation for {col}", ["-", "Mean", "Median", "Mode"], key=f"imp_{col}"
            )

    if st.button("Apply Imputation & Save"):
        df_imputed = df.copy()
        for col, method in impute_methods.items():
            if method == "Mean":
                df_imputed[col].fillna(df[col].mean(), inplace=True)
            elif method == "Median":
                df_imputed[col].fillna(df[col].median(), inplace=True)
            elif method == "Mode":
                df_imputed[col].fillna(df[col].mode()[0], inplace=True)

        new_file = f"curated_{int(time.time())}_{selected_file}"
        new_path = os.path.join(CURATED_DIR, new_file)
        if selected_file.endswith(".csv"):
            df_imputed.to_csv(new_path, index=False)
        else:
            df_imputed.to_excel(new_path, index=False)

        st.success(f"✅ Curated file saved: {new_file}")
else:
    st.info("Upload files to start analysis.")
