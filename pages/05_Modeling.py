import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    from scipy.optimize import nnls
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ----------------------
# Page config
# ----------------------
st.set_page_config(page_title="Modeling", layout="wide")

st.title("📊 Modeling")

# ----------------------
# Storage setup
# ----------------------
STORAGE_DIR = "storage/models"
os.makedirs(STORAGE_DIR, exist_ok=True)

# ----------------------
# Session state init
# ----------------------
if "model_queue" not in st.session_state:
    st.session_state.model_queue = []

# ----------------------
# Dataset selection
# ----------------------
st.subheader("Step 1: Select Dataset")
uploaded_file = st.file_uploader("Upload a CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # ----------------------
    # Target + features
    # ----------------------
    st.subheader("Step 2: Define Model")

    target = st.selectbox("Target variable", df.columns)
    target_lag = st.number_input("Target lag (optional)", min_value=0, value=0, step=1)

    model_name = st.text_input("Model name")

    features = st.multiselect("Independent variables", [c for c in df.columns if c != target])

    force_nonnegative = st.checkbox("Force negative estimates to zero")

    model_type = st.selectbox("Model type", ["OLS", "NNLS", "Ridge", "Lasso"])

    alpha = None
    if model_type in ["Ridge", "Lasso"]:
        alpha = st.number_input("Alpha (regularization strength)", value=1.0, step=0.1)

    if st.button("➕ Add to Queue"):
        if not model_name or not features:
            st.warning("Please provide a model name and select features.")
        elif model_type == "NNLS" and not SCIPY_AVAILABLE:
            st.error("SciPy is required for NNLS. Please install scipy >= 1.10.")
        else:
            spec = {
                "dataset": uploaded_file.name,
                "target": target,
                "target_lag": target_lag,
                "name": model_name,
                "features": features,
                "force_nonnegative": force_nonnegative,
                "model_type": model_type,
                "alpha": alpha,
            }
            st.session_state.model_queue.append(spec)
            st.success(f"Added {model_name} to queue ✅")

# ----------------------
# Queue display
# ----------------------
if st.session_state.model_queue:
    st.subheader("Queued Models")
    st.dataframe(pd.DataFrame(st.session_state.model_queue))

    if st.button("🚀 Run All Models"):
        runs_index_path = os.path.join(STORAGE_DIR, "runs_index.json")
        if os.path.exists(runs_index_path):
            with open(runs_index_path, "r") as f:
                runs_index = json.load(f)
        else:
            runs_index = []

        for spec in st.session_state.model_queue:
            timestamp = int(time.time())
            run_dir = os.path.join(STORAGE_DIR, f"{timestamp}_{spec['name']}")
            os.makedirs(run_dir, exist_ok=True)

            # Prepare data
            X = df[spec["features"]].copy()
            y = df[spec["target"]].copy()

            if spec["target_lag"] > 0:
                y = y.shift(spec["target_lag"]).dropna()
                X = X.iloc[spec["target_lag"]:]

            # Fit model
            if spec["model_type"] == "OLS":
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)

            elif spec["model_type"] == "Ridge":
                model = Ridge(alpha=spec["alpha"])
                model.fit(X, y)
                y_pred = model.predict(X)

            elif spec["model_type"] == "Lasso":
                model = Lasso(alpha=spec["alpha"])
                model.fit(X, y)
                y_pred = model.predict(X)

            elif spec["model_type"] == "NNLS":
                coefs, _ = nnls(X.values, y.values)
                y_pred = X.values @ coefs
                model = {"coefs": coefs.tolist()}

            if spec["force_nonnegative"]:
                y_pred = np.maximum(y_pred, 0)

            # Metrics
            metrics = {
                "r2": float(r2_score(y, y_pred)),
                "rmse": float(mean_squared_error(y, y_pred, squared=False)),
                "mae": float(mean_absolute_error(y, y_pred)),
            }

            # Save outputs
            pd.DataFrame({"y_true": y, "y_pred": y_pred}).to_csv(
                os.path.join(run_dir, "predictions.csv"), index=False
            )
            with open(os.path.join(run_dir, "meta.json"), "w") as f:
                json.dump({"spec": spec, "metrics": metrics}, f, indent=2)

            runs_index.append({"name": spec["name"], "dir": run_dir, "metrics": metrics})

        with open(runs_index_path, "w") as f:
            json.dump(runs_index, f, indent=2)

        st.success("All models have been run and saved ✅")

        # Clear queue
        st.session_state.model_queue = []
