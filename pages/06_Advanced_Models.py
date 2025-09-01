import os
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Advanced Models", layout="wide")

st.title("⚡ Advanced Models")

BASE_DIR = "storage/models"
ADV_DIR = "storage/advanced_models"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(ADV_DIR, exist_ok=True)


# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
def list_saved_models():
    """Return list of saved base models."""
    models = []
    if os.path.exists(BASE_DIR):
        for run in os.listdir(BASE_DIR):
            run_path = os.path.join(BASE_DIR, run)
            if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, "meta.json")):
                models.append(run)
    return models


def load_model(run_name):
    """Load model metadata and predictions."""
    meta_path = os.path.join(BASE_DIR, run_name, "meta.json")
    preds_path = os.path.join(BASE_DIR, run_name, "predictions.csv")
    if not os.path.exists(meta_path):
        return None, None
    with open(meta_path, "r") as f:
        meta = json.load(f)
    preds = pd.read_csv(preds_path) if os.path.exists(preds_path) else None
    return meta, preds


def save_advanced_model(name, adv_type, results, details):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(ADV_DIR, f"{ts}_{name}")
    os.makedirs(run_dir, exist_ok=True)

    results.to_csv(os.path.join(run_dir, "results.csv"), index=False)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump({"name": name, "type": adv_type, "details": details}, f, indent=2)

    st.success(f"Advanced model saved: {run_dir}")


# ---------------------------------------------------
# Main UI
# ---------------------------------------------------
base_models = list_saved_models()
if not base_models:
    st.warning("⚠️ No base models found. Please run 'Modeling' first.")
    st.stop()

selected_model = st.selectbox("Select Base Model", base_models)
meta, preds = load_model(selected_model)

if meta is None:
    st.error("❌ Could not load selected model metadata.")
    st.stop()

st.write("### Base Model Info")
st.json(meta)

adv_type = st.radio(
    "Select Advanced Model Type",
    ["Residual Modelling", "Breakout Modelling", "Pathway Modelling"]
)

# ---------------------------------------------------
# Residual Modelling
# ---------------------------------------------------
if adv_type == "Residual Modelling":
    st.subheader("Residual Modelling")
    available_metrics = list(preds.columns)
    selected_metrics = st.multiselect("Select metrics to redistribute base impact", available_metrics)

    if st.button("Run Residual Model"):
        if not selected_metrics:
            st.error("Please select at least one metric.")
        else:
            X = preds[selected_metrics].values
            y = preds[meta["target"]].values
            model = LinearRegression().fit(X, y)

            results = preds.copy()
            for i, col in enumerate(selected_metrics):
                results[f"{col}_residual_impact"] = model.coef_[i] * results[col]
            results["residual_base"] = model.intercept_

            save_advanced_model("residual", adv_type, results, {"metrics": selected_metrics})


# ---------------------------------------------------
# Breakout Modelling
# ---------------------------------------------------
elif adv_type == "Breakout Modelling":
    st.subheader("Breakout Modelling")
    available_cols = list(preds.columns)
    breakout_target = st.selectbox("Select channel to breakout", available_cols)
    breakout_metrics = st.multiselect("Select metrics to distribute impact", available_cols)

    if st.button("Run Breakout Model"):
        if not breakout_metrics:
            st.error("Please select breakout metrics.")
        else:
            X = preds[breakout_metrics].values
            y = preds[breakout_target].values
            model = LinearRegression().fit(X, y)

            results = preds.drop(columns=[breakout_target]).copy()
            for i, col in enumerate(breakout_metrics):
                results[f"{col}_breakout"] = model.coef_[i] * results[col]
            results[f"{breakout_target}_constant"] = model.intercept_

            save_advanced_model("breakout", adv_type, results,
                                {"target": breakout_target, "metrics": breakout_metrics})


# ---------------------------------------------------
# Pathway Modelling
# ---------------------------------------------------
elif adv_type == "Pathway Modelling":
    st.subheader("Pathway Modelling")
    available_cols = list(preds.columns)
    pathway_target = st.selectbox("Select target channel", available_cols)
    pathway_indep = st.selectbox("Select independent variable", available_cols)

    if st.button("Run Pathway Model"):
        if pathway_target == pathway_indep:
            st.error("Target and independent variable must be different.")
        else:
            X = preds[[pathway_indep]].values
            y = preds[pathway_target].values
            model = LinearRegression().fit(X, y)

            results = preds.copy()
            results[f"{pathway_target}_redistributed"] = model.coef_[0] * results[pathway_indep]
            results[f"{pathway_target}_constant"] = model.intercept_

            save_advanced_model("pathway", adv_type, results,
                                {"target": pathway_target, "independent": pathway_indep})
