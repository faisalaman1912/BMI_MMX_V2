import streamlit as st
st.set_page_config(page_title="Modeling", page_icon="🧪", layout="wide")
st.title("🧪 Modeling")

# Streamlit Modelling page 

# Workflow 

# - Select dataset 

# - Select target variable 

# - Optional: target lag 

# - Model name 

# - Independent variables (multi-select) 

# - Option: force negative predictions to zero 

# - Choose model type: OLS, NNLS, Ridge, Lasso (+ alpha for Ridge/Lasso) 

# - Add to queue (add multiple) 

# - Scrollable queue view 

# - Run all models: fit, compute metrics, save results to storage/models for downstream use 

# - Robust error handling; SciPy optional (for NNLS) 

 

from __future__ import annotations 

 

import os 

import io 

import json 

import time 

from dataclasses import asdict, dataclass 

from datetime import datetime 

from typing import Dict, List, Optional 

 

import numpy as np 

import pandas as pd 

import streamlit as st 

 

# ----- Optional deps 

try: 

    from scipy.optimize import nnls as scipy_nnls  # type: ignore 

    HAVE_SCIPY = True 

except Exception: 

    HAVE_SCIPY = False 

 

try: 

    from sklearn.linear_model import LinearRegression, Ridge, Lasso  # type: ignore 

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # type: ignore 

    HAVE_SKLEARN = True 

except Exception: 

    HAVE_SKLEARN = False 

 

# ---------------- Paths (reuse the same storage layout used elsewhere) ---------------- 

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 

STORAGE_DIR = os.path.join(BASE_DIR, "storage") 

DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets") 

MODELS_DIR = os.path.join(STORAGE_DIR, "models") 

os.makedirs(DATASETS_DIR, exist_ok=True) 

os.makedirs(MODELS_DIR, exist_ok=True) 

LEDGER_PATH = os.path.join(MODELS_DIR, "runs_index.json") 

 

# ---------------- Streamlit Page Config ---------------- 

st.set_page_config(page_title="Modelling", page_icon=":abacus:", layout="wide") 

st.title(":abacus: Modelling") 

 

if not HAVE_SKLEARN: 

    st.error( 

        "scikit-learn is required on this page. Add `scikit-learn>=1.2` to requirements.txt and redeploy.") 

 

# ---------------- Helpers ---------------- 

 

def _list_dataset_files() -> List[Dict]: 

    files = [] 

    if not os.path.exists(DATASETS_DIR): 

        return files 

    for fn in os.listdir(DATASETS_DIR): 

        full = os.path.join(DATASETS_DIR, fn) 

        if not os.path.isfile(full): 

            continue 

        ext = os.path.splitext(fn)[1].lower() 

        if ext not in (".csv", ".xlsx", ".xls"): 

            continue 

        try: 

            stat = os.stat(full) 

            files.append({ 

                "name": fn, 

                "path": full, 

                "ext": ext, 

                "modified_ts": int(stat.st_mtime) 

            }) 

        except Exception: 

            pass 

    files.sort(key=lambda x: x["modified_ts"], reverse=True) 

    return files 

 

 

def _read_df(path: str, ext: str) -> pd.DataFrame: 

    if ext == ".csv": 

        return pd.read_csv(path) 

    else: 

        try: 

            import openpyxl  # noqa: F401 

        except Exception: 

            st.warning("Excel reading requires openpyxl. Install openpyxl>=3.1.2. Trying pandas engine anyway…") 

        return pd.read_excel(path) 

 

 

def _numeric_columns(df: pd.DataFrame) -> List[str]: 

    cols = [] 

    for c in df.columns: 

        if pd.api.types.is_numeric_dtype(df[c]): 

            cols.append(c) 

    return cols 

 

 

def _safe_name(text: str) -> str: 

    return ("".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (text or "")).strip("_") or "model") 

 

 

@dataclass 

class ModelSpec: 

    dataset: str 

    target: str 

    target_lag: int 

    model_name: str 

    features: List[str] 

    force_nonneg_preds: bool 

    model_type: str  # ols|nnls|ridge|lasso 

    alpha: Optional[float] = None  # for ridge/lasso 

 

 

def _apply_target_lag(df: pd.DataFrame, target: str, lag: int) -> pd.DataFrame: 

    if lag and lag > 0: 

        df = df.copy() 

        df[target] = df[target].shift(lag) 

        df = df.dropna().reset_index(drop=True) 

    return df 

 

 

def _fit_and_score(spec: ModelSpec, df: pd.DataFrame) -> Dict: 

    if not HAVE_SKLEARN: 

        raise RuntimeError("scikit-learn is not available.") 

 

    if any(col not in df.columns for col in [spec.target, *spec.features]): 

        missing = [c for c in [spec.target, *spec.features] if c not in df.columns] 

        raise ValueError(f"Missing columns in dataset: {missing}") 

 

    work = _apply_target_lag(df, spec.target, spec.target_lag) 

    if work.empty: 

        raise ValueError("No data remains after applying target lag.") 

 

    X = work[spec.features].astype(float).values 

    y = work[spec.target].astype(float).values 

 

    intercept = 0.0 

    coefs = {} 

 

    mtype = spec.model_type.lower() 

 

    if mtype == "ols": 

        mdl = LinearRegression(fit_intercept=True) 

        mdl.fit(X, y) 

        yhat = mdl.predict(X) 

        intercept = float(mdl.intercept_) 

        for n, v in zip(spec.features, mdl.coef_): 

            coefs[n] = float(v) 

 

    elif mtype in ("ridge", "lasso"): 

        alpha = float(spec.alpha if spec.alpha is not None else 1.0) 

        if mtype == "ridge": 

            mdl = Ridge(alpha=alpha, fit_intercept=True) 

        else: 

            mdl = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000) 

        mdl.fit(X, y) 

        yhat = mdl.predict(X) 

        intercept = float(mdl.intercept_) 

        for n, v in zip(spec.features, mdl.coef_): 

            coefs[n] = float(v) 

 

    elif mtype == "nnls": 

        if not HAVE_SCIPY: 

            raise RuntimeError("NNLS requires SciPy. Add scipy>=1.10 to requirements.txt.") 

        # Include intercept as a nonnegative coefficient by augmenting ones 

        X_aug = np.column_stack([np.ones(len(X)), X]) 

        betas, _ = scipy_nnls(X_aug, y) 

        intercept = float(betas[0]) 

        for n, v in zip(spec.features, betas[1:]): 

            coefs[n] = float(v) 

        yhat = X_aug @ betas 

    else: 

        raise ValueError(f"Unsupported model_type: {spec.model_type}") 

 

    if spec.force_nonneg_preds: 

        yhat = np.clip(yhat, 0, None) 

 

    r2 = float(r2_score(y, yhat)) 

    rmse = float(np.sqrt(mean_squared_error(y, yhat))) 

    mae = float(mean_absolute_error(y, yhat)) 

 

    return { 

        "intercept": intercept, 

        "coefficients": coefs, 

        "metrics": {"r2": r2, "rmse": rmse, "mae": mae}, 

        "y": y, 

        "yhat": yhat, 

        "n_obs": int(len(y)) 

    } 

 

 

def _save_run(spec: ModelSpec, fit: Dict) -> Dict: 

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") 

    safe = _safe_name(spec.model_name) 

    run_id = f"run_{ts}_{int(time.time()*1000)}" 

    run_dir = os.path.join(MODELS_DIR, f"{ts}_{safe}") 

    os.makedirs(run_dir, exist_ok=True) 

 

    # Save predictions 

    preds_path = os.path.join(run_dir, "predictions.csv") 

    pd.DataFrame({"y": fit["y"], "yhat": fit["yhat"]}).to_csv(preds_path, index=False) 

 

    # Save metadata 

    meta = { 

        "id": run_id, 

        "model_name": spec.model_name, 

        "model_type": spec.model_type, 

        "dataset": spec.dataset, 

        "target": spec.target, 

        "target_lag": spec.target_lag, 

        "features": spec.features, 

        "force_nonneg_preds": spec.force_nonneg_preds, 

        "alpha": spec.alpha, 

        "metrics": fit["metrics"], 

        "intercept": fit["intercept"], 

        "coefficients": fit["coefficients"], 

        "n_obs": fit["n_obs"], 

        "predictions_csv": preds_path, 

        "saved_at": datetime.utcnow().isoformat() + "Z", 

    } 

    meta_path = os.path.join(run_dir, "meta.json") 

    with open(meta_path, "w", encoding="utf-8") as w: 

        json.dump(meta, w, indent=2) 

 

    # Append to ledger 

    try: 

        ledger = [] 

        if os.path.exists(LEDGER_PATH): 

            with open(LEDGER_PATH, "r", encoding="utf-8") as r: 

                ledger = json.load(r) 

        ledger.append(meta) 

        with open(LEDGER_PATH, "w", encoding="utf-8") as w: 

            json.dump(ledger, w, indent=2) 

    except Exception: 

        pass 

 

    meta["run_dir"] = run_dir 

    return meta 

 

 

# ---------------- Session State ---------------- 

if "queue" not in st.session_state: 

    st.session_state.queue: List[ModelSpec] = [] 

 

# ---------------- UI ---------------- 

files = _list_dataset_files() 

if not files: 

    st.info("No datasets found. Upload files in Ingestion & Curation Desk first.") 

else: 

    # ===== Model Builder ===== 

    with st.container(): 

        st.subheader("Model Builder") 

        c1, c2 = st.columns([2, 1]) 

        with c1: 

            ds_name = st.selectbox("Select dataset", options=[f["name"] for f in files], index=0, key="mod_ds") 

        selected_file = next((f for f in files if f["name"] == ds_name), None) 

 

        df = None 

        num_cols: List[str] = [] 

        if selected_file: 

            try: 

                df = _read_df(selected_file["path"], selected_file["ext"]) 

                num_cols = _numeric_columns(df) 

            except Exception as e: 

                st.error(f"Failed to read dataset: {e}") 

 

        c3, c4, c5 = st.columns([2, 1, 1]) 

        with c3: 

            target = st.selectbox("Target variable", options=(num_cols if num_cols else ["--"]), key="mod_target") 

        with c4: 

            lag = st.number_input("Target lag (optional)", min_value=0, max_value=365, value=0, step=1, key="mod_lag") 

        with c5: 

            model_type = st.selectbox("Model type", options=["ols", "nnls", "ridge", "lasso"], index=0, key="mod_type") 

 

        c6, c7 = st.columns([2, 1]) 

        with c6: 

            model_name = st.text_input("Model name", value=f"model_{int(time.time())}", key="mod_name") 

        with c7: 

            alpha = None 

            if model_type in ("ridge", "lasso"): 

                alpha = st.number_input("alpha (Ridge/Lasso)", min_value=0.0, value=1.0, step=0.1, key="mod_alpha") 

 

        features = st.multiselect( 

            "Independent variables", 

            options=[c for c in (df.columns.tolist() if df is not None else []) if c != target], 

            default=[c for c in num_cols if c != target][:3], 

            key="mod_features", 

        ) 

 

        c8, c9 = st.columns([1, 3]) 

        with c8: 

            force_nonneg = st.checkbox("Force negative predictions to zero", value=False, key="mod_force") 

        with c9: 

            if model_type == "nnls" and not HAVE_SCIPY: 

                st.warning("NNLS selected but SciPy not installed; this run will fail. Install scipy>=1.10.") 

 

        add_disabled = not (df is not None and target and features and model_name) 

        if st.button("Add to Queue", type="primary", disabled=add_disabled): 

            try: 

                spec = ModelSpec( 

                    dataset=ds_name, 

                    target=target, 

                    target_lag=int(lag or 0), 

                    model_name=model_name, 

                    features=list(features), 

                    force_nonneg_preds=bool(force_nonneg), 

                    model_type=model_type, 

                    alpha=float(alpha) if (model_type in ("ridge", "lasso") and alpha is not None) else None, 

                ) 

                st.session_state.queue.append(spec) 

                st.success(f"Queued: {model_name}") 

            except Exception as e: 

                st.error(f"Could not queue model: {e}") 

 

    st.divider() 

 

    # ===== Queue View ===== 

    st.subheader("Queued Models") 

    if not st.session_state.queue: 

        st.info("No models queued yet.") 

    else: 

        # Represent queue as a table 

        q_rows = [] 

        for i, q in enumerate(st.session_state.queue): 

            q_rows.append({ 

                "#": i + 1, 

                "Model Name": q.model_name, 

                "Type": q.model_type, 

                "Dataset": q.dataset, 

                "Target": q.target, 

                "Lag": q.target_lag, 

                "Features": ", ".join(q.features), 

                "Force>=0": q.force_nonneg_preds, 

                "Alpha": q.alpha if q.alpha is not None else "" 

            }) 

        height = 240 if len(q_rows) > 5 else 180 

        st.dataframe(pd.DataFrame(q_rows), use_container_width=True, height=height) 

        # Remove last / clear buttons 

        cA, cB, cC = st.columns([1,1,6]) 

        with cA: 

            if st.button("Remove last"): 

                try: 

                    removed = st.session_state.queue.pop() 

                    st.warning(f"Removed: {removed.model_name}") 

                except Exception: 

                    pass 

        with cB: 

            if st.button("Clear queue"): 

                st.session_state.queue.clear() 

                st.warning("Queue cleared") 

 

    st.divider() 

 

    # ===== Run All ===== 

    st.subheader("Run All Models") 

    run_btn = st.button("Run all & Save results", type="primary", disabled=(not st.session_state.queue)) 

 

    if run_btn: 

        results_meta: List[Dict] = [] 

        progress = st.progress(0.0, text="Starting…") 

        for idx, spec in enumerate(st.session_state.queue): 

            try: 

                sel = next((f for f in files if f["name"] == spec.dataset), None) 

                if sel is None: 

                    raise FileNotFoundError(f"Dataset not found: {spec.dataset}") 

                df_run = _read_df(sel["path"], sel["ext"]) 

                fit = _fit_and_score(spec, df_run) 

                meta = _save_run(spec, fit) 

                results_meta.append(meta) 

            except Exception as e: 

                st.error(f"Failed {spec.model_name}: {e}") 

            finally: 

                progress.progress((idx + 1) / max(len(st.session_state.queue), 1), text=f"Completed {idx+1}/{len(st.session_state.queue)}") 

        progress.empty() 

 

        if results_meta: 

            st.success(f"Saved {len(results_meta)} run(s) to storage/models. These are now available to Advanced Models.") 

            # Show a compact summary table 

            rows = [] 

            for m in results_meta: 

                rows.append({ 

                    "Model Name": m["model_name"], 

                    "Type": m["model_type"], 

                    "Dataset": m["dataset"], 

                    "Target": m["target"], 

                    "Lag": m["target_lag"], 

                    "R2": m["metrics"].get("r2"), 

                    "RMSE": m["metrics"].get("rmse"), 

                    "MAE": m["metrics"].get("mae"), 

                    "Predictions": os.path.relpath(m["predictions_csv"], STORAGE_DIR) 

                }) 

            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=240) 

 

            # Offer a single JSON of the run summaries for download 

            export_name = f"model_runs_{int(time.time())}.json" 

            st.download_button( 

                label="Download run summaries (JSON)", 

                data=json.dumps(results_meta, indent=2), 

                file_name=export_name, 

                mime="application/json", 

            ) 

 

        # Keep queue but you may choose to clear 

        # st.session_state.queue.clear() 

 

# ---- Footer note ---- 

st.caption("Tip: Results are saved under storage/models/<timestamp>_modelname with predictions.csv and meta.json, and indexed in runs_index.json.") 
