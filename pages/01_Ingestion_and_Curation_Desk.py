# pages/01_Ingestion_and_Curation_Desk.py
# Ingestion & Curation Desk — SAFE MODE (no loops, no brittle paths, no hard deps)
# - Writes to ./storage/datasets (always writable on Streamlit Cloud)
# - Excel support optional; if openpyxl is missing, XLSX is disabled automatically
# - MD5 de-dup to prevent duplicate saves
# - Upload (manual), list, delete, preview, column imputation (Skip/Mean/Median/Mode), save new dataset

import os
import io
import time
from hashlib import md5
from typing import Tuple, Optional

import pandas as pd
import streamlit as st

# ---------- Optional Excel support ----------
try:
    import openpyxl  # for .xlsx read/write
    HAVE_OPENPYXL = True
except Exception:
    HAVE_OPENPYXL = False

# ---------- Storage paths (safe & writable) ----------
APP_ROOT = os.getcwd()
STORAGE_DIR = os.path.join(APP_ROOT, "storage")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

# ---------- Helpers ----------
def _human_kb(size_bytes: int) -> float:
    try:
        return round((size_bytes or 0) / 1024.0, 2)
    except Exception:
        return 0.0

def _safe_filename(name: str) -> str:
    base, ext = os.path.splitext(name)
    safe_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base).strip("_")
    # allow only csv/xlsx
    if ext.lower() not in (".csv", ".xlsx"):
        ext = ".csv"
    return (safe_base or "dataset") + ext

def _unique_path(dirpath: str, filename: str) -> str:
    candidate = os.path.join(dirpath, filename)
    if not os.path.exists(candidate):
        return candidate
    base, ext = os.path.splitext(filename)
    i = 1
    while True:
        cand = os.path.join(dirpath, f"{base}_{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1

def _file_md5_bytes(b: bytes) -> str:
    return md5(b).hexdigest()

def _file_md5_path(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def _list_dataset_files():
    files = []
    for fn in os.listdir(DATASETS_DIR):
        full = os.path.join(DATASETS_DIR, fn)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext not in (".csv", ".xlsx"):
            continue
        try:
            stat = os.stat(full)
            files.append({
                "name": fn,
                "path": full,
                "type": ext.strip("."),
                "size_kb": _human_kb(stat.st_size),
                "modified_ts": int(stat.st_mtime),
            })
        except Exception:
            pass
    files.sort(key=lambda x: x["modified_ts"], reverse=True)
    return files

def _read_preview(path: str, ext: str, nrows: int = 100) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    # ext == ".xlsx"
    if not HAVE_OPENPYXL:
        raise RuntimeError("XLSX preview requires openpyxl. Install openpyxl>=3.1.2 or upload CSV.")
    return pd.read_excel(path, nrows=nrows, engine="openpyxl")

def _read_full(path: str, ext: str) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(path)
    if not HAVE_OPENPYXL:
        raise RuntimeError("Reading XLSX requires openpyxl. Install openpyxl>=3.1.2 or upload CSV.")
    return pd.read_excel(path, engine="openpyxl")

def _quick_shape(path: str, ext: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        if ext == ".csv":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline()
            cols = len(pd.read_csv(io.StringIO(header), nrows=0).columns) if header else 0
            with open(path, "rb") as fb:
                row_lines = sum(1 for _ in fb)
            rows = max(row_lines - 1, 0)
            return rows, cols
        if ext == ".xlsx" and HAVE_OPENPYXL:
            df0 = pd.read_excel(path, nrows=0, engine="openpyxl")
            cols = len(df0.columns)
            df = pd.read_excel(path, usecols=list(df0.columns), engine="openpyxl")
            rows = len(df)
            return rows, cols
    except Exception:
        pass
    try:
        df = _read_full(path, ext)
        return df.shape[0], df.shape[1]
    except Exception:
        return None, None

def _impute_column(series: pd.Series, method: str) -> pd.Series:
    if method == "Skip":
        return series
    if method == "Mean":
        s_num = pd.to_numeric(series, errors="coerce")
        mean_val = s_num.mean()
        if pd.isna(mean_val):
            mode_vals = series.mode(dropna=True)
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else ""
            return series.fillna(fill_val)
        return series.fillna(mean_val)
    if method == "Median":
        s_num = pd.to_numeric(series, errors="coerce")
        med_val = s_num.median()
        if pd.isna(med_val):
            mode_vals = series.mode(dropna=True)
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else ""
            return series.fillna(fill_val)
        return series.fillna(med_val)
    if method == "Mode":
        mode_vals = series.mode(dropna=True)
        fill_val = mode_vals.iloc[0] if not mode_vals.empty else ""
        return series.fillna(fill_val)
    return series

def _save_new_dataset(df: pd.DataFrame, base_name: str, to_xlsx: bool) -> str:
    base_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name).strip("_") or "dataset_imputed"
    ext = "xlsx" if to_xlsx else "csv"
    filename = f"{base_name}.{ext}"
    dest = _unique_path(DATASETS_DIR, filename)
    if to_xlsx:
        if not HAVE_OPENPYXL:
            raise RuntimeError("Saving XLSX requires openpyxl. Install openpyxl>=3.1.2 or save as CSV.")
        df.to_excel(dest, index=False, engine="openpyxl")
    else:
        df.to_csv(dest, index=False)
    return dest

# ---------- UI ----------
st.set_page_config(page_title="Ingestion & Curation Desk", layout="wide")
st.title("Ingestion & Curation Desk")

# Session state
st.session_state.setdefault("_ingested_md5", set())

# ===== 1) Upload files =====
st.subheader("Upload files")

# Only allow XLSX if openpyxl is available
allowed_types = ["csv", "xlsx"] if HAVE_OPENPYXL else ["csv"]
if not HAVE_OPENPYXL:
    st.caption("XLSX features disabled (openpyxl not installed). CSV uploads fully supported.")

uploads = st.file_uploader(
    "Choose one or more files",
    type=allowed_types,
    accept_multiple_files=True,
    key="uploader_files"
)

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    save_uploads = st.button("Save uploaded files", type="primary")
with col_b:
    refresh_files = st.button("Refresh files list")
with col_c:
    clear_selected = st.button("Clear selected files")

if clear_selected:
    # Reset uploader by changing its key
    st.session_state["uploader_files"] = None

if save_uploads and uploads:
    saved_count = 0
    for up in uploads:
        try:
            buf = bytes(up.getbuffer())
            h = _file_md5_bytes(buf)
            if h in st.session_state["_ingested_md5"]:
                continue  # same payload already saved this session

            safe_name = _safe_filename(up.name)
            dest = os.path.join(DATASETS_DIR, safe_name)

            if os.path.exists(dest):
                # If same content on disk, skip; else, use unique suffix
                try:
                    if _file_md5_path(dest) == h:
                        st.session_state["_ingested_md5"].add(h)
                        continue
                except Exception:
                    pass
                dest = _unique_path(DATASETS_DIR, safe_name)

            with open(dest, "wb") as w:
                w.write(buf)

            st.session_state["_ingested_md5"].add(h)
            saved_count += 1
        except Exception as e:
            st.error(f"Failed to save {up.name}: {e}")

    if saved_count:
        st.success(f"Uploaded {saved_count} file(s).")

st.divider()

# ===== 2) Files list with delete =====
st.subheader("Available files")
files = _list_dataset_files()

meta_rows = []
for f in files:
    rows, cols = _quick_shape(f["path"], "." + f["type"])
    meta_rows.append({
        "Name": f["name"],
        "Type": f["type"],
        "Size (KB)": f["size_kb"],
        "Modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f["modified_ts"])),
        "Total Rows": rows if rows is not None else "",
        "Total Columns": cols if cols is not None else "",
    })

h = 320 if len(files) > 5 else (180 if len(files) else 120)
st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, height=h)

c1, c2 = st.columns([3, 1])
with c1:
    del_choice = st.selectbox("Delete a file", options=["-- Select --"] + [f["name"] for f in files])
with c2:
    do_delete = st.button("Delete", disabled=(del_choice == "-- Select --"))

if do_delete:
    target = next((f for f in files if f["name"] == del_choice), None)
    if target:
        try:
            os.remove(target["path"])
            st.warning(f"Deleted: {del_choice}")
        except Exception as e:
            st.error(f"Could not delete {del_choice}: {e}")

st.divider()

# ===== 3) File Analyzer =====
st.subheader("File Analyzer")

if not files:
    st.info("No files available yet. Upload CSV (and XLSX if openpyxl is installed).")
else:
    sel_name = st.selectbox("Pick a file to analyze", options=[f["name"] for f in files])
    sel = next((f for f in files if f["name"] == sel_name), None)

    if sel:
        ext = "." + sel["type"]
        path = sel["path"]

        # Preview
        try:
            preview_df = _read_preview(path, ext, nrows=100)
            st.markdown("**Preview (top 100 rows)**")
            st.dataframe(preview_df, use_container_width=True, height=360)
        except Exception as e:
            st.error(f"Preview failed: {e}")

        # Full read for summary & imputation
        df = None
        try:
            df = _read_full(path, ext)
        except Exception as e:
            st.error(f"Could not load file for analysis: {e}")

        if df is not None:
            null_counts = df.isna().sum()
            null_pct = (df.isna().mean() * 100).round(2)
            summary = pd.DataFrame({
                "Column": df.columns,
                "Datatype": [str(t) for t in df.dtypes],
                "Null Count": null_counts.values,
                "Null %": null_pct.values,
                "Imputation": ["Skip"] * len(df.columns),
            })

            st.markdown("**Column Summary & Imputation**")
            edited = st.data_editor(
                summary,
                use_container_width=True,
                height=320,
                key="summary_editor",
                column_config={
                    "Imputation": st.column_config.SelectboxColumn(
                        "Imputation",
                        help="Choose how to fill null values for each column",
                        options=["Skip", "Mean", "Median", "Mode"],
                        required=True,
                        default="Skip",
                    )
                },
                disabled=["Column", "Datatype", "Null Count", "Null %"],
            )

            st.markdown("**Save as new file**")
            sc1, sc2, sc3 = st.columns([2, 1, 2])
            with sc1:
                new_base = st.text_input("New file name (base)", value=f"{os.path.splitext(sel_name)[0]}_imputed")
            with sc2:
                fmt_opts = ["csv"] + (["xlsx"] if HAVE_OPENPYXL else [])
                fmt = st.selectbox("Format", options=fmt_opts, index=0)
            with sc3:
                save_btn = st.button("Apply imputation & Save", type="primary")

            if save_btn:
                try:
                    out = df.copy()
                    for _, row in edited.iterrows():
                        col = row["Column"]
                        method = row["Imputation"]
                        if method not in ("Skip", "Mean", "Median", "Mode"):
                            method = "Skip"
                        out[col] = _impute_column(out[col], method)

                    saved_path = _save_new_dataset(out, new_base, to_xlsx=(fmt == "xlsx"))
                    st.success(f"Saved new dataset: {os.path.basename(saved_path)}")
                except Exception as e:
                    st.error(f"Failed to save new dataset: {e}")

