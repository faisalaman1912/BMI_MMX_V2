# pages/01_Ingestion_and_Curation_Desk.py
# Ingestion & Curation Desk
# - Upload CSV/XLSX
# - Auto-refresh file list with type, size, modified ts, rows, cols
# - Delete files
# - File Analyzer: preview top 100 (10 rows visible), column summary,
#   per-column imputation (Mean/Median/Mode) and save as a new dataset

import os
import io
import time
import math
import shutil
import pandas as pd
import streamlit as st
from hashlib import md5  # <-- NEW

# ---- If your minimal requirements didn't include Excel support,
# add this line to requirements.txt: openpyxl>=3.1.2
try:
    import openpyxl  # noqa: F401
    HAVE_OPENPYXL = True
except Exception:
    HAVE_OPENPYXL = False

# ---- Paths (reuse storage layout from the minimal scaffold)
# If you used `utils/catalog.py` from the minimal setup, these dirs match that structure.
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")
MODELS_DIR = os.path.join(STORAGE_DIR, "models")
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------- Helper functions ----------------
def _human_kb(size_bytes: int) -> float:
    try:
        return round((size_bytes or 0) / 1024.0, 2)
    except Exception:
        return 0.0

def _safe_filename(name: str) -> str:
    # Remove unsafe characters but keep file extension if present
    base, ext = os.path.splitext(name)
    safe_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base).strip("_")
    ext = ext if ext.lower() in (".csv", ".xlsx", ".xls") else ext
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
    """MD5 for bytes payload."""
    return md5(b).hexdigest()

def _file_md5_path(path: str, chunk_size: int = 1024 * 1024) -> str:
    """MD5 for a file on disk."""
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
        if ext not in (".csv", ".xlsx", ".xls"):
            continue
        try:
            stat = os.stat(full)
            files.append({
                "name": fn,
                "path": full,
                "type": ext.strip("."),
                "size_kb": _human_kb(stat.st_size),
                "modified_ts": int(stat.st_mtime)
            })
        except Exception:
            # Ignore unreadable files but do not crash UI
            pass
    # Sort newest first
    files.sort(key=lambda x: x["modified_ts"], reverse=True)
    return files

def _quick_shape(path: str, ext: str):
    """
    Return (rows, cols) without loading entire file wherever possible.
    Fallback to pandas for reliability.
    """
    try:
        if ext == ".csv":
            # Columns by header, rows by line count - 1 (header)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                header = f.readline()
            cols = len(pd.read_csv(io.StringIO(header), nrows=0).columns) if header else 0
            # Count lines efficiently
            with open(path, "rb") as fb:
                row_lines = sum(1 for _ in fb)
            rows = max(row_lines - 1, 0)
            return rows, cols
        elif ext in (".xlsx", ".xls"):
            if HAVE_OPENPYXL and ext == ".xlsx":
                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                ws = wb.active
                rows = max((ws.max_row or 1) - 1, 0)  # minus header
                cols = ws.max_column or 0
                return rows, cols
            # Fallback to pandas
            df0 = pd.read_excel(path, nrows=0)
            cols = len(df0.columns)
            df = pd.read_excel(path, usecols=list(df0.columns))
            rows = len(df)
            return rows, cols
    except Exception:
        pass
    # Last resort
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        return df.shape[0], df.shape[1]
    except Exception:
        return None, None

def _read_preview(path: str, ext: str, nrows: int = 100) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    else:
        if not HAVE_OPENPYXL and ext == ".xlsx":
            raise RuntimeError("Excel preview requires 'openpyxl'. Add openpyxl>=3.1.2 to requirements.txt.")
        return pd.read_excel(path, nrows=nrows)

def _read_full(path: str, ext: str) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(path)
    else:
        if not HAVE_OPENPYXL and ext == ".xlsx":
            raise RuntimeError("Reading Excel requires 'openpyxl'. Add openpyxl>=3.1.2 to requirements.txt.")
        return pd.read_excel(path)

def _impute_column(series: pd.Series, method: str) -> pd.Series:
    if method == "Skip":
        return series
    if method == "Mean":
        # Mean only for numeric; fallback to Mode if non-numeric or NaN mean
        s_num = pd.to_numeric(series, errors="coerce")
        mean_val = s_num.mean()
        if pd.isna(mean_val):
            # fallback to mode
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
    # default: skip
    return series

def _save_new_dataset(df: pd.DataFrame, base_name: str, ext_choice: str) -> str:
    base_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name).strip("_") or "dataset_imputed"
    filename = f"{base_name}.{ext_choice}"
    dest = _unique_path(DATASETS_DIR, filename)
    if ext_choice == "csv":
        df.to_csv(dest, index=False)
    else:
        # ensure openpyxl present to write xlsx
        if not HAVE_OPENPYXL:
            raise RuntimeError("Writing Excel requires 'openpyxl'. Add openpyxl>=3.1.2 to requirements.txt.")
        df.to_excel(dest, index=False)
    return dest

# ---------------- UI ----------------
st.set_page_config(page_title="Ingestion & Curation Desk", page_icon="📥", layout="wide")
st.title("📥 Ingestion & Curation Desk")

# --- Session guard for duplicate uploads in a single session/run cycle ---  (NEW)
if "_ingested_md5" not in st.session_state:
    st.session_state["_ingested_md5"] = set()

# ===== 1) Upload files =====
st.subheader("Upload files (CSV/XLSX)")
uploads = st.file_uploader(
    "Choose one or more files",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    key="uploader_files"
)

col_up1, col_up2 = st.columns([1, 1])
with col_up1:
    save_uploads = st.button("Save uploaded files", type="primary")
with col_up2:
    refresh_files = st.button("Refresh files list")

if save_uploads and uploads:
    saved_count = 0
    for up in uploads:
        try:
            safe_name = _safe_filename(up.name)
            dest = os.path.join(DATASETS_DIR, safe_name)
            if os.path.exists(dest):
                dest = _unique_path(DATASETS_DIR, safe_name)
            with open(dest, "wb") as w:
                w.write(up.getbuffer())
            saved_count += 1
        except Exception as e:
            st.error(f"Failed to save {up.name}: {e}")
    if saved_count:
        st.success(f"Uploaded {saved_count} file(s). Use 'Refresh files list' to see them below.")

# Manual refresh of the table (user-triggered)
if refresh_files:
    st.rerun()
# ===== 2) Files list with delete =====
st.subheader("Available files")
files = _list_dataset_files()

# Prepare metadata table with rows/cols
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

table_height = 320 if len(files) > 5 else (180 if len(files) else 120)
st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, height=table_height)

col_del1, col_del2 = st.columns([3, 1])
with col_del1:
    del_choice = st.selectbox("Delete a file", options=["-- Select --"] + [f["name"] for f in files])
with col_del2:
    if st.button("Delete", type="secondary", disabled=(del_choice == "-- Select --")):
        target = next((f for f in files if f["name"] == del_choice), None)
        if target:
            try:
                os.remove(target["path"])
                st.warning(f"Deleted: {del_choice}")
                st.rerun()
            except Exception as e:
                st.error(f"Could not delete {del_choice}: {e}")

st.divider()

# ===== 3) File Analyzer =====
st.subheader("File Analyzer")

if not files:
    st.info("No files available yet. Upload CSV/XLSX above.")
else:
    sel_name = st.selectbox("Pick a file to analyze", options=[f["name"] for f in files])
    sel = next((f for f in files if f["name"] == sel_name), None)

    if sel:
        ext = "." + sel["type"]
        path = sel["path"]

        # Preview (top 100 rows) with ~10 visible rows height
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
            # Column summary
            null_counts = df.isna().sum()
            null_pct = (df.isna().mean() * 100).round(2)
            summary = pd.DataFrame({
                "Column": df.columns,
                "Datatype": [str(t) for t in df.dtypes],
                "Null Count": null_counts.values,
                "Null %": null_pct.values,
                "Imputation": ["Skip"] * len(df.columns)
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
            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                new_base = st.text_input("New file name (base)", value=f"{os.path.splitext(sel_name)[0]}_imputed")
            with c2:
                ext_choice = st.selectbox("Format", options=["csv", "xlsx"], index=0)
            with c3:
                btn = st.button("Apply imputation & Save", type="primary")

            if btn:
                try:
                    out = df.copy()
                    for _, row in edited.iterrows():
                        col = row["Column"]
                        method = row["Imputation"]
                        if method not in ("Skip", "Mean", "Median", "Mode"):
                            method = "Skip"
                        out[col] = _impute_column(out[col], method)

                    saved_path = _save_new_dataset(out, new_base, ext_choice)
                    st.success(f"Saved new dataset: {os.path.basename(saved_path)}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save new dataset: {e}")
