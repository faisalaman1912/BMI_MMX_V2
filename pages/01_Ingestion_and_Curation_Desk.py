# pages/1_Data_Upload.py
# Data Upload & File Analyzer (no loops) + Delete ALL
# - Upload CSV/XLSX, then click "Save uploaded files" (no auto-save)
# - Refresh files list (button)
# - Scrollable file inventory with per-file Delete + Delete ALL (guarded)
# - Analyzer: preview top 100 (10 visible), column summary, per-column impute (Mean/Median/Mode), save new file

import os
import io
import time
from hashlib import md5
from typing import List, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional Excel support (XLSX)
try:
    import openpyxl  # for reading/writing .xlsx
    HAVE_OPENPYXL = True
except Exception:
    HAVE_OPENPYXL = False

st.set_page_config(page_title="Data Upload & Analyzer", layout="wide")
st.title("Data Upload & Analyzer")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Session state
st.session_state.setdefault("_uploaded_hashes", set())  # prevent duplicate writes within session
st.session_state.setdefault("_delete_all_armed", False)  # guard for Delete ALL

# ---------------- Helpers ----------------
def _human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    s = float(max(0, n))
    for u in units:
        if s < 1024 or u == units[-1]:
            return "{:,.0f} {}".format(s, u)
        s /= 1024.0

def _safe_filename(name: str) -> str:
    base, ext = os.path.splitext(name)
    safe_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base).strip("_")
    ext = ext.lower()
    if ext not in (".csv", ".xlsx"):
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

def _file_md5_path(path: str, chunk: int = 1024 * 1024) -> str:
    h = md5()
    with open(path, "rb") as f:
        while True:
            c = f.read(chunk)
            if not c:
                break
            h.update(c)
    return h.hexdigest()

def _list_files() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for fn in sorted(os.listdir(DATA_DIR)):
        full = os.path.join(DATA_DIR, fn)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext not in (".csv", ".xlsx"):
            continue
        try:
            stt = os.stat(full)
            out.append({
                "name": fn,
                "path": full,
                "type": ext.strip(".").upper(),
                "size": _human_size(stt.st_size),
                "modified_ts": stt.st_mtime,
            })
        except Exception:
            out.append({"name": fn, "path": full, "type": "?", "size": "?", "modified_ts": 0})
    # newest first
    out.sort(key=lambda x: x["modified_ts"], reverse=True)
    return out

def _excel_engine_required():
    if not HAVE_OPENPYXL:
        raise RuntimeError("XLSX requires 'openpyxl'. Add openpyxl>=3.1.2 to requirements.txt or use CSV.")

def _read_preview(path: str, ext: str, nrows: int = 100) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    _excel_engine_required()
    return pd.read_excel(path, nrows=nrows, engine="openpyxl")

def _read_full(path: str, ext: str) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(path)
    _excel_engine_required()
    return pd.read_excel(path, engine="openpyxl")

def _quick_shape(path: str, ext: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        if ext == ".csv":
            # columns from header; rows by line count - 1
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
    # fallback
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
        m = s_num.mean()
        if pd.isna(m):
            mode_vals = series.mode(dropna=True)
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else ""
            return series.fillna(fill_val)
        return series.fillna(m)
    if method == "Median":
        s_num = pd.to_numeric(series, errors="coerce")
        m = s_num.median()
        if pd.isna(m):
            mode_vals = series.mode(dropna=True)
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else ""
            return series.fillna(fill_val)
        return series.fillna(m)
    if method == "Mode":
        mode_vals = series.mode(dropna=True)
        fill_val = mode_vals.iloc[0] if not mode_vals.empty else ""
        return series.fillna(fill_val)
    return series

def _save_new_dataset(df: pd.DataFrame, base_name: str, to_xlsx: bool) -> str:
    base_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base_name).strip("_") or "dataset_imputed"
    ext = ".xlsx" if to_xlsx else ".csv"
    dest = _unique_path(DATA_DIR, f"{base_name}{ext}")
    if to_xlsx:
        _excel_engine_required()
        df.to_excel(dest, index=False, engine="openpyxl")
    else:
        df.to_csv(dest, index=False)
    return dest

# ---------------- Upload ----------------
st.subheader("Upload files (CSV/XLSX)")

allowed_types = ["csv", "xlsx"]  # allow XLSX upload even if openpyxl missing; we block on read/write only
uploads = st.file_uploader(
    "Choose one or more files",
    type=allowed_types,
    accept_multiple_files=True,
    key="uploader_files",
)

c_up1, c_up2 = st.columns([1, 1])
with c_up1:
    save_uploads = st.button("Save uploaded files", type="primary")
with c_up2:
    refresh_list = st.button("Refresh files list")

if save_uploads and uploads:
    saved = 0
    for uf in uploads:
        try:
            buf = bytes(uf.getbuffer())
            h = _file_md5_bytes(buf)
            if h in st.session_state["_uploaded_hashes"]:
                # same payload already saved in this session
                continue
            safe = _safe_filename(uf.name)
            dest = os.path.join(DATA_DIR, safe)

            if os.path.exists(dest):
                # if same content on disk, skip; else unique suffix
                try:
                    if _file_md5_path(dest) == h:
                        st.session_state["_uploaded_hashes"].add(h)
                        continue
                except Exception:
                    pass
                dest = _unique_path(DATA_DIR, safe)

            with open(dest, "wb") as w:
                w.write(buf)
            st.session_state["_uploaded_hashes"].add(h)
            saved += 1
        except Exception as e:
            st.error(f"Failed to save {uf.name}: {e}")
    if saved:
        st.success(f"Uploaded {saved} file(s).")

st.divider()

# ---------------- Inventory (scrollable if >5) + Delete / Delete ALL ----------------
st.subheader("Saved files")

files = _list_files()
if not files:
    st.info("No files found yet. Upload and click 'Save uploaded files'.")
else:
    # Build metadata with rows/cols
    meta_rows: List[Dict[str, str]] = []
    for f in files:
        rows, cols = _quick_shape(f["path"], os.path.splitext(f["name"])[1].lower())
        meta_rows.append({
            "name": f["name"],
            "type": f["type"],
            "size": f["size"],
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f["modified_ts"])),
            "rows": "" if rows is None else f"{rows:,}",
            "cols": "" if cols is None else f"{cols:,}",
            "path": f["path"],
        })

    # Header
    h1, h2, h3, h4, h5, h6, h7 = st.columns([3, 1, 1, 2, 1, 1, 1])
    h1.markdown("**File**")
    h2.markdown("**Type**")
    h3.markdown("**Size**")
    h4.markdown("**Modified**")
    h5.markdown("**Rows**")
    h6.markdown("**Cols**")
    h7.markdown("**Delete**")

    # Scroll if more than 5 files
    container_height = 320 if len(meta_rows) > 5 else None
    with st.container(height=container_height):
        for row in meta_rows:
            c1, c2, c3, c4, c5, c6, c7 = st.columns([3, 1, 1, 2, 1, 1, 1])
            c1.write(row["name"])
            c2.write(row["type"])
            c3.write(row["size"])
            c4.write(row["modified"])
            c5.write(row["rows"])
            c6.write(row["cols"])
            # per-row delete
            if c7.button("Delete", key=f"del_{row['name']}"):
                try:
                    os.remove(row["path"])
                    st.success(f"Deleted: {row['name']}")
                except Exception as e:
                    st.error(f"Could not delete {row['name']}: {e}")

    # --- Delete ALL with guard ---
    st.markdown("**Danger zone**")
    d1, d2 = st.columns([1, 3])
    with d1:
        if st.button("Delete ALL files", type="secondary"):
            st.session_state["_delete_all_armed"] = True
    with d2:
        if st.session_state["_delete_all_armed"]:
            st.warning("Type **DELETE ALL** to confirm deletion of every file in the data/ folder.")
            confirm = st.text_input("Confirmation text")
            dd1, dd2 = st.columns(2)
            with dd1:
                if st.button("Confirm full delete"):
                    if confirm.strip().upper() == "DELETE ALL":
                        errs = 0
                        for fn in list(os.listdir(DATA_DIR)):
                            fp = os.path.join(DATA_DIR, fn)
                            if os.path.isfile(fp):
                                try:
                                    os.remove(fp)
                                except Exception:
                                    errs += 1
                        st.session_state["_delete_all_armed"] = False
                        if errs == 0:
                            st.success("All files deleted.")
                        else:
                            st.error("Some files could not be deleted.")
                    else:
                        st.error("Confirmation text did not match.")
            with dd2:
                if st.button("Cancel"):
                    st.session_state["_delete_all_armed"] = False

st.divider()

# ---------------- File Analyzer ----------------
st.subheader("File Analyzer")

files = _list_files()
if not files:
    st.info("Nothing to analyze yet.")
else:
    file_options = [f["name"] for f in files]
    sel_name = st.selectbox("Select a file", options=file_options, index=0)
    sel = next((f for f in files if f["name"] == sel_name), None)

    if sel:
        ext = os.path.splitext(sel["name"])[1].lower()
        path = sel["path"]

        # Preview top 100 (10 rows visible ~ about 360px height)
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

        if df is not None and not df.empty:
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
                        default="Skip",
                    )
                },
                disabled=["Column", "Datatype", "Null Count", "Null %"],
            )

            st.markdown("**Save new file**")
            c1, c2, c3 = st.columns([2, 1, 2])
            with c1:
                new_base = st.text_input(
                    "New filename (base)",
                    value=f"{os.path.splitext(sel_name)[0]}_imputed"
                )
            with c2:
                fmt = st.selectbox("Format", options=["csv"] + (["xlsx"] if HAVE_OPENPYXL else []), index=0)
            with c3:
                do_save = st.button("Apply imputation & Save", type="primary")

            if do_save:
                try:
                    out = df.copy()
                    # Apply per-column imputations
                    for _, r in edited.iterrows():
                        col = r["Column"]
                        method = r["Imputation"]
                        if method not in ("Skip", "Mean", "Median", "Mode"):
                            method = "Skip"
                        out[col] = _impute_column(out[col], method)

                    saved_path = _save_new_dataset(out, new_base, to_xlsx=(fmt == "xlsx"))
                    st.success(f"Saved new dataset: {os.path.basename(saved_path)}")
                except Exception as e:
                    st.error(f"Failed to save new dataset: {e}")
