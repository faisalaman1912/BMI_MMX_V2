# pages/02_SQL_Editor.py
# SQL Editor with DuckDB (primary) + SQLite/pandasql (fallback)

import os
import io
import re
import time
import pandas as pd
import streamlit as st

# Soft deps for Excel support
try:
    import openpyxl  # noqa: F401
    HAVE_OPENPYXL = True
except Exception:
    HAVE_OPENPYXL = False

# Try DuckDB first; fallback to pandasql (SQLite)
USE_DUCKDB = True
try:
    import duckdb  # type: ignore
except Exception:
    USE_DUCKDB = False
    try:
        from pandasql import sqldf  # type: ignore
    except Exception:
        sqldf = None  # will show a clear error

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DATASETS_DIR = os.path.join(STORAGE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

def _human_kb(size_bytes: int) -> float:
    try:
        return round((size_bytes or 0) / 1024.0, 2)
    except Exception:
        return 0.0

def _list_dataset_files():
    files = []
    for fn in os.listdir(DATASETS_DIR):
        path = os.path.join(DATASETS_DIR, fn)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext not in (".csv", ".xlsx", ".xls"):
            continue
        try:
            stat = os.stat(path)
            files.append({
                "name": fn,
                "path": path,
                "ext": ext,
                "size_kb": _human_kb(stat.st_size),
                "modified_ts": int(stat.st_mtime),
            })
        except Exception:
            pass
    files.sort(key=lambda x: x["modified_ts"], reverse=True)
    return files

def _safe_table_name(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"\W+", "_", name).strip("_").lower()
    return name or "table"

def _read_df(path: str, ext: str) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(path)
    else:
        if not HAVE_OPENPYXL and ext == ".xlsx":
            raise RuntimeError("Excel support requires 'openpyxl'. Add openpyxl>=3.1.2 to requirements.txt.")
        return pd.read_excel(path)

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

def _save_dataset(df: pd.DataFrame, base_name: str, fmt: str) -> str:
    base_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", base_name).strip("_") or "dataset_from_sql"
    fname = f"{base_name}.{fmt}"
    dest = _unique_path(DATASETS_DIR, fname)
    if fmt == "csv":
        df.to_csv(dest, index=False)
    else:
        if not HAVE_OPENPYXL:
            raise RuntimeError("Writing Excel requires 'openpyxl'. Add openpyxl>=3.1.2 to requirements.txt.")
        df.to_excel(dest, index=False)
    return dest

st.set_page_config(page_title="SQL Editor", page_icon="🧮", layout="wide")
st.title("🧮 SQL Editor")

files = _list_dataset_files()

# 1) Available tables
st.subheader("Available tables")
if not files:
    st.info("No datasets found. Upload files in the Ingestion & Curation Desk or place CSV/XLSX in ./storage/datasets.")
else:
    meta_rows = []
    for f in files:
        meta_rows.append({
            "Table": _safe_table_name(f["name"]),
            "File": f["name"],
            "Type": f["ext"].strip("."),
            "Size (KB)": f["size_kb"],
            "Modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f["modified_ts"])),
        })
    height = 260 if len(meta_rows) > 5 else 180
    st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, height=height)

# 2) Register tables in engine (DuckDB or pandasql env)
tables_info = {}  # {table: [cols]}
pandas_env = {}   # for pandasql

if USE_DUCKDB:
    con = duckdb.connect(database=":memory:")
else:
    if sqldf is None:
        st.error(
            "SQL engine not available. Install either `duckdb` (recommended) or `pandasql`.\n\n"
            "Add to requirements.txt:\n  duckdb>=1.0.0\n  pandasql>=0.7.3\n  openpyxl>=3.1.2\n  xlsxwriter>=3.2.0"
        )
        st.stop()

for f in files:
    try:
        df = _read_df(f["path"], f["ext"])
        tname = _safe_table_name(f["name"])
        # Ensure unique name
        base = tname
        j = 2
        while tname in tables_info:
            tname = f"{base}_{j}"
            j += 1
        tables_info[tname] = list(map(str, df.columns))
        if USE_DUCKDB:
            con.register(tname, df)
        else:
            # pandasql uses env variables as table names
            pandas_env[tname] = df
    except Exception as e:
        st.warning(f"Skipped {f['name']}: {e}")

# 3) Suggestions panel
st.subheader("Suggestions")
with st.expander("Tables & columns (click to insert into editor)", expanded=True):
    if not tables_info:
        st.write("No tables registered.")
    else:
        for t, cols in tables_info.items():
            st.markdown(f"**`{t}`**")
            chunk = 6
            for i in range(0, len(cols), chunk):
                row = st.columns(chunk)
                for j, col in enumerate(cols[i:i+chunk]):
                    key = f"btn_{t}_{i}_{j}"
                    if row[j].button(col, key=key):
                        st.session_state.setdefault("sql_text", "")
                        sep = "" if not st.session_state["sql_text"] or st.session_state["sql_text"][-1].isspace() else " "
                        # For pandasql, table.column also works
                        st.session_state["sql_text"] = f"{st.session_state['sql_text']}{sep}{t}.{col}"

# 4) SQL editor
st.subheader("Editor")
default_table = next(iter(tables_info)) if tables_info else "/*your_table*/"
default_sql = st.session_state.get("sql_text") or f"SELECT * FROM {default_table} LIMIT 100;"
sql = st.text_area("Write SQL here", value=default_sql, key="sql_text", height=220)

c1, c2, c3 = st.columns([1,1,2])
with c1:
    run = st.button("Run query", type="primary")
with c2:
    clear = st.button("Clear editor")
with c3:
    example = st.button("Insert example JOIN")

if clear:
    st.session_state["sql_text"] = ""
    st.rerun()

if example and len(tables_info) >= 2:
    tnames = list(tables_info.keys())[:2]
    a, b = tnames[0], tnames[1]
    join_col = tables_info[a][0] if tables_info[a] else "id"
    st.session_state["sql_text"] = f"SELECT a.*, b.*\nFROM {a} a\nJOIN {b} b ON a.{join_col} = b.{join_col}\nLIMIT 100;"
    st.rerun()

# 5) Execute & show results
result_df = None
if run:
    if not sql.strip():
        st.warning("Enter a SQL statement.")
    else:
        try:
            if USE_DUCKDB:
                result_df = con.execute(sql).df()
            else:
                # pandasql uses local/global env; supply our table dict
                result_df = sqldf(sql, pandas_env | globals())
            st.success(f"Returned {len(result_df)} rows")
            st.dataframe(result_df, use_container_width=True, height=360)
        except Exception as e:
            if not USE_DUCKDB:
                st.error(
                    f"Query failed under SQLite fallback. "
                    f"SQLite is less feature-complete than DuckDB (no FULL OUTER JOIN, limited functions). "
                    f"Error: {e}"
                )
            else:
                st.error(f"Query failed: {e}")

# 6) Save & download
st.subheader("Save / Download")
col_a, col_b, col_c = st.columns([2,1,2])

with col_a:
    out_name = st.text_input("Save as (base name)", value="dataset_from_sql")
with col_b:
    out_fmt = st.selectbox("Format", options=["csv", "xlsx"])
with col_c:
    save_btn = st.button("Save dataset")

if result_df is not None:
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, index=False)
    st.download_button(
        "Download result (Excel)",
        data=xbuf.getvalue(),
        file_name="query_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if save_btn:
    if result_df is None:
        st.warning("Run a query first to generate results.")
    else:
        try:
            saved_path = _save_dataset(result_df, out_name, out_fmt)
            st.success(f"Saved new dataset: {os.path.basename(saved_path)}")
            st.info("You can now see it under 'Available tables' after a refresh.")
        except Exception as e:
            st.error(f"Failed to save dataset: {e}")
