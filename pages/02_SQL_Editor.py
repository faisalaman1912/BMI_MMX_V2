# 02_SQL_Editor.py
# Streamlit SQL Editor page with DuckDB (preferred) or SQLite fallback.
# Features
# - Auto-registers available/saved tables from a data directory (CSV/Parquet/Feather)
# - Sidebar list of tables with vertical scroll and schema viewer
# - SQL editor with simple suggestions helper
# - Run queries and view results
# - Save result as a new dataset (CSV/Parquet)
# - Download result as Excel
# - Robust error handling; avoids hard dependency on duckdb

import os
import io
import time
import glob
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Try to import duckdb; if unavailable, we'll fall back to SQLite
try:
    import duckdb  # type: ignore
    DUCKDB_AVAILABLE = True
except Exception:
    DUCKDB_AVAILABLE = False

# -------------------------------
# Configuration
# -------------------------------
st.set_page_config(page_title="SQL Editor", layout="wide")

DATA_DIR = st.secrets.get("DATA_DIR", "data/curated")  # where previous steps saved datasets
DERIVED_DIR = st.secrets.get("DERIVED_DIR", "data/derived")  # where we save new datasets
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DERIVED_DIR, exist_ok=True)

st.sidebar.title("SQL Editor")
st.sidebar.caption(
    "Available tables are auto-registered from CSV/Parquet/Feather in the data folder."
)

# -------------------------------
# Engine Abstraction
# -------------------------------
class SQLEngine:
    def __init__(self, prefer_duckdb: bool = True):
        self.engine = None
        self.kind = None  # 'duckdb' or 'sqlite'

        if prefer_duckdb and DUCKDB_AVAILABLE:
            self.kind = 'duckdb'
            # DuckDB in-memory; we'll register lazy scans of files
            self.engine = duckdb.connect(database=':memory:')
            # Make pandas prints nicer
            self.engine.execute("PRAGMA threads=4;")
        else:
            self.kind = 'sqlite'
            self.engine = sqlite3.connect(':memory:')

    # --- Registration of external files as tables ---
    def register_file_table(self, table_name: str, file_path: str):
        ext = os.path.splitext(file_path)[1].lower()
        if self.kind == 'duckdb':
            if ext == '.csv':
                # Create a view over the file (lazy scan)
                self.engine.execute(
                    f"CREATE OR REPLACE VIEW {duckdb_identifier(table_name)} AS SELECT * FROM read_csv_auto('{file_path.replace("'", "''")}');"
                )
            elif ext in ('.parquet', '.pq'):
                self.engine.execute(
                    f"CREATE OR REPLACE VIEW {duckdb_identifier(table_name)} AS SELECT * FROM parquet_scan('{file_path.replace("'", "''")}');"
                )
            elif ext in ('.feather', '.ft'):  # Feather/Arrow
                self.engine.execute(
                    f"CREATE OR REPLACE VIEW {duckdb_identifier(table_name)} AS SELECT * FROM read_ipc('{file_path.replace("'", "''")}');"
                )
            else:
                # Fallback: read via pandas then create relation
                df = read_file_to_df(file_path)
                self.register_dataframe(table_name, df)
        else:
            # SQLite: load into memory using pandas
            df = read_file_to_df(file_path)
            self.register_dataframe(table_name, df)

    def register_dataframe(self, table_name: str, df: pd.DataFrame):
        safe = sqlite_identifier(table_name) if self.kind == 'sqlite' else duckdb_identifier(table_name)
        if self.kind == 'duckdb':
            self.engine.register(table_name, df)
            # Make a stable view name equal to the table_name
            self.engine.execute(f"CREATE OR REPLACE VIEW {safe} AS SELECT * FROM {duckdb_identifier(table_name)};")
        else:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)

    # --- Introspection ---
    def list_tables(self) -> List[str]:
        if self.kind == 'duckdb':
            q = """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema IN ('main')
            ORDER BY table_name
            """
            return [r[0] for r in self.engine.execute(q).fetchall()]
        else:
            q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            return [r[0] for r in self.engine.execute(q).fetchall()]

    def get_columns(self, table_name: str) -> List[Tuple[str, Optional[str]]]:
        # returns list of (column_name, data_type)
        if self.kind == 'duckdb':
            q = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name.replace("'", "''")}' ORDER BY ordinal_position;"
            return [(r[0], r[1]) for r in self.engine.execute(q).fetchall()]
        else:
            q = f"PRAGMA table_info({sqlite_identifier(table_name)});"  # cid, name, type, notnull, dflt_value, pk
            return [(r[1], r[2]) for r in self.engine.execute(q).fetchall()]

    # --- Execution ---
    def run_query(self, sql: str) -> pd.DataFrame:
        if self.kind == 'duckdb':
            return self.engine.execute(sql).df()
        else:
            return pd.read_sql_query(sql, self.engine)

# -------------------------------
# Helpers
# -------------------------------

def duckdb_identifier(name: str) -> str:
    # Quote identifiers that contain non-alphanumeric chars
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
        return name
    return f'"{name.replace("\"", "\"\"")}"'


def sqlite_identifier(name: str) -> str:
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
        return name
    return f'"{name.replace("\"", "\"\"")}"'


def read_file_to_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path)
    if ext in ('.parquet', '.pq'):
        return pd.read_parquet(path)
    if ext in ('.feather', '.ft'):
        return pd.read_feather(path)
    # Best-effort fallback
    return pd.read_csv(path)


def scan_data_dir(root: str) -> Dict[str, str]:
    """Return mapping of table_name -> file_path from data directory.
    Table name is the file stem; duplicates last-write-wins.
    """
    patterns = ["*.csv", "*.parquet", "*.pq", "*.feather", "*.ft"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(root, p)))
    mapping = {}
    for f in sorted(files, key=lambda p: os.path.getmtime(p)):
        table = os.path.splitext(os.path.basename(f))[0]
        mapping[table] = f
    return mapping


# -------------------------------
# Load/Refresh tables into engine
# -------------------------------
@st.cache_data(show_spinner=False)
def _scan_paths_cached(root: str):
    return scan_data_dir(root)


def load_tables(engine: SQLEngine, root: str) -> Dict[str, str]:
    mapping = _scan_paths_cached(root)
    for t, p in mapping.items():
        engine.register_file_table(t, p)
    return mapping

# -------------------------------
# UI Components
# -------------------------------
@st.cache_data(show_spinner=False)
def _schema_snapshot(kind: str, _refresh_key: int) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    # Cache will bust when _refresh_key changes
    eng = st.session_state["_engine"]
    meta = {}
    for t in eng.list_tables():
        meta[t] = eng.get_columns(t)
    return meta


def render_sidebar_tables(schema: Dict[str, List[Tuple[str, Optional[str]]]]):
    st.sidebar.subheader("Available Tables")
    filter_text = st.sidebar.text_input("Filter tables", "")
    names = sorted(schema.keys())
    if filter_text:
        names = [n for n in names if filter_text.lower() in n.lower()]
    with st.sidebar.container(height=280):
        for name in names:
            with st.sidebar.expander(name, expanded=False):
                cols = schema[name]
                st.caption(f"{len(cols)} columns")
                for c, t in cols:
                    st.code(f"{name}.{c} ({t})", language="text")
                st.button("Copy SELECT *", key=f"copy_{name}", on_click=_inject_sql, args=(f"SELECT * FROM {name} LIMIT 100;",))


def _inject_sql(template: str):
    st.session_state["sql_input"] = template


def suggestion_helper(sql_text: str, schema: Dict[str, List[Tuple[str, Optional[str]]]]) -> List[str]:
    """Very simple suggester: if the last token partially matches, propose table/column names."""
    try:
        last = re.split(r"[^A-Za-z0-9_\.]", sql_text.strip())[-1]
    except Exception:
        last = ""
    suggestions = []
    if not last:
        return suggestions

    # Suggest tables
    for t in schema.keys():
        if t.lower().startswith(last.lower()) and t not in suggestions:
            suggestions.append(t)

    # Suggest columns with table prefix variants
    for t, cols in schema.items():
        for c, _ in cols:
            full = f"{t}.{c}"
            if c.lower().startswith(last.lower()) and c not in suggestions:
                suggestions.append(c)
            if full.lower().startswith(last.lower()) and full not in suggestions:
                suggestions.append(full)
    return suggestions[:20]

# -------------------------------
# App State
# -------------------------------
if "_engine" not in st.session_state:
    st.session_state["_engine"] = SQLEngine(prefer_duckdb=True)
    st.session_state["_refresh_key"] = 0

engine: SQLEngine = st.session_state["_engine"]

# (Re)load tables
files_map = load_tables(engine, DATA_DIR)

# Schema snapshot (cached)
schema = _schema_snapshot(engine.kind, st.session_state["_refresh_key"])  # table -> [(col, type), ...]

render_sidebar_tables(schema)

st.markdown("## SQL Editor")

# Editor with helpful placeholder
placeholder = (
    """-- Tips:\n"
    "-- 1) Click a table in the sidebar to see columns.\n"
    "-- 2) Click 'Copy SELECT *' to insert a template.\n"
    f"-- 3) Engine: {engine.kind.upper()} | Data dir: {DATA_DIR}\n\n"
    "-- Example:\n"
    "-- SELECT a.col1, b.col2\n"
    "-- FROM table_a a\n"
    "-- JOIN table_b b ON a.id = b.id\n"
    "-- WHERE a.date >= '2025-01-01'\n"
    "-- LIMIT 100;\n"
)

sql_text = st.text_area(
    "Write your SQL query",
    key="sql_input",
    height=200,
    placeholder=placeholder,
)

# Suggestion helper UI
if sql_text:
    sugg = suggestion_helper(sql_text, schema)
    if sugg:
        st.caption("Suggestions (click to insert):")
        scols = st.columns(min(4, len(sugg)))
        for i, s in enumerate(sugg):
            if st.button(s, key=f"sug_{i}"):
                st.session_state["sql_input"] = (sql_text + (" " if not sql_text.endswith(" ") else "") + s)
                st.rerun()

run_col, save_col, refresh_col = st.columns([1,1,1])

with run_col:
    run_clicked = st.button("▶ Run Query", type="primary")
with save_col:
    st.session_state.setdefault("_save_name", "")
    save_name = st.text_input("Save result as (table name)", key="_save_name", placeholder="my_new_dataset")
with refresh_col:
    if st.button("🔁 Refresh Tables"):
        # Bust caches and rescan
        st.cache_data.clear()
        st.session_state["_refresh_key"] += 1
        st.rerun()

result_df: Optional[pd.DataFrame] = None
error_text: Optional[str] = None

if run_clicked and sql_text.strip():
    start = time.time()
    try:
        result_df = engine.run_query(sql_text)
        elapsed = time.time() - start
        st.success(f"Query OK in {elapsed:.2f}s — {len(result_df):,} rows")
    except Exception as e:
        error_text = str(e)
        st.error("Query failed. See details below.")
        with st.expander("Error details", expanded=False):
            st.code(error_text, language="text")

# Show results
if isinstance(result_df, pd.DataFrame):
    st.dataframe(result_df, use_container_width=True, height=420)

    # Downloads
    with st.expander("Download Result"):
        # CSV
        csv_bytes = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv_bytes, file_name="query_result.csv", mime="text/csv")
        # Excel
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine='xlsxwriter') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Result')
        st.download_button("Download Excel", data=xbuf.getvalue(), file_name="query_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Save as dataset (to both engine and disk)
    if save_name:
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", save_name.strip())
        if st.button("💾 Save as Dataset"):
            try:
                # Register to engine as an in-memory table
                engine.register_dataframe(safe_name, result_df)
                # Persist to disk (Parquet and CSV)
                out_parquet = os.path.join(DERIVED_DIR, f"{safe_name}.parquet")
                out_csv = os.path.join(DERIVED_DIR, f"{safe_name}.csv")
                result_df.to_parquet(out_parquet, index=False)
                result_df.to_csv(out_csv, index=False)
                # Clear scan cache & rescan so new table appears
                st.cache_data.clear()
                st.session_state["_refresh_key"] += 1
                st.success(f"Saved as '{safe_name}' in {DERIVED_DIR} (CSV & Parquet) and registered in the SQL engine.")
            except Exception as e:
                st.error(f"Save failed: {e}")

# Footer help
with st.expander("Help / Troubleshooting"):
    st.markdown(
        f"""
        **Engine:** {engine.kind.upper()}  
        **Data folder:** `{DATA_DIR}`  
        **Derived folder:** `{DERIVED_DIR}`

        **Tips**
        - If you see `ModuleNotFoundError: duckdb`, the page will fall back to SQLite automatically. For best performance and larger-than-memory CSV/Parquet queries, add `duckdb` to your `requirements.txt`.
        - Supported file types auto-registered: CSV (`*.csv`), Parquet (`*.parquet`/`*.pq`), Feather (`*.feather`/`*.ft`). Table name is the file name (without extension).
        - Click **Refresh Tables** after adding/removing files in the data folders.
        - Results are saved to `{DERIVED_DIR}` in both CSV and Parquet and also registered as an in-memory table for immediate querying.
        - The suggestions are lightweight (prefix-based). Use the sidebar to inspect schemas.
        """
    )
