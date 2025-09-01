# -*- coding: utf-8 -*-
# 02_SQL_Editor.py — SAFE build for Py 3.13/Streamlit Cloud
# Notes:
# - Removes emoji in button labels (can break some terminals/parsers)
# - Simplifies f-strings/escaping to avoid AST parse edge cases
# - UTF-8 header added; keeps unicode in comments only
# - Falls back to SQLite if DuckDB unavailable
# - No optional deps required for core features; Excel download falls back gracefully

import os
import io
import glob
import re
import time
import sqlite3
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Try DuckDB; fallback to SQLite
try:
    import duckdb  # type: ignore
    DUCKDB_AVAILABLE = True
except Exception:
    DUCKDB_AVAILABLE = False

# ---------------------------------
# Config
# ---------------------------------
st.set_page_config(page_title="SQL Editor", layout="wide")
DATA_DIR = st.secrets.get("DATA_DIR", "data/curated")
DERIVED_DIR = st.secrets.get("DERIVED_DIR", "data/derived")
os.makedirs(DATA_DIR, exist_ok=True)
(os.path.isdir(DERIVED_DIR) and True) or os.makedirs(DERIVED_DIR, exist_ok=True)

st.sidebar.title("SQL Editor")
st.sidebar.caption("Auto-registers CSV/Parquet/Feather from data folder as tables.")

# ---------------------------------
# Utilities
# ---------------------------------

def _duck_ident(name: str) -> str:
    # Quote if needed using double-quotes
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return name
    return '"' + name.replace('"', '""') + '"'


def _sqlite_ident(name: str) -> str:
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return name
    return '"' + name.replace('"', '""') + '"'


def _read_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext in (".feather", ".ft"):
        return pd.read_feather(path)
    return pd.read_csv(path)


def _scan_dir(root: str) -> Dict[str, str]:
    pats = ["*.csv", "*.parquet", "*.pq", "*.feather", "*.ft"]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(os.path.join(root, p)))
    mapping: Dict[str, str] = {}
    for f in sorted(files, key=lambda x: os.path.getmtime(x)):
        mapping[os.path.splitext(os.path.basename(f))[0]] = f
    return mapping

# ---------------------------------
# Engine wrapper
# ---------------------------------
class SQLEngine:
    def __init__(self, prefer_duckdb: bool = True):
        if prefer_duckdb and DUCKDB_AVAILABLE:
            self.kind = "duckdb"
            self.engine = duckdb.connect(database=":memory:")
            self.engine.execute("PRAGMA threads=4;")
        else:
            self.kind = "sqlite"
            self.engine = sqlite3.connect(":memory:")

    def register_file(self, tbl: str, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if self.kind == "duckdb":
            t = _duck_ident(tbl)
            p = path.replace("'", "''")
            if ext == ".csv":
                self.engine.execute("CREATE OR REPLACE VIEW %s AS SELECT * FROM read_csv_auto('%s');" % (t, p))
            elif ext in (".parquet", ".pq"):
                self.engine.execute("CREATE OR REPLACE VIEW %s AS SELECT * FROM parquet_scan('%s');" % (t, p))
            elif ext in (".feather", ".ft"):
                self.engine.execute("CREATE OR REPLACE VIEW %s AS SELECT * FROM read_ipc('%s');" % (t, p))
            else:
                df = _read_df(path)
                self.register_df(tbl, df)
        else:
            df = _read_df(path)
            self.register_df(tbl, df)

    def register_df(self, tbl: str, df: pd.DataFrame) -> None:
        if self.kind == "duckdb":
            self.engine.register(tbl, df)
            self.engine.execute("CREATE OR REPLACE VIEW %s AS SELECT * FROM %s;" % (_duck_ident(tbl), _duck_ident(tbl)))
        else:
            df.to_sql(tbl, self.engine, if_exists="replace", index=False)

    def list_tables(self) -> List[str]:
        if self.kind == "duckdb":
            q = (
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema IN ('main') ORDER BY table_name"
            )
            return [r[0] for r in self.engine.execute(q).fetchall()]
        q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        return [r[0] for r in self.engine.execute(q).fetchall()]

    def columns(self, tbl: str) -> List[Tuple[str, Optional[str]]]:
        if self.kind == "duckdb":
            q = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '%s' ORDER BY ordinal_position;" % (tbl.replace("'", "''"),)
            return [(r[0], r[1]) for r in self.engine.execute(q).fetchall()]
        q = "PRAGMA table_info(%s);" % _sqlite_ident(tbl)
        return [(r[1], r[2]) for r in self.engine.execute(q).fetchall()]

    def query(self, sql: str) -> pd.DataFrame:
        if self.kind == "duckdb":
            return self.engine.execute(sql).df()
        return pd.read_sql_query(sql, self.engine)

# ---------------------------------
# Cacheable helpers
# ---------------------------------
@st.cache_data(show_spinner=False)
def _cached_scan(root: str) -> Dict[str, str]:
    return _scan_dir(root)


def _load_all(engine: SQLEngine, root: str) -> Dict[str, str]:
    mp = _cached_scan(root)
    for t, p in mp.items():
        engine.register_file(t, p)
    return mp

@st.cache_data(show_spinner=False)
def _snapshot_schema(_kind: str, _bust: int) -> Dict[str, List[Tuple[str, Optional[str]]]]:
    eng: SQLEngine = st.session_state["_engine"]
    out: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    for t in eng.list_tables():
        out[t] = eng.columns(t)
    return out

# ---------------------------------
# UI
# ---------------------------------
if "_engine" not in st.session_state:
    st.session_state["_engine"] = SQLEngine(prefer_duckdb=True)
    st.session_state["_bust"] = 0

engine: SQLEngine = st.session_state["_engine"]
_files = _load_all(engine, DATA_DIR)
schema = _snapshot_schema(engine.kind, st.session_state["_bust"])  # table -> cols

# Sidebar tables
st.sidebar.subheader("Available Tables")
flt = st.sidebar.text_input("Filter tables", "")
names = sorted(schema.keys())
if flt:
    names = [n for n in names if flt.lower() in n.lower()]
with st.sidebar.container(height=280):
    for nm in names:
        with st.sidebar.expander(nm, expanded=False):
            cols = schema[nm]
            st.caption("%d columns" % len(cols))
            for c, t in cols:
                st.code("%s.%s (%s)" % (nm, c, t), language="text")
            st.button("Copy SELECT *", key="copy_%s" % nm, on_click=lambda s="SELECT * FROM %s LIMIT 100;" % nm: st.session_state.__setitem__("sql_in", s))

st.markdown("## SQL Editor")
placeholder = (
    "-- Tips:\n"
    "-- 1) Use the sidebar to inspect schemas.\n"
    "-- 2) Click 'Copy SELECT *' to insert a template.\n"
    "-- 3) Engine: %s | Data dir: %s\n\n"
    "-- Example:\n"
    "-- SELECT a.col1, b.col2\n"
    "-- FROM table_a a\n"
    "-- JOIN table_b b ON a.id = b.id\n"
    "-- WHERE a.date >= '2025-01-01'\n"
    "-- LIMIT 100;\n" % (engine.kind.upper(), DATA_DIR)
)

sql_text = st.text_area("Write your SQL query", key="sql_in", height=200, placeholder=placeholder)

# Lightweight suggestions
if sql_text:
    try:
        last_tok = re.split(r"[^A-Za-z0-9_\.]", sql_text.strip())[-1]
    except Exception:
        last_tok = ""
    sugg: List[str] = []
    if last_tok:
        for tname in schema.keys():
            if tname.lower().startswith(last_tok.lower()):
                sugg.append(tname)
        for tname, cols in schema.items():
            for c, _ in cols:
                if c.lower().startswith(last_tok.lower()) and c not in sugg:
                    sugg.append(c)
                full = tname + "." + c
                if full.lower().startswith(last_tok.lower()) and full not in sugg:
                    sugg.append(full)
    if sugg:
        st.caption("Suggestions (click to insert):")
        n = min(4, len(sugg))
        cs = st.columns(n)
        for i, s in enumerate(sugg[:20]):
            if cs[i % n].button(s, key="sug_%d" % i):
                st.session_state["sql_in"] = (sql_text + (" " if not sql_text.endswith(" ") else "") + s)
                st.rerun()

col_run, col_save, col_ref = st.columns([1, 1, 1])
run_clicked = col_run.button("Run Query")
save_name = col_save.text_input("Save result as (table)", key="_save_name", placeholder="my_new_dataset")
if col_ref.button("Refresh Tables"):
    st.cache_data.clear()
    st.session_state["_bust"] += 1
    st.rerun()

result_df: Optional[pd.DataFrame] = None
if run_clicked and sql_text and sql_text.strip():
    t0 = time.time()
    try:
        result_df = engine.query(sql_text)
        st.success("Query OK in %.2fs — %s rows" % (time.time() - t0, format(len(result_df), ",")))
    except Exception as e:
        st.error("Query failed. See details below.")
        with st.expander("Error details", expanded=False):
            st.code(str(e), language="text")

if isinstance(result_df, pd.DataFrame):
    st.dataframe(result_df, use_container_width=True, height=420)

    with st.expander("Download Result"):
        # CSV
        st.download_button("Download CSV", data=result_df.to_csv(index=False).encode("utf-8"), file_name="query_result.csv", mime="text/csv")
        # Excel (best-effort)
        try:
            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Result")
            st.download_button("Download Excel", data=xbuf.getvalue(), file_name="query_result.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as _ex:
            st.info("Excel export not available (missing dependency). CSV download still works.")

    if save_name:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", save_name.strip())
        if st.button("Save as Dataset"):
            try:
                engine.register_df(safe, result_df)
                out_parquet = os.path.join(DERIVED_DIR, safe + ".parquet")
                out_csv = os.path.join(DERIVED_DIR, safe + ".csv")
                try:
                    result_df.to_parquet(out_parquet, index=False)
                except Exception:
                    pass
                result_df.to_csv(out_csv, index=False)
                st.cache_data.clear()
                st.session_state["_bust"] += 1
                st.success("Saved as '%s' in %s (CSV% sParquet) and registered in the engine." % (safe, DERIVED_DIR, ", " if os.path.exists(out_parquet) else " without "))
            except Exception as e:
                st.error("Save failed: %s" % str(e))

with st.expander("Help / Troubleshooting"):
    st.markdown(
        """
**Engine**: DuckDB (if installed) else SQLite  
**Data folder**: `data/curated` (override via `st.secrets["DATA_DIR"]`)  
**Derived folder**: `data/derived` (override via `st.secrets["DERIVED_DIR"]`)

Tips
- If you saw `ModuleNotFoundError: duckdb`, the app automatically uses SQLite.
- Supported auto-registered types: CSV, Parquet, Feather (Parquet/Feather need pyarrow).
- Click **Refresh Tables** after adding/removing files in the data folders.
- Saved datasets land in `data/derived` and are registered in-memory for immediate querying.
        """
    )
