
import time
import pandas as pd
import streamlit as st
from utils.catalog import recent_files, recent_models

st.set_page_config(page_title="Home | BMI MMX V1", page_icon="🏠", layout="wide")

st.markdown(
    """
    <div style="background-color:#f8f6f2;border-radius:16px;padding:24px;margin-bottom:18px;text-align:center;border:1px solid #eee;">
      <h2 style="margin:0;font-weight:600;">Marketing Mix Modelling by BlueMatter</h2>
    </div>
    """, unsafe_allow_html=True,
)

left, right = st.columns(2)

with left:
    st.subheader("📁 Recent Uploaded Files")
    files = recent_files(5)
    if not files:
        st.info("No files found. Drop CSV/XLSX into ./storage/datasets to see them here.")
    else:
        rows = []
        for f in files:
            rows.append({
                "Name": f.get("name"),
                "Type": f.get("type"),
                "Size (KB)": round((f.get("size", 0) or 0)/1024, 2),
                "Modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(f.get("modified_ts", 0) or 0)),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)

with right:
    st.subheader("🧠 Latest Saved Models")
    models = recent_models(5)
    if not models:
        st.info("No models found. Place model JSON artifacts in ./storage/models.")
    else:
        rows = []
        for m in models:
            rows.append({
                "Model": m.get("model_name") or m.get("name"),
                "Algo": m.get("algo"),
                "Saved At": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m.get("ts", m.get("modified_ts", 0)))),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)
