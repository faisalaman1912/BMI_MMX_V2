
import os, time, json
from typing import List, Dict

BASE = os.path.dirname(os.path.dirname(__file__))
STORAGE = os.path.join(BASE, "storage")
DATASETS_DIR = os.path.join(STORAGE, "datasets")
MODELS_DIR   = os.path.join(STORAGE, "models")

def _ensure_dirs():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

def _file_meta(path: str) -> Dict:
    stat = os.stat(path)
    size = stat.st_size
    mtime = int(stat.st_mtime)
    name = os.path.splitext(os.path.basename(path))[0]
    ext  = os.path.splitext(path)[1].lower().strip(".")
    return {
        "name": name,
        "path": path,
        "type": ext or "file",
        "size": size,
        "modified_ts": mtime
    }

def recent_files(n: int = 5) -> List[Dict]:
    _ensure_dirs()
    items = []
    for fn in os.listdir(DATASETS_DIR):
        full = os.path.join(DATASETS_DIR, fn)
        if os.path.isfile(full):
            items.append(_file_meta(full))
    items.sort(key=lambda x: x["modified_ts"], reverse=True)
    return items[:n]

def recent_models(n: int = 5) -> List[Dict]:
    _ensure_dirs()
    items = []
    for fn in os.listdir(MODELS_DIR):
        full = os.path.join(MODELS_DIR, fn)
        if os.path.isfile(full) and full.lower().endswith(".json"):
            meta = _file_meta(full)
            try:
                with open(full, "r", encoding="utf-8") as f:
                    js = json.load(f)
                meta["model_name"] = js.get("spec", {}).get("name") or js.get("name") or meta["name"]
                meta["algo"] = js.get("spec", {}).get("algo") or js.get("algo")
                meta["ts"] = js.get("ts", meta["modified_ts"])
            except Exception:
                meta["model_name"] = meta["name"]
                meta["algo"] = None
                meta["ts"] = meta["modified_ts"]
            items.append(meta)
    items.sort(key=lambda x: x.get("ts", x["modified_ts"]), reverse=True)
    return items[:n]
