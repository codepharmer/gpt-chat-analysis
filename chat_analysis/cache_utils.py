from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional


def get_cache_dir(log_path: str, cache_root: Optional[str] = None) -> Path:
    """Return directory for caching chunk summaries."""
    base = Path(log_path)
    name = base.stem + "_cache"
    if cache_root:
        cache_dir = Path(cache_root) / name
    else:
        cache_dir = base.with_name(name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_file(cache_dir: Path, chunk: str) -> Path:
    h = hashlib.md5(chunk.encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.json"


def load_chunk(cache_dir: Path, chunk: str) -> Optional[Dict[str, Any]]:
    path = _cache_file(cache_dir, chunk)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_chunk(cache_dir: Path, chunk: str, data: Dict[str, Any]) -> None:
    path = _cache_file(cache_dir, chunk)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
