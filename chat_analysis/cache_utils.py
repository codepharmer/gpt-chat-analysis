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


def clear_cache(cache_dir: Path) -> bool:
    """Clear all cached files in the specified cache directory."""
    try:
        if cache_dir.exists():
            import shutil

            shutil.rmtree(cache_dir)
            print(f"Cleared cache directory: {cache_dir}")
            return True
        else:
            print(f"Cache directory does not exist: {cache_dir}")
            return False
    except Exception as e:
        print(f"Error clearing cache directory {cache_dir}: {e}")
        return False


def clear_all_caches(log_path: str, cache_root: Optional[str] = None) -> bool:
    """Clear the cache directory for a specific log file."""
    cache_dir = get_cache_dir(log_path, cache_root)
    return clear_cache(cache_dir)


def clear_python_cache(project_root: Optional[str] = None) -> bool:
    """Clear Python bytecode cache (__pycache__) directories."""
    import shutil
    from pathlib import Path

    if project_root is None:
        project_root = Path(__file__).parent.parent
    else:
        project_root = Path(project_root)

    try:
        cleared_count = 0
        for pycache_dir in project_root.rglob("__pycache__"):
            if pycache_dir.is_dir():
                shutil.rmtree(pycache_dir)
                print(f"Cleared Python cache: {pycache_dir}")
                cleared_count += 1

        if cleared_count == 0:
            print("No Python cache directories found")
        else:
            print(f"Cleared {cleared_count} Python cache directories")
        return True
    except Exception as e:
        print(f"Error clearing Python cache: {e}")
        return False
