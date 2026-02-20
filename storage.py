import json
import os
from pathlib import Path


def get_output_base_dir():
    base = (os.getenv("OUTPUT_DIR") or "").strip()
    if not base:
        return None
    return Path(base).expanduser().resolve()


def ensure_scan_dir(scan_id):
    base = get_output_base_dir()
    if base is None:
        return None
    scan_dir = (base / scan_id).resolve()
    base.mkdir(parents=True, exist_ok=True)
    scan_dir.mkdir(parents=True, exist_ok=True)
    return scan_dir


def write_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_artifact_path(scan_id: str, filename: str):
    base = get_output_base_dir()
    if base is None:
        return None
    safe_scan_id = "".join(ch for ch in scan_id if ch.isalnum())
    if not safe_scan_id:
        return None
    candidate = (base / safe_scan_id / filename).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        return None
    return candidate
