from __future__ import annotations

import uuid
from pathlib import Path


def unique_file_path(directory: Path, suffix: str) -> Path:
    return directory / f"{uuid.uuid4().hex}{suffix}"


def safe_suffix(filename: str | None, default: str = ".bin") -> str:
    if not filename:
        return default

    suffix = Path(filename).suffix.lower()
    return suffix if suffix else default
