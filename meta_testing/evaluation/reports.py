from __future__ import annotations

from typing import Mapping
import json
from pathlib import Path


def save_report(path: str | Path, report: Mapping[str, object]) -> None:
    Path(path).write_text(json.dumps(dict(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
