from __future__ import annotations
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # MetaDrive-only installations sometimes omit PyYAML.
    yaml = None


def _scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "None"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [_scalar(x) for x in value[1:-1].split(",") if x.strip()]
    if (value.startswith("\"")
            and value.endswith("\"")) or (value.startswith("'")
                                          and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Small loader for this repository's scalar/list mapping config files.

    It is intentionally not a general YAML implementation; PyYAML is used when
    installed. This fallback keeps the real MetaDrive entry points runnable in
    lean simulator environments.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"unsupported YAML line: {raw}")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def load_config(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) if yaml is not None else _minimal_yaml_load(
        text)


def dump_json(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(value), indent=2, default=str),
                          encoding="utf-8")


def write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def versions() -> dict[str, str]:
    result = {"python": sys.version, "platform": platform.platform()}
    for name in ("torch", "stable_baselines3", "gymnasium", "metadrive"):
        try:
            module = __import__(name)
            result[name] = str(
                getattr(module, "__version__",
                        "installed (version attribute unavailable)"))
        except ImportError:
            result[name] = "not installed"
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        result["git_commit"] = "unknown"
    return result
