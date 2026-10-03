"""JSON serialization and configuration for the retained method chain."""
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "methods/srd_tnp_bqd/configs/s01.yaml"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def config_at(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
