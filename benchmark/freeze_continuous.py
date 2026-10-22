"""Separate preregistration for continuous search, following development validation."""

from dataclasses import asdict
import hashlib
import json

from .build_collision_models import model_path
from .continuous_runs import CONFIGURATIONS, source_contract
from .continuous_selection import METHODS
from .develop_continuous import OUTPUT, SETTINGS, WORKERS as DEVELOPMENT_WORKERS
from .evaluate_continuous import WORKERS as FINAL_WORKERS
from .protocol import DATA_ROOT, DEFAULT, ROOT, validate_sut_freeze
from .studies import CONTINUOUS_TASKS


def main():
    finite = DATA_ROOT / "frozen_protocol.json"
    if not finite.is_file():
        raise ValueError("Continuous extension requires the qualified frozen main protocol")
    validate_sut_freeze(json.loads(finite.read_text(encoding="utf-8")))
    development = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
    if (development["status"] != "complete" or len(development["runs"]) != 40
            or development["settings"] != SETTINGS or not SETTINGS["ordinary_only"]
            or development["source_contract"] != source_contract()
            or development["configurations"] != CONFIGURATIONS
            or development["execution_workers"] != DEVELOPMENT_WORKERS
            or any(row["queries"] != development["settings"]["budget"] for row in development["runs"])):
        raise ValueError("Continuous mechanisms have not completed the frozen development matrix")
    models = []
    for _, kind in CONTINUOUS_TASKS:
        for method in ("history_response", "pooled_ft"):
            path = model_path(method, "all", kind)
            models.append({"path": str(path.relative_to(ROOT)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"protocol": asdict(DEFAULT), "tasks": CONTINUOUS_TASKS, "methods": METHODS,
                "search_domain": {"development": "ordinary_only", "final": "all_physically_valid"},
                "configurations": CONFIGURATIONS, "source_contract": source_contract(),
                "execution_workers": {"development": DEVELOPMENT_WORKERS, "final": FINAL_WORKERS},
                "models": models, "main_protocol_sha256": hashlib.sha256(finite.read_bytes()).hexdigest(),
                "development_report": str((OUTPUT / "report.json").relative_to(ROOT)),
                "development_report_sha256": hashlib.sha256((OUTPUT / "report.json").read_bytes()).hexdigest()}
    with (DATA_ROOT / "frozen_continuous.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
