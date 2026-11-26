"""Published paper cohort. The original execution registries remain reproducible."""

import json
from dataclasses import replace

from .protocol import ROOT


RUNS = ROOT / "results"
REVISED = RUNS / "dqn_continual_learning"
PUBLICATION = REVISED / "publication.json"


def published_release():
    if not PUBLICATION.is_file():
        return None
    release = json.loads(PUBLICATION.read_text(encoding="utf-8"))
    if release.get("status") != "published":
        raise ValueError("DQN publication receipt is incomplete")
    return release


def paper_study(original):
    if (original.parent == RUNS and original.name in ("k_shot_adaptation", "budget_discovery", "weight_adaptation")
            and published_release()):
        return REVISED / original.name
    return original


def regression_policy(name):
    if name == "dqn" and published_release():
        return "dqn_continual"
    return name


def create_controller(sut_id):
    """Load a controller from the same cohort used by the published plots."""
    from controllers.highway.catalog import FORMAL_BY_ID
    from controllers.highway.factory import create

    release = published_release() if sut_id == "target_dqn" else None
    if not release:
        return create(sut_id)
    from controllers.highway.contracts import runtime_contract
    from controllers.highway.decision_interface import DecisionInterface
    from controllers.highway.learning import LearnedController
    from .qualification import verify_course_evidence

    sut = replace(FORMAL_BY_ID[sut_id], training_run=release["training_run"])
    gate = ROOT / release["qualification"]
    record = json.loads(gate.read_text(encoding="utf-8"))
    contract = runtime_contract(sut)
    if (contract != record["runtime_contract"] or contract["checkpoint_sha256"] != release["checkpoint_sha256"]
            or record["courses"] != verify_course_evidence(gate.parent)["courses"]):
        raise ValueError("Published DQN identity or driving capability evidence changed")
    return DecisionInterface(LearnedController(sut))
