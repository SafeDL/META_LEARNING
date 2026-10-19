"""The retained frozen response baselines, with one paid-risk interface."""
import numpy as np

from methods.baselines.hgt.oracle import RemoteOracle
from methods.baselines.hgt.scenarios import parameter_cells
from methods.baselines.hgt.search import TestingSession, select_session
from methods.baselines.ras_frt_uq.unified import select_from_responses

from .calibration import calibrators
from .config import BUDGET, SOURCE_WEIGHTS
from .history_session import RiskTestingSession
from .kernel import covariance
from .log_probability import LogProbabilityTestingSession
from .risk_session import RiskConditionedTestingSession


def select_task(connection, task):
    method, x, seed = task["method"], task["x"], task["seed"]
    oracle = RemoteOracle(connection)
    if method == "ras_frt_uq":
        return select_from_responses(x, parameter_cells(x), task["ras"], oracle, seed)
    if method == "frozen_original":
        return select_session(TestingSession(x, seed, mode="global_feedback", prior=task["prior"]), oracle)
    if method == "previous_best":
        prediction = task["original"]
        session = RiskTestingSession(x, covariance(x, prediction[0]), *prediction,
                                     np.asarray(SOURCE_WEIGHTS), calibrators())
    elif method == "matched_collision_gp":
        risk, collision = task["historical_prediction"]
        calibration = {int(key): value for key, value in task["matched_calibrators"].items()}
        session = RiskTestingSession(x, covariance(x, risk), risk, collision,
                                     np.full(risk.shape[1], 1 / risk.shape[1]),
                                     calibration, mode="calibrated")
    elif method == "behavior_posterior":
        risk, logits = task["grid_prediction"]
        session = LogProbabilityTestingSession(x, risk, logits, task["discrepancy"])
    elif method == "risk_conditioned":
        risk, logits = task["grid_prediction"]
        session = RiskConditionedTestingSession(x, risk, logits, task["discrepancy"],
                                                task["decoder"], collision_only=False,
                                                budget=BUDGET)
    else:
        raise ValueError(f"Unknown frozen baseline: {method}")
    selected = []
    while (index := session.next_index()) is not None:
        session.observe(oracle.query(index).risk)
        selected.append(index)
    return {"selected_indices": selected, "queries": session.records}
