"""A risk-only testing session, shared by deployment and ablation experiments."""
import numpy as np
import torch

from .archive import RiskArchive
from .config import BUDGET, ROOT
from .gp import RiskGP, clipped_risk_ei
from .history import historical_risk, load_history, split_indices, subset
from .io import read_json
from .scenarios import parameter_cells
from .train import load_kernel


class TestingSession:
    def __init__(self, x, seed=11, budget=BUDGET, *, mode=None, prior=None, risk_only=True):
        torch.set_num_threads(1)
        self.x = np.asarray(x, dtype=float)
        self.cells = parameter_cells(self.x)
        self.budget = budget
        if mode is None:
            mode = read_json(ROOT / "models/selection.json")["active_kernel"]
        if prior is None:
            history = load_history()
            train, _ = split_indices(next(iter(history.values()))["x"])
            prior = historical_risk(subset(history, train), self.x)
        with torch.no_grad():
            covariance = load_kernel(seed, mode).covariance(self.x).numpy()
        self.gp = RiskGP(covariance, prior)
        self.archive = None if risk_only else RiskArchive()
        self.remaining = np.ones(len(self.x), dtype=bool)
        self.pending = None
        self.count = 0
        self.risk_only = risk_only

    def next_index(self):
        if self.pending is not None:
            raise RuntimeError("observe the pending query before requesting another")
        if self.count >= self.budget or not self.remaining.any():
            return None
        threshold = (self.archive.threshold + self.archive.elite_quality[self.cells]
                     if not self.risk_only and self.count % 3 == 0 else 0.)
        scores = clipped_risk_ei(self.gp.mean, self.gp.variance, threshold)
        scores[~self.remaining] = -np.inf
        self.pending = int(np.argmax(scores))
        return self.pending

    def observe(self, risk):
        if self.pending is None:
            raise RuntimeError("request a query before supplying feedback")
        index = self.pending
        if risk is not None:
            self.gp.observe(index, risk)
            if self.archive is not None:
                self.archive.observe(int(self.cells[index]), risk)
        self.remaining[index] = False
        self.pending = None
        self.count += 1


def select_session(session, oracle):
    records = []
    for _ in range(session.budget):
        index = session.next_index()
        if index is None:
            break
        observation = oracle.query(index)
        session.observe(observation.risk if observation.valid_risk else None)
        records.append({"index": index, "query_number": len(records) + 1,
                        "risk": observation.risk, "cell": int(session.cells[index])})
    return {"selected_indices": [row["index"] for row in records], "queries": records}
