"""Native unit trajectories shared by transfer execution and audit tests."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from controllers.highway.control import Control
from experiments.history import build_source
from experiments.history_transfer_experiment import pooled_history
from experiments.rss import measurement_contract, trajectory_margin
from experiments.scenarios import DOMAINS, sample
from experiments.transfer_settings import MethodSettings
from simulator.highway.collision_env import execute
from testing_methods.baselines.scenario_fuzz.graphs import graph
from testing_methods.baselines.scenario_fuzz.model import ScenarioRiskNetwork
from testing_methods.common.response import ResponseArchive
from testing_methods.history_transfer.adaptation import FitSettings, OnlineSettings
from testing_methods.history_transfer.inference import TARGET_PARAMETERS
from testing_methods.history_transfer.shared_private import SharedPrivateKernel


class Coast:
    def act(self, world):
        return Control(0.0, 0.0)


class NativeBank:
    def __init__(self):
        self.candidates = sample("car_following", 8, "new", replicate=991)
        bounds = DOMAINS["car_following"].bounds
        self.candidates[0] = replace(
            self.candidates[0],
            parameters={
                "ego_speed_mps": bounds["ego_speed_mps"][1],
                "lead_speed_mps": bounds["lead_speed_mps"][0],
                "gap_m": bounds["gap_m"][0],
            },
        )
        self.executions = {
            s.scenario_id: execute(Coast(), s, duration_s=0.4) for s in self.candidates
        }
        self.manifest = {
            "sut_id": "executor_fixture",
            "environment_seed": 81001,
            "contract": {"episode_duration_s": 0.4},
            "attempted_calls": 8,
            "successful_calls": 8,
            "failed_or_interrupted_calls": 0,
        }
        self.domain = None
        self.touched = []

    def __call__(self, scene):
        self.touched.append(scene.scenario_id)
        return self.executions[scene.scenario_id]

    def read_paid(self, index):
        return self.executions[self.candidates[index].scenario_id]

    @property
    def snapshot_fingerprint(self):
        return hashlib.sha256(
            json.dumps(
                [
                    {"feedback": p.feedback.to_dict(), "trajectory": p.trajectory}
                    for p in self.executions.values()
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()


@pytest.fixture(scope="module")
def native_inputs():
    torch.set_num_threads(1)
    torch.manual_seed(1409)
    bank = NativeBank()
    paid = [bank.read_paid(i) for i in range(8)]
    history = [
        ResponseArchive(
            source_id=f"native_unit_source_{i}",
            family="unit_fixture",
            lineage=f"unit_{i}",
            role="history",
            scenarios=bank.candidates,
            collision=np.array([p.feedback.collision for p in paid]),
            safety=np.array(
                [[p.feedback.safety.min_clearance_m, p.feedback.safety.min_ttc_s] for p in paid]
            ),
            rss_margin=np.array([trajectory_margin(p.trajectory)["minimum_margin"] for p in paid]),
            rss_contract=measurement_contract(),
            provenance={
                "record_ids": [f"unit_{i}_{j}" for j in range(8)],
                "scope": "Actual native unit trajectories, not formal history",
            },
        )
        for i in range(8)
    ]
    # Explicitly untrained unit models check orchestration equivalence, not adequacy.
    models = (SharedPrivateKernel(3, deep=True), SharedPrivateKernel(3, deep=True))
    sources = build_source(models[1], history, bank.candidates)
    pooled = pooled_history(history, {"scope": "native unit fixture"})
    fit = FitSettings(2, 0.003, TARGET_PARAMETERS)
    settings = MethodSettings(OnlineSettings(fit, fit, -1.0), fit)
    classifier = ScenarioRiskNetwork()
    classifier.fit_normalization([graph(s) for s in bank.candidates])
    classifier.eval()
    return bank, history, models, sources, pooled, settings, classifier
