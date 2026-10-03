import os
import subprocess
import sys

import numpy as np
import pytest

from methods.srd_tnp_bqd.data import HistoryOutput, PosteriorOutput
from methods.srd_tnp_bqd.nonstationary_kernel import ResidualKernel
from methods.srd_tnp_bqd.oracle import ContinuousOracle
from methods.srd_tnp_bqd.qd import expected_archive_improvement, posterior_risk_mean
from methods.srd_tnp_bqd.acquisition import Selector, mode_at
from methods.srd_tnp_bqd.common import DEFAULT_CONFIG, config_at


CONFIG = config_at(DEFAULT_CONFIG)


def test_mix_preserves_first_batch_period_and_query_counts():
    modes = [mode_at(t) for t in range(1, 201)]
    assert modes[:6] == ["eai", "risk", "risk"] * 2
    assert modes.count("eai") == 67
    assert modes.count("risk") == 133
    with pytest.raises(ValueError):
        mode_at(0)


def test_saturated_cell_keeps_global_risk_channel():
    posterior = PosteriorOutput(np.array([[4.0], [0.5]]), np.zeros((2, 1)))
    assert np.argmax(expected_archive_improvement(
        posterior.mean[:, 0], posterior.latent_var[:, 0], np.array([0.99, 0.5])
    )) == 1
    assert np.argmax(posterior_risk_mean(
        posterior.mean[:, 0], posterior.latent_var[:, 0]
    )) == 0


def test_feedback_updates_gp_without_using_labels_or_mutating_history():
    x = np.random.default_rng(1).uniform(size=(8, 4))
    prior = HistoryOutput(np.zeros((8, 1)), np.zeros((8, 32)))
    kernel = ResidualKernel(CONFIG["residual_gp"])
    paths = []
    for label in (0, 1):
        oracle = ContinuousOracle([str(i) for i in range(8)],
                                  lambda i: (label, 0.6, True), 3)
        result = Selector(prior, kernel).run(x, np.arange(8), oracle, 3)
        paths.append(result["selected_indices"])
        assert len(oracle.queried) == len(set(paths[-1])) == 3
        assert all(row["e"] == row["z"] - row["m"] for row in result["queries"])
        assert result["final_mean"] != prior.m[:, 0].tolist()
    assert paths[0] == paths[1]
    assert not prior.m.any() and not prior.h.any()


def test_invalid_risk_is_charged_without_entering_gp_or_archive():
    x = np.random.default_rng(2).uniform(size=(6, 4))
    prior = HistoryOutput(np.zeros((6, 1)), np.zeros((6, 32)))
    oracle = ContinuousOracle([str(i) for i in range(6)],
                              lambda i: (1, None, False), 3)
    result = Selector(prior, ResidualKernel(CONFIG["residual_gp"])).run(
        x, np.arange(6), oracle, 3
    )
    assert len(oracle.queried) == 3
    assert result["final_mean"] == [0.0] * 6
    assert all(row["e"] is None and row["archive_gain"] == 0 for row in result["queries"])


def test_selector_process_blocks_development_and_undisclosed_target_responses():
    code = '''
import numpy as np
from methods.srd_tnp_bqd import experiment
from methods.srd_tnp_bqd.s01 import ROOT, TARGET, TARGET_ROOT
paths = [
    ROOT / 'measurements/D' / (TARGET + '.risk.jsonl'),
    TARGET_ROOT / (TARGET + '.jsonl'),
]
assert all(p.exists() for p in paths)
class Connection:
    def send(self, value): pass
    def close(self): pass
def probe(limits):
    blocked = 0
    for path in paths:
        try:
            with path.open(): pass
        except PermissionError:
            blocked += 1
    print('BLOCKED', blocked)
    assert blocked == 2
    raise RuntimeError('probe complete before model loading')
experiment.threadpool_limits = probe
experiment.worker(Connection(), {'method': 'SRD_TNP_BQD', 'seed': 11}, np.zeros((1,4)), np.zeros(1,int))
'''
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert "BLOCKED 2" in result.stdout
