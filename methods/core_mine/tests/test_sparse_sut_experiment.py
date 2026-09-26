from methods.core_mine.sparse_sut_experiment import (
    BENIGN_PER_MODE, CHALLENGE_PER_MODE, MODES, PER_MODE, SparseBank,
    configure_proposal, qualification_gate, sparse_scenarios, tasks_from_bank,
)
import numpy as np


def test_sparse_proposal_is_balanced_and_outcome_independent() -> None:
    anchors, modes, controls, regimes = sparse_scenarios(17)
    assert anchors.shape == (len(MODES) * PER_MODE, 2)
    assert controls.shape == (len(anchors), 2)
    for mode in MODES:
        mask = modes == mode
        assert mask.sum() == PER_MODE
        assert (regimes[mask] == "benign").sum() == BENIGN_PER_MODE
        assert (regimes[mask] == "challenge").sum() == CHALLENGE_PER_MODE


def test_leave_one_sut_out_never_uses_target_as_source() -> None:
    anchors, modes, controls, regimes = sparse_scenarios(17)
    size = len(anchors); names = ("a", "b", "c", "d")
    collisions = np.zeros((4, size), dtype=bool); collisions[0, 0] = True
    bank = SparseBank(anchors, modes, controls, regimes, names, collisions, collisions.copy(),
                      np.zeros_like(collisions), np.full((4, size), np.inf), np.ones((4, size)), np.ones((4, size), dtype=bool))
    tasks = tasks_from_bank(bank, 17)
    assert len(tasks) == 4 and all(task.source_y.shape[0] == 3 for task in tasks)
    assert tasks[0].target_event[0] and not tasks[1].target_event[0]


def test_continuous_proposal_has_only_outcome_blind_audit_strata() -> None:
    configure_proposal("v3_continuous_borderline")
    try:
        anchors, modes, controls, regimes = sparse_scenarios(19)
        assert anchors.shape == (len(MODES) * PER_MODE, 2)
        assert controls.shape == (len(anchors), 2)
        assert set(regimes) == {"gap_q1", "gap_q2", "gap_q3", "gap_q4"}
        for mode in MODES:
            assert (modes == mode).sum() == PER_MODE
    finally:
        configure_proposal("v2_far_headway")


def test_sparse_continuous_proposal_preserves_mode_balanced_pool() -> None:
    configure_proposal("v4_sparse_continuous")
    try:
        anchors, modes, controls, regimes = sparse_scenarios(29)
        assert anchors.shape == (len(MODES) * PER_MODE, 2)
        assert controls.shape == (len(anchors), 2)
        assert set(regimes) == {"gap_q1", "gap_q2", "gap_q3", "gap_q4"}
        assert all((modes == mode).sum() == PER_MODE for mode in MODES)
    finally:
        configure_proposal("v2_far_headway")


def test_qualification_gate_requires_nonshared_multimode_failures(tmp_path) -> None:
    configure_proposal("v3_continuous_borderline")
    try:
        anchors, modes, controls, regimes = sparse_scenarios(23)
        names = ("a", "b", "c", "d"); size = len(anchors)
        collision = np.zeros((4, size), dtype=bool)
        def point(mode, quartile):
            return int(np.flatnonzero((modes == mode) & (regimes == quartile))[0])
        # Every target has two modes where one source agrees on failure and
        # another remains safe; no three-SUT group is jointly redundant.
        left = [point("fast_intrusion", "gap_q1"), point("cutin_braking", "gap_q2")]
        right = [point("lead_braking", "gap_q3"), point("stop_and_go", "gap_q4")]
        collision[0, left] = True; collision[2, left] = True
        collision[1, right] = True; collision[3, right] = True
        bank = SparseBank(anchors, modes, controls, regimes, names, collision,
                          np.zeros_like(collision), np.zeros_like(collision),
                          np.full((4, size), np.inf), np.ones((4, size)),
                          np.ones((4, size), dtype=bool))
        result = qualification_gate([(23, bank)], tmp_path)
        assert result["passed"]
    finally:
        configure_proposal("v2_far_headway")
