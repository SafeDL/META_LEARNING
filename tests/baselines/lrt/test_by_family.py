"""Independent historical inputs, common initialization and measured costs."""

import numpy as np
import pytest

from methods.baselines.dart.encoder import fixed_context
from methods.baselines.dart.selection import select as dart_select
from methods.baselines.hgt.baseline import select as baseline_select
from methods.baselines.hgt.history import split_indices
from methods.baselines.hgt.search import TestingSession as RiskSession, select_session
from benchmark.initialization import initial_indices
from benchmark.timing import QueryClock
from methods.baselines.cg_brt import select as cg_select
from methods.baselines.lrt.evaluation.by_family import workspace_for
from methods.common.prior import visible_history
from methods.baselines.lrt.selection import select
from methods.baselines.lrt.teacher import branch_return, replay_reference
from tests.baselines.lrt.test_method import Oracle, problem


def test_each_family_filters_before_splitting_and_context_sampling():
    for family in ("cutin", "brake"):
        workspace = workspace_for(family)
        x = next(iter(workspace.history.values()))["x"]
        assert x.shape == (1024, 5)
        assert (x[:, 4] == workspace.family).all()
        train, validation = split_indices(x)
        assert (len(train), len(validation)) == (819, 205)
        assert len(workspace.archives) == 3
        sparse = visible_history(workspace.history, "sparse", 11)
        context = fixed_context(sparse, 11)
        for source in context.values():
            assert len(source["x"]) == 128
            assert (source["x"][:, 4] == workspace.family).all()


def test_encoder_probes_follow_the_single_family_budget_fraction(monkeypatch):
    import torch
    from methods.baselines.dart import objectives

    observed = []
    original = objectives.probe_probability

    def capture(*args):
        observed.append(int(args[-2]))
        return original(*args)

    monkeypatch.setattr(objectives, "probe_probability", capture)
    priors = torch.linspace(0.1, 0.9, 48, dtype=torch.float64)[:, None].repeat(1, 2)
    covariance = torch.eye(48, dtype=torch.float64) * 0.03
    probability = torch.full((2,), 0.5, dtype=torch.float64)
    truth = priors[:, 0]
    objectives.training_loss(covariance, priors, probability, truth, 0, pool_size=1024)
    assert observed[0] == 39
    observed.clear()
    objectives.training_loss(covariance, priors, probability, truth, 0)
    assert observed[0] == 43


@pytest.mark.parametrize(
    "method", ("lrt", "dart", "cg_brt", "hgt", "gp_ei", "rf_bo", "uniform_random")
)
def test_risk_only_methods_share_initialization_and_complete_unique_queries(method):
    common, prior, truth = problem()
    indices = initial_indices(len(truth), 3, 11)
    oracle = Oracle(truth)
    clock = QueryClock()
    if method == "lrt":
        result = select(
            **common,
            oracle=oracle,
            transfer_prior=prior,
            value_model=None,
            variant="no_value_learning",
            budget=9,
            initial_queries=3,
            initial_indices=indices,
            clock=clock,
        )
        assert (
            abs(
                sum(d["loss_charged"] for d in result["decisions"])
                + result["loss_budget_remaining"]
                - 0.9
            )
            < 1e-10
        )
    elif method in ("dart", "cg_brt"):
        selector = dart_select if method == "dart" else cg_select
        extra = (
            {"proposal_covariance": prior.enhanced_covariance}
            if method == "dart"
            else {}
        )
        result = selector(
            **common,
            oracle=oracle,
            budget=9,
            initial_queries=3,
            initial_indices=indices,
            clock=clock,
            **extra,
        )
    elif method == "hgt":
        session = RiskSession(
            common["x"],
            budget=9,
            prior=common["fixed_prior"],
            covariance=common["covariance"],
            initial_indices=indices,
        )
        result = select_session(session, oracle, risk_only=True, clock=clock)
    else:
        from methods.baselines.hgt.baseline import SETTINGS

        settings = {
            **SETTINGS,
            "initial_queries": 3,
            "rf_trees": 3,
            "rf_jackknife_folds": 2,
        }
        result = baseline_select(
            method,
            common["x"],
            np.zeros(len(truth), int),
            oracle,
            11,
            budget=9,
            settings=settings,
            cell_count=1,
            initial_indices=indices,
            risk_only=True,
            clock=clock,
        )
    assert result["selected_indices"][:3] == indices
    assert len(set(result["selected_indices"])) == 9
    assert len(clock.steps) == 9 and min(clock.steps) >= 0


def test_branch_replay_preserves_explicit_initialization_and_accounting():
    common, prior, truth = problem()
    common["initial_indices"] = initial_indices(len(truth), 3, 11)
    reference, available = replay_reference(common, [], 9, 3)
    candidate, _ = reference.recommend(available, 0)
    before = reference.model.indices.copy()
    found, selected = branch_return(reference, available, candidate, truth)
    assert selected[:3] == common["initial_indices"]
    assert len(set(selected)) == 9
    assert found == int((truth[selected] > 0.5).sum())
    assert reference.model.indices == before


def test_cost_prefix_includes_initialization_and_censors_unreached_targets():
    from methods.baselines.lrt.evaluation.family_report import (
        break_even,
        hit_query,
        task_cost,
    )

    trace = {
        "positives": 4,
        "found": [0] * 3 + [1] * 10 + [2] * 20 + [3] * 167,
        "execution_s": [2.0] * 200,
        "timing": {"task_initialization_s": 5.0, "query_compute_s": [0.1] * 200},
    }
    assert hit_query(trace, 0.8) is None
    assert hit_query(trace, 0.5) == 14
    trace["positives"] = 1
    assert hit_query(trace, 0.8) == 10
    assert task_cost(trace, 10) == {
        "compute_s": 6.0,
        "execution_s": 20.0,
        "queries": 10,
    }
    assert break_even(100, 10) == 11
    assert break_even(100, -10) is None


def test_execution_cost_sensitivity_respects_both_saving_directions():
    from methods.baselines.lrt.evaluation.family_report import execution_time_condition

    assert execution_time_condition(1000, 10, 2) == {
        "direction": "above",
        "seconds": 10.0,
    }
    assert execution_time_condition(1000, -30, -2) == {
        "direction": "below",
        "seconds": 10.0,
    }
    assert execution_time_condition(1000, 10, -2)["direction"] == "none"
    assert execution_time_condition(1000, -30, 0)["direction"] == "all"


def test_unilateral_spending_keeps_censored_outcomes_and_is_symmetric():
    from methods.baselines.lrt.evaluation.family_report import unilateral_spending

    trace = {
        "execution_s": [1.0] * 200,
        "timing": {"task_initialization_s": 1.0, "query_compute_s": [0.1] * 200},
    }
    left = unilateral_spending(trace, trace, 10, None, 0, 0)
    assert left["baseline_hit_query"] is None
    assert left["lower_cost_achiever"] == "lrt"
    right = unilateral_spending(trace, trace, None, 10, 0, 0)
    assert right["lrt_hit_query"] is None
    assert right["lower_cost_achiever"] == "baseline"
    expensive = unilateral_spending(trace, trace, 10, None, 100000, 0)
    assert expensive["lower_cost_achiever"] == "unresolved"


@pytest.mark.parametrize("field", ("risk", "execution"))
def test_saved_record_audit_rejects_changed_feedback_or_execution_cost(field):
    from methods.baselines.lrt.evaluation.family_verify import verify_observations

    risk = np.where(np.arange(256) % 37 == 0, 0.7, 0.1)
    ids = np.asarray([str(i) for i in range(256)])
    durations = np.linspace(1, 2, 256)
    selected = np.arange(200)
    trace = {
        "method": "uniform_random",
        "selected_indices": selected.tolist(),
        "queries": [{"index": int(i), "risk": float(risk[i])} for i in selected],
        "disclosures": [
            {"index": int(i), "risk": float(risk[i]), "collision": None}
            for i in selected
        ],
        "found": np.cumsum(risk[selected] > 0.5).tolist(),
        "scenario_ids": ids[selected].tolist(),
        "execution_s": durations[selected].tolist(),
        "timing": {
            "task_initialization_s": 1.0,
            "query_compute_s": [0.1] * 200,
            "response_transport_s": [0.001] * 200,
        },
    }
    verify_observations(trace, risk, ids, durations, list(range(10)))
    if field == "risk":
        trace["queries"][0]["risk"] = trace["disclosures"][0]["risk"] = 0.8
    else:
        trace["execution_s"][0] += 0.5
    with pytest.raises(AssertionError):
        verify_observations(trace, risk, ids, durations, list(range(10)))


def test_isolated_worker_finishes_200_paid_queries_with_only_risk_feedback():
    import multiprocessing as mp
    from benchmark.risk_channel import disclose_risk
    from methods.baselines.lrt.evaluation.by_family import selector_worker

    x = np.column_stack((np.random.default_rng(11).random((256, 4)), np.zeros(256)))
    truth = np.clip(x[:, 0] + 0.2 * x[:, 1], 0, 1)
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=selector_worker, args=(child,))
    process.start()
    child.close()
    try:
        task = {
            "method": "gp_ei",
            "common": {"x": x, "initial_indices": initial_indices(256, 10, 11)},
        }
        result = disclose_risk(parent, task, truth, process)
        assert len(set(result["selected_indices"])) == 200
        assert len(result["timing"]["query_compute_s"]) == 200
        assert len(result["timing"]["response_transport_s"]) == 200
        assert all(row["collision"] is None for row in result["disclosures"])
    finally:
        if process.is_alive():
            parent.send(("stop", None))
        parent.close()
        process.join(timeout=15)
        if process.is_alive():
            process.terminate()
            process.join()
        assert process.exitcode == 0


def test_complete_report_does_not_claim_savings_for_identical_queries(tmp_path):
    from methods.baselines.hgt.io import write_json
    from methods.baselines.lrt.evaluation.by_family import METHODS, SUT_IDS
    from methods.baselines.lrt.evaluation.family_report import family_summary, render

    folder = tmp_path / "cutin"
    write_json(
        folder / "build_cost.json",
        {
            "stages": {
                name: {"status": "complete", "elapsed_s": 10.0}
                for name in ("kernel", "encoder", "branch_labels", "value")
            }
        },
    )
    queries = [{"index": i, "risk": 0.9 if i < 4 else 0.1} for i in range(200)]
    decisions = [
        {
            "selected_index": i,
            "reference_index": i,
            "loss_charged": 0.0,
            "posterior_replacement_loss": 0.0,
        }
        for i in range(200)
    ]
    for pool in (0, 1, 2):
        write_json(
            folder / "pool_cache" / f"pool_{pool}.json",
            {"base": 1.0, "encoder": 1.0, "archive": 1.0},
        )
        for sut in SUT_IDS:
            for method in METHODS:
                trace = {
                    "method": method,
                    "selected_indices": list(range(200)),
                    "queries": queries,
                    "found": [min(i + 1, 4) for i in range(200)],
                    "positives": 4,
                    "H200": 4,
                    "recall": 1.0,
                    "execution_s": [1.0] * 200,
                    "decisions": decisions,
                    "loss_budget_remaining": 20.0,
                    "timing": {
                        "task_initialization_s": 1.0,
                        "query_compute_s": [0.1] * 200,
                    },
                }
                write_json(
                    folder
                    / "trajectories"
                    / f"pool_{pool}"
                    / sut
                    / f"{method}_11.json",
                    trace,
                )
    summary = family_summary(folder)
    assert summary["trajectory_count"] == 144
    assert all(row["accepted"] for row in summary["acceptance"].values())
    for row in summary["costs"]:
        assert row["mean_saved_queries"] == 0
        assert row["reuse"]["100"]["saving_pct"] < 0
        assert row["break_even_tasks"] is None
    assert "CG-BRT" in render([summary])


def test_costs_keep_equal_pool_weights_when_reach_rates_differ(monkeypatch):
    from methods.baselines.lrt.evaluation import family_report

    monkeypatch.setattr(family_report, "SUT_IDS", ("a", "b"))
    rows = {}
    for pool in (0, 1, 2):
        for sut in ("a", "b"):
            for method in ("lrt", "cg_brt"):
                hit = 40 if (pool, sut, method) == (1, "a", "cg_brt") else 10
                if (pool, sut, method) == (1, "b", "lrt"):
                    hit = 201
                rows[pool, sut, method] = {
                    "positives": 1,
                    "found": [int(i >= hit) for i in range(1, 201)],
                    "execution_s": [1.0] * 200,
                    "timing": {
                        "task_initialization_s": 0.0,
                        "query_compute_s": [0.1] * 200,
                    },
                }
    build = {
        "stages": {
            name: {"elapsed_s": 0.0}
            for name in ("kernel", "encoder", "branch_labels", "value")
        }
    }
    preparation = {pool: dict(base=0, encoder=0, archive=0) for pool in (0, 1, 2)}
    result = family_report.paired_costs(rows, build, preparation, "cg_brt", 0.8)
    assert result["outcomes"]["both_reached"] == 5
    assert result["outcomes"]["baseline_only"] == 1
    assert result["mean_saved_queries"] == pytest.approx(10.0)
    assert result["reuse"]["100"]["lrt_s"] == pytest.approx(11.0)
    assert result["reuse"]["100"]["baseline_s"] == pytest.approx(22.0)
    assert result["reuse"]["100"]["saving_pct"] == pytest.approx(50.0)


@pytest.mark.parametrize("method", ("cg_brt", "dart", "lrt"))
def test_default_decisions_match_retained_original_source(method):
    import types
    from zipfile import ZipFile
    from methods.baselines.lrt.config import ROOT
    from methods.common.snapshots import current_path, current_source

    common, prior, truth = problem()
    package = (
        "methods.baselines.cg_brt"
        if method == "cg_brt"
        else ("methods.baselines.dart" if method == "dart" else "methods.baselines.lrt")
    )
    archive_path = (
        ROOT / "frozen_method.zip"
        if method == "lrt"
        else ROOT.parent / "dart/frozen_inputs.zip"
    )
    expected = package.replace(".", "/") + "/selection.py"
    with ZipFile(archive_path) as archive:
        name = next(
            name
            for name in archive.namelist()
            if current_path(name.removeprefix("source/")) == expected
        )
        source = current_source(
            archive.read(name), name.removeprefix("source/")
        ).decode("utf-8")
    module = types.ModuleType(package + ".retained_selection")
    module.__package__ = package
    exec(compile(source, str(archive_path) + ":" + name, "exec"), module.__dict__)
    kwargs = {"budget": 12, "initial_queries": 2}
    selector = cg_select
    if method == "dart":
        selector = dart_select
        kwargs["proposal_covariance"] = prior.enhanced_covariance
    elif method == "lrt":
        selector = select
        kwargs.update(
            transfer_prior=prior, value_model=None, variant="no_value_learning"
        )
    original = module.select(**common, oracle=Oracle(truth), **kwargs)
    current = selector(**common, oracle=Oracle(truth), **kwargs)
    assert current["selected_indices"] == original["selected_indices"]
    assert current["queries"] == original["queries"]
