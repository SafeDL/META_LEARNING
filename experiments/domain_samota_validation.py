"""SAMOTA author-kernel/PR comparisons and an isolated development-bank first run."""

import ast
import urllib.request
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from controllers.highway.catalog import BY_ID
from testing_methods.baselines.samota.pool import RadialNetwork, run

from .collision_metrics import collision_recall, summarize
from .construction import load_construction
from .discovery_config import write_json
from .executions import ExecutionBank, execution_contract
from .initialization import initial_indices
from .protocol import DEFAULT, ROOT, TEST_RECORDS_ROOT
from .query import QueryLedger, run_isolated
from .source_registration import specification_matches

OUTPUT = ROOT / "experiment_outputs/samota/collision/discovery/domain_comparison/stage_a"
AUTHOR_ARCHIVE = "https://ndownloader.figshare.com/files/34050833"


def author_checks():
    payload = urllib.request.urlopen(AUTHOR_ARCHIVE, timeout=30).read()
    archive = ZipFile(BytesIO(payload))
    pr_name = next(name for name in archive.namelist() if name.endswith("/lib/PR.py"))
    tree = ast.parse(archive.read(pr_name).decode("utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Polynomial_Regression"
    )
    namespace = {"np": np, "numpy": np, "preprocessing": preprocessing}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), pr_name, "exec"
        ),
        namespace,
    )
    rng = np.random.default_rng(11)
    x, y, query = rng.normal(size=(32, 16)), rng.random(32), rng.normal(size=(13, 16))
    reference = namespace[cls.name](degree=2, cluster=np.column_stack((x, y)).tolist())
    model = make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()).fit(x, y)
    pr_error = float(
        np.max(
            np.abs(
                np.clip(model.predict(query), 0, 1)
                - np.array([reference.predict(point) for point in query])
            )
        )
    )
    rbf_name = next(name for name in archive.namelist() if name.endswith("/lib/RBF.py"))
    tree = ast.parse(archive.read(rbf_name).decode("utf-8"))
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RBFLayer"
    )
    call = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "call"
    )
    backend = SimpleNamespace(
        expand_dims=lambda value: np.expand_dims(value, -1),
        transpose=np.transpose,
        exp=np.exp,
        sum=np.sum,
    )
    namespace = {"K": backend}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[call], type_ignores=[])), rbf_name, "exec"
        ),
        namespace,
    )
    points = rng.normal(size=(9, 5)).astype(np.float32)
    radial = RadialNetwork(points, centers=10)
    source_layer = SimpleNamespace(
        centers=radial.centers.detach().numpy(), betas=radial.betas.detach().numpy()
    )
    basis = namespace["call"](source_layer, points)
    expected = basis @ radial.output.weight.detach().numpy().T + radial.output.bias.detach().numpy()
    with torch.no_grad():
        rbf_error = float(np.max(np.abs(radial(torch.tensor(points)).numpy() - expected[:, 0])))
    if pr_error > 1e-10 or rbf_error > 1e-6:
        raise ValueError(
            "SAMOTA's PR or trainable radial basis differs from the author implementation"
        )
    return {
        "source": AUTHOR_ARCHIVE,
        "figshare_article": 16468530,
        "article_version": 4,
        "file_id": 34050833,
        "same_input_pr_prediction_error": pr_error,
        "same_input_radial_forward_error": rbf_error,
        "keras_training_bitwise_equivalence_claimed": False,
        "kriging_correction": "predict on standardized inputs, matching the fitted feature coordinates",
    }


def main():
    torch.set_num_threads(1)
    checks = author_checks()
    kind, target, seed = "cut_in", "development_idm_mobil", 11
    _, development = load_construction(kind)
    archive = next(a for a in development if a.source_id == target)
    bank = ExecutionBank(TEST_RECORDS_ROOT / "physical/development" / kind / target)
    if bank.candidates != archive.scenarios or not specification_matches(
        bank.manifest["contract"], execution_contract(BY_ID[target])
    ):
        raise ValueError("SAMOTA validation requires its qualified original development bank")
    initial = initial_indices(len(bank.candidates), DEFAULT.initial_count, seed)
    ledger = QueryLedger(bank.candidates, DEFAULT.budget, bank)
    result = run_isolated(
        run,
        {
            "candidates": bank.candidates,
            "initial_indices": initial,
            "budget": DEFAULT.budget,
            "seed": seed,
        },
        ledger,
    )
    write_json(
        OUTPUT / "first_run.json",
        {
            "stage": "A mechanism validation; not the frozen development comparison",
            "target_id": target,
            "scenario": kind,
            "algorithm_seed": seed,
            "initial_indices": initial,
            "result": result,
            "query_ledger": ledger.records,
            "metrics": {
                **summarize(bank.candidates, result["queries"], ledger.trajectories),
                **collision_recall(bank.candidates, archive.collision.tolist(), result["queries"]),
            },
            "new_physical_calls": 0,
        },
    )
    report = {
        "status": "passed",
        "author_checks": checks,
        "queries": len(ledger.records),
        "initialization_included": True,
        "new_physical_calls": 0,
        "coverage_restart_adaptation_confirmed_by_user": True,
        "coverage_restart_count": len(result["coverage_restarts"]),
        "phases_executed": sorted({q["reason"] for q in result["queries"]}),
        "formal_comparison_frozen": False,
    }
    write_json(OUTPUT / "audit.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    main()
