"""Build from historical responses or reuse the required family-specific bases."""

import shutil

import torch
from threadpoolctl import threadpool_limits

from methods.baselines.dart.train import prepare_cases, train_seed as train_encoder
from methods.baselines.hgt.history import load_history
from methods.baselines.hgt.io import read_json, write_json
from methods.baselines.hgt.train import train_seed as train_kernel
from benchmark.timing import now
from methods.common.workspace import HistoricalWorkspace

from .config import ARCHIVE_ROOT, BASELINE_ROOT, ROOT, SEED
from .data import construction_components, generate
from .fit import fit
from .train import train


def workspace_for(family):
    code = {"cutin": 0, "brake": 1}[family]
    history = {
        name: {
            key: array[source["x"][:, 4] == code].copy()
            for key, array in source.items()
        }
        for name, source in load_history().items()
    }
    return HistoricalWorkspace(ROOT / family, code, history)


def stage(workspace, name, required, action):
    path = workspace.root / "build_cost.json"
    ledger = read_json(path) if path.exists() else {"stages": {}, "physical_calls": 0}
    previous = ledger["stages"].get(name, {})
    if previous.get("status") == "complete" and all(p.exists() for p in required):
        return
    started = now()
    ledger["stages"][name] = {"status": "running"}
    write_json(path, ledger)
    try:
        action()
    except BaseException as error:
        ledger["stages"][name].update(status="failed", error=repr(error))
        raise
    else:
        ledger["stages"][name]["status"] = "complete"
    finally:
        ledger["stages"][name]["elapsed_s"] = (
            previous.get("elapsed_s", 0.0) + now() - started
        )
        write_json(path, ledger)
    print(f"BUILD {workspace.root.name} {name} {now() - started:.2f}s", flush=True)


def ensure_base(workspace, kind):
    directory = workspace.kernel_root if kind == "kernel" else workspace.encoder_root
    filename = "seed_11.pt" if kind == "kernel" else "decision_11.pt"
    source = BASELINE_ROOT / workspace.root.name
    existing = source / kind / "models" / filename
    destination = directory / "models" / filename
    if existing.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(existing, destination)
        report = existing.with_name("training_decision_11.json")
        if kind == "encoder" and report.exists():
            shutil.copy2(report, destination.with_name(report.name))
        cost = read_json(source / "build_cost.json")["stages"][kind]["elapsed_s"]
        write_json(
            directory / "reuse.json",
            {"source": str(existing), "standalone_construction_s": cost},
        )
    elif kind == "kernel":
        train_kernel(
            workspace.history, SEED, root=directory, modes=("global_feedback",)
        )
    else:
        train_encoder(
            SEED,
            "decision",
            prepare_cases(
                workspace.history,
                SEED,
                root=directory,
                kernel_root=workspace.kernel_root,
                common_initial=True,
            ),
            root=directory,
        )


def build_family(workspace):
    for kind, filename in (("kernel", "seed_11.pt"), ("encoder", "decision_11.pt")):
        stage(
            workspace,
            kind,
            [workspace.root / kind / "models" / filename],
            lambda kind=kind: ensure_base(workspace, kind),
        )

    def prior():
        existing = ARCHIVE_ROOT / workspace.root.name / "prior.json"
        if existing.exists():
            shutil.copy2(existing, workspace.root / "prior.json")
        else:
            fit(workspace)

    stage(workspace, "prior", [workspace.root / "prior.json"], prior)
    expected = [
        workspace.root / "historical" / f"{name}_{condition}.pt"
        for name in workspace.history
        for condition in workspace.archives
    ]
    stage(
        workspace,
        "historical",
        [*expected, workspace.root / "policy.json"],
        lambda: generate(workspace),
    )
    stage(
        workspace,
        "cost_measurement",
        [workspace.root / "historical_cost.json"],
        lambda: construction_components(workspace),
    )
    stage(
        workspace,
        "count_network",
        [workspace.root / "model.pt", workspace.root / "training.json"],
        lambda: train(workspace),
    )


def main():
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for family in ("cutin", "brake"):
            build_family(workspace_for(family))


if __name__ == "__main__":
    main()
