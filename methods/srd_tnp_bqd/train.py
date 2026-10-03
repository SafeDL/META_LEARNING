"""Two-stage historical training on the original five-source S01 benchmark."""
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score
import torch
from threadpoolctl import threadpool_limits

from .benchmark import CONFIG, SEEDS
from .common import append_jsonl, read_json, write_json
from .data import SourceContext
from .discrepancy import conditional_distribution, gaussian_nll
from .s01 import COUNT, ROOT, SOURCES, configuration, response_guard, source_contexts
from .historical import HistoricalModel
from .nonstationary_kernel import ResidualKernel


TRAINING = CONFIG["training"]


def setup(seed):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training required")
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def split_indices():
    order = np.random.default_rng(TRAINING["split_seed"]).permutation(COUNT)
    count = TRAINING["spatial_train_count"]
    return order[:count], order[count:]


def reconstruction(mean, target):
    return (mean - target).square().mean()


def subset(c, indices):
    return SourceContext(c.source_id, c.x[indices], c.z[indices], c.valid[indices])


def checkpoint(path, model, kernel, step, metrics=None):
    torch.save({"model": model.state_dict(), "kernel": kernel.state_dict(), "step": step,
                "model_config": model.config, "kernel_config": configuration()["residual_gp"],
                "validation": metrics}, path)


def load_modules(path):
    state = torch.load(path, map_location="cuda", weights_only=False)
    model = HistoricalModel(state["model_config"]).cuda()
    kernel = ResidualKernel(configuration()["residual_gp"]).cuda()
    model.load_state_dict(state["model"])
    kernel.load_state_dict(state["kernel"])
    return model, kernel


@torch.no_grad()
def validate(model, kernel, sources, train, validation, stage, targets=None):
    model.eval()
    kernel.eval()
    report = []
    for target in (sources if targets is None else targets):
        contexts = [subset(c, train) for c in sources if stage == "source" and c.source_id == target.source_id
                    or stage != "source" and c.source_id != target.source_id]
        prior = model.predict(contexts, target.x[validation])
        m = torch.as_tensor(prior.m, device="cuda", dtype=torch.float64)
        h = torch.as_tensor(prior.h, device="cuda", dtype=torch.float64)
        x = torch.as_tensor(target.x[validation], device="cuda", dtype=torch.float64)
        z = torch.as_tensor(target.z[validation], device="cuda", dtype=torch.float64)
        for count in ((0,) if stage == "source" else (0, 10, 50, 199)):
            mean, covariance = conditional_distribution(kernel, x, h, m, z[:count], count)
            target_z = z[count:]
            nll, _ = gaussian_nll(mean, covariance, target_z)
            truth = expit(target_z.cpu().numpy()[:, 0]) > .5
            scores = mean.cpu().numpy()[:, 0]
            ap = float(average_precision_score(truth, scores)) if truth.any() and (~truth).any() else None
            errors = (mean - target_z).abs().reshape(-1)
            covered = (errors <= 1.96 * covariance.diag().sqrt()).cpu().numpy()
            report.append({"source": target.source_id, "support": count, "tail_ap": ap,
                           "balanced_mse": float(reconstruction(mean, target_z)),
                           "nll": float(nll), "rmse_z": float(errors.square().mean().sqrt()),
                           "coverage_95": float(covered.mean()),
                           "high_risk_coverage95": float(covered[truth].mean()) if truth.any() else None,
                           "ordinary_coverage95": float(covered[~truth].mean()) if (~truth).any() else None})
    return {"tail_ap": float(np.mean([r["tail_ap"] for r in report if r["tail_ap"] is not None])),
            "balanced_mse": float(np.mean([r["balanced_mse"] for r in report])),
            "nll": float(np.mean([r["nll"] for r in report])),
            "rmse_z": float(np.mean([r["rmse_z"] for r in report])), "groups": report}


def support_paths(model, kernel, sources, train):
    from .acquisition import Selector
    from .s01 import CELL_COUNT, cells_for, load_scenes
    from .oracle import ContinuousOracle
    rng = np.random.default_rng(11)
    paths = {}
    cells = cells_for(load_scenes("A"))[train]
    for target in sources:
        model.eval()
        prior = model.predict([subset(c, train) for c in sources if c.source_id != target.source_id], target.x[train])
        oracle = ContinuousOracle([str(i) for i in train],
                                 lambda i, c=target: (None, float(expit(c.z[train[i], 0])), True), 199)
        result = Selector(prior, kernel, cell_count=CELL_COUNT).run(target.x[train], cells, oracle, 199)
        paths[target.source_id] = {"random": rng.permutation(train)[:199].tolist(),
                                  "history": train[np.argsort(-prior.m[:, 0], kind="stable")[:199]].tolist(),
                                  "mix": train[result["selected_indices"]].tolist()}
    # Selector freezes its kernel; stage-two training explicitly re-enables it.
    kernel.requires_grad_(True)
    return paths


def train_stage(directory, stage, model, kernel, sources, train, validation, seed, fixed_steps=None,
                validation_targets=None, freeze_kernel=False):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + (1000 if stage == "transfer" else 0))
    paths = support_paths(model, kernel, sources, train) if stage == "transfer" else None
    if paths is not None:
        write_json(directory / "support_paths.json", paths)
    if freeze_kernel:
        kernel.freeze()
    params = list(model.parameters()) + (list(kernel.parameters()) if stage == "transfer" and not freeze_kernel else [])
    optimizer = torch.optim.AdamW(params, lr=3e-4 if stage == "source" else 1e-4, weight_decay=0)
    counts = TRAINING["support_checkpoints"]
    best, best_step, stale = None, 0, 0
    coverage = np.zeros((len(sources), len(sources[0].x)), int)
    started = time.perf_counter()
    max_steps = TRAINING["stage1_max_steps" if stage == "source" else "stage2_max_steps"]
    for step in range(1, (max_steps if fixed_steps is None else fixed_steps) + 1):
        model.train()
        kernel.train()
        optimizer.zero_grad(set_to_none=True)
        j = (step - 1) % len(sources)
        target = sources[j]
        size = (256, 512, 1024, len(train))[((step - 1) // len(sources)) % 4]
        if stage == "source":
            order = rng.permutation(train)
            query, context = order[:64], order[64:64 + size]
            coverage[j, context] += 1
            m, _ = model([subset(target, context)], target.x[query])
            z = model.as_tensor(target.z[query])
            loss = reconstruction(m, z) + 1e-4 * sum(p.square().sum() for p in model.parameters())
        else:
            count = counts[((step - 1) // len(sources)) % len(counts)]
            strategy = ("random", "history", "mix")[((step - 1) // (len(sources) * len(counts))) % 3]
            support = np.asarray(paths[target.source_id][strategy][:count], int)
            query = rng.choice(np.setdiff1d(train, support), 64, replace=False)
            contexts = []
            for k, c in enumerate(sources):
                if k != j:
                    selected = rng.permutation(train)[:size]
                    contexts.append(subset(c, selected))
                    coverage[k, selected] += 1
            ids = np.r_[support, query]
            m, h = model(contexts, target.x[ids])
            z = torch.as_tensor(target.z[ids], device="cuda", dtype=torch.float64)
            mean, covariance = conditional_distribution(kernel, target.x[ids], h, m, z[:count], count)
            loss, _ = gaussian_nll(mean, covariance, z[count:])
            same_source = rng.permutation(train)
            reconstruct_ids, context_ids = same_source[:64], same_source[64:320]
            reconstructed, _ = model([subset(target, context_ids)], target.x[reconstruct_ids])
            loss = loss + .1 * reconstruction(reconstructed, model.as_tensor(target.z[reconstruct_ids]))
            loss = loss + 1e-4 * sum(p.square().sum() for p in kernel.gate.parameters())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite {stage} loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.)
        optimizer.step()
        if step % TRAINING["validation_interval"] == 0 or step == fixed_steps:
            metrics = validate(model, kernel, sources, train, validation, stage, validation_targets) if fixed_steps is None else None
            criterion = (metrics["nll"], metrics["rmse_z"]) if metrics else None
            if metrics is None or best is None or criterion < best:
                best, best_step, stale = criterion, step, 0
                checkpoint(directory / "best.pt", model, kernel, step, metrics)
            else:
                stale += 1
            append_jsonl(directory / "curve.jsonl", {"step": step, "loss": float(loss.detach()), "validation": metrics})
            print(directory.parent.name, stage, step, float(loss.detach()), metrics and metrics["tail_ap"], flush=True)
            if fixed_steps is None and stale >= TRAINING["patience"]:
                break
    state = torch.load(directory / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(state["model"])
    kernel.load_state_dict(state["kernel"])
    if not np.all(coverage[:, train] > 0):
        raise ValueError("unseen training context record")
    result = {"best_step": best_step, "executed_steps": step, "best_criterion": best,
              "elapsed_s": time.perf_counter() - started, "context_coverage_min": int(coverage[:, train].min())}
    write_json(directory / "completed.json", result)
    return result



def main():
    response_guard()
    if all((ROOT / "model" / f"seed_{seed}.pt").exists() for seed in SEEDS):
        frozen = read_json(ROOT / "model/frozen_model.json")
        assert frozen["D_used_for_selection"] is False
        print("Reuse the five unchanged S01 checkpoints; no retraining or target selection.")
        return
    config = configuration()
    train, validation = split_indices()
    sources = source_contexts()
    completed = []
    for held_out in sources:
        setup(11)
        model = HistoricalModel(config["model"]).cuda()
        kernel = ResidualKernel(config["residual_gp"]).cuda()
        allowed = [c for c in sources if c.source_id != held_out.source_id]
        directory = ROOT / "training" / f"fold_{held_out.source_id}"
        record = {}
        for stage in ("source", "transfer"):
            record[stage] = train_stage(directory / stage, stage, model, kernel, allowed,
                                        train, validation, 11,
                                        validation_targets=[held_out] if stage == "transfer" else None)
        completed.append((held_out.source_id, record))
    steps = {stage: int(np.median([r[stage]["best_step"] for _, r in completed]))
             for stage in ("source", "transfer")}
    best = min(completed, key=lambda item: item[1]["transfer"]["best_criterion"])
    kernel_path = ROOT / "training" / f"fold_{best[0]}" / "transfer/best.pt"
    write_json(ROOT / "model/A_development_selection.json",
               {"final_steps": steps, "selected_frozen_kernel_fold": best[0],
                "criterion": "outer_fold_unqueried_nll_then_rmse", "D_used": False})
    for seed in SEEDS:
        setup(seed)
        model = HistoricalModel(config["model"]).cuda()
        kernel = ResidualKernel(config["residual_gp"]).cuda()
        directory = ROOT / "training" / f"final_seed_{seed}"
        for stage in ("source", "transfer"):
            if stage == "transfer":
                state = torch.load(kernel_path, map_location="cuda", weights_only=False)
                kernel.load_state_dict(state["kernel"])
            train_stage(directory / stage, stage, model, kernel, sources,
                        np.arange(COUNT), validation, seed, steps[stage], freeze_kernel=stage == "transfer")
        model.freeze()
        kernel.freeze()
        checkpoint(ROOT / "model" / f"seed_{seed}.pt", model, kernel, steps["transfer"])
    write_json(ROOT / "model/frozen_model.json",
               {"status": "FROZEN", "final_steps": steps, "D_used_for_selection": False,
                "sources": list(SOURCES), "risk_threshold": .5})


if __name__ == "__main__":
    main()
