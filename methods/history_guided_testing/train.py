"""Train physical kernels on measured histories; select each on historical validation."""
import time

import numpy as np
import torch

from .config import FEEDBACK, GROUPS, ROOT, SEEDS, TRAINING, group_of
from .history import historical_risk, load_history, split_indices, subset
from .io import write_json
from .kernel import MODES, RiskKernel, conditional_risk


def validation_cases(history, seed):
    x = next(iter(history.values()))["x"]
    train, validation = split_indices(x)
    support = np.random.default_rng(seed).permutation(train)[:100]
    indices = np.r_[support, validation]
    cases = []
    for name, source in history.items():
        prior = historical_risk(subset(history, train, group_of(name)), x[indices])
        cases.append((name, x[indices], prior, source["risk"][indices]))
    return cases


def validate(kernel, cases):
    grouped = {group: [] for group in GROUPS}
    kernel.eval()
    with torch.no_grad():
        for name, x, prior, truth in cases:
            for count in (20, 50):
                indices = np.r_[np.arange(count), np.arange(100, len(x))]
                mean, _ = conditional_risk(kernel, x[indices], prior[indices], truth[:count], count)
                error = np.sqrt(np.mean((mean.cpu().numpy().clip(0, 1) - truth[100:]) ** 2))
                grouped[group_of(name)].append(float(error))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def train_seed(history, seed):
    destination = ROOT / "models" / f"seed_{seed}.pt"
    if destination.exists():
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    train, _ = split_indices(next(iter(history.values()))["x"])
    histories = subset(history, train)
    kernels = {mode: RiskKernel(mode).cuda() for mode in MODES}
    optimizers = {mode: torch.optim.AdamW(kernel.parameters(), lr=TRAINING["lr"], weight_decay=0)
                  for mode, kernel in kernels.items()}
    cases = validation_cases(history, seed)
    best = {mode: np.inf for mode in MODES}
    selected, states, curve = {}, {}, []
    started = time.perf_counter()
    for step in range(1, TRAINING["steps"] + 1):
        group = tuple(GROUPS)[(step - 1) % len(GROUPS)]
        names = GROUPS[group]
        name = names[((step - 1) // len(GROUPS)) % len(names)]
        source = histories[name]
        count = TRAINING["support_counts"][((step - 1) // len(GROUPS)) % 6]
        indices = rng.choice(len(train), count + TRAINING["query_count"], replace=False)
        context_indices = rng.choice(len(train), TRAINING["context_count"], replace=False)
        sources = {key: {"x": value["x"][context_indices], "risk": value["risk"][context_indices]}
                   for key, value in histories.items() if group_of(key) != group}
        prior = historical_risk(sources, source["x"][indices])
        truth = torch.as_tensor(source["risk"][indices], device="cuda", dtype=torch.float64)
        losses = {}
        for mode, kernel in kernels.items():
            kernel.train()
            optimizers[mode].zero_grad(set_to_none=True)
            mean, covariance = conditional_risk(kernel, source["x"][indices], prior, truth[:count], count)
            factor = torch.linalg.cholesky(covariance + (FEEDBACK["noise_variance"] + 1e-8) *
                                          torch.eye(len(mean), device="cuda"))
            residual = (truth[count:] - mean)[:, None]
            quadratic = (residual * torch.cholesky_solve(residual, factor)).sum()
            nll = .5 * (quadratic + 2 * torch.log(factor.diag()).sum()) / len(mean)
            loss = nll + 5 * (mean - truth[count:]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite historical training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kernel.parameters(), 5.)
            optimizers[mode].step()
            losses[mode] = float(loss.detach())
        if step % TRAINING["validation_interval"] == 0:
            errors = {mode: validate(kernel, cases) for mode, kernel in kernels.items()}
            for mode, error in errors.items():
                if error < best[mode]:
                    best[mode], selected[mode] = error, step
                    states[mode] = {key: value.detach().cpu().clone() for key, value in kernels[mode].state_dict().items()}
            curve.append({"step": step, "loss": losses, "validation_rmse": errors})
            print("Kernel training", seed, step, {mode: round(value, 4) for mode, value in errors.items()}, flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kernels": states, "seed": seed, "selected_steps": selected,
                "validation_rmse": best}, destination)
    write_json(ROOT / "models" / f"training_{seed}.json", {
        "executed_steps": TRAINING["steps"], "gradient_updates": TRAINING["steps"] * len(MODES),
        "selected_steps": selected, "validation_rmse": best, "curve": curve,
        "elapsed_s": time.perf_counter() - started,
        "parameter_counts": {mode: sum(p.numel() for p in kernel.parameters()) for mode, kernel in kernels.items()},
        "stopping": "fixed training budget, historical-validation checkpoint selection",
    })


def load_kernel(seed, mode="adaptive"):
    state = torch.load(ROOT / "models" / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
    kernel = RiskKernel(mode)
    kernel.load_state_dict(state["kernels"][mode])
    return kernel.eval().requires_grad_(False)


def train():
    if not torch.cuda.is_available():
        raise RuntimeError("Run in the metadrive Conda environment with CUDA")
    history = load_history()
    for seed in SEEDS:
        train_seed(history, seed)


if __name__ == "__main__":
    train()
