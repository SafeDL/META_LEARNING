"""Run baselines under one frozen taskbook, casebooks, and metric protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from pearl_learning.src.baselines import (
    BASELINE_NAMES, OracleTaskObservation, PooledLogicalMergeEnv, write_baseline_manifest,
)
from pearl_learning.src.casebook import load_casebook
from pearl_learning.src.checkpoint import load_checkpoint
from pearl_learning.src.io import content_hash, read_config, write_json
from pearl_learning.src.metrics import summarize
from pearl_learning.src.pearl_agent import PEARLAgent
from pearl_learning.src.task_env import LogicalMergeEnv
from pearl_learning.src.taskbook import load_taskbook, taskbook_payload


def _evaluation_tasks(taskbook: Mapping[str, list[Any]]) -> list[Any]:
    return list(taskbook["meta_test_template"]) + list(taskbook["meta_test_logical"])


def _evaluate_sac(model: Any, task: Any, config: Mapping[str, Any], cases: list[Mapping[str, Any]],
                  transform: Callable[[np.ndarray, Any], np.ndarray] | None = None) -> dict[str, Any]:
    """Evaluate one deterministic policy on the frozen query cases only."""
    env = LogicalMergeEnv(task, config, cases)
    records: list[dict[str, Any]] = []
    try:
        for case in cases:
            observation, _ = env.reset(options={"case": case})
            terminated = truncated = False
            while not (terminated or truncated):
                model_input = transform(observation, task) if transform else observation
                action, _ = model.predict(model_input, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(action)
            records.append(env.episode_record())
    finally:
        env.close()
    return {"summary": summarize(records, case_metadata={str(case["case_id"]): case for case in cases}), "records": records}


def _evaluate_no_context(agent: PEARLAgent, task: Any, config: Mapping[str, Any],
                         cases: list[Mapping[str, Any]]) -> dict[str, Any]:
    """The PEARL ablation: query every episode from the fixed unit-normal prior."""
    env = LogicalMergeEnv(task, config, cases)
    records: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            mu, _ = agent.prior()
            for case in cases:
                observation, _ = env.reset(options={"case": case})
                terminated = truncated = False
                while not (terminated or truncated):
                    obs = torch.as_tensor(observation[None], dtype=torch.float32, device=agent.device)
                    action = agent.act(obs, mu, deterministic=True)[0].detach().cpu().numpy()
                    observation, _, terminated, truncated, _ = env.step(action)
                records.append(env.episode_record())
    finally:
        env.close()
    return {"summary": summarize(records, case_metadata={str(case["case_id"]): case for case in cases}), "records": records, "context": "unit_normal_prior"}


def _save_metrics(output: Path, payload: Mapping[str, Any]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, dict(payload))
    return output


def _implementation_hash() -> str:
    """Fingerprint executable baseline/environment code, including dirty edits."""
    package = Path(__file__).resolve().parents[1]
    paths = sorted((package / "src").rglob("*.py")) + [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _bind_training_protocol(root: Path, protocol: Mapping[str, Any], resume: bool) -> Path:
    """Refuse to resume models trained under different code/data semantics."""
    path = root / "training_protocol.json"
    reusable_artifacts = root.exists() and any(
        item.suffix == ".zip" or "partial" in item.name for item in root.rglob("*") if item.is_file()
    )
    if resume and reusable_artifacts:
        if not path.exists():
            raise SystemExit(f"refusing legacy resume without a bound training protocol: {root}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(protocol):
            raise SystemExit(f"refusing incompatible baseline resume: {root}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(protocol))
    return path


def _new_sac(env: Any, seed: int) -> Any:
    from stable_baselines3 import SAC
    return SAC("MlpPolicy", env, seed=seed, verbose=0)


def _latest_sac_checkpoint(root: Path, prefix: str) -> Path | None:
    candidates = list(root.glob(f"{prefix}_*_steps.zip"))
    if not candidates:
        return None
    def steps(path: Path) -> int:
        suffix = path.stem.removeprefix(f"{prefix}_").removesuffix("_steps")
        return int(suffix)
    return max(candidates, key=steps)


def _partial_payload(path: Path, protocol: Mapping[str, Any], resume: bool, field: str) -> dict[str, Any]:
    """Load resumable metrics only when their complete protocol still matches.

    A model can be continued to a larger step budget, but metrics collected
    from the earlier model must never be relabelled as results from the larger
    budget.  Legacy partial files have no protocol and are intentionally
    invalidated here.
    """
    if not resume or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != dict(protocol):
        return {}
    value = payload.get(field, {})
    if not isinstance(value, dict):
        raise ValueError(f"partial baseline field {field!r} must be a mapping")
    return value


def _partial_protocol_matches(path: Path, protocol: Mapping[str, Any], resume: bool) -> bool:
    if not resume or not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("protocol") == dict(protocol)


def _write_partial(path: Path, protocol: Mapping[str, Any], **payload: Any) -> Path:
    return _save_metrics(path, {"protocol": dict(protocol), **payload})


def _learn_sac(model: Any, total_steps: int, checkpoint_root: Path | None, checkpoint_prefix: str, checkpoint_interval_steps: int = 0) -> None:
    remaining = max(0, int(total_steps) - int(model.num_timesteps))
    if not remaining:
        return
    callback = None
    if checkpoint_root is not None and checkpoint_interval_steps:
        from stable_baselines3.common.callbacks import CheckpointCallback
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        callback = CheckpointCallback(save_freq=int(checkpoint_interval_steps), save_path=str(checkpoint_root), name_prefix=checkpoint_prefix)
    model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=callback)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--taskbook", required=True); parser.add_argument("--casebook-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", choices=BASELINE_NAMES, required=True); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--env-steps", type=int, required=True)
    parser.add_argument("--pretrain-steps", type=int, help="pooled pre-training budget; defaults to --env-steps")
    parser.add_argument(
        "--pooled-steps-per-task", type=int,
        help="for a pooled SAC policy, allocate this many training steps to each meta-train task",
    )
    parser.add_argument("--per-task-policy-root", help="reuse frozen per-task SAC policies when building the cross-task transfer matrix")
    parser.add_argument("--pooled-pretrain-model", help="reuse a frozen pooled SAC checkpoint for pooled_finetune_sac")
    parser.add_argument("--resume", action="store_true", help="resume reusable per-task artifacts for supported long-running baselines")
    parser.add_argument("--pearl-checkpoint", help="required by pearl_no_context")
    parser.add_argument("--checkpoint-interval-steps", type=int, default=0, help="save resumable SAC checkpoints every N steps for per-task training")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal-run", action="store_true", help="explicitly authorize a non-smoke baseline run after a separate resource plan has been approved")
    args = parser.parse_args()
    if not args.smoke and not args.formal_run:
        parser.error("non-smoke baseline runs are disabled by default; use --smoke for a flow check, or pass --formal-run only after approving a separate experiment and resource plan")
    if args.checkpoint_interval_steps < 0:
        raise ValueError("--checkpoint-interval-steps must be non-negative")
    if args.pooled_steps_per_task is not None and args.pooled_steps_per_task <= 0:
        raise ValueError("--pooled-steps-per-task must be positive")
    cfg = read_config(args.config)
    taskbook = load_taskbook(args.taskbook)
    taskbook_hash = content_hash(taskbook_payload(taskbook))
    train_tasks, evaluation_tasks = taskbook["meta_train"], _evaluation_tasks(taskbook)
    all_tasks = list(train_tasks) + list(taskbook["meta_validation"]) + evaluation_tasks
    casebooks = {task.task_id: load_casebook(task, args.casebook_root) for task in all_tasks}
    case_hashes = {task_id: content_hash(book) for task_id, book in casebooks.items()}
    query_cases = lambda task: casebooks[task.task_id]["test_query"][:1 if args.smoke else None]
    root = Path(args.output) / args.baseline
    artifacts: dict[str, str] = {}
    checkpoint_hash: str | None = None
    partial_protocol = {
        "taskbook_hash": taskbook_hash,
        "config_hash": content_hash(cfg),
        "casebook_hashes": case_hashes,
        "seed": int(args.seed),
        "environment_steps": int(args.env_steps),
        "smoke": bool(args.smoke),
        "implementation_hash": _implementation_hash(),
    }
    training_protocol = {
        key: value for key, value in partial_protocol.items()
        if key not in {"environment_steps"}
    }
    training_protocol["baseline"] = args.baseline
    training_protocol_path = _bind_training_protocol(root, training_protocol, args.resume)
    artifacts["training_protocol"] = str(training_protocol_path)
    training_budget: dict[str, int | str] = {"scope": "single_policy", "total_environment_steps": int(args.env_steps)}

    if args.baseline == "pearl_no_context":
        if not args.pearl_checkpoint:
            raise SystemExit("pearl_no_context requires --pearl-checkpoint")
        device = torch.device("cuda" if torch.cuda.is_available() and cfg["experiment"].get("device") != "cpu" else "cpu")
        agent = PEARLAgent(int(cfg["environment"]["observation_dim"]), int(cfg["environment"]["action_dim"]), cfg, device)
        checkpoint = load_checkpoint(args.pearl_checkpoint, agent, device)
        if checkpoint["taskbook_hash"] != taskbook_hash:
            raise SystemExit("PEARL checkpoint belongs to a different frozen taskbook")
        before = agent.parameter_hash()
        result = {task.task_id: _evaluate_no_context(agent, task, cfg, query_cases(task)) for task in evaluation_tasks}
        if agent.parameter_hash() != before:
            raise RuntimeError("no-context baseline changed PEARL parameters")
        checkpoint_hash = json.loads(Path(args.pearl_checkpoint).with_suffix(".manifest.json").read_text(encoding="utf-8"))["checkpoint_hash"]
        artifacts["metrics"] = str(_save_metrics(root / "no_context_metrics.json", {"protocol": "fixed_unit_normal_prior", "tasks": result}))

    else:
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise SystemExit("stable_baselines3 is required for SAC baselines") from exc

        train_books = {task.task_id: casebooks[task.task_id] for task in train_tasks}
        if args.baseline in {"per_task_sac", "cross_task_policy_matrix"}:
            training_budget = {
                "scope": "per_task_independent" if args.baseline == "per_task_sac" else "evaluation_only",
                "per_task_environment_steps": int(args.env_steps),
                "total_environment_steps": int(args.env_steps) * len(train_tasks) if args.baseline == "per_task_sac" else 0,
            }
            policies: dict[str, Any] = {}
            partial_path = root / "per_task_partial_metrics.json"
            partial_metrics = (
                _partial_payload(partial_path, partial_protocol, args.resume, "tasks")
                if args.baseline == "per_task_sac" else {}
            )
            for index, task in enumerate(train_tasks):
                env = LogicalMergeEnv(task, cfg, train_books[task.task_id]["train_pool"])
                try:
                    source = Path(args.per_task_policy_root) / f"{task.task_id}.zip" if args.per_task_policy_root else None
                    if args.baseline == "cross_task_policy_matrix" and source:
                        if not source.exists():
                            raise SystemExit(f"missing frozen per-task SAC policy: {source}")
                        model = SAC.load(source, device="auto")
                        artifacts[f"policy:{task.task_id}"] = str(source)
                    else:
                        target = root / "policies" / f"{task.task_id}.zip"; target.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint_root = root / "checkpoints" / task.task_id if args.baseline == "per_task_sac" and args.checkpoint_interval_steps else None
                        partial_checkpoint = _latest_sac_checkpoint(checkpoint_root, task.task_id) if args.baseline == "per_task_sac" and args.resume and checkpoint_root is not None else None
                        if args.baseline == "per_task_sac" and args.resume and target.exists():
                            model = SAC.load(target, env=env, device="auto")
                        elif args.baseline == "per_task_sac" and partial_checkpoint is not None:
                            model = SAC.load(partial_checkpoint, env=env, device="auto")
                        else:
                            model = _new_sac(env, args.seed + index)
                        _learn_sac(model, args.env_steps, checkpoint_root, task.task_id, args.checkpoint_interval_steps)
                        if args.baseline in {"per_task_sac", "cross_task_policy_matrix"}:
                            model.save(target)
                        artifacts[f"policy:{task.task_id}"] = str(target)
                    # The per-task baseline evaluates and persists one policy
                    # at a time.  Retaining every completed GPU model here
                    # grows device memory linearly with the number of tasks;
                    # only the cross-task matrix needs the in-memory mapping.
                    if args.baseline == "cross_task_policy_matrix":
                        policies[task.task_id] = model
                finally:
                    env.close()
                if args.baseline == "per_task_sac" and task.task_id not in partial_metrics:
                    partial_metrics[task.task_id] = _evaluate_sac(model, task, cfg, query_cases(task))
                    _write_partial(partial_path, partial_protocol, tasks=partial_metrics)
                if args.baseline == "per_task_sac":
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if args.baseline == "per_task_sac":
                artifacts["metrics"] = str(_save_metrics(root / "per_task_metrics.json", {"tasks": partial_metrics}))
            else:
                partial_path = root / "cross_task_partial_matrix.json"
                matrix = _partial_payload(partial_path, partial_protocol, args.resume, "matrix")
                for policy_id, model in policies.items():
                    rows = matrix.setdefault(policy_id, {})
                    for task in evaluation_tasks:
                        if task.task_id not in rows:
                            rows[task.task_id] = _evaluate_sac(model, task, cfg, query_cases(task))
                            _write_partial(
                                partial_path, partial_protocol,
                                policy_tasks=list(policies),
                                evaluation_tasks=[item.task_id for item in evaluation_tasks],
                                matrix=matrix,
                            )
                artifacts["matrix"] = str(_save_metrics(root / "cross_task_matrix.json", {"policy_tasks": list(policies), "evaluation_tasks": [task.task_id for task in evaluation_tasks], "matrix": matrix}))

        elif args.baseline in {"topology_conditioned_pooled_sac", "oracle_task_conditioned_sac"}:
            pooled_steps_per_task = int(args.pooled_steps_per_task) if args.pooled_steps_per_task is not None else None
            pooled_total_steps = int(args.env_steps) if pooled_steps_per_task is None else pooled_steps_per_task * len(train_tasks)
            training_budget = {
                "scope": "pooled_shared",
                "per_task_environment_steps": pooled_steps_per_task if pooled_steps_per_task is not None else pooled_total_steps / len(train_tasks),
                "total_environment_steps": pooled_total_steps,
            }
            pooled = PooledLogicalMergeEnv(train_tasks, cfg, train_books, args.seed)
            geometry_ids = [task.geometry_id for task in all_tasks]
            env = OracleTaskObservation(pooled, geometry_ids) if args.baseline == "oracle_task_conditioned_sac" else pooled
            try:
                target = root / "model.zip"; target.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_root = root / "checkpoints" if args.checkpoint_interval_steps else None
                partial_checkpoint = (
                    _latest_sac_checkpoint(checkpoint_root, args.baseline)
                    if args.resume and checkpoint_root is not None else None
                )
                if args.resume and target.exists():
                    model = SAC.load(target, env=env, device="auto")
                elif partial_checkpoint is not None:
                    model = SAC.load(partial_checkpoint, env=env, device="auto")
                else:
                    model = _new_sac(env, args.seed)
                _learn_sac(
                    model, pooled_total_steps, checkpoint_root, args.baseline,
                    args.checkpoint_interval_steps,
                )
                model.save(target)
                artifacts["model"] = str(target)
            finally:
                env.close()
            transform = None
            if args.baseline == "oracle_task_conditioned_sac":
                positions = {geometry_id: index for index, geometry_id in enumerate(geometry_ids)}
                transform = lambda observation, task: np.concatenate([np.asarray(observation, dtype=np.float32), np.eye(len(geometry_ids), dtype=np.float32)[positions[task.geometry_id]]])
            # The non-privileged pooled baseline is also evaluated on the
            # frozen meta-train queries.  This is the matched comparator for
            # per-task SAC in the heterogeneity audit; held-out metrics remain
            # present in the same artifact.
            report_tasks = list(train_tasks) + evaluation_tasks if args.baseline == "topology_conditioned_pooled_sac" else evaluation_tasks
            partial_path = root / "pooled_partial_metrics.json"
            pooled_protocol = {**partial_protocol, "pooled_total_environment_steps": pooled_total_steps}
            metrics = _partial_payload(partial_path, pooled_protocol, args.resume, "tasks")
            for task in report_tasks:
                if task.task_id not in metrics:
                    metrics[task.task_id] = _evaluate_sac(model, task, cfg, query_cases(task), transform)
                    _write_partial(partial_path, pooled_protocol, tasks=metrics)
            artifacts["metrics"] = str(_save_metrics(root / "pooled_metrics.json", {"tasks": metrics}))

        elif args.baseline == "scratch_sac":
            training_budget = {
                "scope": "heldout_task_independent",
                "per_task_environment_steps": int(args.env_steps),
                "task_count": len(evaluation_tasks),
                "total_environment_steps": int(args.env_steps) * len(evaluation_tasks),
            }
            partial_path = root / "scratch_partial_metrics.json"
            metrics = _partial_payload(partial_path, partial_protocol, args.resume, "tasks")
            for index, task in enumerate(evaluation_tasks):
                env = LogicalMergeEnv(task, cfg, casebooks[task.task_id]["test_support"])
                target = root / "policies" / f"{task.task_id}.zip"
                try:
                    if args.resume and target.exists():
                        model = SAC.load(target, env=env, device="auto")
                    else:
                        model = _new_sac(env, args.seed + index)
                    checkpoint_root = root / "checkpoints" / task.task_id if args.checkpoint_interval_steps else None
                    _learn_sac(
                        model, args.env_steps, checkpoint_root, task.task_id,
                        args.checkpoint_interval_steps,
                    )
                    target.parent.mkdir(parents=True, exist_ok=True); model.save(target)
                    artifacts[f"policy:{task.task_id}"] = str(target)
                finally:
                    env.close()
                if task.task_id not in metrics:
                    metrics[task.task_id] = _evaluate_sac(model, task, cfg, query_cases(task))
                    _write_partial(partial_path, partial_protocol, support_steps=args.env_steps, tasks=metrics)
            artifacts["metrics"] = str(_save_metrics(root / "scratch_metrics.json", {"support_steps": args.env_steps, "tasks": metrics}))

        elif args.baseline == "pooled_finetune_sac":
            if args.pooled_pretrain_model:
                base_path = Path(args.pooled_pretrain_model)
                if not base_path.exists():
                    raise SystemExit(f"missing frozen pooled SAC checkpoint: {base_path}")
                pretrain_steps = int(SAC.load(base_path, device="auto").num_timesteps)
                artifacts["pooled_pretrain"] = str(base_path)
            else:
                pretrain_steps = int(args.pretrain_steps or args.env_steps)
                pooled = PooledLogicalMergeEnv(train_tasks, cfg, train_books, args.seed)
                try:
                    base = _new_sac(pooled, args.seed); base.learn(total_timesteps=pretrain_steps)
                    base_path = root / "pooled_pretrain.zip"; base_path.parent.mkdir(parents=True, exist_ok=True); base.save(base_path); artifacts["pooled_pretrain"] = str(base_path)
                finally:
                    pooled.close()
            training_budget = {
                "scope": "pooled_pretrain_plus_heldout_task_finetune",
                "pooled_pretrain_environment_steps": pretrain_steps,
                "per_task_finetune_environment_steps": int(args.env_steps),
                "finetune_task_count": len(evaluation_tasks),
                "total_finetune_environment_steps": int(args.env_steps) * len(evaluation_tasks),
            }
            partial_path = root / "pooled_finetune_partial_metrics.json"
            finetune_protocol = {**partial_protocol, "pretrain_steps": pretrain_steps, "pooled_pretrain": str(base_path)}
            metrics = _partial_payload(partial_path, finetune_protocol, args.resume, "tasks")
            resume_finetuned = _partial_protocol_matches(
                partial_path, finetune_protocol, args.resume,
            )
            for index, task in enumerate(evaluation_tasks):
                env = LogicalMergeEnv(task, cfg, casebooks[task.task_id]["test_support"])
                target = root / "finetuned" / f"{task.task_id}.zip"
                checkpoint_root = root / "checkpoints" / task.task_id if args.checkpoint_interval_steps else None
                partial_checkpoint = (
                    _latest_sac_checkpoint(checkpoint_root, task.task_id)
                    if args.resume and checkpoint_root is not None else None
                )
                try:
                    if resume_finetuned and target.exists():
                        model = SAC.load(target, env=env, device="auto")
                    elif partial_checkpoint is not None:
                        model = SAC.load(partial_checkpoint, env=env, device="auto")
                    else:
                        model = SAC.load(base_path, env=env, device="auto")
                        model.set_random_seed(args.seed + index)
                    _learn_sac(
                        model, pretrain_steps + int(args.env_steps),
                        checkpoint_root,
                        task.task_id, args.checkpoint_interval_steps,
                    )
                    target.parent.mkdir(parents=True, exist_ok=True); model.save(target)
                    artifacts[f"finetuned:{task.task_id}"] = str(target)
                finally:
                    env.close()
                if task.task_id not in metrics:
                    metrics[task.task_id] = _evaluate_sac(model, task, cfg, query_cases(task))
                    _write_partial(
                        partial_path, finetune_protocol,
                        pretrain_steps=pretrain_steps, support_steps=args.env_steps, tasks=metrics,
                    )
            artifacts["metrics"] = str(_save_metrics(root / "pooled_finetune_metrics.json", {"pretrain_steps": pretrain_steps, "support_steps": args.env_steps, "tasks": metrics}))

        else:
            raise RuntimeError(f"unhandled baseline {args.baseline}")

    write_baseline_manifest(
        args.output, name=args.baseline, taskbook_hash=taskbook_hash, seed=args.seed, env_steps=args.env_steps,
        smoke=args.smoke, artifacts=artifacts, config_hash=content_hash(cfg), casebook_hashes=case_hashes,
        checkpoint_hash=checkpoint_hash, training_budget=training_budget,
        implementation_hash=str(partial_protocol["implementation_hash"]),
    )


if __name__ == "__main__":
    main()
