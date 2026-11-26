"""Continue DQN on paid failures while rehearsing successful general driving skills."""

import argparse
import hashlib
import random
import time
from collections import Counter
from copy import copy
from dataclasses import asdict, replace
from functools import partial

import numpy as np
import torch
from scipy.stats import qmc
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch.nn import functional as F

from controllers.highway.decision_interface import DecisionInterface
from controllers.highway.learning import ACTION_INTERFACE, DiscreteDriver
from controllers.highway.train import ModelController
from simulator.highway.collision_env import CollisionWorld, execute
from simulator.highway.repair_env import RepairEnv

from experiments.formal_scenarios import FREE_BOUNDS, coordinate_stream, development_coordinates
from experiments.history_transfer_experiment import freeze
from experiments.parallel import COLLECTION_WORKERS, initialize_worker
from experiments.protocol import ROOT
from experiments.qualification import COURSES, DEVELOPMENT_COURSES, ENVIRONMENT_SEED, course_result, qualify, verify_course_evidence
from experiments.regression.execution import checkpoint_identity
from experiments.regression.preflight import DQN_CHECKPOINTS
from experiments.regression.protocol import DQN_WARMUP, TRAINING_STEPS, numeric_coordinate, repair_subset, sample_block
from experiments.schema import Scenario
from experiments.scenarios import DOMAINS
from experiments.storage import read_json

STUDY = ROOT / "results/dqn_continual_learning"
ANCESTOR_HISTORY = STUDY / "ancestor_history"
STRATIFIED = STUDY / "stratified_rehearsal"
CURRICULUM = STUDY / "driving_curriculum"
PHYSICAL_ACTIONS = STUDY / "physical_action_rehearsal"
STEPS = (8192, 16384, 32768, 65536, 98304, TRAINING_STEPS)
SPLITS = {"training": (128, 75101), "selection": (64, 75103), "blind_base": (128, 75107),
          **{f"blind_R{i}": (64, 75109 + 2 * i) for i in range(4)}}


def driving_outcomes(model, scenes):
    collisions, violations = [], []
    for scene in scenes:
        policy = DecisionInterface(ModelController(model))
        feedback = execute(policy, scene, environment_seed=81001).feedback
        if feedback.collision:
            collisions.append(scene.scenario_id)
        if feedback.safety.violations:
            violations.append(scene.scenario_id)
    return {"collisions": collisions, "violations": violations}


def used_coordinates(kind):
    excluded = set(development_coordinates(kind))
    for split in ("history", "target"):
        excluded.update(tuple(r["independent"][n] for n in FREE_BOUNDS[kind]) for r in coordinate_stream(kind, split))
    reference = read_json(ANCESTOR_HISTORY / kind / "protocol.json")
    for block in [reference["history"], *reference["candidates"]]:
        excluded.update(numeric_coordinate(Scenario(**r)) for r in block)
    # Exclude the deterministic first-round diagnostic blocks (seeds 73121 and 73123).
    sample_block(kind, 64, 73121, excluded, "repair_selection")
    sample_block(kind, 128, 73123, excluded, "repair_holdout")
    return excluded


def register_cases():
    path = STUDY / "coordinates.json"
    registered = {}
    for kind in FREE_BOUNDS:
        excluded = used_coordinates(kind)
        registered[kind] = {key: [asdict(s) for s in sample_block(kind, size, seed, excluded, key)]
                            for key, (size, seed) in SPLITS.items()}
    return freeze(path, registered)


def register_stratified():
    first = register_cases()
    registered = {}
    for kind in FREE_BOUNDS:
        excluded = used_coordinates(kind)
        for block in first[kind].values():
            excluded.update(numeric_coordinate(Scenario(**r)) for r in block)
        registered[kind] = [asdict(s) for s in sample_block(kind, 64, 75131, excluded, "stratified_confirmation")]
    return freeze(STRATIFIED / "coordinates.json", registered)


def cases(registered, split, kinds):
    return [Scenario(**s) for k in kinds for s in registered[k][split]]


def ordinary_courses(split):
    """Generate the five fixed driving tasks with independently varying physical parameters."""
    exponent, seed = {"training": (5, 75201), "development": (3, 75211), "confirmation": (3, 75221)}[split]
    courses = []
    for index, course in enumerate(DEVELOPMENT_COURSES):
        domain = DOMAINS[course.scenario.kind]
        names = list(domain.bounds)
        bounds = dict(domain.bounds)
        if course.lane_change or course.initial_rear_guard:
            for key in ("rear_gap_m", "rear_speed_mps", "adjacent_front_gap_m"):
                low, high = bounds[key]
                midpoint = (low + high) / 2
                upper = course.lane_change == (key != "rear_speed_mps")
                bounds[key] = (midpoint, high) if upper else (low, midpoint)
        sampler = qmc.Sobol(len(names), scramble=True, seed=seed + index * 100)
        for i, point in enumerate(sampler.random_base2(exponent)):
            params = {n: round(bounds[n][0] + u * (bounds[n][1] - bounds[n][0]), 6)
                      for n, u in zip(names, point)}
            if course.name == "normal_following":
                params["event_time_s"] = course.duration_s
            scene = replace(course.scenario, scenario_id=f"ordinary:{split}:{course.name}:{i}", parameters=params)
            domain.validate(scene)
            courses.append(replace(course, scenario=scene))
    return courses


def register_curriculum():
    groups = {key: ordinary_courses(key) for key in ("training", "development", "confirmation")}
    def coordinate(course):
        return course.scenario.kind, tuple(sorted(course.scenario.parameters.items()))
    seen = {coordinate(c) for c in (*COURSES, *DEVELOPMENT_COURSES, *legacy_rehearsal_courses())}
    for block in groups.values():
        for course in block:
            key = coordinate(course)
            if key in seen:
                raise ValueError("A driving course shares previous physical coordinates")
            seen.add(key)
    kind = "cut_in_onset"
    excluded = used_coordinates(kind)
    for block in read_json(STUDY / "coordinates.json")[kind].values():
        excluded.update(numeric_coordinate(Scenario(**r)) for r in block)
    excluded.update(numeric_coordinate(Scenario(**r)) for r in read_json(STRATIFIED / "coordinates.json")[kind])
    return freeze(CURRICULUM / "coordinates.json", {
        "courses": {key: [asdict(c) for c in block] for key, block in groups.items()},
        "cut_in_onset": [asdict(s) for s in sample_block(kind, 64, 75231, excluded, "curriculum_confirmation")]})


def driving_skills(model, courses):
    results = qualify(lambda: ModelController(model), courses=courses)["courses"]
    return [{**row, "scenario_id": c.scenario.scenario_id} for row, c in zip(results, courses, strict=True)]


def retains_skills(parent, candidate):
    if [(r["course"], r["scenario_id"]) for r in parent] != [(r["course"], r["scenario_id"]) for r in candidate]:
        raise ValueError("Skill assessment requires paired physical courses")
    old = Counter(r["course"] for r in parent if r["passed"])
    new = Counter(r["course"] for r in candidate if r["passed"])
    return all(new[k] >= old[k] for k in old)


def rehearsal_indices(count, action_groups, device):
    if not action_groups:
        return torch.randint(count, (64,), device=device)
    uniform = torch.randint(count, (32,), device=device)
    quota, remainder = divmod(32, len(action_groups))
    groups = [indices[torch.randint(len(indices), (quota + (i < remainder),), device=device)]
              for i, indices in enumerate(action_groups)]
    return torch.cat([uniform, *groups])


def action_labels(driver, world, selected):
    """Use deployed controls and both persistent setpoints to identify equivalent actions."""
    outcomes = []
    for action in range(len(ACTION_INTERFACE["actions"])):
        candidate = copy(driver)
        command = candidate.command(action, world)
        outcomes.append((command, candidate.target_lane, candidate.target_speed))
    result = outcomes[selected]
    intent = 1 if result[1:] == (driver.target_lane, driver.target_speed) else selected
    return [value == result for value in outcomes], intent


def demonstration_loss(q_values, accepted):
    best = q_values.masked_fill(~accepted, -torch.inf).max(dim=1).values
    margins = torch.full_like(q_values, .8).masked_fill(accepted, 0.)
    return ((q_values + margins).max(dim=1).values - best).mean()


class RehearsalDQN(DQN):
    """One-step SB3 DQN with a large-margin loss on safe parent demonstrations."""

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), "examples", "accepted_actions", "action_examples", "exploration_schedule"]

    def train(self, gradient_steps, batch_size=256):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with torch.no_grad():
                future = self.q_net_target(batch.next_observations).max(dim=1, keepdim=True).values
                target = batch.rewards + (1 - batch.dones) * self.gamma * future
            prediction = self.q_net(batch.observations).gather(1, batch.actions.long())
            sample = rehearsal_indices(len(self.examples), self.action_examples, self.device)
            q = self.q_net(self.examples[sample])
            imitation = demonstration_loss(q, self.accepted_actions[sample])
            loss = F.smooth_l1_loss(prediction, target) + imitation
            self.policy.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps


def legacy_rehearsal_courses():
    for course in DEVELOPMENT_COURSES:
        for offset in (-5, -3, 3, 5):
            params = dict(course.scenario.parameters)
            if course.scenario.kind == "front_brake":
                params["gap_m"] += offset
                params["deceleration_mps2"] += .125 * offset
            else:
                params["lead_gap_m"] += offset
                params["adjacent_front_gap_m"] -= offset
            scene = replace(course.scenario, scenario_id=f"rehearsal:{course.name}:{offset}", parameters=params)
            yield replace(course, scenario=scene)


def rehearsal_examples(model, courses=None, *, physical_actions=False):
    observations, actions, records, masks, intents = [], [], [], [], []
    for course in legacy_rehearsal_courses() if courses is None else courses:
        scene = course.scenario
        world = CollisionWorld(scene, lane_centers=course.lane_centers,
                               duration_s=course.duration_s, environment_seed=75119)
        driver = DiscreteDriver()
        obs, act, accepted, effective = [], [], [], []
        try:
            while not world.terminal:
                visible = world.visible_world()
                obs.append(driver.observation(visible))
                act.append(int(model.predict(obs[-1], deterministic=True)[0]))
                if physical_actions:
                    mask, intent = action_labels(driver, visible, act[-1])
                    accepted.append(mask)
                    effective.append(intent)
                world.advance(driver.command(act[-1], visible))
            result = course_result(world, course)
        finally:
            world.close()
        if result["passed"]:
            observations.extend(obs)
            actions.extend(act)
            masks.extend(accepted)
            intents.extend(effective)
        records.append({"scene": asdict(scene), **result})
    if not actions:
        raise ValueError("No successful rehearsal trajectories for this parent")
    actions = torch.tensor(actions)
    accepted = torch.tensor(masks) if physical_actions else F.one_hot(actions, len(ACTION_INTERFACE["actions"])).bool()
    intents = torch.tensor(intents) if physical_actions else actions
    return torch.as_tensor(np.stack(observations)), actions, records, accepted, intents


def closed_loop_score(model, scenes):
    results = driving_outcomes(model, scenes)
    return {**results, "n": len(scenes)}


def check_lane_change():
    """Intervene only on the left action in the existing development lane-change course."""
    initialize_worker()
    path = STUDY / "base/selected/policy.zip"
    model = DQN.load(path, device="cpu")
    _, actions, _, _, _ = rehearsal_examples(model)
    course = next(c for c in DEVELOPMENT_COURSES if c.name == "legal_lane_change")
    outcomes = {}
    for block_left in (False, True):
        world = CollisionWorld(course.scenario, lane_centers=course.lane_centers,
                               duration_s=course.duration_s, environment_seed=ENVIRONMENT_SEED)
        driver = DiscreteDriver()
        requested, applied = [], []
        try:
            while not world.terminal:
                visible = world.visible_world()
                action = int(model.predict(driver.observation(visible), deterministic=True)[0])
                requested.append(action)
                applied.append(1 if block_left and action == 0 else action)
                world.advance(driver.command(applied[-1], visible))
            outcomes["suppress_left" if block_left else "policy"] = {
                "requested": requested, "applied": applied, **course_result(world, course)}
        finally:
            world.close()
    return freeze(STUDY / "base/development_lane_change.json", {
        "checkpoint_sha256": checkpoint_identity(path), "environment_seed": ENVIRONMENT_SEED,
        "course": asdict(course), "rehearsal_actions": dict(Counter(actions.tolist())), **outcomes})


def save_training_point(model, env, directory):
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "policy.zip")
    model.save_replay_buffer(directory / "replay.pkl")
    workers = env.env_method("curriculum_state")
    torch.save({"numpy": np.random.get_state(), "torch": torch.get_rng_state(), "python": random.getstate(),
                "actions": model.action_space.np_random.bit_generator.state,
                "environments": [s["rng"] for s in workers]}, directory / "random.pt")
    return {"sha256": checkpoint_identity(directory / "policy.zip"), "workers": workers,
            "optimizer_updates": model._n_updates, "environment_steps": model.num_timesteps,
            "replay_position": model.replay_buffer.pos, "replay_full": model.replay_buffer.full}


def certify_continuation(directory):
    report = read_json(directory.parent / "selection.json")
    expected = report["state"]
    if not report["step"] or checkpoint_identity(directory / "policy.zip") != expected["sha256"]:
        raise ValueError("A retained optimizer and buffer must belong to the selected checkpoint")
    model = DQN.load(directory / "policy.zip", device="cpu")
    model.load_replay_buffer(directory / "replay.pkl")
    if (model.num_timesteps != expected["environment_steps"] or model._n_updates != expected["optimizer_updates"]
            or (model.replay_buffer.pos, model.replay_buffer.full) != (expected["replay_position"], expected["replay_full"])):
        raise ValueError("Selected network, optimizer or replay state differs from the training point")
    fingerprints = {}
    for name in ("replay.pkl", "random.pt"):
        digest = hashlib.sha256()
        with (directory / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        fingerprints[name] = digest.hexdigest()
    return freeze(directory / "continuation.json", {"checkpoint_sha256": expected["sha256"], **fingerprints})


def train_update(parent, directory, current, historical, selection, *, inherited=None, stratify_actions=False,
                 broad_curriculum=False, physical_actions=False):
    """Select an actual training point; unopened confirmation labels never enter this function."""
    parent_sha = checkpoint_identity(parent)
    specification = {"parent_sha256": parent_sha, "current": [asdict(s) for s in current],
                     "historical": [asdict(s) for s in historical], "selection": [asdict(s) for s in selection],
                     "training_steps": list(STEPS), "seed": 6119}
    if stratify_actions:
        specification["rehearsal_sampling"] = "32 uniform plus 32 stratified by observed action"
    if broad_curriculum:
        specification["driving_courses"] = {k: [asdict(c) for c in ordinary_courses(k)]
                                            for k in ("training", "development")}
    if physical_actions:
        specification["supervision"] = "Equal control, next target lane and next target speed are equivalent"
        specification["rehearsal_sampling"] = "32 uniform plus 32 stratified by actual target change"
    freeze(directory / "protocol.json", specification)
    receipt = directory / "selection.json"
    if receipt.exists():
        report = read_json(receipt)
        path = directory / "selected/policy.zip" if report["step"] else parent
        if checkpoint_identity(path) != report["published_sha256"]:
            raise ValueError("Selected checkpoint differs from its frozen training receipt")
        return report
    started = time.perf_counter()
    initialize_worker()
    if inherited is not None:
        certify_continuation(inherited)
    teacher = DQN.load(parent, device="cpu")
    examples, actions, demos, masks, intents = rehearsal_examples(
        teacher, ordinary_courses("training") if broad_curriculum else None, physical_actions=physical_actions)
    dev_courses = ordinary_courses("development") if broad_curriculum else []
    dev_parent = driving_skills(teacher, dev_courses) if broad_curriculum else []
    original = closed_loop_score(teacher, selection)
    best = (len(original["collisions"]), len(original["violations"]))
    rank, step, state = best, 0, None
    results = []
    env = SubprocVecEnv([partial(RepairEnv, current, historical, set(), preparation=not historical)
                          for _ in range(8)], start_method="spawn")
    try:
        model = RehearsalDQN.load(parent, env=env, device="cpu", learning_rate=.00005, gamma=.99,
                                  buffer_size=200000, batch_size=256, train_freq=4, gradient_steps=1,
                                  target_update_interval=10000)
        model.examples, model.accepted_actions = examples, masks
        model.action_examples = ([torch.where(intents == a)[0] for a in intents.unique()]
                                 if stratify_actions else [])
        start = model.num_timesteps
        if inherited is None:
            env.seed(6119)
            random.seed(6119)
            torch.manual_seed(6119)
            np.random.seed(6119)
            model.action_space.seed(6119)
        else:
            model.load_replay_buffer(inherited / "replay.pkl")
            rng = torch.load(inherited / "random.pt", map_location="cpu", weights_only=False)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            model.action_space.np_random.bit_generator.state = rng["actions"]
            env._reset_seeds()
            for i, generator in enumerate(rng["environments"]):
                env.env_method("restore_random_state", generator, indices=i)
        warmup = DQN_WARMUP if inherited is None else 0
        model.learning_starts = start + warmup
        model.exploration_rate = .2 if warmup else .05
        model.exploration_schedule = lambda _: .05 + (.15 * max(0., 1 - max(
            0, model.num_timesteps - start - warmup) / 26214) if warmup else 0.)
        for limit in STEPS:
            model.learn(total_timesteps=start + limit - model.num_timesteps, reset_num_timesteps=False)
            capability = qualify(lambda: ModelController(model), courses=DEVELOPMENT_COURSES)
            row = {"step": limit, "capability": capability}
            if broad_curriculum:
                row["driving_skills"] = driving_skills(model, dev_courses)
            if capability["qualified"] and (not broad_curriculum or retains_skills(dev_parent, row["driving_skills"])):
                observed = closed_loop_score(model, selection)
                row["outcomes"] = observed
                score = len(observed["collisions"]), len(observed["violations"])
                if score[0] < best[0] and score[1] <= best[1] and score < rank:
                    state = save_training_point(model, env, directory / "selected")
                    rank, step = score, limit
            results.append(row)
            print(directory.name, limit, "courses", sum(r["passed"] for r in capability["courses"]),
                  "selected step", step, flush=True)
        report = {"parent_sha256": parent_sha, "parent_outcomes": original, "rehearsal": demos,
                  "rehearsal_examples": len(actions), "candidates": results, "step": step,
                  "state": state, "actual_online_steps": model.num_timesteps - start,
                  "gradient_updates": model._n_updates - teacher._n_updates,
                  "elapsed_s": time.perf_counter() - started,
                  "published_sha256": state["sha256"] if state else parent_sha}
        if broad_curriculum:
            report["parent_driving_skills"] = dev_parent
        if physical_actions:
            report["physical_action_labels"] = {
                "original": actions.bincount(minlength=5).tolist(), "effective": intents.bincount(minlength=5).tolist(),
                "multiple_acceptable": int((masks.sum(1) > 1).sum())}
        return freeze(receipt, report)
    finally:
        env.close()


def confirm(parent_path, selected_path, scenes, directory):
    path = directory / "confirmation.json"
    identities = {"parent_sha256": checkpoint_identity(parent_path),
                  "selected_sha256": checkpoint_identity(selected_path)}
    freeze(directory / "confirmation_inputs.json", {**identities, "scenes": [asdict(s) for s in scenes]})
    if path.exists():
        saved = read_json(path)
        if any(saved[key] != value for key, value in identities.items()):
            raise ValueError("Confirmation belongs to different driving policies")
        for kind in {s.kind for s in scenes}:
            ids = {s.scenario_id for s in scenes if s.kind == kind}
            for arm in ("before", "after"):
                row = saved[arm][kind]
                if row["n"] != len(ids) or (set(row["collisions"]) | set(row["violations"])) - ids:
                    raise ValueError("Stored outcomes differ from the registered confirmation cases")
        if saved["capability"]["qualified"]:
            evidence = verify_course_evidence(directory / "qualification")
            if evidence["courses"] != saved["capability"]["courses"]:
                raise ValueError("Qualification results differ from the confirmation")
        return saved
    parent = DQN.load(parent_path, device="cpu")
    current = DQN.load(selected_path, device="cpu")
    capability = qualify(lambda: ModelController(current), save_root=directory / "qualification")
    after, before = {}, {}
    for kind in dict.fromkeys(s.kind for s in scenes):
        block = [s for s in scenes if s.kind == kind]
        before[kind] = closed_loop_score(parent, block)
        after[kind] = closed_loop_score(current, block)
    count = lambda rows, field: sum(len(r[field]) for r in rows.values())
    return freeze(path, {
        **identities,
        "before": before, "after": after, "capability": capability,
        "passed": (capability["qualified"] and count(after, "collisions") < count(before, "collisions")
                   and count(after, "violations") <= count(before, "violations"))})


def train_base():
    initialize_worker()
    registered = register_cases()
    parent = DQN_CHECKPOINTS["parent"]
    directory = STUDY / "base"
    model = DQN.load(parent, device="cpu")
    training = cases(registered, "training", FREE_BOUNDS)
    feedback = directory / "training_feedback.json"
    if not feedback.exists():
        freeze(feedback, {"parent_sha256": checkpoint_identity(parent), **closed_loop_score(model, training)})
    recorded = read_json(feedback)
    if recorded["parent_sha256"] != checkpoint_identity(parent):
        raise ValueError("Base training failures belong to a different parent")
    hits = set(recorded["collisions"])
    result = train_update(parent, directory, [s for s in training if s.scenario_id in hits], [],
                          cases(registered, "selection", FREE_BOUNDS))
    if result["step"]:
        final = confirm(parent, directory / "selected/policy.zip", cases(registered, "blind_base", FREE_BOUNDS), directory)
        print("base independent confirmation", final["passed"], flush=True)
    else:
        print("base has no qualified development improvement", flush=True)


def train_reference(kind, workers, *, stratified=False, broad_curriculum=False, physical_actions=False):
    from experiments.regression.fixed_inputs import paid_reference_records
    from experiments.regression.audit import verify_execution
    from experiments.regression.queries import query_reference_round
    from experiments.regression.run_fixed import collect_all

    initialize_worker()
    registered = register_cases()
    base = STUDY / "base"
    parent_state = base / "selected"
    parent = parent_state / "policy.zip"
    result = read_json(base / "confirmation.json")
    if not result["passed"] or checkpoint_identity(parent) != result["selected_sha256"]:
        raise ValueError("Reference repair requires a confirmed new base policy")
    original_root = STUDY / kind / "reference"
    root = original_root
    if physical_actions:
        prior = CURRICULUM / "cut_in_onset/reference"
        if read_json(prior / "chain.json")["status"] != "no_development_gain" or (prior / "R0/confirmation.json").exists():
            raise ValueError("Physical-action repair requires an unopened confirmation cohort")
        broad_curriculum = True
    if broad_curriculum:
        if read_json(STRATIFIED / "cut_in_onset/reference/R0/confirmation.json")["passed"]:
            raise ValueError("The previous independent repair already qualified")
        register_curriculum()
        study_root = PHYSICAL_ACTIONS if physical_actions else CURRICULUM
        if kind != "cut_in_onset" and not read_json(study_root / "cut_in_onset/reference/R0/acceptance.json")["passed"]:
            raise ValueError("The broader driving curriculum has no confirmed primary repair")
        root = study_root / kind / "reference"
    elif stratified:
        for scene in FREE_BOUNDS:
            prior_chain = read_json(STUDY / scene / "reference/chain.json")
            if any(r["independent_gain"] for r in prior_chain["rounds"]):
                raise ValueError("The original update protocol already passed independent repair confirmation")
        if kind != "cut_in_onset" and not read_json(STRATIFIED / "cut_in_onset/reference/R0/confirmation.json")["passed"]:
            raise ValueError("The primary stratified repair has not passed independent confirmation")
        root = STRATIFIED / kind / "reference"
    reference = read_json(ANCESTOR_HISTORY / kind / "protocol.json")
    protocol = freeze(root / "protocol.json", reference)
    jobs = [(parent, Scenario(**row), original_root / "history/parent" / f"{i:04d}.json.gz", "history", "dqn")
            for i, row in enumerate(protocol["history"])]
    if not (stratified or broad_curriculum):
        collect_all(jobs, workers, f"{kind} new base history")
    old = []
    for _, scene, path, _, _ in jobs:
        row = read_json(path)
        verify_execution(row, asdict(scene), result["selected_sha256"])
        if row["feedback"]["collision"]:
            old.append(scene)
    current, historical = repair_subset(old), []
    rounds = []
    for iteration in range(4):
        version = f"R{iteration}"
        if not current:
            return freeze(root / "chain.json", {"status": "no_paid_failure", "rounds": rounds})
        directory = root / version
        row = train_update(parent, directory, current, historical, cases(registered, "selection", [kind]),
                           inherited=parent_state, stratify_actions=stratified or broad_curriculum,
                           broad_curriculum=broad_curriculum, physical_actions=physical_actions)
        state = directory / "selected" if row["step"] else parent_state
        selected = state / "policy.zip"
        if broad_curriculum and iteration == 0 and kind == "cut_in_onset" and not row["step"]:
            return freeze(root / "chain.json", {"status": "no_development_gain", "rounds": [{
                "version": version, "selected_step": 0, "checkpoint": selected.relative_to(ROOT).as_posix(),
                "sha256": checkpoint_identity(selected), "independent_gain": None}]})
        if (stratified or broad_curriculum) and iteration > 0:
            path = directory / "release.json"
            if path.exists():
                sealed = read_json(path)
                if sealed["selected_sha256"] != checkpoint_identity(selected):
                    raise ValueError("Released weights differ from their driving evidence")
            else:
                model = DQN.load(selected, device="cpu")
                capability = qualify(lambda: ModelController(model), save_root=directory / "qualification")
                sealed = freeze(path, {"selected_sha256": checkpoint_identity(selected),
                                       "capability": capability, "passed": None})
        else:
            if stratified or broad_curriculum:
                cohort = CURRICULUM if broad_curriculum and kind == "cut_in_onset" else STRATIFIED
                scenes = [Scenario(**r) for r in read_json(cohort / "coordinates.json")[kind]]
            else:
                scenes = cases(registered, f"blind_{version}", [kind])
            sealed = confirm(parent, selected, scenes, directory)
            if broad_curriculum:
                courses = ordinary_courses("confirmation")
                baseline = driving_skills(DQN.load(parent, device="cpu"), courses)
                revised = driving_skills(DQN.load(selected, device="cpu"), courses)
                acceptance = freeze(directory / "acceptance.json", {"parent_sha256": checkpoint_identity(parent),
                    "selected_sha256": sealed["selected_sha256"], "before": baseline, "after": revised,
                    "passed": sealed["passed"] and retains_skills(baseline, revised)})
                sealed = {**sealed, "passed": acceptance["passed"]}
        rounds.append({"version": version, "selected_step": row["step"],
                       "checkpoint": selected.relative_to(ROOT).as_posix(),
                       "sha256": sealed["selected_sha256"], "independent_gain": sealed["passed"]})
        print(kind, version, "independent gain", sealed["passed"], flush=True)
        if not sealed["capability"]["qualified"]:
            return freeze(root / "chain.json", {"status": "unqualified", "rounds": rounds})
        if (stratified or broad_curriculum) and kind == "cut_in_onset" and iteration == 0 and not sealed["passed"]:
            return freeze(root / "chain.json", {"status": "no_independent_gain", "rounds": rounds})
        if iteration == 3:
            break
        parent, parent_state = selected, state
        pool = [Scenario(**s) for s in protocol["candidates"][iteration]]
        query_root = root
        previous = original_root / "random/queries" / f"result_{iteration + 1}.json"
        if (stratified or broad_curriculum) and previous.exists() and read_json(previous)["specification"]["checkpoint_sha256"] == sealed["selected_sha256"]:
            query_root = original_root
        else:
            query_reference_round(pool, protocol["initial_indices"], selected, root / "random/queries",
                                  iteration=iteration + 1, execution_root=root, policy="dqn")
        rounds[-1]["query_root"] = query_root.relative_to(ROOT).as_posix()
        records = paid_reference_records(query_root, protocol, iteration + 1, sealed["selected_sha256"])
        historical = repair_subset(old)
        current = [Scenario(**r["scenario"]) for r in records if r["feedback"]["collision"]]
        old.extend(current)
    return freeze(root / "chain.json", {"status": "complete", "rounds": rounds})


def stratified_development():
    kind = "cut_in_onset"
    original = read_json(STUDY / kind / "reference/R0/protocol.json")
    inherited = STUDY / "base/selected"
    if original["parent_sha256"] != checkpoint_identity(inherited / "policy.zip"):
        raise ValueError("Development comparison requires the same parent and paid repair feedback")
    return train_update(inherited / "policy.zip", STRATIFIED / kind / "reference/R0",
                        [Scenario(**s) for s in original["current"]], [],
                        [Scenario(**s) for s in original["selection"]], inherited=inherited, stratify_actions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--scene", choices=FREE_BOUNDS, help="Continue a scene from the confirmed general base")
    task.add_argument("--stratified-development", action="store_true", help="Diagnose rare-action retention on development cases")
    task.add_argument("--stratified-scene", choices=FREE_BOUNDS, help="Run the registered rare-action repair confirmation")
    task.add_argument("--curriculum-scene", choices=FREE_BOUNDS, help="Assess the preregistered broader driving curriculum")
    task.add_argument("--physical-actions", choices=FREE_BOUNDS, help="Train with deployed action-equivalence labels")
    parser.add_argument("--workers", type=int, default=COLLECTION_WORKERS, help="Native history collection processes")
    options = parser.parse_args()
    if options.physical_actions:
        train_reference(options.physical_actions, options.workers, physical_actions=True)
    elif options.curriculum_scene:
        train_reference(options.curriculum_scene, options.workers, broad_curriculum=True)
    elif options.stratified_development:
        stratified_development()
    elif options.stratified_scene:
        train_reference(options.stratified_scene, options.workers, stratified=True)
    elif options.scene:
        train_reference(options.scene, options.workers)
    else:
        train_base()
