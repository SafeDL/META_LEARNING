"""Continual Double DQN repair using the admitted public attention policy."""

import time
from copy import deepcopy
from dataclasses import asdict

import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F

from controllers.highway.adapted_agents import AttentionPolicy, attention_network, attention_observation
from controllers.highway.decision_interface import DecisionInterface, public_world
from controllers.highway.learning import DiscreteDriver
from experiments.history_transfer_experiment import freeze
from experiments.protocol import episode_duration
from experiments.public_ads.train import TrainingWorld
from experiments.qualification import DEVELOPMENT_COURSES, course_result, qualify
from experiments.regression.dqn_learning import (
    STEPS, action_labels, demonstration_loss, ordinary_courses, rehearsal_indices, retains_skills,
)
from experiments.regression.execution import checkpoint_identity
from experiments.regression.protocol import DQN_DECAY, DQN_SEED, DQN_WARMUP
from experiments.storage import read_json
from simulator.highway.collision_env import CollisionWorld, execute
from simulator.highway.repair_env import RepairEnv

ENVIRONMENTS, CAPACITY, BATCH = 8, 15000, 256
TRAIN_EVERY, TARGET_EVERY = 32, 10000
LR, GAMMA = 0.00005, 0.99


def load_attention(checkpoint):
    network = attention_network()
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    network.load_state_dict(state["model"], strict=True)
    if any(not torch.isfinite(p).all() for p in network.parameters()):
        raise ValueError("Nonfinite attention checkpoint")
    return network.eval()


def controller(network):
    return DecisionInterface(AttentionPolicy(network))


class AttentionRepairEnv(RepairEnv):
    """The same repair reward and native worlds, with attention observations."""

    courses = None

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if self.courses is None:
            self.courses = ordinary_courses("training")
        if self.world is not None:
            self.world.close()
        if self.np_random.random() < .5:
            category = "ordinary"
            course = self.courses[int(self.np_random.integers(len(self.courses)))]
            scene, lanes, duration = course.scenario, course.lane_centers, course.duration_s
            environment_seed = int(self.np_random.integers(1_000_000))
        else:
            category = "historical" if self.historical and self.np_random.random() < .5 else "current"
            points = self.historical if category == "historical" else self.current
            scene = points[int(self.np_random.integers(len(points)))]
            lanes, duration, environment_seed = None, episode_duration(scene.kind), 81001
            self.reset_points[scene.scenario_id] += 1
        if (scene.kind, tuple(sorted(scene.parameters.items()))) in self.excluded:
            raise ValueError("Reserved coordinates entered attention repair")
        self.active_category = category
        self.episodes[category] += 1
        self.world = TrainingWorld(scene, environment_seed=environment_seed, duration_s=duration, lane_centers=lanes)
        self.driver = DiscreteDriver()
        return self.entities(), {}

    def entities(self):
        return attention_observation(public_world(self.world.visible_world()))


def demonstrations(teacher):
    obs, masks, labels, evidence = [], [], [], []
    for course in ordinary_courses("training"):
        world = CollisionWorld(course.scenario, lane_centers=course.lane_centers,
                               duration_s=course.duration_s, environment_seed=75119)
        driver = DiscreteDriver()
        episode = []
        try:
            with torch.inference_mode():
                while not world.terminal:
                    visible = public_world(world.visible_world())
                    entities = attention_observation(visible)
                    selected = int(teacher(torch.from_numpy(entities[None])).argmax())
                    acceptable, intent = action_labels(driver, visible, selected)
                    episode.append((entities, acceptable, intent))
                    world.advance(driver.command(selected, visible))
            result = course_result(world, course)
            if result["passed"]:
                for entities, acceptable, intent in episode:
                    obs.append(entities)
                    masks.append(acceptable)
                    labels.append(intent)
            evidence.append({"scenario_id": course.scenario.scenario_id, **result})
        finally:
            world.close()
    if not obs:
        raise ValueError("There are no successful parent attention demonstrations")
    return np.stack(obs), np.asarray(masks), np.asarray(labels), evidence


def development_skills(model, courses):
    rows = qualify(lambda: controller(model), courses=courses)["courses"]
    return [{**row, "scenario_id": c.scenario.scenario_id} for row, c in zip(rows, courses, strict=True)]


def outcomes(network, selection):
    rows = []
    for scene in selection:
        result = execute(controller(network), scene, environment_seed=81001).feedback
        rows.append(dict(scenario_id=scene.scenario_id, collision=result.collision,
                         violations=result.safety.violations))
    return rows


def rank(rows):
    return sum(r["collision"] for r in rows), sum(bool(r["violations"]) for r in rows)


class AttentionUpdate:
    def __init__(self, parent, device):
        self.device = device
        self.network = load_attention(parent).to(device)
        self.target = deepcopy(self.network).eval()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.replay = [
            np.zeros((CAPACITY, 15, 7), dtype=np.float32),
            np.zeros((CAPACITY, 15, 7), dtype=np.float32),
            np.zeros(CAPACITY, dtype=np.int64),
            np.zeros(CAPACITY, dtype=np.float32),
            np.zeros(CAPACITY, dtype=np.float32),
        ]
        self.seen, self.gradient_updates = 0, 0
        self.rng = np.random.default_rng(DQN_SEED)
        torch.manual_seed(DQN_SEED)
        saved = torch.load(parent, weights_only=True, map_location="cpu")
        if "optimizer" in saved:
            self.target.load_state_dict(saved["target"])
            self.optimizer.load_state_dict(saved["optimizer"])
            for array, value in zip(self.replay, saved["replay"], strict=True):
                array[:] = value.numpy()
            self.seen = saved["seen"]
            self.rng.bit_generator.state = saved["rng"]
            torch.set_rng_state(saved["torch_rng"])
            if str(device) == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng"])
        self.environment_rng = saved.get("environment_rng")

    def checkpoint(self, path, environments):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(dict(model=self.network.state_dict(), target=self.target.state_dict(),
                        optimizer=self.optimizer.state_dict(), replay=[torch.from_numpy(a) for a in self.replay],
                        seen=self.seen, rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if str(self.device) == "cuda" else None,
                        environment_rng=[env.curriculum_state()["rng"] for env in environments]), temporary)
        temporary.replace(path)

    def action(self, states, epsilon):
        with torch.no_grad():
            actions = self.network(torch.as_tensor(states, device=self.device)).argmax(1).cpu().numpy()
        explore = self.rng.random(len(states)) < epsilon
        actions[explore] = self.rng.integers(5, size=int(explore.sum()))
        return actions

    def record(self, states, successors, actions, rewards, done):
        size = len(states)
        slots = (self.seen + np.arange(size)) % CAPACITY
        for data, values in zip(self.replay, (states, successors, actions, rewards, done), strict=True):
            data[slots] = values
        self.seen += size

    def update(self, demonstrations):
        indices = self.rng.choice(min(self.seen, CAPACITY), BATCH, replace=False)
        state, successor, action, reward, stop = [
            torch.as_tensor(a[indices], device=self.device) for a in self.replay
        ]
        with torch.no_grad():
            best_action = self.network(successor).argmax(1, keepdim=True)
            expected = reward + GAMMA * (1 - stop) * self.target(successor).gather(1, best_action).squeeze(1)
        q_values = self.network(state).gather(1, action[:, None]).squeeze(1)
        examples, accepted, groups = demonstrations
        subset = rehearsal_indices(len(examples), groups, self.device)
        loss = F.smooth_l1_loss(q_values, expected) + demonstration_loss(self.network(examples[subset]), accepted[subset])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite attention repair loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.gradient_updates += 1


def train_round(parent, current, historical, selection, excluded, directory, *, device):
    specification = dict(parent_sha256=checkpoint_identity(parent),
                         current=[asdict(s) for s in current], historical=[asdict(s) for s in historical],
                         selection=[asdict(s) for s in selection], steps=STEPS, device=device,
                         environments=ENVIRONMENTS, replay_capacity=CAPACITY, batch=BATCH,
                         learning_rate=LR, gamma=GAMMA, update_every=TRAIN_EVERY, target_every=TARGET_EVERY,
                         margin=0.8, margin_batch={"uniform": 32, "stratified_by_effective_action": 32})
    freeze(directory / "training_protocol.json", specification)
    completed = directory / "training.json"
    if completed.exists():
        saved = read_json(completed)
        if saved["checkpoint_sha256"] != checkpoint_identity(directory / "policy.pt" if saved["step"] else parent):
            raise ValueError("Attention checkpoint changed after repair")
        return saved
    started = time.perf_counter()
    teacher = load_attention(parent)
    training_obs, masks, intents, demo_evidence = demonstrations(teacher)
    examples = torch.as_tensor(training_obs, device=device)
    accepted = torch.as_tensor(masks, device=device)
    labels = torch.as_tensor(intents, device=device)
    groups = [torch.where(labels == i)[0] for i in labels.unique()]
    development = ordinary_courses("development")
    parent_skills = development_skills(teacher, development)
    parent_outcomes = outcomes(teacher, selection)
    parent_rank = rank(parent_outcomes)
    best, step, checks = parent_rank, 0, []
    update = AttentionUpdate(parent, device)
    envs = [AttentionRepairEnv(current, historical, excluded, preparation=not historical)
            for _ in range(ENVIRONMENTS)]
    warmup = 0 if update.seen else DQN_WARMUP
    try:
        states = []
        for i, env in enumerate(envs):
            if update.environment_rng is not None:
                env.restore_random_state(update.environment_rng[i])
            state, _ = env.reset(seed=None if update.environment_rng is not None else DQN_SEED + i)
            states.append(state)
        states = np.stack(states)
        for position in range(ENVIRONMENTS, STEPS[-1] + 1, ENVIRONMENTS):
            epsilon = .05 + (.15 * max(0., 1 - max(0, position - warmup) / DQN_DECAY) if warmup else 0)
            actions = update.action(states, epsilon)
            results = [env.step(a) for env, a in zip(envs, actions, strict=True)]
            successors = np.stack([env.entities() for env in envs])
            rewards = [r[1] for r in results]
            done = [r[2] or r[3] for r in results]
            update.record(states, successors, actions, rewards, done)
            if position > warmup and position % TRAIN_EVERY == 0:
                update.update((examples, accepted, groups))
            if update.seen % TARGET_EVERY == 0:
                update.target.load_state_dict(update.network.state_dict())
            for i, (env, ended) in enumerate(zip(envs, done, strict=True)):
                if ended:
                    successors[i], _ = env.reset()
            states = successors
            if position not in STEPS:
                continue
            candidate = deepcopy(update.network).to("cpu").eval()
            skill = development_skills(candidate, development)
            courses = qualify(lambda: controller(candidate), courses=DEVELOPMENT_COURSES)
            observed = outcomes(candidate, selection)
            score = rank(observed)
            check = dict(step=position, development_courses=courses, skills=skill, outcomes=observed,
                         rank=score, gate_retains_skills=retains_skills(parent_skills, skill))
            if (courses["qualified"] and check["gate_retains_skills"] and
                    score[0] < parent_rank[0] and score[1] <= parent_rank[1] and score < best):
                evidence = directory / "screened" / str(position)
                formal = qualify(lambda: controller(candidate), save_root=evidence)
                check["formal"] = formal
                if formal["qualified"]:
                    update.checkpoint(directory / "policy.pt", envs)
                    best, step = score, position
            checks.append(check)
            print(directory, position, score, "selected", step, flush=True)
        selected = directory / "policy.pt" if step else parent
        capability = qualify(lambda: controller(load_attention(selected)), save_root=directory / "qualification")
        if not capability["qualified"]:
            raise ValueError("Attention publication did not pass all five formal courses")
        return freeze(completed, dict(
            checkpoint_sha256=checkpoint_identity(selected), step=step, parent_outcomes=parent_outcomes,
            parent_skills=parent_skills, candidates=checks, qualification=capability, demonstrations=demo_evidence,
            rehearsal_examples=len(examples), actual_online_steps=STEPS[-1],
            gradient_updates=update.gradient_updates,
            curriculum=[env.curriculum_state() for env in envs], elapsed_s=time.perf_counter() - started,
        ))
    finally:
        for env in envs:
            env.close()
