from types import SimpleNamespace

import numpy as np

from mvr.training import trainers


def test_interaction_prior_visits_one_task_per_episode_in_balanced_epochs(monkeypatch) -> None:
    tasks = [SimpleNamespace(task_id=f"task-{index}") for index in range(3)]
    calls = []
    updates = []

    class FakeOnline:
        def run(self, task, budget, **kwargs):
            calls.append((task.task_id, budget, kwargs["episode_index_offset"]))
            return SimpleNamespace(inner_transitions=(), episodes=(SimpleNamespace(),))

    monkeypatch.setattr(trainers, "build_online", lambda *args, **kwargs: FakeOnline())
    monkeypatch.setattr(
        trainers,
        "_update_inner",
        lambda *args: updates.append(None),
    )
    monkeypatch.setattr(trainers, "_inner_metrics", lambda *args: {})
    config = {
        "seed": 11,
        "training": {"step_budget": 60},
            "interaction_prior": {
            "episodes_per_task": 2,
            "updates_per_episode": 8,
            "batch_size": 64,
        },
    }

    metrics, _ = trainers.train_interaction_prior(None, tasks, config, None, None)

    assert len(calls) == 6
    assert all(budget == 1 for _, budget, _ in calls)
    assert [offset for _, _, offset in calls] == [0, 0, 0, 1, 1, 1]
    assert {task_id for task_id, _, _ in calls[:3]} == {task.task_id for task in tasks}
    assert {task_id for task_id, _, _ in calls[3:]} == {task.task_id for task in tasks}
    assert len(updates) == 6
    assert metrics["balanced_sampling_epochs"] == 2
    assert metrics["requested_optimizer_updates"] == 48


def test_interaction_prior_uses_random_actions_before_warmup_updates(monkeypatch) -> None:
    task = SimpleNamespace(task_id="task-0")
    providers = []
    updates = []

    class FakeOnline:
        def run(self, _task, _budget, **kwargs):
            providers.append(kwargs["inner_action_provider"])
            return SimpleNamespace(inner_transitions=(), episodes=(SimpleNamespace(),))

    monkeypatch.setattr(trainers, "build_online", lambda *args, **kwargs: FakeOnline())
    monkeypatch.setattr(
        trainers,
        "_update_inner",
        lambda *args: updates.append(None),
    )
    monkeypatch.setattr(trainers, "_inner_metrics", lambda *args: {})
    config = {
        "seed": 11,
        "training": {"step_budget": 60},
        "interaction_prior": {
            "episodes_per_task": 6,
            "updates_per_episode": 8,
            "batch_size": 64,
            "warmup_episodes": 5,
        },
    }

    trainers.train_interaction_prior(None, [task], config, None, None)

    assert all(provider is not None for provider in providers[:5])
    assert providers[5] is None
    warmup_action = providers[0](np.zeros(1, dtype=np.float32))
    expected = np.random.default_rng(11).uniform(-1.0, 1.0, size=(4,)).astype(
        np.float32
    )
    assert warmup_action.shape == (4,)
    assert warmup_action.dtype == np.float32
    np.testing.assert_allclose(warmup_action, expected)
    assert np.all((-1.0 <= warmup_action) & (warmup_action <= 1.0))
    assert len(updates) == 1
