from types import SimpleNamespace

from mvr.training import trainers


def test_inner_pretrain_visits_one_task_per_episode_in_balanced_epochs(monkeypatch) -> None:
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
        "inner": {
            "episodes_per_task": 2,
            "updates_per_episode": 8,
            "batch_size": 64,
        },
    }

    metrics, _ = trainers.train_inner(None, tasks, config, None, None)

    assert len(calls) == 6
    assert all(budget == 1 for _, budget, _ in calls)
    assert [offset for _, _, offset in calls] == [0, 0, 0, 1, 1, 1]
    assert {task_id for task_id, _, _ in calls[:3]} == {task.task_id for task in tasks}
    assert {task_id for task_id, _, _ in calls[3:]} == {task.task_id for task in tasks}
    assert len(updates) == 6
    assert metrics["balanced_sampling_epochs"] == 2
    assert metrics["requested_optimizer_updates"] == 48
