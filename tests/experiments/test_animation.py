import gzip
import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Text
from PIL import Image, ImageSequence

from controllers.highway.control import Control
from experiments.animation import save_gif
from experiments.render_joint import place_actor_labels, render_execution, select_cases
from experiments.scenarios import TEST_SCENARIOS, sample
from experiments.storage import write_json
from simulator.highway.collision_env import execute


class Brake:
    def act(self, world):
        return Control(-8, 0)


def test_close_vehicle_labels_are_separate_and_keep_actual_leader_positions():
    figure, road = plt.subplots(figsize=(7.1, 2))
    road.set(xlim=(-20, 140), ylim=(11, -3))
    actors = {
        "ego": {"x": 0, "y": 4},
        "obstacle": {"x": 1, "y": 4},
        "lead": {"x": 2, "y": 4},
        "adjacent_front": {"x": 139, "y": 0},
    }
    labels = {
        name: road.annotate(
            name,
            (0, 0),
            xytext=(0.5, 1.04),
            textcoords="axes fraction",
            fontsize=8,
            ha="center",
            va="bottom",
            arrowprops={"arrowstyle": "-"},
        )
        for name in actors
    }
    try:
        figure.canvas.draw()
        place_actor_labels(road, labels, actors, {"ego", "lead", "obstacle"})
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        boxes = sorted(
            (Text.get_window_extent(label, renderer) for label in labels.values()),
            key=lambda box: box.x0,
        )
        assert all(left.x1 < right.x0 for left, right in zip(boxes, boxes[1:]))
        assert boxes[0].x0 >= road.bbox.x0
        assert boxes[-1].x1 <= road.bbox.x1
        assert all(box.y0 > road.bbox.y1 for box in boxes)
        assert labels["adjacent_front"].get_text() == "Adjacent front *"
        for name, label in labels.items():
            assert label.xy == (actors[name]["x"], actors[name]["y"])
    finally:
        plt.close(figure)


@pytest.mark.parametrize("kind", TEST_SCENARIOS)
def test_gif_keeps_every_physics_frame_at_real_time_and_retains_raw_evidence(tmp_path, kind):
    scene = sample(kind, 1, "new")[0]
    execution = execute(Brake(), scene, duration_s=0.3)
    provenance = {"task_id": "render_test", "query": 1}
    static = kind == "car_following"
    result = render_execution(
        scene, execution.trajectory, tmp_path, provenance=provenance, static=static
    )
    with Image.open(tmp_path / result["gif"]) as gif:
        durations = [frame.info["duration"] for frame in ImageSequence.Iterator(gif)]
        assert gif.n_frames == len(execution.trajectory)
        gif.seek(0)
        initial = np.asarray(gif.convert("RGB")).copy()
        gif.seek(gif.n_frames - 1)
        assert np.any(initial != np.asarray(gif.convert("RGB")))
    assert durations[:-1] == [50] * (len(execution.trajectory) - 1)
    assert sum(durations[:-1]) == 300
    assert durations[-1] == result["final_hold_ms"] == 500
    assert result["playback_speed"] == 1
    payload = json.loads(gzip.decompress((tmp_path / f"{kind}.execution.json.gz").read_bytes()))
    assert payload["trajectory"] == execution.trajectory
    assert payload["provenance"] == provenance
    assert len(list(tmp_path.glob("*.png"))) == int(static)
    if static:
        with Image.open(tmp_path / f"{kind}.png") as figure:
            assert figure.info["dpi"] == pytest.approx((600, 600), abs=0.1)
    assert not list(tmp_path.glob("*.csv"))


def test_gif_follows_irregular_timestamps_instead_of_assuming_constant_frame_rate(tmp_path):
    images = [Image.new("RGB", (8, 8), color) for color in ("red", "green", "blue")]
    path = tmp_path / "timing.gif"
    save_gif(images, [2.0, 2.05, 2.2], path)
    with Image.open(path) as gif:
        assert [frame.info["duration"] for frame in ImageSequence.Iterator(gif)] == [50, 150, 500]


def test_cases_follow_registered_ads_order_then_paid_order_and_keep_missing_outcomes(tmp_path):
    kind = "car_following"
    scene = sample(kind, 1, "new")[0]
    write_json(
        tmp_path / "reports/protocol.json",
        {
            "suite": "pilot",
            "systems": [{"sut_id": "z_first"}, {"sut_id": "a_second"}],
            "config": {"scenarios": [kind, "cut_out"]},
        },
    )
    rows = []
    for system in ("a_second", "z_first"):
        relative = f"methods/joint/{system}/result.json"
        rows.append(
            {"method": "joint", "sut_id": system, "scenario": kind, "result_path": relative}
        )
        path = tmp_path / "methods/joint" / system / "completed.json.gz"
        path.parent.mkdir(parents=True)
        task = {
            "completed": True,
            "sut_id": system,
            "task_id": system,
            "original_role": "independent",
            "executions": [
                {
                    "query": query,
                    "record_id": f"{system}:{query}",
                    "scenario": asdict(scene),
                    "trajectory": [{"query": query}],
                    "feedback": {"collision": collision},
                }
                for query, collision in ((3, False), (2, True), (1, True))
            ],
        }
        path.write_bytes(gzip.compress(json.dumps(task).encode()))
    rows.append({"method": "random", "sut_id": "z_first", "scenario": kind})
    write_json(tmp_path / "reports/summary.json", {"status": "complete", "runs": rows})
    cases = select_cases(tmp_path)
    assert cases[kind]["collision"]["provenance"]["record_id"] == "z_first:1"
    assert cases[kind]["safe"]["provenance"]["record_id"] == "z_first:3"
    assert cases["cut_out"] == {"collision": None, "safe": None}
    write_json(tmp_path / "reports/summary.json", {"status": "running", "runs": rows})
    with pytest.raises(ValueError, match="completed registered"):
        select_cases(tmp_path)


def sequential_case_archives(root):
    kind = "car_following"
    scene = sample(kind, 1, "new")[0]
    write_json(
        root / "reports/protocol.json",
        {
            "suite": "sequential",
            "systems": [{"sut_id": "a_second"}, {"sut_id": "z_first"}],
            "config": {"scenarios": [kind, "cut_out"]},
        },
    )
    rows, paths = [], {}
    for system, ordinal in (("a_second", 2), ("z_first", 1)):
        rows.append(
            {
                "regime": "update_shared",
                "branch_id": "update_shared:11",
                "seed": 11,
                "ordinal": ordinal,
                "sut_id": system,
                "scenario": kind,
                "result_path": f"methods/update_shared/11/queries/{ordinal:02d}_{system}/{kind}/result.json",
            }
        )
        task = {
            "completed": True,
            "completed_order": ordinal,
            "branch_id": "update_shared:11",
            "sut_id": system,
            "task_id": system,
            "original_role": "independent",
            "model_snapshot": {"target_model_version": "target", "encoder_version": "encoder"},
            "executions": [
                {
                    "query": query,
                    "record_id": f"{system}:{query}",
                    "scenario": asdict(scene),
                    "trajectory": [{"query": query}],
                    "feedback": {"collision": collision},
                }
                for query, collision in ((3, False), (2, True), (1, True))
            ],
        }
        path = (
            root / "methods/update_shared/11/completed" / f"{ordinal:02d}_{system}_{kind}.json.gz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(task).encode()))
        paths[system] = path
    # These controls have no case archives and must never be read as the full method.
    rows.extend({"regime": regime} for regime in ("fixed_history", "append_sources"))
    write_json(root / "reports/summary.json", {"status": "complete", "runs": rows})
    return paths


def test_sequential_cases_use_update_branch_chronology_and_keep_model_provenance(tmp_path):
    sequential_case_archives(tmp_path)
    cases = select_cases(tmp_path)
    collision = cases["car_following"]["collision"]
    assert collision["provenance"]["record_id"] == "z_first:1"
    assert collision["provenance"]["branch_id"] == "update_shared:11"
    assert collision["provenance"]["ordinal"] == 1
    assert collision["provenance"]["model_snapshot"] == {
        "target_model_version": "target",
        "encoder_version": "encoder",
    }
    assert collision["trace"] == [{"query": 1}]
    assert cases["car_following"]["safe"]["provenance"]["record_id"] == "z_first:3"
    assert cases["cut_out"] == {"collision": None, "safe": None}


@pytest.mark.parametrize(
    "field,value", [("branch_id", "append_sources:11"), ("completed_order", 3)]
)
def test_sequential_case_archive_cannot_cross_branch_or_completion_order(tmp_path, field, value):
    path = sequential_case_archives(tmp_path)["z_first"]
    task = json.loads(gzip.decompress(path.read_bytes()))
    task[field] = value
    path.write_bytes(gzip.compress(json.dumps(task).encode()))
    with pytest.raises(ValueError, match="chronological task snapshot"):
        select_cases(tmp_path)


def test_candidate_scale_cases_require_one_registered_pool(tmp_path):
    write_json(tmp_path / "reports/summary.json", {"status": "complete", "runs": []})
    write_json(
        tmp_path / "reports/protocol.json",
        {"suite": "sensitivity", "candidate_sizes": [1000, 2000, 4000]},
    )
    with pytest.raises(ValueError, match="single registered candidate size"):
        select_cases(tmp_path)
