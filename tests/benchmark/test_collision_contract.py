from dataclasses import fields

import numpy as np
import pytest

from benchmark.protocol import check_roles
from benchmark.scenarios import sample
from benchmark.storage import load_archive, save_archive
from controllers.highway.catalog import SUTS
from methods.common.response import ResponseArchive
from simulation.highway.observation import VehicleState, observe, policy_observation


def test_twenty_objects_and_learning_runs_have_explicit_lineage():
    assert len(SUTS) == 20
    assert len({s.sut_id for s in SUTS}) == 20
    for family in ("rule", "optimization", "search", "learning"):
        group = [s for s in SUTS if s.family == family]
        assert [sum(s.role == role for s in group)
                for role in ("history", "development", "target")] == [2, 1, 2]
        history = {s.lineage for s in group if s.role == "history"}
        assert history == {s.lineage for s in group if s.role == "target"}
    learned = [s for s in SUTS if s.family == "learning"]
    assert len({s.training_seed for s in learned}) == 5
    assert len({s.training_run for s in learned}) == 5
    history = tuple(s for s in SUTS if s.role == "history")
    development = tuple(s for s in SUTS if s.role == "development")
    with pytest.raises(ValueError, match="Held family"):
        check_roles(history, development, held_family="rule")


def test_geometric_occlusion_is_shared_and_tracks_reveal():
    ego = VehicleState("ego", 0, 0, 25, 0)
    lead = VehicleState("lead", 20, 0, 25, 0)
    obstacle = VehicleState("obstacle", 55, 0, 0, 0)
    world = observe(0, [ego, lead, obstacle])
    assert [v.actor_id for v in world.vehicles] == ["lead"]
    moved = VehicleState("lead", 20, 4, 25, 0)
    revealed = observe(1, [ego, moved, obstacle])
    assert "obstacle" in [v.actor_id for v in revealed.vehicles]
    assert set(f.name for f in fields(world)) == {
        "time_s", "ego", "vehicles", "lane_centers", "speed_limit_mps"}
    assert policy_observation(world).shape == (82,)
    assert not np.array_equal(policy_observation(world), policy_observation(revealed))
    assert not hasattr(world, "road")
    assert not hasattr(world, "event_time_s")


def test_new_archive_round_trip_requires_true_collision_provenance(tmp_path):
    scenes = sample("cut_in", 16, "history")
    archive = ResponseArchive("a", "rule", "idm_mobil", "history", scenes,
                              np.arange(16) % 3 == 0, np.ones((16, 2)))
    path = tmp_path / "archive.npz"
    save_archive(path, archive, provenance={"qualification": "fixture", "physics_hz": 20})
    restored = load_archive(path)
    assert restored.scenarios == scenes
    np.testing.assert_array_equal(restored.collision, archive.collision)
    with pytest.raises(ValueError, match="Final targets"):
        ResponseArchive("target", "rule", "idm_mobil", "target", scenes,
                        archive.collision, archive.safety)
    with pytest.raises(ValueError, match="Boolean"):
        ResponseArchive("risk", "rule", "idm_mobil", "history", scenes,
                        np.ones(16) * 0.9, archive.safety)
