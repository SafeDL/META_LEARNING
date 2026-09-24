import numpy as np

from highway_env_benchmark.data.generate_anchor_bank import FUNCTIONAL_MODES
from method_chains.function_conditioned_routing.benchmark import (
    ARCHETYPE_NAMES,
    RELEASE_CODES,
    functional_releases,
    generate_functional_scenarios,
)


def test_functional_releases_have_real_balanced_module_sources() -> None:
    releases = functional_releases()
    assert len(releases) == 6
    assert len({release.name for release in releases}) == 6
    for release in releases:
        assert set(release.modules) == set(FUNCTIONAL_MODES)
        assert set(release.modules.values()).issubset(ARCHETYPE_NAMES)
    for column in RELEASE_CODES.T:
        assert np.array_equal(np.bincount(column, minlength=3), np.asarray([2, 2, 2]))


def test_functional_scenarios_are_balanced_and_controlled() -> None:
    anchors, modes, controls = generate_functional_scenarios(20, 17)
    assert anchors.shape == (20, 2)
    assert controls.shape == (20, 2)
    assert np.all((controls >= 0.0) & (controls <= 1.0))
    assert {mode: int(np.sum(modes == mode)) for mode in FUNCTIONAL_MODES} == {
        mode: 4 for mode in FUNCTIONAL_MODES
    }
