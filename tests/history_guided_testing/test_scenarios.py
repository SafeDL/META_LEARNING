import numpy as np

from methods.history_guided_testing.history import split_indices
from methods.history_guided_testing.scenarios import parameter_cells, scene_library


def test_family_counts_and_spatial_split():
    scenes = scene_library("target")
    x = np.asarray([scene["numeric_input"] for scene in scenes])
    assert len(scenes) == 2048
    assert all(sum(x[:, 4] == family) == 1024 for family in (0, 1))
    train, validation = split_indices(x)
    assert len(train) == 1638 and len(validation) == 410
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(2048))
    cells = parameter_cells(x)
    assert cells[:1024].max() < 256 and cells[1024:].min() >= 256
