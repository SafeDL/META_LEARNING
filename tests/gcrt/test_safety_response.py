import numpy as np

from benchmark.scenarios import features, sample
from methods.common.response import ResponseArchive
from methods.gcrt.safety_response import safety_coordinates


def test_safety_coordinates_use_source_training_statistics_and_preserve_query_order():
    scenarios = sample('cut_in',32,'history')
    x = features(scenarios)
    archive = ResponseArchive('history_unit','rule','unit','history',scenarios,
                              x[:,0] > .6,
                              np.column_stack((10 + 10*x[:,1],1 + 3*x[:,2])))
    actual = safety_coordinates([archive],x[:7])
    reversed_actual = safety_coordinates([archive],x[:7][::-1])
    assert actual.shape == (7,3)
    assert np.isfinite(actual).all()
    assert (np.abs(actual) <= 1 + 1e-14).all()
    np.testing.assert_allclose(actual,reversed_actual[::-1],atol=0,rtol=0)
