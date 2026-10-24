import numpy as np
import pytest

from benchmark.scenarios import features, sample
from methods.gcrt.response_geometry import kinematic_features, predict_fields, response_fields


@pytest.mark.parametrize('kind,extra',[('cut_in',6),('cut_out',8)])
def test_relative_motion_coordinates_are_public_and_order_preserving(kind,extra):
    scenarios = sample(kind,8,'history')
    x = kinematic_features(scenarios)
    raw = features(scenarios)
    assert x.shape == (8,raw.shape[1]+extra)
    np.testing.assert_allclose(x[:,:raw.shape[1]],raw)
    np.testing.assert_allclose(kinematic_features(scenarios[::-1]),x[::-1],atol=0,rtol=0)
    assert np.isfinite(x).all()
    assert (np.abs(x[:,raw.shape[1]:]) <= 1).all()
    p = scenarios[0].parameters
    closing = p.get('closing_speed_mps',p['ego_speed_mps']-p.get('lead_speed_mps',0))
    projected = (p['gap_m']-closing*(p['event_time_s']+p['lateral_duration_s']/2))/50
    assert x[0,raw.shape[1]+1] == pytest.approx(np.clip(projected,-1,1))


def test_structured_prediction_rejects_development_or_final_sources():
    class Archive:
        role = 'development'
    with pytest.raises(ValueError,match='historical'):
        predict_fields(Archive(),np.arange(4),np.zeros((4,2)),kinematic=True)
    with pytest.raises(ValueError,match='historical'):
        response_fields([Archive()],sample('cut_in',4,'history'))
