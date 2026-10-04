import numpy as np
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from sut_algorithms.highway_env.perception import PerceptionProfile, PerceivedIDMVehicle, PerceivedFVDMVehicle

def perceived_vehicle(profile):
    road=Road(RoadNetwork.straight_road_network(2,speed_limit=40))
    ego=PerceivedIDMVehicle(road,np.array([120.,0]),speed=25,target_lane_index=("0","1",0),profile=profile)
    front=Vehicle(road,np.array([150.,0]),speed=20)
    road.vehicles=[ego,front]
    return ego,front


def test_continuous_delay_holds_later_speed_changes():
    ego,front=perceived_vehicle(PerceptionProfile("test","IDM",perception_delay_s=.6))
    assert ego.perceived_front() is None
    ego.elapsed=.6
    sensed=ego.perceived_front()
    assert sensed.speed==20
    front.speed=5
    ego.elapsed=.7
    sensed=ego.perceived_front()
    assert sensed.speed==20
    np.testing.assert_allclose(sensed.position,[164,0])
    ego.elapsed=1.3
    sensed=ego.perceived_front()
    assert sensed.speed==5
    front.position[0]=300
    assert sensed.position[0]<300


def test_prediction_uses_observable_lateral_motion():
    ego,front=perceived_vehicle(PerceptionProfile("test","IDM",perception_mode="constant_velocity",prediction_horizon_s=2))
    front.position=np.array([150.,4]);front.heading=-.06
    # No scripted destination or schedule exists on this plain Vehicle.
    assert ego.perceived_front() is not None
    front.heading=0
    ego.elapsed=.05
    assert ego.perceived_front() is None
    front.position=np.array([110.,4]);front.heading=-.06
    ego.elapsed=.1
    assert ego.perceived_front() is None


def test_fvdm_has_a_physical_standstill_equilibrium():
    road=Road(RoadNetwork.straight_road_network(2,speed_limit=40))
    profile=PerceptionProfile("test","FVDM",target_speed=23,desired_gap=6)
    ego=PerceivedFVDMVehicle(road,np.array([120.,0]),speed=0,target_lane_index=("0","1",0),profile=profile)
    front=Vehicle(road,np.array([131.,0]),speed=0)
    assert ego.acceleration(ego,front)==0
    front.position[0]=130
    assert ego.acceleration(ego,front)==0
    ego.speed=2
    assert ego.acceleration(ego,front)<0
    # The original formula still illustrates the old creeping failure, without
    # changing any original profile or serialized build fingerprint.
    from sut_algorithms.highway_env.idm_profiles import ProfiledFVDMVehicle, SUTProfile
    original=ProfiledFVDMVehicle(road,np.array([120.,0]),speed=0,profile=SUTProfile("old","FVDM",desired_gap=6))
    assert original.acceleration(original,front)>0


