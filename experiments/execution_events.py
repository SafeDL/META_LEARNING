"""Observable event/response times reconstructed outside the SUT decision interface."""


def analyze_events(scenario, trajectory):
    if not trajectory:
        raise ValueError("Event reconstruction needs the complete execution trace")
    onset = scenario.parameters.get("event_time_s")
    initial = {a["actor_id"]: a for a in trajectory[0]["actors"]}
    cut_in = scenario.kind in ("cut_in", "cut_in_interaction")
    scripted_role = "cut_in" if cut_in else "lead"
    measured_onset, completed, revealed, maneuver = None, None, None, None
    first_brake, after_event, after_reveal = None, None, None
    for index, frame in enumerate(trajectory):
        time_s = frame["time_s"]
        control_time = trajectory[max(0, index - 1)]["time_s"]
        states = {a["actor_id"]: a for a in frame["actors"]}
        ego = states["ego"]
        if frame["control"]["acceleration"] < -0.5:
            if first_brake is None:
                first_brake = control_time
            if onset is not None and time_s >= onset and after_event is None:
                after_event = max(control_time, onset)
            if revealed is not None and after_reveal is None:
                after_reveal = max(control_time, revealed)
        if revealed is None and "obstacle" in frame["visible_ids"]:
            revealed = time_s
            if frame["control"]["acceleration"] < -0.5:
                after_reveal = time_s
        actor = states[scripted_role]
        if scenario.kind == "front_brake":
            if measured_onset is None and actor["speed"] < initial[scripted_role]["speed"] - 0.01:
                measured_onset = time_s
            if completed is None and actor["speed"] <= 0.02:
                completed = time_s
        elif scenario.kind in ("cut_in", "cut_in_interaction", "cut_out"):
            if measured_onset is None and abs(actor["y"] - initial[scripted_role]["y"]) >= 0.01:
                measured_onset = time_s
            destination = 4 if cut_in else 0
            if measured_onset is not None and completed is None:
                if abs(actor["y"] - destination) < 0.2 and abs(actor["heading"]) < 0.05:
                    completed = time_s
        if maneuver is None and abs(ego["y"] - initial["ego"]["y"]) > 0.3:
            maneuver = time_s
    return {
        "scheduled_event_s": onset,
        "measured_background_onset_s": measured_onset,
        "background_complete_s": completed,
        "obstacle_visible_s": revealed,
        "ego_lane_change_onset_s": maneuver,
        "first_ego_brake_s": first_brake,
        "ego_brake_at_or_after_event_s": after_event,
        "brake_latency_from_scheduled_event_s": (
            after_event - onset if after_event is not None else None
        ),
        "ego_brake_at_or_after_reveal_s": after_reveal,
        "brake_latency_from_reveal_s": after_reveal - revealed
        if after_reveal is not None
        else None,
    }
