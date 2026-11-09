"""BehAVExplor temporal statistics and clustered abstract-trace distance."""

import numpy as np
from sklearn.cluster import KMeans


def motion_trace(trajectory, *, interval=0.1):
    """Derive eight ego motion attributes from this paid execution only."""
    if not trajectory:
        raise ValueError("BehAVExplor requires the paid actual trajectory")
    times = np.array([f["time_s"] for f in trajectory], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Behavior timestamps must be finite and strictly increasing")
    ego = [next(a for a in f["actors"] if a["actor_id"] == "ego") for f in trajectory]
    speed = np.array([a["speed"] for a in ego])
    heading = np.unwrap([a["heading"] for a in ego])
    forward = np.column_stack((np.cos(heading), np.sin(heading)))
    lateral = np.column_stack((-np.sin(heading), np.cos(heading)))
    velocity = speed[:, None] * forward
    acceleration = np.zeros_like(velocity)
    yaw_rate = np.zeros_like(speed)
    sign = np.ones_like(speed)
    if len(times) > 1:
        dt = np.diff(times)
        acceleration[:-1] = np.diff(velocity, axis=0) / dt[:, None]
        yaw_rate[:-1] = np.diff(heading) / dt
        sign[:-1] = np.where(np.diff(speed) < 0, -1, 1)
        acceleration[-1], yaw_rate[-1], sign[-1] = acceleration[-2], yaw_rate[-2], sign[-2]
    attributes = np.column_stack(
        (
            speed,
            np.linalg.norm(acceleration, axis=1) * sign,
            (velocity * forward).sum(1),
            (velocity * lateral).sum(1),
            (acceleration * forward).sum(1),
            (acceleration * lateral).sum(1),
            np.rad2deg(yaw_rate),
            np.rad2deg(heading),
        )
    )
    # Include the actual terminal frame, without inventing a terminal stop.
    aligned_times = np.r_[np.arange(times[0], times[-1], interval), times[-1]]
    return np.column_stack([np.interp(aligned_times, times, a) for a in attributes.T])


def temporal_features(trace, *, window=10):
    """Eight statistics per motion attribute; pad each trailing window per §3.2.1."""
    trace = np.asarray(trace, dtype=float)
    if trace.ndim != 2 or len(trace) == 0 or window < 3 or not np.isfinite(trace).all():
        raise ValueError("Temporal features need finite motion rows and a window of at least 3")
    padded = np.pad(trace, ((0, window - 1), (0, 0)), mode="edge")
    segments = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    changes = np.diff(segments, axis=-1)
    standard_deviation = segments.std(-1)
    complexity = np.divide(
        np.sqrt((changes**2).sum(-1)),
        standard_deviation,
        out=np.zeros_like(standard_deviation),
        where=standard_deviation > 0,
    )
    statistics = (
        segments.mean(-1),
        segments.var(-1),
        segments.min(-1),
        segments.max(-1),
        changes.mean(-1),
        np.abs(changes).mean(-1),
        (segments[..., :-2] * segments[..., 1:-1] * segments[..., 2:]).mean(-1),
        complexity,
    )
    return np.stack(statistics, axis=-1).reshape(len(trace), -1)


def trace_distance(left, right):
    """Equation (9): aligned Hamming mismatches plus unmatched duration."""
    common, longest = min(len(left), len(right)), max(len(left), len(right))
    if common == 0:
        raise ValueError("Abstract behavior traces must be nonempty")
    return float((np.count_nonzero(left[:common] != right[:common]) + longest - common) / longest)


class BehaviorMiner:
    def __init__(self, *, clusters=10, threshold=0.4, seed=11):
        self.clusters = clusters
        self.threshold = threshold
        self.seed = seed
        self.features = []
        self.traces = []
        self.model = None
        self.rebuilds = 0

    def add(self, feature):
        self.features.append(feature)
        data = np.concatenate(self.features)
        count = min(self.clusters, len(np.unique(data, axis=0)))
        self.model = KMeans(n_clusters=count, n_init=10, random_state=self.seed).fit(data)
        self.traces = [self.model.predict(x) for x in self.features]
        self.rebuilds += 1

    def initialize(self, features):
        self.features = list(features[:-1])
        self.add(features[-1])

    def feedback(self, feature):
        trace = self.model.predict(feature)
        distance = min(trace_distance(trace, previous) for previous in self.traces)
        return distance > self.threshold, distance
