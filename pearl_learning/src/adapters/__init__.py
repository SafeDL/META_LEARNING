from .base import LogicalScenarioAdapter
from .bottleneck import BottleneckMergeAdapter
from .on_ramp import OnRampMergeAdapter

ADAPTERS = {
    "on_ramp_merge": OnRampMergeAdapter,
    "lane_drop_merge": BottleneckMergeAdapter,
    "bottleneck_merge": BottleneckMergeAdapter,
    "y_merge": OnRampMergeAdapter,
}


def adapter_for(logical_type: str) -> LogicalScenarioAdapter:
    try:
        return ADAPTERS[logical_type]()
    except KeyError as exc:
        raise ValueError(f"no adapter for {logical_type}") from exc
