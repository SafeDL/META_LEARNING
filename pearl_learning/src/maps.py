"""Current MetaDrive map definitions for bottleneck, lane-drop, and Y merge."""
from __future__ import annotations

def bottleneck_env_class() -> type:
    """Return a MetaDriveEnv subclass backed by actual Merge/Split blocks."""
    from metadrive.component.map.pg_map import PGMap
    from metadrive.component.pgblock.bottleneck import Merge, Split
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.manager.pg_map_manager import PGMapManager
    from metadrive.utils import Config

    class LogicalBottleneckMap(PGMap):
        def _generate(self) -> None:
            cfg = self.config
            parent, world = self.engine.worldNP, self.engine.physics_world
            first = FirstPGBlock(self.road_network, cfg[self.LANE_WIDTH], int(cfg["bottle_lane_num"]), parent, world, length=float(cfg["exit_length"]))
            self.blocks.append(first)
            merge = Merge(1, first.get_socket(0), self.road_network, random_seed=1, ignore_intersection_checking=False)
            merge.construct_from_config({"lane_num": int(cfg["bottle_lane_num"]) - int(cfg["neck_lane_num"]), "length": float(cfg["neck_length"])}, parent, world)
            self.blocks.append(merge)
            split = Split(2, merge.get_socket(0), self.road_network, random_seed=1, ignore_intersection_checking=False)
            split.construct_from_config({"lane_num": int(cfg["bottle_lane_num"]) - int(cfg["neck_lane_num"]), "length": float(cfg["exit_length"])}, parent, world)
            self.blocks.append(split)

    class LogicalBottleneckMapManager(PGMapManager):
        def reset(self) -> None:
            if not self.spawned_objects:
                road_map = self.spawn_object(LogicalBottleneckMap, map_config=self.engine.global_config["map_config"], random_seed=None)
            else:
                road_map = next(iter(self.spawned_objects.values()))
            self.load_map(road_map)

    class LogicalBottleneckEnv(MetaDriveEnv):
        @staticmethod
        def default_config() -> Config:
            return MetaDriveEnv.default_config().update({
                "map_config": {"exit_length": 60, "bottle_lane_num": 3, "neck_lane_num": 1, "neck_length": 32, "lane_num": 3}
            }, allow_add_new_key=True)

        def setup_engine(self) -> None:
            super().setup_engine()
            self.engine.update_manager("map_manager", LogicalBottleneckMapManager())

    return LogicalBottleneckEnv


def y_merge_env_class() -> type:
    """A separately named two-arm Y merge environment.

    It deliberately uses a 2->1 ``Merge`` block, whose two explicit incoming
    lane sequences meet on the same downstream road.  This is not the random
    ``"r"`` map previously (and incorrectly) labelled as a Y merge.
    """
    from metadrive.component.map.pg_map import PGMap
    from metadrive.component.pgblock.bottleneck import Merge
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.envs.metadrive_env import MetaDriveEnv
    from metadrive.manager.pg_map_manager import PGMapManager
    from metadrive.utils import Config

    class LogicalYMergeMap(PGMap):
        def _generate(self) -> None:
            cfg = self.config
            parent, world = self.engine.worldNP, self.engine.physics_world
            first = FirstPGBlock(
                self.road_network, cfg[self.LANE_WIDTH], int(cfg["bottle_lane_num"]), parent, world,
                length=float(cfg["exit_length"]),
            )
            self.blocks.append(first)
            # Unlike the bottleneck map, this graph deliberately ends after the
            # two inbound arms join the one shared downstream road: no Split
            # block is appended after the merge.
            merge = Merge(1, first.get_socket(0), self.road_network, random_seed=1, ignore_intersection_checking=False)
            merge.construct_from_config(
                {"lane_num": int(cfg["bottle_lane_num"]) - int(cfg["neck_lane_num"]), "length": float(cfg["neck_length"])},
                parent, world,
            )
            self.blocks.append(merge)

    class LogicalYMergeMapManager(PGMapManager):
        def reset(self) -> None:
            if not self.spawned_objects:
                road_map = self.spawn_object(LogicalYMergeMap, map_config=self.engine.global_config["map_config"], random_seed=None)
            else:
                road_map = next(iter(self.spawned_objects.values()))
            self.load_map(road_map)

    class LogicalYMergeEnv(MetaDriveEnv):
        @staticmethod
        def default_config() -> Config:
            return MetaDriveEnv.default_config().update({
                "map_config": {"exit_length": 60, "bottle_lane_num": 2, "neck_lane_num": 1, "neck_length": 32, "lane_num": 2}
            }, allow_add_new_key=True)

        def setup_engine(self) -> None:
            super().setup_engine()
            self.engine.update_manager("map_manager", LogicalYMergeMapManager())

    return LogicalYMergeEnv
