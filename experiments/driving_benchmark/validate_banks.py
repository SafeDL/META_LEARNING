"""Audit available response banks against scenes, physical logs and smoke runs."""
import numpy as np

from methods.history_guided_testing.io import read_json, read_rows, write_json

from .config import OUTPUT, POOL_SEEDS, SUT_IDS
from .pools import scene_pool


def main():
    smoke = read_json(OUTPUT / "interface_audit.json")["measurements"]
    banks, missing = [], []
    for pool_id in range(len(POOL_SEEDS)):
        scenes = scene_pool(pool_id)
        expected_x = np.asarray([scene["numeric_input"] for scene in scenes])
        expected_ids = np.asarray([scene["scenario_id"] for scene in scenes])
        for sut_id in SUT_IDS:
            folder = OUTPUT / f"pool_{pool_id}" / sut_id
            if not (folder / "responses.npz").exists():
                missing.append({"pool": pool_id, "sut_id": sut_id})
                continue
            logs = read_rows(folder / "measurements.jsonl")
            if len(logs) != 2048 or len({row["scenario_id"] for row in logs}) != 2048:
                raise ValueError("Bank log size or unique scene count differs")
            mapping = {row["scenario_id"]: row for row in logs}
            ordered = [mapping[scene["scenario_id"]] for scene in scenes]
            with np.load(folder / "responses.npz", allow_pickle=False) as bank:
                np.testing.assert_array_equal(bank["x"], expected_x)
                np.testing.assert_array_equal(bank["scenario_id"], expected_ids)
                np.testing.assert_array_equal(bank["collision"], [row["collision"] for row in ordered])
                np.testing.assert_array_equal(bank["risk"], [row["risk"] for row in ordered])
                np.testing.assert_array_equal(bank["risk_components"], [row["risk_components"] for row in ordered])
                if not np.all(np.isfinite(bank["risk"]) & (bank["risk"] >= 0) & (bank["risk"] <= 1)):
                    raise ValueError("Invalid finite risk in benchmark")
                collisions = [int(np.count_nonzero(bank["collision"][expected_x[:, 4] == family]))
                              for family in (0, 1)]
                matched = 0
                if pool_id == 0:
                    for row in smoke:
                        if row["sut_id"] != sut_id:
                            continue
                        index = row["index"]
                        if float(bank["risk"][index]) != row["risk"] or bool(
                                bank["collision"][index]) != row["collision"]:
                            raise ValueError("Bank differs from the prior physical smoke execution")
                        matched += 1
                banks.append({"pool": pool_id, "sut_id": sut_id, "scenes": 2048,
                              "family_collisions": collisions, "smoke_replays_matched": matched})
    result = {"available_banks_passed": banks, "missing_banks": missing,
              "complete": not missing, "scope": "all rows of each available bank; no additional simulation"}
    write_json(OUTPUT / "bank_audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
