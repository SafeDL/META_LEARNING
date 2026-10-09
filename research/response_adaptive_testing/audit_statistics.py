"""Independently rebuild co-primary metrics, exact tests and bootstrap intervals."""
import numpy as np

from methods.history_guided_testing.io import read_json, write_json

from .config import OUTPUT
from .confirmation import CONFIRMATION, verify_lock


def signed_sums(values):
    sums = np.zeros((1, values.shape[1]))
    for row in values:
        sums = np.concatenate((sums - row, sums + row))
    return sums


def exact_p_meet_in_middle(differences):
    count = len(differences)
    split = count // 2
    left = signed_sums(differences[:split])
    right = signed_sums(differences[split:])
    observed = np.abs(differences.sum(0))
    tolerance = count * 1e-12
    probabilities = []
    for column, threshold in enumerate(observed):
        if threshold <= tolerance:
            probabilities.append(1.0)
            continue
        ordered = np.sort(right[:, column])
        positive = len(ordered) - np.searchsorted(
            ordered, threshold - tolerance - left[:, column], side="left")
        negative = np.searchsorted(ordered,
                                   -threshold + tolerance - left[:, column],
                                   side="right")
        extreme = np.sum(positive, dtype=np.int64) + np.sum(negative,
                                                            dtype=np.int64)
        probabilities.append(float(extreme / 2**count))
    return np.asarray(probabilities)


def bootstrap_weights(profiles):
    rng = np.random.default_rng(20261007)
    count = 20000
    weights = np.zeros((count, len(profiles)), dtype=np.int16)
    for controller in ("IDM", "FVDM"):
        indices = np.array([
            i for i, profile in enumerate(profiles)
            if profile["controller"] == controller
        ])
        draws = rng.integers(0, len(indices), (count, len(indices)))
        for local, index in enumerate(indices):
            weights[:, index] = (draws == local).sum(1)
    assert np.all(weights.sum(1) == len(profiles))
    return weights


def main():
    protocol = read_json(CONFIRMATION / "protocol.json")
    verify_lock(protocol)
    summary = read_json(CONFIRMATION / "summary.json")
    physical = read_json(OUTPUT / "confirmation_audit/summary.json")
    assert physical["all_physical_banks_audited"]
    assert summary["protocol"] == protocol
    methods, profiles = protocol["methods"], protocol["profiles"]
    values = np.zeros((len(profiles), len(methods), 2))
    disclosures = 0
    for profile_index, profile in enumerate(profiles):
        rows = {method: [] for method in methods}
        for replicate in range(protocol["replicates_per_profile"]):
            folder = CONFIRMATION / profile["name"] / f"pool_{replicate}"
            with np.load(folder / "responses.npz") as bank:
                total = int(np.count_nonzero(bank["collision"]))
                for method in methods:
                    for seed in protocol["seeds"]:
                        result = read_json(folder / "selection" /
                                           f"{method}_{seed}.json")
                        selected = result["selected_indices"]
                        assert len(selected) == len(
                            set(selected)) == protocol["budget"]
                        assert [
                            row["index"] for row in result["observations"]
                        ] == selected
                        for observation in result["observations"]:
                            index = observation["index"]
                            if method == "ras_frt_uq":
                                assert observation == {
                                    "index": index,
                                    "collision": bool(bank["collision"][index])
                                }
                            else:
                                assert observation == {
                                    "index": index,
                                    "continuous_risk":
                                    float(bank["risk"][index])
                                }
                        curve = np.cumsum(bank["collision"][selected])
                        assert curve.tolist() == result["curve"]
                        rows[method].append([
                            curve.mean() / total if total else 0,
                            curve[-1] / total if total else 1
                        ])
                        disclosures += len(selected)
        for method_index, method in enumerate(methods):
            assert len(rows[method]) == len(
                protocol["seeds"]) * protocol["replicates_per_profile"]
            values[profile_index, method_index] = np.mean(rows[method], axis=0)
            saved = summary["per_profile"][method][profile_index]
            np.testing.assert_allclose(
                values[profile_index, method_index],
                [saved["normalized_area"], saved["recall"]],
                atol=1e-12,
                rtol=0)
    columns = [(control, metric) for control in protocol["primary_controls"]
               for metric in ("normalized_area", "recall")]
    differences = np.column_stack([
        values[:, methods.index("candidate"), metric_index] -
        values[:, methods.index(control), metric_index]
        for control in protocol["primary_controls"] for metric_index in (0, 1)
    ])
    probabilities = exact_p_meet_in_middle(differences)
    order = sorted(range(len(probabilities)),
                   key=lambda index: probabilities[index])
    adjusted = np.zeros(len(probabilities))
    running = 0
    for rank, index in enumerate(order):
        running = max(running, probabilities[index] * (len(order) - rank))
        adjusted[index] = min(1, running)
    weights = bootstrap_weights(profiles)
    intervals = np.quantile(weights @ differences / len(profiles),
                            [0.025, 0.975],
                            axis=0).T
    for column, (control, metric) in enumerate(columns):
        saved = summary["comparisons"][control][metric]
        np.testing.assert_allclose(saved["per_profile_differences"],
                                   differences[:, column],
                                   atol=1e-12,
                                   rtol=0)
        assert abs(saved["mean_difference"] -
                   differences[:, column].mean()) < 1e-12
        assert abs(saved["exact_two_sided_p"] - probabilities[column]) < 1e-12
        assert abs(saved["holm_p"] - adjusted[column]) < 1e-12
        np.testing.assert_allclose(saved["bootstrap_95_ci"],
                                   intervals[column],
                                   atol=1e-12,
                                   rtol=0)
    success = bool(
        np.all(differences.mean(0) > 0)
        and np.all(adjusted < protocol["round_alpha"])
        and np.all(intervals[:2, 0] > 0))
    assert summary["success"] == success
    assert summary["selector_disclosures_audited"] == disclosures
    assert disclosures == len(
        profiles) * protocol["replicates_per_profile"] * len(
            protocol["seeds"]) * len(methods) * protocol["budget"]
    value = {
        "primary_metrics_rebuilt_from_full_banks": True,
        "exact_sign_tests_verified_by_half_sum_search": True,
        "bootstrap_verified_by_multinomial_counts": True,
        "holm_and_success_gate_verified": True,
        "statistical_units": len(profiles),
        "disclosures": disclosures,
        "success": success,
        "exact_p": probabilities.tolist(),
        "holm_p": adjusted.tolist(),
        "bootstrap_95_ci": intervals.tolist()
    }
    write_json(OUTPUT / "confirmation_audit/statistics.json", value)
    print("INDEPENDENT STATISTICAL AUDIT",
          success,
          probabilities.tolist(),
          flush=True)


if __name__ == "__main__":
    main()
