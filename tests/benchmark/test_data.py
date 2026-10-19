"""Measured costs are joined by stable scenario identifiers."""


def test_archived_costs_join_by_scenario_id_instead_of_file_order(
    tmp_path, monkeypatch
):
    import json
    from benchmark import data

    records = [
        {"scenario_id": "second", "elapsed_s": 3.0},
        {"scenario_id": "first", "elapsed_s": 1.0},
    ]
    (tmp_path / "measurements.jsonl").write_text(
        "\n".join(json.dumps(row) for row in records), encoding="utf-8"
    )
    monkeypatch.setattr(data, "bank_path", lambda *args: tmp_path / "responses.npz")
    assert data.measured_seconds(0, "sut", ["first", "second"]).tolist() == [
        1.0,
        3.0,
    ]
