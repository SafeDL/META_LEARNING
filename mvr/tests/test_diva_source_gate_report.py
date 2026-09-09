from __future__ import annotations

import json

from mvr.scripts.report_diva_source_gate import run


def test_source_gate_stops_before_target_when_every_domain_fails_g1(tmp_path) -> None:
    analyses = []
    observations = []
    for domain in ("first", "second"):
        analysis_path = tmp_path / f"{domain}_analysis.json"
        analysis_path.write_text(
            json.dumps(
                {
                    "logical_domain_id": domain,
                    "summary": {
                        "observations": 4,
                        "formal_valid_rate": 1.0,
                        "posterior_eligible_rate": 0.5,
                        "event_rate": 0.5,
                        "common_eligible_anchors": 1,
                    },
                    "g1_source_viability": {"pass": False},
                    "headroom": [{"gain": 0.0}, {"gain": 0.0}],
                }
            ),
            encoding="utf-8",
        )
        observation_path = tmp_path / f"{domain}.jsonl"
        observation_path.write_text(
            "\n".join(json.dumps({"elapsed_seconds": value}) for value in (1.0, 2.0)) + "\n",
            encoding="utf-8",
        )
        analyses.append(str(analysis_path))
        observations.append(str(observation_path))

    report = run(analyses, observations, str(tmp_path / "gate.json"))

    assert report["physical_calls"] == 8
    assert [row["logical_domain_id"] for row in report["domains"]] == ["first", "second"]
    assert report["selected_domain"] is None
    assert report["decision"] == "stop_before_source_expansion_and_target_evaluation"
