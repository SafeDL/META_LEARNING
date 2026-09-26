from metadrive_sim_env.evaluation.fewshot_inner import summarize_outcomes, valid_critical_score


def test_valid_critical_score_uses_the_formal_outcome_contract() -> None:
    assert valid_critical_score({"is_valid_episode": True, "valid_target_collision": True}) == 1.0
    assert valid_critical_score({
        "is_valid_episode": True,
        "valid_critical_near_miss": True
    }) == 0.5
    assert valid_critical_score({"is_valid_episode": False, "valid_target_collision": True}) == 0.0


def test_summary_counts_invalid_episodes_without_awarding_them_score() -> None:
    report = summarize_outcomes([
        {
            "is_valid_episode": True,
            "valid_target_collision": True
        },
        {
            "is_valid_episode": False,
            "valid_target_collision": True
        },
    ])
    assert report == {
        "episodes": 2.0,
        "valid_critical_score_mean": 0.5,
        "cumulative_valid_critical_score": 1.0,
        "invalid_rate": 0.5,
    }
