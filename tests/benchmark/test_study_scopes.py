from collections import Counter

from controllers.highway.catalog import BY_ID
from benchmark.studies import FAMILIES, SCOPES, tasks


def test_strict_holdouts_exclude_the_family_from_sources_and_development():
    for family in FAMILIES:
        scope = SCOPES[family]
        assert len(scope.history_ids) == 6
        assert len(scope.development_ids) == 3
        assert all(BY_ID[name].family != family for name in (*scope.history_ids, *scope.development_ids))


def test_source_swap_changes_one_group_with_equal_records_and_common_development():
    for target_family in FAMILIES:
        for alternative in FAMILIES:
            if target_family == alternative:
                continue
            a = SCOPES[f"swap_{target_family}_{alternative}_{target_family}"]
            b = SCOPES[f"swap_{target_family}_{alternative}_{alternative}"]
            assert len(a.history_ids) == len(b.history_ids) == 6
            assert len(set(a.history_ids) & set(b.history_ids)) == 4
            assert a.development_ids == b.development_ids
            assert len(a.development_ids) == 2
            assert all(BY_ID[n].family == target_family for n in set(a.history_ids) - set(b.history_ids))
            assert all(BY_ID[n].family == alternative for n in set(b.history_ids) - set(a.history_ids))


def test_all_registered_targets_conditions_and_mechanisms_have_explicit_tasks():
    plan = tasks()
    counts = Counter(t.study for t in plan)
    assert counts == {"overall": 3840, "strict_family": 3840, "source_swap": 2880, "mechanisms": 320}
    assert len(set(plan)) == len(plan)
    assert {t.condition for t in plan if t.study == "overall"} == {"reuse", "new", "hard"}
    assert len({t.target_id for t in plan if t.study == "overall"}) == 8
