"""Aggregation and JSON scalar regressions for historical tree reporting."""

import json

from benchmark.discovery_tree_screening_report import mean_by_lineage


def test_equal_lineages_and_missing_recall():
    rows = [{'lineage': 'a', 'recall': 1.}, {'lineage': 'a', 'recall': .5},
            {'lineage': 'a', 'recall': None}, {'lineage': 'b', 'recall': .25}]
    assert mean_by_lineage(rows, 'recall') == .5
    assert mean_by_lineage([{'lineage': 'a', 'recall': None}], 'recall') is None


def test_summary_comparisons_have_json_native_types():
    value = mean_by_lineage([{'lineage': 'a', 'value': .75}, {'lineage': 'b', 'value': 1.}], 'value')
    encoded = json.dumps({'mean': value, 'higher': value > .8}, allow_nan=False)
    assert json.loads(encoded) == {'mean': .875, 'higher': True}
