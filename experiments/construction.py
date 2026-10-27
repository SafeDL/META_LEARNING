"""One role-checked archive boundary for every historical method and baseline."""

import json

from controllers.highway.catalog import SUTS

from .executions import execution_contract, require_qualified
from .protocol import DEFAULT, TEST_RECORDS_ROOT
from .scenarios import sample
from .source_registration import specification_matches
from .storage import load_archive


def load_construction(kind, held_family=None):
    groups = {}
    for role in ("history", "development"):
        coordinates = sample(kind, DEFAULT.candidate_count, role)
        archives = [
            load_archive(p) for p in sorted((TEST_RECORDS_ROOT / role / kind).glob("*.npz"))
        ]
        expected = {s.sut_id: s for s in SUTS if s.role == role}
        if {a.source_id for a in archives} != set(expected) or len(archives) != len(expected):
            raise ValueError(
                "Construction requires exactly the registered 8 historical/4 development objects"
            )
        for archive in archives:
            sut = expected[archive.source_id]
            if (archive.role, archive.family, archive.lineage) != (role, sut.family, sut.lineage):
                raise ValueError("Archive family/lineage/role differs from the registered object")
            if not archive.provenance.get("qualified"):
                raise ValueError("Only independently qualified responses may build formal models")
            require_qualified(sut)
            contract = json.loads(json.dumps(execution_contract(sut, kind=kind)))
            if not specification_matches(archive.provenance["contract"], contract):
                raise ValueError(
                    "Archive controller version or policy weights differ from the registered object"
                )
            if archive.scenarios != coordinates:
                raise ValueError(
                    "Archive coordinates/count differ from the registered role-specific pool"
                )
        groups[role] = [a for a in archives if a.family != held_family]
    return groups["history"], groups["development"]
