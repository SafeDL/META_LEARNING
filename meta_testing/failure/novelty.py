from __future__ import annotations

from dataclasses import dataclass, field

from .signature import FailureSignature


@dataclass
class NoveltyTracker:
    seen: set[str] = field(default_factory=set)

    def observe(self, signature: FailureSignature) -> bool:
        if not signature.valid:
            return False
        novel = signature.signature_id not in self.seen
        self.seen.add(signature.signature_id)
        return novel
