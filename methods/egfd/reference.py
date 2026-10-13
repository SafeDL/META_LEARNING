"""Independent replay of the paid risk-only reference."""


class ReferenceReplay:
    def __init__(self, session):
        self.session = session
        self.order = []

    def head(self):
        if self.session.pending is None:
            return self.session.next_index()
        return int(self.session.pending)

    def advance(self, known):
        while (index := self.head()) is not None and index in known:
            self.session.observe(known[index][0])
            self.order.append(index)

    def certificate(self, known):
        extra = [index for index in known if self.session.remaining[index]]
        negative = [index for index in extra if not known[index][1]]
        return {"reference_prefix_length": len(self.order), "extra_indices": extra,
                "negative_extra_indices": negative, "negative_extra_count": len(negative),
                "discovery_gain_lower": -len(negative)}
