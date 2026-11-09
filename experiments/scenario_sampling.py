"""Deterministic product-domain designs without ADS identities or test responses."""

from scipy.stats import qmc

from .schema import Scenario


def sample_candidates(kind, domain, *, count, seed, namespace, feasible, max_draws, existing=()):
    """Generate coordinates; the caller supplies declared geometry/motion constraints.

    Mutation quantization is not applied to the continuous design. Constraint
    exceptions stop preparation; they are never converted to safe responses.
    """
    if type(count) is not int or count < 1 or type(max_draws) is not int or max_draws < count:
        raise ValueError("A positive candidate count must fit the explicit draw limit")
    if type(seed) is not int or not namespace:
        raise ValueError("Candidate preparation requires an integer seed and an ID namespace")
    continuous = len(domain.bounds)
    sampler = qmc.Sobol(continuous + len(domain.categories), scramble=True, seed=seed)
    candidates, seen = [], {scene.coordinate() for scene in existing}
    duplicates, rejected = 0, 0
    for index in range(max_draws):
        point = sampler.random(1)[0]
        parameters = {
            name: float(lower + value * (upper - lower))
            for (name, (lower, upper)), value in zip(domain.bounds.items(), point[:continuous])
        }
        categories = {
            name: options[min(int(value * len(options)), len(options) - 1)]
            for (name, options), value in zip(domain.categories.items(), point[continuous:])
        }
        scene = Scenario(f"{kind}:{namespace}:{index:06d}", kind, parameters, categories)
        domain.validate(scene)
        coordinate = scene.coordinate()
        if coordinate in seen:
            duplicates += 1
            continue
        seen.add(coordinate)
        if not feasible(scene):
            rejected += 1
            continue
        candidates.append(scene)
        if len(candidates) == count:
            return candidates, {
                "method": "scrambled Sobol over independent parameter ranges",
                "scenario": kind,
                "namespace": namespace,
                "seed": seed,
                "requested_count": count,
                "accepted_count": len(candidates),
                "examined_points": index + 1,
                "geometry_rejections": rejected,
                "duplicate_coordinates": duplicates,
                "max_draws": max_draws,
                "definition": domain.to_dict(),
            }
    raise ValueError(
        f"Candidate preparation exhausted {max_draws} Sobol points: accepted={len(candidates)}/{count}, "
        f"geometry_rejections={rejected}, duplicates={duplicates}"
    )
