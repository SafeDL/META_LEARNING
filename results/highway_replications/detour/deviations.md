# Deviations from original DETOUR

DETOUR-Scenario-H replaces curvature-distance ranking with frozen Cut-in input features because this harness uses straight roads. Road curvature compression is independently tested and plotted. The failure oracle is collision (not lane departure). The no-known-failure case uses declared seeded-uniform fallback. Static selections do not update executed/failure counts.
