"""Stable method names used in comparison tables and figures.

Recorded experiment IDs are retained so that existing trajectories keep their
original provenance. Reports resolve them to these method names.
"""

NAMES = {
    "gcrt": "GCRT",
    "lrt": "LRT",
    "dart": "DART",
    "cg_brt": "CG-BRT",
    "cg_cbm_fd": "CG-CBM-FD",
    "frozen_original": "HGT",
    "previous_best": "HR-GP",
    "matched_collision_gp": "Matched Collision GP",
    "risk_conditioned": "Risk-Conditioned Response GP",
    "behavior_posterior": "Behavior Posterior",
    "ras_frt_uq": "RAS-FRT-UQ",
    "gp_ei": "GP-EI",
    "gp_ucb": "GP-UCB",
    "rf_bo": "RF-BO",
    "bop_elites": "BOP-Elites",
    "knn_history": "Historical kNN Ranking",
    "farthest_first": "Farthest-First",
    "uniform_random": "Uniform Random",
    "bas": "BAS",
    "fixed_mean": "Fixed-Mean GP",
    "bayesian_sources": "Gaussian Source Mixture",
    "mixture_scale": "Student-t Source Mixture",
    "single_scale": "Student-t Fixed Mean",
    "event_signs_protected": "Protected Event-Sign Mixture",
    "student_protected": "Protected Student-t Mixture",
    "student_unprotected": "Student-t Mixture without Protection",
    "gaussian_unprotected": "Gaussian Mixture without Protection",
}

DART_VARIANTS = {
    "cg_brt": NAMES["cg_brt"],
    "full": NAMES["dart"],
    "physical_kernel": "DART: Physical Kernel",
    "conventional_training": "DART: Conventional Training",
    "greedy": "DART: Marginal Probability",
    "unprotected": "DART: No Loss Budget",
    "no_spatial_planning": "DART: No Spatial Lookahead",
}
