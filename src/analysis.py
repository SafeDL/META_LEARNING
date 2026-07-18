"""
Post-hoc analysis module.

Provides:
  - compute_ate()          : Average Treatment Effect with 95% bootstrap CI
  - maneuver_vs_return()   : H_maneuver ↔ episode-return Spearman correlation
  - diversity_analysis()   : DBSCAN trajectory clustering for policy diversity
  - load_results()         : Aggregate results.json files into a DataFrame
  - ate_criterion_passed() : Pre-registered ATE pass/fail check (>+10% AUC, 95% CI)

Pre-registered ATE criterion:
  ATE(MAML − joint-pretrain) > +10% return AUC, 95% CI excludes zero
"""

import glob
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.cluster import DBSCAN
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# ─── Result loading ───────────────────────────────────────────────────────────

def load_results(results_root: str) -> pd.DataFrame:
    """
    Walk *results_root*, read all results.json files, return a DataFrame.
    Expected path pattern:
      results_root/<algorithm>/<topology>_<init_mode>_seed<N>/results.json
    """
    EXCLUDE_DIRS = {"sanity", "ablation", "logs", "tables"}
    records = []
    for path in glob.glob(os.path.join(results_root, "**", "results.json"),
                          recursive=True):
        # Skip sanity / ablation / auxiliary dirs
        rel = os.path.relpath(path, results_root)
        top_dir = rel.split(os.sep)[0]
        if top_dir in EXCLUDE_DIRS:
            continue
        with open(path) as f:
            r = json.load(f)
        # Skip entries with no algorithm or None algorithm
        if not r.get("algorithm"):
            continue
        r["results_path"] = path
        records.append(r)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    # Only keep known algorithms
    df = df[df["algorithm"].isin(["ppo", "sac", "td3"])].reset_index(drop=True)
    return df


# ─── ATE computation ─────────────────────────────────────────────────────────

def compute_ate(
    treated: np.ndarray,
    control: np.ndarray,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute Average Treatment Effect = E[treated] − E[control].

    Bootstrap 95% CI (percentile method).

    Parameters
    ----------
    treated : 1-D array of AUC values for treated condition (e.g. MAML)
    control : 1-D array of AUC values for control condition (e.g. joint-pretrain)
    n_bootstrap : number of bootstrap resamples
    ci_level : confidence level
    seed : RNG seed for reproducibility

    Returns
    -------
    dict with keys: ate, ci_low, ci_high, p_value, significant
    """
    rng = np.random.default_rng(seed)
    ate = float(np.mean(treated) - np.mean(control))

    # Bootstrap
    boot_ates = []
    for _ in range(n_bootstrap):
        t_ = rng.choice(treated, size=len(treated), replace=True)
        c_ = rng.choice(control, size=len(control), replace=True)
        boot_ates.append(float(np.mean(t_) - np.mean(c_)))
    boot_ates = np.array(boot_ates)

    alpha   = 1 - ci_level
    ci_low  = float(np.percentile(boot_ates, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_ates, 100 * (1 - alpha / 2)))

    # One-sided p-value (H0: ATE <= 0)
    p_value = float(np.mean(boot_ates <= 0))

    return {
        "ate":        ate,
        "ci_low":     ci_low,
        "ci_high":    ci_high,
        "p_value":    p_value,
        "significant": bool(ci_low > 0),
    }


def ate_criterion_passed(ate_result: dict, threshold: float = 0.10) -> bool:
    """
    Pre-registered criterion:
      ATE > +threshold AND 95% CI excludes zero (ci_low > 0).
    """
    return ate_result["ate"] > threshold and ate_result["ci_low"] > 0


# ─── Maneuver vs return analysis (H_maneuver) ───────────────────────────────

def maneuver_vs_return(
    maneuver_entropies: np.ndarray,
    episode_returns: np.ndarray,
) -> dict:
    """
    Spearman correlation between maneuver entropy H_maneuver and episode return.

    High positive correlation → diverse maneuver repertoire correlates with
    higher adversarial return (rules out "generic aggression" confound).

    Parameters
    ----------
    maneuver_entropies : shape (N,) — per-episode maneuver entropy
    episode_returns    : shape (N,) — per-episode return

    Returns
    -------
    dict with: rho, p_value, n
    """
    assert len(maneuver_entropies) == len(episode_returns), "Length mismatch"
    rho, p = stats.spearmanr(maneuver_entropies, episode_returns)
    return {
        "rho":     float(rho),
        "p_value": float(p),
        "n":       len(maneuver_entropies),
    }


# ─── Diversity analysis (DBSCAN) ─────────────────────────────────────────────

def diversity_analysis(
    trajectory_features: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 3,
) -> dict:
    """
    Cluster trajectory features with DBSCAN.
    trajectory_features : shape (N, D) — one row per episode
                          e.g. mean(obs) or maneuver histogram per episode

    Returns
    -------
    dict with: n_clusters, noise_fraction, cluster_sizes, labels
    """
    if not _SKLEARN_OK:
        raise ImportError("scikit-learn is required for diversity_analysis()")

    db     = DBSCAN(eps=eps, min_samples=min_samples).fit(trajectory_features)
    labels = db.labels_
    n      = len(labels)
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_frac = float(np.sum(labels == -1) / max(n, 1))

    cluster_sizes = {}
    for lbl in set(labels):
        if lbl != -1:
            cluster_sizes[int(lbl)] = int(np.sum(labels == lbl))

    return {
        "n_clusters":      n_clusters,
        "noise_fraction":  noise_frac,
        "cluster_sizes":   cluster_sizes,
        "labels":          labels.tolist(),
    }


# ─── Full analysis pipeline ──────────────────────────────────────────────────

def run_full_analysis(
    results_root: str,
    out_dir: str,
    n_bootstrap: int = 10_000,
) -> dict:
    """
    Load all results, compute ATE table, print summary.
    Saves analysis_results.json to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = load_results(results_root)
    if df.empty:
        print("[Analysis] No results found.")
        return {}

    summary = {}

    # ATE: MAML vs random (per topology, per algorithm)
    for algo in df["algorithm"].unique() if "algorithm" in df.columns else []:
        for topo in df["topology"].unique():
            sub     = df[(df["topology"] == topo)]
            if "algorithm" in df.columns:
                sub = sub[sub["algorithm"] == algo]

            maml_auc  = sub[sub["init_mode"] == "maml"]["return_auc"].values
            joint_auc = sub[sub["init_mode"] == "joint"]["return_auc"].values
            rand_auc  = sub[sub["init_mode"] == "random"]["return_auc"].values

            key = f"{algo}_{topo}"
            summary[key] = {}

            if len(maml_auc) >= 2 and len(joint_auc) >= 2:
                ate_mj = compute_ate(maml_auc, joint_auc, n_bootstrap=n_bootstrap)
                summary[key]["maml_vs_joint"] = ate_mj
                passed = ate_criterion_passed(ate_mj)
                print(f"[ATE] {key}: MAML-vs-Joint ATE={ate_mj['ate']:.3f} "
                      f"CI=[{ate_mj['ci_low']:.3f}, {ate_mj['ci_high']:.3f}] "
                      f"{'PASS' if passed else 'FAIL'}")

            if len(maml_auc) >= 2 and len(rand_auc) >= 2:
                ate_mr = compute_ate(maml_auc, rand_auc, n_bootstrap=n_bootstrap)
                summary[key]["maml_vs_random"] = ate_mr

    out_path = os.path.join(out_dir, "analysis_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Analysis] Summary saved → {out_path}")
    return summary
