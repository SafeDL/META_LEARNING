#!/usr/bin/env python
"""
Aggregate all completed experiment results and produce paper-ready tables.

Outputs:
  results/tables/main_table.csv        — AUC return by (algo, topology, init_mode)
  results/tables/ate_table.csv         — ATE + CI for each (algo, topology) cell
  results/tables/ablation_table.csv    — R060/R061/R062 comparison
  results/tables/summary.json          — full structured summary

Usage:
  python scripts/compute_results.py --results_root results
  python scripts/compute_results.py --results_root results --latex  # also print LaTeX tables
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import load_results, compute_ate, ate_criterion_passed

ALGORITHMS  = ["ppo", "sac", "td3"]
TOPOLOGIES  = ["roundabout", "y_junction"]       # held-out test topologies (B1)
TRAIN_TOPOS = ["highway", "merge", "t_junction", "intersection"]
INIT_MODES  = ["maml", "joint", "random"]


def build_main_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build mean ± std AUC return table:
      rows    = (algorithm, init_mode)
      columns = topology
    """
    records = []
    algos   = [a for a in (df["algorithm"].unique() if "algorithm" in df.columns else ["ppo"])
               if a in ("ppo", "sac", "td3")]
    for algo in sorted(algos):
        for init_mode in INIT_MODES:
            row = {"algorithm": algo, "init_mode": init_mode}
            for topo in (TOPOLOGIES + TRAIN_TOPOS):
                mask = (df["topology"] == topo) & (df["init_mode"] == init_mode)
                if "algorithm" in df.columns:
                    mask &= (df["algorithm"] == algo)
                vals = df[mask]["return_auc"].values
                if len(vals) > 0:
                    row[topo] = f"{np.mean(vals):.2f}±{np.std(vals):.2f}"
                else:
                    row[topo] = "—"
            records.append(row)
    return pd.DataFrame(records)


def build_ate_table(df: pd.DataFrame) -> pd.DataFrame:
    """ATE(MAML−joint) table for paper."""
    records = []
    algos   = [a for a in (df["algorithm"].unique() if "algorithm" in df.columns else ["ppo"])
               if a in ("ppo", "sac", "td3")]
    for algo in sorted(algos):
        for topo in (TOPOLOGIES + TRAIN_TOPOS):
            mask = (df["topology"] == topo)
            if "algorithm" in df.columns:
                mask &= (df["algorithm"] == algo)
            maml  = df[mask & (df["init_mode"] == "maml")]["return_auc"].values
            joint = df[mask & (df["init_mode"] == "joint")]["return_auc"].values
            rand  = df[mask & (df["init_mode"] == "random")]["return_auc"].values

            if len(maml) >= 2 and len(joint) >= 2:
                ate = compute_ate(maml, joint)
                ok  = ate_criterion_passed(ate)
                records.append({
                    "algorithm":  algo,
                    "topology":   topo,
                    "ATE_mj":     round(ate["ate"], 3),
                    "CI_low":     round(ate["ci_low"], 3),
                    "CI_high":    round(ate["ci_high"], 3),
                    "p_value":    round(ate["p_value"], 4),
                    "criterion":  "PASS" if ok else "FAIL",
                    "n_maml":     len(maml),
                    "n_joint":    len(joint),
                })
    return pd.DataFrame(records)


def build_ablation_table(results_root: str) -> pd.DataFrame:
    """R060/R061/R062 comparison."""
    records = []
    ablation_dir = os.path.join(results_root, "ablation")
    if not os.path.exists(ablation_dir):
        return pd.DataFrame()

    variant_map = {
        "r060": "encoder-only (proposed)",
        "r061": "full-network transfer",
        "r062": "frozen encoder",
    }
    for prefix, label in variant_map.items():
        aucs = []
        for name in os.listdir(ablation_dir):
            if not name.startswith(prefix):
                continue
            res_path = os.path.join(ablation_dir, name, "results.json")
            if os.path.exists(res_path):
                with open(res_path) as f:
                    r = json.load(f)
                aucs.append(r.get("return_auc", np.nan))
        if aucs:
            records.append({
                "variant":   label,
                "mean_auc":  round(float(np.mean(aucs)), 3),
                "std_auc":   round(float(np.std(aucs)), 3),
                "n_seeds":   len(aucs),
            })
    return pd.DataFrame(records)


def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to LaTeX table string."""
    try:
        return df.to_latex(index=False, caption=caption, label=label,
                           escape=True, float_format="%.3f")
    except Exception:
        return str(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--out_dir",      default="results/tables")
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--latex",        action="store_true",
                        help="Print LaTeX table sources to stdout")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_results(args.results_root)
    if df.empty:
        print("[compute_results] No results.json files found in", args.results_root)
        sys.exit(0)

    print(f"[compute_results] Loaded {len(df)} runs from {args.results_root}")

    # ── Main table ─────────────────────────────────────────────────────────────
    main_tbl = build_main_table(df)
    main_path = os.path.join(args.out_dir, "main_table.csv")
    main_tbl.to_csv(main_path, index=False)
    print(f"\nMain AUC table ({len(main_tbl)} rows) → {main_path}")
    print(main_tbl.to_string(index=False))

    # ── ATE table ──────────────────────────────────────────────────────────────
    ate_tbl  = build_ate_table(df)
    ate_path = os.path.join(args.out_dir, "ate_table.csv")
    if not ate_tbl.empty:
        ate_tbl.to_csv(ate_path, index=False)
        print(f"\nATE table ({len(ate_tbl)} rows) → {ate_path}")
        print(ate_tbl.to_string(index=False))

        # Pre-registered criterion summary
        n_pass  = int((ate_tbl["criterion"] == "PASS").sum())
        n_total = len(ate_tbl)
        print(f"\nPre-registered ATE criterion: {n_pass}/{n_total} cells PASS")
    else:
        print("\n[ATE] Insufficient data for ATE table (need ≥2 seeds per cell).")

    # ── Ablation table ─────────────────────────────────────────────────────────
    abl_tbl  = build_ablation_table(args.results_root)
    abl_path = os.path.join(args.out_dir, "ablation_table.csv")
    if not abl_tbl.empty:
        abl_tbl.to_csv(abl_path, index=False)
        print(f"\nAblation table → {abl_path}")
        print(abl_tbl.to_string(index=False))

    # ── LaTeX output ───────────────────────────────────────────────────────────
    if args.latex:
        print("\n% ─── Main table (LaTeX) ─────────────────────────────────────────")
        print(df_to_latex(main_tbl, "Return AUC by algorithm and topology.",
                          "tab:main_results"))
        if not ate_tbl.empty:
            print("\n% ─── ATE table (LaTeX) ──────────────────────────────────────────")
            print(df_to_latex(ate_tbl, r"ATE(MAML$-$Joint-pretrain) with 95\% bootstrap CI.",
                              "tab:ate_results"))

    # ── Full JSON summary ──────────────────────────────────────────────────────
    summary = {
        "n_runs":           len(df),
        "algorithms":       list(df["algorithm"].unique()) if "algorithm" in df.columns else [],
        "topologies":       list(df["topology"].unique()),
        "init_modes":       list(df["init_mode"].unique()) if "init_mode" in df.columns else [],
        "ate_criterion_passed_cells": n_pass if not ate_tbl.empty else None,
        "ate_criterion_total_cells":  n_total if not ate_tbl.empty else None,
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary → {summary_path}")


if __name__ == "__main__":
    main()
