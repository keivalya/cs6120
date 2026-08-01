#!/usr/bin/env python3
"""core/analyze/kinematic_divergence.py — Kinematic trajectory divergence analysis.

Computes the Euclidean end-effector position error e(t) over time step t in [0, T]
between canonical (original) rollouts and perturbed rollouts (wrong_object, wrong_action,
para_object, para_action, para_compositional) sharing the exact same initial state S_0.

Outputs:
  - paper/rq3_divergence.csv
  - paper/rq3_divergence.png
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]

def load_trajectories(results_dir: Path):
    """Load eef_traj indexed by (task_id, episode) -> condition -> traj."""
    trajs = defaultdict(dict)
    for cond_dir in results_dir.iterdir():
        if not cond_dir.is_dir():
            continue
        cond = cond_dir.name
        for seed_dir in cond_dir.glob("seed*"):
            for jsonl_file in seed_dir.glob("*.jsonl"):
                if jsonl_file.name == "episodes.jsonl":
                    continue
                for line in jsonl_file.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line.replace("\x00", "").strip())
                    except Exception:
                        continue
                    key = (rec["task_id"], rec.get("reset_state_hash"))
                    if key[1] is None:
                        continue
                    if "eef_traj" in rec and rec["eef_traj"]:
                        trajs[key][cond] = np.array(rec["eef_traj"])
    return trajs

def compute_divergence(trajs):
    """Compute mean e(t) curves for each condition vs original."""
    cond_curves = defaultdict(list)
    cond_tdiv = defaultdict(list)

    for (tid, ep), cdict in trajs.items():
        if "original" not in cdict:
            continue
        orig = cdict["original"]
        for cond, pert in cdict.items():
            if cond == "original":
                continue
            min_len = min(len(orig), len(pert))
            if min_len < 10:
                continue
            err = np.linalg.norm(orig[:min_len] - pert[:min_len], axis=1)
            cond_curves[cond].append(err)
            
            # Divergence timestep: first step where error > 0.05 meters (5cm)
            above = np.where(err > 0.05)[0]
            if len(above) > 0:
                cond_tdiv[cond].append(above[0])
            else:
                cond_tdiv[cond].append(min_len)

    summary = {}
    for cond, curves in cond_curves.items():
        # Truncate / pad to common length 300
        max_len = 300
        arr = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            l = min(len(c), max_len)
            arr[i, :l] = c[:l]
        
        mean_curve = np.nanmean(arr, axis=0)
        std_curve = np.nanstd(arr, axis=0)
        mean_tdiv = np.mean(cond_tdiv[cond]) if cond_tdiv[cond] else 300
        summary[cond] = {
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "mean_tdiv": mean_tdiv,
            "n": len(curves)
        }
    return summary

def main():
    results_root = REPO_ROOT / "data" / "results"
    all_summary = {}

    for model in ["smolvla", "openvla", "openvla_oft"]:
        mdir = results_root / model / "libero_goal"
        if not mdir.exists():
            continue
        trajs = load_trajectories(mdir)
        summary = compute_divergence(trajs)
        all_summary[model] = summary

    # Write CSV
    csv_file = REPO_ROOT / "paper" / "rq3_divergence.csv"
    lines = ["model,condition,n_pairs,mean_tdiv_step,final_error_m"]
    for model, msummary in all_summary.items():
        for cond, data in msummary.items():
            final_err = data["mean_curve"][~np.isnan(data["mean_curve"])][-1] if len(data["mean_curve"]) > 0 else 0.0
            lines.append(f"{model},{cond},{data['n']},{data['mean_tdiv']:.2f},{final_err:.4f}")
    csv_file.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_file}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = ["smolvla", "openvla", "openvla_oft"]
    titles = ["SmolVLA (500M)", "OpenVLA (7B)", "OpenVLA-OFT (7.5B)"]

    colors = {
        "wrong_object": "#d62728",
        "wrong_action": "#ff7f0e",
        "para_object": "#1f77b4",
        "para_action": "#2ca02c",
        "para_compositional": "#9467bd",
        "blank": "#7f7f7f"
    }

    for idx, model in enumerate(models):
        ax = axes[idx]
        ax.set_title(titles[idx], fontsize=14, fontweight="bold")
        if model in all_summary:
            msummary = all_summary[model]
            for cond, data in msummary.items():
                if cond not in colors:
                    continue
                curve = data["mean_curve"]
                steps = np.arange(len(curve))
                ax.plot(steps, curve, label=f"{cond} (t_div={data['mean_tdiv']:.0f})", color=colors[cond], linewidth=2)
        ax.set_xlabel("Rollout Timestep t", fontsize=12)
        if idx == 0:
            ax.set_ylabel("EEF Divergence Error e(t) [meters]", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plot_file = REPO_ROOT / "paper" / "rq3_divergence.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"wrote {plot_file}")

if __name__ == "__main__":
    main()
