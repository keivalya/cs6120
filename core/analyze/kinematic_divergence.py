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
        # How many pairs actually contribute to each step of that mean. Pairs are
        # truncated to min(len(orig), len(pert)), and the median pair is only
        # ~10-12 steps long, so the tail of mean_curve is an average over a
        # shrinking and NON-RANDOM subset: the episodes that ran longest, which
        # are disproportionately the failures. Reporting the last value of the
        # curve beside the full pair count -- which is what `final_error_m` did --
        # credits a 4-episode number to n=55.
        support = np.count_nonzero(~np.isnan(arr), axis=0)
        # Every counted pair has min_len >= 10 (enforced above), so step index 9 is
        # the last horizon supported by 100% of the pairs. That makes e10 the one
        # displacement number the reported n genuinely backs.
        E_STEP = 9
        half = np.flatnonzero(support >= max(1, 0.5 * len(curves)))
        summary[cond] = {
            "mean_curve": mean_curve,
            "std_curve": std_curve,
            "mean_tdiv": mean_tdiv,
            "n": len(curves),
            "support": support,
            "e10": float(mean_curve[E_STEP]),
            "n_e10": int(support[E_STEP]),
            # Last step at which at least half the pairs still contribute; the
            # curve is drawn only this far.
            "t_half": int(half[-1]) if len(half) else 0,
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
    lines = ["model,condition,n_pairs,mean_tdiv_step,e10_m,n_at_e10,t_half_support"]
    for model, msummary in all_summary.items():
        for cond, data in msummary.items():
            lines.append(f"{model},{cond},{data['n']},{data['mean_tdiv']:.2f},"
                         f"{data['e10']:.4f},{data['n_e10']},{data['t_half']}")
    csv_file.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_file}")

    # ---- Plot -------------------------------------------------------------
    # Only the three conditions Section 4.7 actually argues about. The previous
    # version drew six, chosen by whatever happened to be in a `colors` dict --
    # which silently EXCLUDED `nonsense` even though the prose names it, and
    # included the para_* conditions, which exist for smolvla and openvla_oft but
    # not openvla, so the three panels were not comparable. Same conditions in
    # every panel, and every condition in the text.
    CONDS = [("blank", "#2a78d6"),        # categorical slot 1
             ("nonsense", "#1baf7a"),     # slot 3 -- slots 1-3 validate all-pairs
             ("wrong_action", "#eb6834")]  # slot 2, the one with the shape
    INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    models = ["smolvla", "openvla", "openvla_oft"]
    titles = {"smolvla": "SmolVLA (0.45B)", "openvla": "OpenVLA (7B)",
              "openvla_oft": "OpenVLA-OFT (7.5B)"}
    present = [m for m in models if m in all_summary]

    fig, axes = plt.subplots(len(present), 1, figsize=(3.34, 3.7),
                             sharex=True, sharey=True)
    if len(present) == 1:
        axes = [axes]

    for ax, model in zip(axes, present):
        msummary = all_summary[model]
        for cond, color in CONDS:
            data = msummary.get(cond)
            if data is None:
                continue
            # Draw only as far as half the pairs reach. Past that the mean is an
            # average over the longest-running episodes and the curve says more
            # about episode length than about divergence.
            stop = data["t_half"] + 1
            curve = np.asarray(data["mean_curve"], dtype=float)[:stop]
            steps = np.arange(len(curve))
            ax.plot(steps, curve, color=color, linewidth=1.4, zorder=3,
                    solid_capstyle="round")
            # t_div is the number the table reports; show WHERE it falls on the
            # curve rather than printing it in a legend divorced from the shape.
            t = data["mean_tdiv"]
            if np.isfinite(t) and 0 <= int(round(t)) < len(curve):  # noqa: E501
                ax.plot(t, curve[int(round(t))], "o", color=color, markersize=3.2,
                        markeredgecolor="white", markeredgewidth=0.7, zorder=4)
        ax.set_title(titles[model], fontsize=7, fontweight="bold", color=INK,
                     loc="left", pad=3)
        ax.yaxis.grid(True, color=MUTED, lw=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("rollout timestep $t$", fontsize=7)
    axes[-1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # t is a step index
    axes[len(present) // 2].set_ylabel("end-effector divergence $e(t)$ [m]", fontsize=7)

    handles = [plt.Line2D([0], [0], color=c, lw=1.4) for _, c in CONDS]
    axes[0].legend(handles, [c.replace("_", r"\_") and c for c, _ in CONDS],
                   fontsize=6.4, ncol=3, frameon=False, loc="lower left",
                   bbox_to_anchor=(0.0, 1.22), handlelength=1.4,
                   columnspacing=1.2, labelcolor=INK_2)
    # Anchored to the bottom axes, not the figure: under `bbox="tight"` a negative
    # figure-fraction y is measured against a canvas that is about to be cropped,
    # which stranded this line an inch below the x-label.
    axes[-1].text(0.5, -0.60, "dot marks $t_{\\mathrm{div}}$; curves stop where "
                  "half the pairs have ended",
                  transform=axes[-1].transAxes, fontsize=6.2, color=INK_2,
                  style="italic", ha="center", va="top")

    fig.subplots_adjust(hspace=0.45)
    plot_file = REPO_ROOT / "paper" / "rq3_divergence.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"wrote {plot_file}")

if __name__ == "__main__":
    main()
