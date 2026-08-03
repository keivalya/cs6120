#!/usr/bin/env python3
"""core/analyze/make_paper_figures.py — Extract environment frames and generate paper figures.

Outputs:
  - paper/figures/fig_scene_overview.png
  - paper/figures/fig_qualitative_grid.png
"""
import os
import subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FFMPEG = "/opt/homebrew/bin/ffmpeg"

def extract_frame(video_path: Path, frame_idx: int, output_path: Path) -> bool:
    """Extract a single frame from video using ffmpeg."""
    cmd = [
        FFMPEG, "-y", "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vframes", "1", str(output_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return output_path.exists()

def extract_n_frames(video_path: Path, n_frames: int = 5) -> list:
    """Extract n_frames evenly spaced from MP4 video."""
    cmd = [
        "/opt/homebrew/bin/ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        total_packets = int(res.stdout.strip())
    except Exception:
        total_packets = 50
    
    indices = np.linspace(0, max(0, total_packets - 1), n_frames, dtype=int)
    frames = []
    tmp_dir = REPO_ROOT / "paper" / "tmp_frames"
    tmp_dir.mkdir(exist_ok=True)
    
    for i, idx in enumerate(indices):
        out_f = tmp_dir / f"frame_{i}.png"
        if extract_frame(video_path, idx, out_f):
            img = mpimg.imread(out_f)
            frames.append(img)
    return frames

def build_scene_overview_figure():
    """Create Fig 1: Initial state S_0 screenshot and Causal Isolation diagram."""
    v_dir = REPO_ROOT / "data" / "videos"
    sample_video = v_dir / "task7_turn_on_the_stove" / "original" / "success_ep1.mp4"
    if not sample_video.exists():
        sample_video = list(v_dir.glob("**/*.mp4"))[0]
    
    ref_img = REPO_ROOT / "paper" / "tmp_frames" / "frame_1.png"
    if ref_img.exists():
        s0_img = mpimg.imread(ref_img)
    else:
        tmp_img = REPO_ROOT / "paper" / "tmp_s0.png"
        extract_frame(sample_video, 10, tmp_img)
        s0_img = mpimg.imread(tmp_img)
    # Rotate 180 degrees as requested
    s0_img = np.rot90(s0_img, 2)

    
    fig = plt.figure(figsize=(10, 4.2), dpi=300)
    
    # Left subplot: Environment Initial State S_0
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(s0_img)
    ax1.set_title("(a) LIBERO-Goal Initial Visual State $S_0$", fontsize=11, fontweight="bold", pad=8)
    ax1.axis("off")


    # Right subplot: Scene-Fixed Causal Isolation Framework Diagram
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("(b) Scene-Fixed Causal Isolation Protocol", fontsize=11, fontweight="bold", pad=8)

    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    def draw_box(ax, x, y, w, h, text, title="", box_color="#f8fafc", edge_color="#64748b", text_color="#0f172a", fontsize=8.5, title_size=9):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.15",
                           ec=edge_color, fc=box_color, lw=1.5, zorder=2)
        ax.add_patch(p)
        if title:
            ax.text(x + w/2, y + h - 0.35, title, ha="center", va="top",
                    fontsize=title_size, fontweight="bold", color=text_color, zorder=3)
            ax.text(x + w/2, y + 0.3, text, ha="center", va="bottom",
                    fontsize=fontsize, color=text_color, zorder=3)
        else:
            ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color=text_color, zorder=3)

    def draw_arrow(ax, x1, y1, x2, y2, color="#475569", style="-|>", lw=1.5):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle=f"{style},head_length=5,head_width=3.5",
                                color=color, lw=lw, zorder=1)
        ax.add_patch(arrow)

    # 1. Root Node: Fixed State S_0
    draw_box(ax2, 0.2, 4.0, 2.4, 2.0, "Hashed qpos/qvel\nState Invariant",
             title="Fixed Scene $S_0$", box_color="#e0f2fe", edge_color="#0284c7", text_color="#0369a1")

    # 2. Middle Branch Nodes: Instruction Perturbations
    draw_box(ax2, 3.4, 7.8, 3.2, 1.4, "\"turn on the stove\"", title="Canonical ($I_{\\text{orig}}$)",
             box_color="#dcfce7", edge_color="#16a34a", text_color="#15803d")

    draw_box(ax2, 3.4, 5.4, 3.2, 1.4, "blank / nonsense", title="Destructive ($I_{\\text{dest}}$)",
             box_color="#fee2e2", edge_color="#dc2626", text_color="#b91c1c")

    draw_box(ax2, 3.4, 3.0, 3.2, 1.4, "wrong_action / task", title="Misleading ($I_{\\text{mis}}$)",
             box_color="#fef3c7", edge_color="#d97706", text_color="#b45309")

    draw_box(ax2, 3.4, 0.6, 3.2, 1.4, "object / action / comp.", title="Paraphrase ($I_{\\text{para}}$)",
             box_color="#f3e8ff", edge_color="#9333ea", text_color="#6b21a8")

    # 3. Right Branch Nodes: Target Metrics
    draw_box(ax2, 7.3, 7.8, 2.4, 1.4, "TSR Baseline", box_color="#f1f5f9", edge_color="#94a3b8", text_color="#334155", fontsize=8)
    draw_box(ax2, 7.3, 5.4, 2.4, 1.4, "CSS Metric", box_color="#f1f5f9", edge_color="#94a3b8", text_color="#334155", fontsize=8)
    draw_box(ax2, 7.3, 3.0, 2.4, 1.4, "OAR / Compliance", box_color="#f1f5f9", edge_color="#94a3b8", text_color="#334155", fontsize=8)
    draw_box(ax2, 7.3, 0.6, 2.4, 1.4, "PRIDE Score", box_color="#f1f5f9", edge_color="#94a3b8", text_color="#334155", fontsize=8)

    # 4. Arrows from S0 to Perturbations
    draw_arrow(ax2, 2.6, 5.5, 3.4, 8.5, color="#0284c7")
    draw_arrow(ax2, 2.6, 5.2, 3.4, 6.1, color="#0284c7")
    draw_arrow(ax2, 2.6, 4.8, 3.4, 3.7, color="#0284c7")
    draw_arrow(ax2, 2.6, 4.5, 3.4, 1.3, color="#0284c7")

    # 5. Arrows from Perturbations to Metrics
    draw_arrow(ax2, 6.6, 8.5, 7.3, 8.5, color="#16a34a")
    draw_arrow(ax2, 6.6, 6.1, 7.3, 6.1, color="#dc2626")
    draw_arrow(ax2, 6.6, 3.7, 7.3, 3.7, color="#d97706")
    draw_arrow(ax2, 6.6, 1.3, 7.3, 1.3, color="#9333ea")

    plt.tight_layout()
    out_file = REPO_ROOT / "paper" / "figures" / "fig_scene_overview.png"
    out_file.parent.mkdir(exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_file}")


def build_qualitative_grid_figure():
    """Create Fig: Qualitative temporal rollout comparison grid."""
    v_dir = REPO_ROOT / "data" / "videos"
    
    conditions = [
        ("Canonical Prompt (original)", v_dir / "task7_turn_on_the_stove" / "original" / "success_ep1.mp4"),
        ("Destructive Blank (blank)", v_dir / "task7_turn_on_the_stove" / "blank" / "failure_ep1.mp4"),
        ("Destructive Nonsense (nonsense)", v_dir / "task7_turn_on_the_stove" / "nonsense" / "failure_ep1.mp4"),
        ("Misleading Task (wrong_task)", v_dir / "task0_open_the_middle_drawer_of_the_cabinet" / "wrong_task" / "failure_ep1.mp4"),
    ]
    
    num_cols = 5
    fig, axes = plt.subplots(len(conditions), num_cols, figsize=(12, 8.5), dpi=300)
    
    for row_idx, (title, video_path) in enumerate(conditions):
        frames = extract_n_frames(video_path, num_cols) if video_path.exists() else []
        
        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(frames):
                frame_rot = np.rot90(frames[col_idx], 2)
                ax.imshow(frame_rot)
            else:
                ax.fill([0, 1, 1, 0], [0, 0, 1, 1], color="#f3f4f6")
            ax.axis("off")

            
            if row_idx == 0:
                step_t = int(col_idx * 75)
                ax.set_title(f"Step $t = {step_t}$", fontsize=10, fontweight="bold")
            if col_idx == 0:
                ax.text(-0.15, 0.5, title, transform=ax.transAxes,
                        fontsize=9.5, fontweight="bold", va="center", ha="right")

    plt.suptitle("Qualitative Rollout Snapshots Across Instruction Conditions", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0.18, 0.02, 0.98, 0.95])
    out_file = REPO_ROOT / "paper" / "figures" / "fig_qualitative_grid.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_file}")

def main():
    build_scene_overview_figure()
    build_qualitative_grid_figure()

if __name__ == "__main__":
    main()
