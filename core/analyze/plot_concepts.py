#!/usr/bin/env python3
"""core/analyze/plot_concepts.py — the two figures the argument actually needs.

The paper's thesis is informational: on LIBERO-Goal the nouns carry all 3.32 bits
of task identity and the verb carries none conditional on them, so a policy that
ignores the verb is optimal rather than deficient. Every figure the paper had was
a success-rate bar chart, which cannot show that. These two can:

  fig_information.png  what each word class tells you about task identity, and
                       the partition over the ten tasks that produces it
  fig_removal.png      the removal test: damage tracks bits destroyed, not how
                       much text was edited
  fig_verbsplit.png    the suite-wide verb probe against its own two halves: the
                       pooled number is the average of a decisive effect and
                       seven non-events, which is how the effect stays hidden

Both read the generated CSVs rather than any typed-in number, and both apply the
SAME scene-fixed gate as make_tables.py — paper/qualitative_grid.png and the RQ3
figure disagreed with the tables about which models were reportable, which is the
kind of contradiction a reviewer finds before we do.

Usage:  python core/analyze/plot_concepts.py
"""
from __future__ import annotations

import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "paper"
SUITE = "libero_goal"

sys.path.insert(0, str(REPO_ROOT / "core" / "analyze"))
import instruction_information as ii  # noqa: E402  (verb_set/noun_set live there)

# dataviz reference palette, categorical slots 1-2 and the text tokens. Slots 1-3
# are the documented all-pairs-safe subset in both modes; we use two of them.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#b8b7b0"
# Neutral, for a bar that is an AGGREGATE of the other two rather than a third
# category. It is deliberately outside the categorical ramp: giving the pooled
# probe its own hue would imply it is a peer measurement, which is the reading
# the figure exists to argue against.
GREY = "#8a8983"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def scene_fixed_ok(model: str) -> bool:
    """The gate make_tables.py applies. A model that fails it is not plotted."""
    p = PAPER / f"scene_fixed_check_{model}_{SUITE}.json"
    if not p.exists():
        return False
    return bool(json.loads(p.read_text()).get("pass"))


def read_csv(name: str) -> list[dict]:
    with (PAPER / name).open() as f:
        return list(csv.DictReader(f))


def originals() -> dict[int, str]:
    rows = [json.loads(l) for l in (REPO_ROOT / "data" / "instructions"
                                    / f"{SUITE}.jsonl").open() if l.strip()]
    return {r["task_id"]: r["instruction"] for r in rows if r["condition"] == "original"}


def token_edit_distance(a: str, b: str) -> int:
    """Word-level Levenshtein. Plain token-position diffing overstates a deletion:
    dropping the leading verb shifts every later word, scoring 7 changes for what
    is one removed token."""
    x, y = a.split(), b.split()
    prev = list(range(len(y) + 1))
    for i, xi in enumerate(x, 1):
        cur = [i]
        for j, yj in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (xi != yj)))
        prev = cur
    return prev[-1]


def fig_information() -> None:
    """Panel A: residual uncertainty. Panel B: why — the induced partition."""
    inst = originals()
    verb_groups: dict[tuple, list[int]] = collections.defaultdict(list)
    noun_groups: dict[tuple, list[int]] = collections.defaultdict(list)
    for t, s in sorted(inst.items()):
        verb_groups[tuple(sorted(ii.verb_set(s)))].append(t)
        noun_groups[tuple(sorted(ii.noun_set(s)))].append(t)

    n = len(inst)
    import math
    h_task = math.log2(n)
    h_given_verbs = sum(len(g) / n * math.log2(len(g)) for g in verb_groups.values())
    h_given_nouns = sum(len(g) / n * math.log2(len(g)) for g in noun_groups.values())

    # STACKED, not side by side: as a full-width figure* this reserved a two-column
    # float band and left the page around it about 40% empty, which cost a page we
    # did not have against the ACL limit. One column, two rows, same content.
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(3.34, 3.05), gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.75})

    # --- Panel A: one series, so one hue and no legend box.
    labels = ["no instruction\n$H(\\mathrm{task})$",
              "given the verb set\n$H(\\mathrm{task}\\mid V)$",
              "given the noun set\n$H(\\mathrm{task}\\mid N)$"]
    vals = [h_task, h_given_verbs, h_given_nouns]
    ypos = range(len(vals))
    ax_a.barh(ypos, vals, height=0.52, color=BLUE, zorder=3)
    for i, v in enumerate(vals):
        # A zero bar has no mark to read, so the value must be written.
        ax_a.text(v + 0.09, i, f"{v:.2f}", va="center", ha="left",
                  fontsize=7.2, fontweight="bold", color=INK)
    ax_a.set_yticks(list(ypos), labels, fontsize=6.8)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 3.95)
    ax_a.set_xlabel("bits of task identity still unresolved", fontsize=7)
    ax_a.xaxis.grid(True, color=MUTED, lw=0.4, alpha=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.set_title("(a) what each word class resolves", fontsize=7.2,
                   fontweight="bold", color=INK, loc="left", pad=4)

    # --- Panel B: the partition that produces those numbers.
    rows = [("by noun set", sorted(noun_groups.items(), key=lambda kv: kv[1][0]), BLUE),
            ("by verb set", sorted(verb_groups.items(), key=lambda kv: -len(kv[1])), ORANGE)]
    gap = 0.055  # the 2px surface gap between adjacent fills, in data units
    for r, (row_label, groups, color) in enumerate(rows):
        y = 1 - r
        x = 0.0
        for key, tasks in groups:
            w = len(tasks)
            ax_b.add_patch(Rectangle((x + gap / 2, y - 0.36), w - gap, 0.72,
                                     facecolor=color, edgecolor="none", zorder=3))
            if w > 1:  # only the collapsed group gets a direct label; the rest would collide
                ax_b.text(x + w / 2, y, f"“{'+'.join(key)}” {w}/{n}",
                          ha="center", va="center", fontsize=6.0,
                          fontweight="bold", color="white", zorder=4)
            x += w
        ax_b.text(-0.25, y, row_label, ha="right", va="center", fontsize=7, color=INK_2)
        ax_b.text(10.25, y, f"{len(groups)} group{'s' if len(groups) > 1 else ''}",
                  ha="left", va="center", fontsize=7, fontweight="bold", color=INK)
    ax_b.set_xlim(-4.2, 13.6)
    ax_b.set_ylim(-1.95, 1.7)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for s in ax_b.spines.values():
        s.set_visible(False)
    ax_b.set_title("(b) the partition each induces over the ten tasks", fontsize=7.2,
                   fontweight="bold", color=INK, loc="left", pad=4)
    ax_b.text(5.0, -1.15, "each cell is one task; a wide cell is a group\nthe word "
                          "class cannot tell apart",
              ha="center", va="center", fontsize=6.2, color=INK_2, style="italic",
              linespacing=1.3)

    out = PAPER / "fig_information.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig_removal() -> None:
    """Damage tracks bits destroyed, not how much text changed."""
    abl = [r for r in read_csv("rq_ablation.csv") if scene_fixed_ok(r["model"])]
    if not abl:
        raise SystemExit("no model passes the scene-fixed gate; nothing to plot")
    model = "openvla_oft" if any(r["model"] == "openvla_oft" for r in abl) else abl[0]["model"]
    rows = [r for r in abl if r["model"] == model and r["condition"] != "original"]
    if not rows:
        raise SystemExit(f"no ablation rows for {model}")

    inst = originals()
    per_cond = {}
    for r in [json.loads(l) for l in (REPO_ROOT / "data" / "instructions"
                                      / f"{SUITE}.jsonl").open() if l.strip()]:
        per_cond.setdefault(r["condition"], {})[r["task_id"]] = r["instruction"]

    def words_changed(cond: str) -> float:
        got = per_cond.get(cond, {})
        d = [token_edit_distance(inst[t], got[t]) for t in inst if t in got]
        return sum(d) / len(d) if d else float("nan")

    rows.sort(key=lambda r: float(r["delta_pp_matched"]))
    conds = [r["condition"] for r in rows]
    deltas = [-float(r["delta_pp_matched"]) for r in rows]  # as a signed drop
    bits = [float(r["bits_removed"]) for r in rows]
    colors = [BLUE if b == 0 else ORANGE for b in bits]

    fig, ax = plt.subplots(figsize=(3.34, 2.05))
    ypos = range(len(rows))
    ax.barh(ypos, deltas, height=0.56, color=colors, zorder=3)
    for i, d in enumerate(deltas):
        # Bars run leftward from 0, so "inside the bar" is d < x < 0. Putting the
        # label at d - 1.8 lands it on the surface, where white ink is invisible.
        inside = d < -20
        ax.text(d + 2.0, i, f"{d:.1f} pp", va="center", ha="left",
                fontsize=7, fontweight="bold",
                color="white" if inside else INK, zorder=4)
    ax.set_yticks(list(ypos),
                  [f"\\texttt{{{c}}}".replace("\\texttt{", "").replace("}", "")
                   + f"\n{words_changed(c):.1f} words changed" for c in conds],
                  fontsize=6.8)
    ax.set_xlim(min(deltas) * 1.10, 16)  # headroom for the smallest bar's label
    ax.axvline(0, color=INK_2, lw=0.7, zorder=2)
    ax.set_xlabel("change in task success rate (pp)", fontsize=7)
    ax.xaxis.grid(True, color=MUTED, lw=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [Rectangle((0, 0), 1, 1, facecolor=BLUE),
               Rectangle((0, 0), 1, 1, facecolor=ORANGE)]
    ax.legend(handles, ["0.00 bits removed", "3.32 bits removed"],
              fontsize=6.6, loc="upper left", bbox_to_anchor=(0.02, -0.30),
              ncol=2, frameon=False, handlelength=1.1,
              handleheight=0.85, borderpad=0.2, labelcolor=INK_2)
    ax.set_title("removal cost tracks information, not edit size", fontsize=7.6,
                 fontweight="bold", color=INK, loc="left", pad=6)

    out = PAPER / "fig_removal.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (model={model})")


MODEL_LABEL = {"smolvla": "SmolVLA\n0.45B",
               "openvla": "OpenVLA\n7B",
               "openvla_oft": "OpenVLA-OFT\n7.5B"}
MODEL_ORDER = ["smolvla", "openvla", "openvla_oft"]


def _p_label(p: float) -> str:
    """Match the prose: an exact p where it is readable, a bound where it is not."""
    import math
    if p < 0.001:
        return f"$p<10^{{{math.ceil(math.log10(p)):d}}}$"
    if p < 0.01:
        return f"$p={p:.3f}$"
    return f"$p={p:.2f}$"


def fig_verbsplit() -> None:
    """Why the standard verb probe reports nothing on a suite where verbs matter.

    The pooled `wrong_action` number is an n-weighted average of the three antonym
    tasks (which change the goal) and the seven near-synonym tasks (which do not).
    Plotting all three side by side is the whole argument of Section 4.6: the
    aggregate sits between two findings of opposite sign, so a study that reports
    only the aggregate cannot see either.
    """
    split = [r for r in read_csv("instruction_information.csv")
             if r["section"] == "wrong_action_split" and scene_fixed_ok(r["model"])]
    if not split:
        raise SystemExit("no wrong_action_split rows pass the scene-fixed gate")

    by_model: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in split:
        by_model[r["model"]][r["subset"]] = r
    models = [m for m in MODEL_ORDER if {"antonym", "near-syn"} <= set(by_model.get(m, {}))]
    if not models:
        raise SystemExit("no model has both halves of the verb split")

    # The pooled bar is DERIVED here (n-weighted over the two halves) rather than
    # read from rq1_causal.csv, so the three bars are guaranteed to be the same
    # arithmetic. It is then checked against the table's own number: if the figure
    # and Table 3 ever disagree about wrong_action, this raises instead of shipping
    # a plot that contradicts the text.
    causal = {r["model"]: r for r in read_csv("rq1_causal.csv")
              if r["condition"] == "wrong_action"}

    rows: list[tuple] = []          # (y, value, ci, color, hollow, label, p)
    yticks: list[tuple[float, str]] = []
    group_mid: dict[str, float] = {}
    y = 0.0
    for m in models:
        ant, syn = by_model[m]["antonym"], by_model[m]["near-syn"]
        n_a, n_s = int(ant["n"]), int(syn["n"])
        pooled = -(n_a * float(ant["delta_pp"]) + n_s * float(syn["delta_pp"])) / (n_a + n_s)
        if m in causal:
            expect = -float(causal[m]["delta_TSR_pp"])
            if abs(pooled - expect) > 0.15:
                raise SystemExit(
                    f"{m}: pooled split {pooled:.2f} pp disagrees with rq1_causal.csv "
                    f"{expect:.2f} pp — the figure and the table are not the same data")

        # n is a property of the subset, not of the model (3 antonym tasks and 7
        # near-synonym tasks, 20 episodes each), so it goes in the caption once
        # rather than onto nine tick labels -- that width is better spent on bars.
        entries = [
            ("suite-wide probe", pooled, None, GREY, False, None),
            ("antonym", -float(ant["delta_pp"]),
             (float(ant["ci_lo"]), float(ant["ci_hi"])), ORANGE,
             float(ant["mcnemar_p"]) >= 0.05, float(ant["mcnemar_p"])),
            ("near-synonym", -float(syn["delta_pp"]),
             (float(syn["ci_lo"]), float(syn["ci_hi"])), BLUE,
             float(syn["mcnemar_p"]) >= 0.05, float(syn["mcnemar_p"])),
        ]
        for label, val, ci, color, hollow, p in entries:
            rows.append((y, val, ci, color, hollow, p))
            yticks.append((y, label))
            y += 1.0
        group_mid[m] = y - 2.0
        y += 0.85  # the gap that makes the three models read as three groups

    # Text columns sit right of the zero line. The whisker cap on a short bar
    # reaches a few pp past it, so the delta column starts clear of the widest of
    # them rather than immediately at zero.
    X_DELTA, X_P = 8.0, 22.0

    fig, ax = plt.subplots(figsize=(3.34, 2.75))
    for yy, val, ci, color, hollow, p in rows:
        ax.barh(yy, val, height=0.62, zorder=3,
                facecolor="white" if hollow else color,
                edgecolor=color, linewidth=0.9 if hollow else 0)
        if ci is not None:
            ax.errorbar(val, yy, xerr=[[val - ci[0]], [ci[1] - val]], fmt="none",
                        ecolor=INK_2, elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=4)
        ax.text(X_DELTA, yy, f"{val:+.1f}", va="center", ha="left",
                fontsize=6.8, fontweight="bold", color=INK, zorder=4)
        if p is not None:
            ax.text(X_P, yy, _p_label(p), va="center", ha="left",
                    fontsize=6.4, color=INK_2 if p < 0.05 else MUTED, zorder=4)

    ax.set_yticks([t[0] for t in yticks], [t[1] for t in yticks], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(-50, 45)
    ax.set_ylim(rows[-1][0] + 0.85, -1.5)
    ax.axvline(0, color=INK_2, lw=0.7, zorder=2)
    ax.set_xticks([-40, -30, -20, -10, 0])
    ax.set_xlabel("change in task success rate (pp)", fontsize=7)
    ax.xaxis.grid(True, color=MUTED, lw=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for m, mid in group_mid.items():
        ax.text(-0.40, mid, MODEL_LABEL[m], transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=7, fontweight="bold",
                color=INK, linespacing=1.35)

    ax.text(X_DELTA, -1.25, "$\\Delta$TSR", fontsize=6.4, color=INK_2,
            ha="left", va="center")
    ax.text(X_P, -1.25, "McNemar", fontsize=6.4, color=INK_2, ha="left", va="center")
    ax.set_title("the suite-wide verb probe averages a real\neffect against seven non-events",
                 fontsize=7.6, fontweight="bold", color=INK, loc="left", pad=5)
    # Identity is carried by the tick labels, so the only thing left to explain is
    # the hollow fill -- which is the significance encoding, not a fourth series.
    # It goes UNDER the axis: above it, it collided with the two-line title.
    ax.text(0.0, -0.175, "hollow = not distinguishable from no edit ($p\\geq0.05$)",
            transform=ax.transAxes, fontsize=6.2, color=INK_2, style="italic",
            ha="left", va="top")

    out = PAPER / "fig_verbsplit.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  ({len(models)} models)")


if __name__ == "__main__":
    fig_information()
    fig_removal()
    fig_verbsplit()
