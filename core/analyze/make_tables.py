#!/usr/bin/env python3
"""core/analyze/make_tables.py — generate the paper's result tables FROM the CSVs.

WHY THIS EXISTS
---------------
paper/paper_acl.tex carried ~45 measured numbers transcribed by hand across 12
regions, with no \\input of any generated file. That is how it came to claim
"$N=40$ per condition" while paper/rq1_causal.csv says wrong_object is n=28 and
openvla/repeated is n=20 — and hand-transcription is the same mechanism that let
fabricated numbers survive review. Every table the paper shows is emitted here
from the CSV instead, so a stale number becomes impossible rather than unlikely.

Writes paper/tables/*.tex, each containing ONE tabular environment (no caption,
no label — those stay in paper_acl.tex where they can be edited):

    rq1_causal.tex        wide per-model view, the paper's current shape
    rq1_causal_full.tex   long form: every model x condition with n and 95% CI
    rq1_n_note.tex        one sentence stating the true per-condition n
    rq2_para.tex          paraphrase axes, WITH the n column
    rq3_divergence.tex    kinematic divergence, WITH the n_pairs column

Usage: python core/analyze/make_tables.py       (CPU only, reads paper/*.csv)
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "paper"
OUT = REPORT / "tables"

PRETTY = {"smolvla": "SmolVLA (0.45B)", "openvla": "OpenVLA (7B)",
          "openvla_oft": "OpenVLA-OFT (7.5B)"}
MODEL_ORDER = ["smolvla", "openvla", "openvla_oft"]


def scene_status(model: str, suite: str = "libero_goal") -> dict:
    """The per-model scene-fixed verdict, read from core/run/aggregate.py's output.

    The causal contrast is only interpretable where this passes, so the tables mark
    the models where it does not instead of presenting every column as equivalent.
    A missing file is treated as unverified, never as a pass.
    """
    import json
    p = REPORT / f"scene_fixed_check_{model}_{suite}.json"
    if not p.exists():
        return {"state": "unverified", "detail": "no check on record"}
    d = json.loads(p.read_text())
    comparable = d.get("n_keys_comparable", d.get("n_keys_checked", 0))
    vacuous = d.get("n_keys_vacuous", 0)
    if not d.get("pass"):
        return {"state": "failed",
                "detail": f"{d.get('n_mismatches', 0)} of {comparable} comparable "
                          f"episodes started from different scenes across conditions"}
    if vacuous > comparable:
        return {"state": "thin",
                "detail": f"passes on {comparable} comparable episodes, but {vacuous} "
                          f"further episodes exist in only one condition and are "
                          f"unchecked"}
    return {"state": "passed", "detail": f"{comparable} comparable episodes, 0 mismatches"}


MARK = {"failed": r"$^{\ddagger}$", "thin": r"$^{\dagger}$", "unverified": r"$^{\ddagger}$",
        "passed": ""}

# A cell we deliberately leave empty because the measurement is not trustworthy yet.
# Visually distinct from "--", which means "this cell has no data at all".
PENDING = r"\textemdash"
PENDING_ROW = r"\textit{(grid regenerating)}"

SHOW_UNVERIFIED = False  # --show-unverified flips this, for internal inspection only


def suppressed(model: str) -> bool:
    """True when a model's numbers must be WITHHELD from the paper, not merely marked.

    A footnote saying "confounded, we do not interpret this" does not survive contact
    with a reader: the number is still on the page, still quotable, and this project
    has already shipped one table whose numbers were wrong while a caveat sat beside
    them. So a model that fails the scene-fixed check renders as em-dashes until the
    check passes. The verdict comes from the data, so the blanks fill themselves in
    on the next refresh --- nobody has to remember to unblank the paper.
    """
    return (not SHOW_UNVERIFIED) and scene_status(model)["state"] in ("failed", "unverified")


def model_label(model: str) -> str:
    return PRETTY[model] + MARK[scene_status(model)["state"]]


def prose_macros(rq1: list[dict], abl: list[dict]) -> None:
    """LaTeX macros for the handful of numbers the PROSE quotes.

    Tables have been generated from the CSVs since the hand-transcription incident,
    but the abstract and discussion still quoted figures by hand, which is the same
    failure with a smaller blast radius -- the abstract said "roughly sixty points"
    while the table said 74.4. Anything the prose states about a measured quantity
    comes from here, so it moves when the data moves. A macro whose model is
    suppressed renders as an em-dash and will look obviously wrong in the PDF, which
    is the intended behaviour: the sentence around it needs rewriting, not the number.
    """
    by1 = {(r["model"], r["condition"]): r for r in rq1}
    byA = {(r["model"], r["condition"]): r for r in abl}
    out = []

    def emit(name: str, model: str, value, fmt: str) -> None:
        if suppressed(model) or value is None:
            out.append(rf"\newcommand{{\{name}}}{{\textemdash}}")
        else:
            out.append(rf"\newcommand{{\{name}}}{{{format(value, fmt)}}}")

    for name, model, cond in [("oftBaseline", "openvla_oft", "original"),
                              ("oftBlank", "openvla_oft", "blank"),
                              ("oftNonsense", "openvla_oft", "nonsense"),
                              ("smolBaseline", "smolvla", "original"),
                              ("ovBaseline", "openvla", "original")]:
        r = by1.get((model, cond))
        tsr = fnum(r["TSR"]) if r else None
        emit(name, model, None if tsr is None else 100 * tsr, ".1f")
    for name, model, cond in [("oftCssBlank", "openvla_oft", "blank"),
                              ("oftCssNonsense", "openvla_oft", "nonsense")]:
        r = by1.get((model, cond))
        emit(name, model, fnum(r["CSS"]) if r else None, ".2f")
    for name, model, cond in [("oftVerbDrop", "openvla_oft", "verb_dropped"),
                              ("oftNounMask", "openvla_oft", "nouns_masked")]:
        r = byA.get((model, cond))
        d = fnum(r.get("delta_pp_matched")) if r else None
        emit(name, model, None if d is None else abs(d), ".1f")
    write("macros.tex", "\n".join(out) + "\n")


def draft_status() -> None:
    """A sentence naming the models whose cells are currently blank, or nothing.

    Generated rather than written, so that when the last grid passes its check this
    sentence becomes empty on the next refresh and the paper stops describing itself
    as provisional. A hand-written "OpenVLA is still running" would outlive the fact.
    """
    pending = [PRETTY[m] for m in MODEL_ORDER if suppressed(m)]
    if not pending:
        write("draft_status.tex", "%% all models verified; no pending note needed\n")
        return
    names = pending[0] if len(pending) == 1 else (
        ", ".join(pending[:-1]) + " and " + pending[-1])
    verb = "is" if len(pending) == 1 else "are"
    write("draft_status.tex",
          f"\\paragraph{{Status of this draft.}} The causal grid for {names} {verb} "
          f"being regenerated after the scene-fixed failure described in "
          f"Section~\\ref{{sec:scenefixed}}, so every cell for "
          f"{'that model' if len(pending) == 1 else 'those models'} is left blank "
          f"(\\textemdash) throughout. We report the models that pass the check and "
          f"withhold the ones that do not, rather than printing a confounded number "
          f"beside a caveat. The tables are generated from the result files, so they "
          f"fill in as the runs land.\n")


def scene_note() -> None:
    """One generated sentence per non-passing model, for the table captions."""
    bits = []
    for m in MODEL_ORDER:
        st = scene_status(m)
        if st["state"] == "passed":
            continue
        sym = {"failed": r"$\ddagger$", "unverified": r"$\ddagger$",
               "thin": r"$\dagger$"}[st["state"]]
        verb = ("scene-fixed check FAILS" if st["state"] in ("failed", "unverified")
                else "scene-fixed check passes only thinly")
        bits.append(f"{sym}~{PRETTY[m]}: {verb} --- {st['detail']}.")
    body = (" ".join(bits) if bits else
            "All models pass the scene-fixed check on every comparable episode.")
    if bits:
        body += (" A $\\ddagger$ model's causal contrasts are confounded by differing "
                 "initial scenes, so its cells are left blank (\\textemdash) rather "
                 "than printed with a caveat: the grid is being regenerated and the "
                 "tables fill in automatically once the check passes. A $\\dagger$ "
                 "model's grid is incomplete rather than broken.")
    write("scene_note.tex", body + "\n")
COND_ORDER = ["original", "blank", "nonsense", "wrong_action", "wrong_object",
              "wrong_task", "repeated"]
COND_PRETTY = {"original": "Baseline", "blank": "Blank", "nonsense": "Nonsense",
               "wrong_action": "Wrong Action", "wrong_object": "Wrong Object",
               "wrong_task": "Wrong Task", "repeated": "Repeated"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Deliberately not the normal approximation: at these cell sizes the headline
    wrong_object rate is 1 success in 14, where a Wald interval runs off the end of
    [0,1] and understates the uncertainty. Wilson stays inside the unit interval
    and is honest about small n, which is the entire point of reporting it.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def read_csv(name: str) -> list[dict]:
    path = REPORT / name
    if not path.exists():
        print(f"[make_tables] SKIP: {path} not found", file=sys.stderr)
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(s, default=None):
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def pct(x, nd=1):
    return "--" if x is None else f"{100 * x:.{nd}f}\\%"


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        "% GENERATED by core/analyze/make_tables.py — do not edit by hand.\n"
        "% Re-run after any change to paper/*.csv.\n" + body
    )
    print(f"[make_tables] wrote {(OUT / name).relative_to(REPO_ROOT)}")


def rq1_tables(rows: list[dict]) -> None:
    if not rows:
        return
    by = {(r["model"], r["condition"]): r for r in rows}

    # --- wide view: the shape the paper already uses -------------------------
    lines = [r"\begin{tabular}{l" + "c" * len(COND_ORDER) + "}", r"\toprule",
             r"\textbf{Model} & " +
             " & ".join(rf"\textbf{{{COND_PRETTY[c]}}}" for c in COND_ORDER) + r" \\",
             r"\midrule"]
    for m in MODEL_ORDER:
        if suppressed(m):
            lines.append(f"{model_label(m)} & " +
                         " & ".join([PENDING] * len(COND_ORDER)) + r" \\")
            continue
        cells = []
        for c in COND_ORDER:
            r = by.get((m, c))
            if not r:
                cells.append("--")
                continue
            tsr, std, css = fnum(r["TSR"]), fnum(r["TSR_std_over_seeds"]), fnum(r["CSS"])
            cell = pct(tsr)
            if std:  # only when seeds actually disagree; std=0 adds nothing
                cell += rf"$\pm${100 * std:.1f}"
            if css is not None:
                cell += f" ({css:.2f})"
            cells.append(cell)
        lines.append(f"{model_label(m)} & " + " & ".join(cells) + r" \\")
    # n per condition, pooled over seeds. Show every distinct value when the models
    # disagree (openvla/repeated is seed-7 only, n=20, while the others are 40) —
    # collapsing to one number here is how "N=40 per condition" got into the paper.
    n_cells = []
    for c in COND_ORDER:
        ns = sorted({int(fnum(by[(m, c)]["n_total"], 0)) for m in MODEL_ORDER
                     if (m, c) in by and not suppressed(m)})
        n_cells.append("/".join(str(n) for n in ns) if ns else "--")
    lines += [r"\midrule",
              r"\textit{episodes per cell} & " + " & ".join(n_cells) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    write("rq1_causal.tex", "\n".join(lines) + "\n")

    # --- long view: n and a CI for every cell, which the wide view cannot fit -
    lines = [r"\begin{tabular}{llrcc}", r"\toprule",
             r"\textbf{Model} & \textbf{Condition} & \textbf{$n$} & "
             r"\textbf{TSR (95\% CI)} & \textbf{CSS} \\", r"\midrule"]
    for i, m in enumerate(MODEL_ORDER):
        if i:
            lines.append(r"\midrule")
        if suppressed(m):
            lines.append(f"{model_label(m)} & {PENDING_ROW} & {PENDING} & "
                         f"{PENDING} & {PENDING} \\\\")
            continue
        for c in COND_ORDER:
            r = by.get((m, c))
            if not r:
                continue
            n = int(fnum(r["n_total"], 0))
            tsr = fnum(r["TSR"])
            css = fnum(r["CSS"])
            lo, hi = wilson(round(tsr * n), n) if (tsr is not None and n) else (None, None)
            ci = "--" if lo is None else f"[{100 * lo:.1f}, {100 * hi:.1f}]"
            lines.append(f"{model_label(m)} & \\texttt{{{c.replace('_', chr(92) + '_')}}} & {n} & "
                         f"{pct(tsr)} {ci} & {'--' if css is None else f'{css:.2f}'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("rq1_causal_full.tex", "\n".join(lines) + "\n")

    # --- the note that replaces the false "N=40 per condition" ---------------
    counts = {}
    for c in COND_ORDER:
        ns = {int(fnum(by[(m, c)]["n_total"], 0)) for m in MODEL_ORDER
              if (m, c) in by and not suppressed(m)}
        counts[c] = ns
    parts = []
    for c in COND_ORDER:
        ns = counts[c]
        shown = "/".join(str(n) for n in sorted(ns)) if ns else r"\textemdash"
        parts.append(rf"\texttt{{{c.replace('_', chr(92) + '_')}}} $n={shown}$")
    seeds = sorted({s for r in rows for s in r["seeds"].split(";")})
    write("rq1_n_note.tex",
          "Episode counts differ by condition and model, so there is no single $N$: " +
          ", ".join(parts) + rf". Seeds {', '.join(seeds)}." + "\n")


def rq2_table(rows: list[dict]) -> None:
    if not rows:
        return
    axis_pretty = {"para_object": "Object", "para_action": "Action",
                   "para_compositional": "Compositional", "ALL": r"\textbf{All axes}"}
    # `scenes` is not decoration: SmolVLA's 2963 compositional episodes cover only 20
    # distinct initial scenes, so n badly overstates the independent variation behind
    # the estimate. The paper says it reports scenes alongside n, so the table must.
    lines = [r"\begin{tabular}{llrrccc}", r"\toprule",
             r"\textbf{Model} & \textbf{Paraphrase Axis} & \textbf{$n$} & "
             r"\textbf{scenes} & \textbf{TSR} & "
             r"\textbf{$\Delta$TSR (pp)} & \textbf{PRIDE} \\", r"\midrule"]
    for i, m in enumerate(MODEL_ORDER):
        mr = [r for r in rows if r["model"] == m]
        if not mr:
            continue
        if i:
            lines.append(r"\midrule")
        if suppressed(m):
            lines.append(f"{model_label(m)} & {PENDING_ROW} & {PENDING} & {PENDING} & "
                         f"{PENDING} & {PENDING} & {PENDING} \\\\")
            continue
        for r in mr:
            scenes = fnum(r.get("n_scenes"))
            lines.append(
                f"{PRETTY[m]} & {axis_pretty.get(r['axis'], r['axis'])} & {int(fnum(r['n'], 0))} & "
                f"{'--' if scenes is None else int(scenes)} & "
                f"{pct(fnum(r['TSR']), 2)} & $-{fnum(r['delta_TSR_pp'], 0):.2f}$ & "
                f"{fnum(r['PRIDE'], 0):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("rq2_para.tex", "\n".join(lines) + "\n")


def rq3_table(rows: list[dict]) -> None:
    if not rows:
        return
    lines = [r"\begin{tabular}{llrcc}", r"\toprule",
             r"\textbf{Model} & \textbf{Condition} & \textbf{pairs} & "
             r"\textbf{$t_{\text{div}}$ (samples)} & \textbf{$e_{10}$ [m]} \\", r"\midrule"]
    for m in MODEL_ORDER:
        mr = [x for x in rows if x["model"] == m]
        if mr and suppressed(m):
            lines.append(f"{model_label(m)} & {PENDING_ROW} & {PENDING} & "
                         f"{PENDING} & {PENDING} \\\\")
            continue
        for r in mr:
            cond = r["condition"].replace("_", chr(92) + "_")
            # e10 is quoted at a fixed 10-step horizon that every counted pair
            # reaches, so the `pairs` column on this row is the n behind BOTH
            # numbers. Guard it rather than trust it: the previous column was the
            # tail of the mean curve, which n did not back.
            n_pairs, n_e10 = int(fnum(r["n_pairs"], 0)), int(fnum(r["n_at_e10"], 0))
            if n_e10 != n_pairs:
                raise SystemExit(f"rq3 {m}/{r['condition']}: e10 rests on {n_e10} "
                                 f"pairs but the table would print {n_pairs}")
            lines.append(
                f"{model_label(m)} & \\texttt{{{cond}}} & {n_pairs} & "
                f"{fnum(r['mean_tdiv_step'], 0):.2f} & {fnum(r['e10_m'], 0):.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("rq3_divergence.tex", "\n".join(lines) + "\n")


def wordclass_table(rows: list[dict]) -> None:
    """The antonym vs near-synonym split behind the RQ2 reframe.

    This was hand-written into paper_acl.tex. Those five numbers move whenever the
    grid is regenerated, and hand-transcribed numbers are exactly what produced this
    project's earlier false claims -- so they are generated here instead, with n in
    the table. Rows come from core/analyze/instruction_information.py.
    """
    split = [r for r in rows if r["section"] == "wrong_action_split"]
    if not split:
        return
    by = {(r["model"], r["subset"]): r for r in split}
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"\textbf{Model} & \multicolumn{2}{c}{\textbf{antonym}} & "
             r"\multicolumn{2}{c}{\textbf{near-synonym}} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
             r" & $\Delta$TSR & $p$ & $\Delta$TSR & $p$ \\", r"\midrule"]
    for m in MODEL_ORDER:
        if suppressed(m):
            lines.append(f"{model_label(m)} & " + " & ".join([PENDING] * 4) + r" \\")
            continue
        cells = []
        for subset in ("antonym", "near-syn"):
            r = by.get((m, subset))
            if r is None:
                cells += ["--", "--"]
                continue
            # Explicit columns now, rather than scraping the note string.
            delta = fnum(r.get("delta_pp"))
            if delta is None:  # older CSVs kept it only in the note
                for tok in (r.get("note") or "").split():
                    if tok.startswith("delta_pp="):
                        delta = fnum(tok.split("=", 1)[1])
            n = int(fnum(r["n"], 0))
            # The p-value replaces n here: n is fixed by the suite (three antonym
            # tasks, seven near-synonyms) and says nothing a reader needs, whereas
            # whether the gap is distinguishable from zero is the entire question.
            # OpenVLA-OFT's 5.0 pp antonym gap has p=0.45 on 60 pairs; printing it
            # bare invited exactly the over-reading an earlier draft committed.
            pcell = _pfmt(fnum(r.get("mcnemar_p")))
            if delta is None:
                cells += ["--", "--"]
            elif abs(delta) < 0.05:  # avoid rendering a signed "-0.0"
                cells += [r"\phantom{$-$}$0.0$ pp", pcell]
            else:
                cells += [f"$-{delta:.1f}$ pp", pcell]
        lines.append(f"{model_label(m)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("rq2_wordclass.tex", "\n".join(lines) + "\n")


def _pfmt(pv: float | None) -> str:
    """A p-value the way a table should carry one: never "0.000", never 12 digits."""
    if pv is None:
        return "--"
    if pv < 1e-10:
        return r"$<\!10^{-10}$"
    if pv < 0.01:
        exp = int(f"{pv:e}".split("e")[1])
        return rf"$10^{{{exp}}}$"
    return f"${pv:.2f}$"


def ablation_table(rows: list[dict]) -> None:
    """The word-class removal test — the paper's decisive experiment.

    Rows from core/analyze/make_ablation_csv.py. The baseline here is scene-matched: the
    ablation conditions are produced in the same process as `original`, so they share
    initial scenes and the delta is a clean within-scene contrast (unlike the
    paraphrase pools, cf. Section 'scenefixed').
    """
    if not rows:
        return
    order = ["original", "verb_dropped", "nouns_masked", "blank"]
    pretty = {"original": r"\texttt{original} (reference)",
              "verb_dropped": r"\texttt{verb\_dropped}",
              "nouns_masked": r"\texttt{nouns\_masked}",
              "blank": r"\texttt{blank}"}
    lines = [r"\begin{tabular}{llrrccr}", r"\toprule",
             r"\textbf{Model} & \textbf{Condition} & \textbf{bits} & "
             r"\textbf{$n$} & \textbf{TSR} & \textbf{$\Delta$ [95\% CI]} & "
             r"\textbf{$p$} \\",
             r"\midrule"]
    any_row = False
    for i, m in enumerate(MODEL_ORDER):
        mr = [r for r in rows if r["model"] == m and
              any(r["condition"] == a for a in ("verb_dropped", "nouns_masked"))]
        if not mr:
            continue  # a model with no ablation data yet contributes no block
        if any_row:
            lines.append(r"\midrule")
        any_row = True
        if suppressed(m):
            lines.append(f"{model_label(m)} & {PENDING_ROW} & {PENDING} & {PENDING} & "
                         f"{PENDING} & {PENDING} & {PENDING} \\\\")
            continue
        for cond in order:
            r = next((x for x in rows if x["model"] == m and x["condition"] == cond), None)
            if r is None:
                continue
            d = fnum(r.get("delta_pp_matched"))
            # Paired CI and McNemar, from make_ablation_csv.py. The design matches
            # every perturbed episode to its baseline on initial state, so the
            # interval is over PAIRS; reporting a bare delta throws that away and
            # leaves the decisive experiment with no uncertainty at all.
            lo, hi = fnum(r.get("delta_ci_lo")), fnum(r.get("delta_ci_hi"))
            pv = fnum(r.get("mcnemar_p"))
            if d is None:
                dcell, pcell = "--", "--"
            elif cond == "original":
                dcell, pcell = r"\phantom{$-$}$0.0$", "--"
            else:
                ci = "" if lo is None else f" {{\\scriptsize $[{-hi:+.1f}, {-lo:+.1f}]$}}"
                dcell = f"${-d:+.1f}${ci}"
                pcell = _pfmt(pv)
            lines.append(
                f"{model_label(m)} & {pretty[cond]} & {r['bits_removed']} & "
                f"{int(fnum(r['n'], 0))} & {pct(fnum(r['TSR']), 1)} & {dcell} & {pcell} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if any_row:
        write("rq_ablation.tex", "\n".join(lines) + "\n")
        # A one-sentence coverage note, so partial data cannot read as complete.
        cov = []
        for m in MODEL_ORDER:
            r = next((x for x in rows if x["model"] == m and
                      x["condition"] == "nouns_masked"), None)
            if r and not suppressed(m):
                cov.append(f"{PRETTY[m]} on {int(fnum(r['n_tasks'], 0))} of 10 tasks")
        write("rq_ablation_note.tex",
              ("Coverage: " + "; ".join(cov) + ". " if cov else "") +
              "The ablation conditions are generated in the same process as "
              "\\texttt{original}, so the $\\Delta$ column is a within-scene contrast.\n")


def main() -> None:
    global SHOW_UNVERIFIED
    if "--show-unverified" in sys.argv:
        SHOW_UNVERIFIED = True
        print("[make_tables] --show-unverified: printing numbers that FAIL the "
              "scene-fixed check. Do NOT ship these tables.", file=sys.stderr)
    prose_macros(read_csv("rq1_causal.csv"), read_csv("rq_ablation.csv"))
    rq1_tables(read_csv("rq1_causal.csv"))
    rq2_table(read_csv("rq2_paraphrase.csv"))
    rq3_table(read_csv("rq3_divergence.csv"))
    wordclass_table(read_csv("instruction_information.csv"))
    ablation_table(read_csv("rq_ablation.csv"))
    scene_note()
    draft_status()
    for m in MODEL_ORDER:
        st = scene_status(m)
        held = "  [NUMBERS WITHHELD FROM TABLES]" if suppressed(m) else ""
        print(f"[make_tables] scene-fixed {m}: {st['state']} -- {st['detail']}{held}")
    print("\n[make_tables] done. In paper_acl.tex, replace each hand-written tabular with:\n"
          "    \\input{tables/rq1_causal}      (and rq1_n_note in the caption)\n"
          "    \\input{tables/rq2_para}\n"
          "    \\input{tables/rq3_divergence}")


if __name__ == "__main__":
    main()
