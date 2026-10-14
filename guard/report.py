"""Plain-text (Markdown) report of a run: raw counts, failure classes, and each failure with its reason."""

from guard.metrics import summarize

SIM_BANNER = ("> **Simulated model.** These results measure how well the evaluators detect planted defects. "
              "They say nothing about how a real language model performs. Run with ANTHROPIC_API_KEY for that.\n")


def _frac(t):
    return f"{t['passed']} / {t['total']}"


def render(outcomes, cases, model, condition="fresh index", title="AgentGuard report"):
    s = summarize(outcomes)
    by_id = {c.id: c for c in cases}
    lines = [f"# {title}", "", f"Model: `{model}`. Index: {condition}.", ""]
    if model.startswith("simulated"):
        lines += [SIM_BANNER]
    lines += [f"**Overall: {_frac(s['overall'])} passed.** "
              f"Dev {_frac(s['by_split'].get('dev', {'passed': 0, 'total': 0}))}, "
              f"held out {_frac(s['by_split'].get('heldout', {'passed': 0, 'total': 0}))}.", "",
              f"Wrong answers delivered to the user: **{s['unsafe']}**. "
              f"Failures that were safe abstentions or questions: {s['safe_failures']}. "
              f"Paraphrase groups stable: {s['consistency']['stable']} / {s['consistency']['groups']}.", "",
              "## By category", "", "| Category | Passed |", "|---|---|"]
    lines += [f"| {k} | {_frac(v)} |" for k, v in s["by_category"].items()]
    lines += ["", "## Failure classes", ""]
    if s["failure_classes"]:
        lines += ["| Class | Cases |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["failure_classes"].items()]
        lines += ["", "## Every failure", "", "| Case | Split | Class | Delivered wrong answer | Question | Why |", "|---|---|---|---|---|---|"]
        for o in outcomes:
            if not o.passed:
                q = by_id[o.case_id].question if o.case_id in by_id else ""
                lines.append(f"| {o.case_id} | {o.split} | {o.failure_class} | {'yes' if o.unsafe else 'no'} | {q} | {o.detail} |")
    else:
        lines.append("No failures.")
    lines.append("")
    return "\n".join(lines)


def render_calibration(rows):
    lines = ["# Evaluator calibration", "", SIM_BANNER, "| Planted defect | Expected class | Cases changed | In expected class | Other classes | Reaches user | Caught |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['defect']} | {r['expected_class']} | {r['changed']} | {r['in_expected_class']} | "
                     f"{r['other_classes'] or '-'} | {'yes' if r['reaches_user'] else 'no'} | {'yes' if r['caught'] else 'NO'} |")
    lines.append("")
    return "\n".join(lines)
