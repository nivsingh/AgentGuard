"""Run the same cases against several models and compare them on identical ground truth."""

from guard.metrics import summarize
from guard.runner import run_suite


def compare_models(cases, docs, db_path, models, options=None, only=None):
    """models: {label: llm}. Returns {label: [Outcome]}. Each model sees exactly the same cases and the same index."""
    return {label: run_suite(cases, llm, docs, db_path, options=options, only=only) for label, llm in models.items()}


def render(results, cases):
    labels = list(results)
    sums = {k: summarize(v) for k, v in results.items()}
    lines = ["# Model comparison", "", "Same cases, same ground truth, same agent. Counts are raw.", "",
             "| Model | Passed | Dev | Held out | Wrong answers delivered | Median latency (ms) | Tokens in / out |", "|---|---|---|---|---|---|---|"]
    for k in labels:
        s = sums[k]
        dev, held = s["by_split"].get("dev"), s["by_split"].get("heldout")
        lines.append(f"| {k} | {s['overall']['passed']} / {s['overall']['total']} | "
                     f"{dev['passed'] if dev else '-'} / {dev['total'] if dev else '-'} | "
                     f"{held['passed'] if held else '-'} / {held['total'] if held else '-'} | {s['unsafe']} | "
                     f"{round(s['median_latency_ms'], 1) if s['median_latency_ms'] is not None else '-'} | "
                     f"{s['tokens']['input']} / {s['tokens']['output']} |")
    cats = sorted({c for s in sums.values() for c in s["by_category"]})
    lines += ["", "| Category | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(f"{sums[k]['by_category'][c]['passed']} / {sums[k]['by_category'][c]['total']}" if c in sums[k]["by_category"] else "-" for k in labels) + " |")
    classes = sorted({c for s in sums.values() for c in s["failure_classes"]})
    if classes:
        lines += ["", "| Failure class | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        lines += [f"| {c} | " + " | ".join(str(sums[k]["failure_classes"].get(c, 0)) for k in labels) + " |" for c in classes]
    if any(k.startswith("simulated") for k in labels):
        lines += ["", "> Rows labeled simulated are stand-ins that test the evaluators. They are not a model comparison."]
    return "\n".join(lines) + "\n"
