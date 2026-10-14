"""Run an experiment the way the project's rules require.

A change is tried on the 60 dev cases, then confirmed on the 40 held-out cases. It is KEPT only when
    dev improves, held out does not get worse, and no case that passed before now fails.
The held-out cases are never used to choose or tune a change. Results are raw counts.
"""

import statistics
from dataclasses import dataclass, field

from agent.config import BASELINE
from guard.metrics import compare
from guard.runner import run_suite
from guard.simulated_model import SimulatedModel


@dataclass
class Experiment:
    name: str
    hypothesis: str
    change: dict  # options that differ from BASELINE
    defects: dict = field(default_factory=dict)  # simulated defects active during the experiment
    metric: str = "passed"  # "passed" or "unsafe"
    base: dict = None  # options to compare against; BASELINE when omitted
    adopted: bool = True  # whether the change went into agent.config.RECOMMENDED (a KEEP can still be left out, with a note)
    note: str = ""


def _tally(outcomes, split):
    rows = [o for o in outcomes if o.split == split]
    return {"passed": sum(o.passed for o in rows), "total": len(rows), "unsafe": sum(o.unsafe for o in rows)}


def measure(outcomes):
    lat = [o.latency_ms for o in outcomes if o.latency_ms]
    return {"dev": _tally(outcomes, "dev"), "heldout": _tally(outcomes, "heldout"),
            "median_latency_ms": round(statistics.median(lat), 2) if lat else None}


def decide(exp, before, after, regressions):
    key = "passed" if exp.metric == "passed" else "unsafe"
    sign = 1 if key == "passed" else -1  # fewer unsafe answers is better
    dev_gain = sign * (after["dev"][key] - before["dev"][key])
    held_gain = sign * (after["heldout"][key] - before["heldout"][key])
    if regressions:
        return "REJECT", f"{len(regressions)} previously passing case(s) now fail: {', '.join(regressions[:5])}."
    if held_gain < 0:
        return "REJECT", "held out got worse, so the dev gain does not generalize."
    if dev_gain <= 0:
        return "REJECT", "no measurable gain on dev."
    note = "" if held_gain > 0 else " Held out did not change (it may already be at its ceiling), so the gain is confirmed only as no harm."
    return "KEEP", f"dev improved and held out did not get worse.{note}"


def run_experiment(exp, cases, docs, db_path):
    llm = lambda: SimulatedModel(cases, exp.defects)  # noqa: E731  a fresh model per run
    start = exp.base or BASELINE
    base = run_suite(cases, llm(), docs, db_path, options=start)
    cand = run_suite(cases, llm(), docs, db_path, options={**start, **exp.change})
    before, after = measure(base), measure(cand)
    regressions, fixes = compare(base, cand)
    decision, reason = decide(exp, before, after, regressions)
    return {"experiment": exp, "before": before, "after": after, "regressions": regressions, "fixes": fixes,
            "decision": decision, "reason": reason}


def format_result(r):
    e, b, a = r["experiment"], r["before"], r["after"]
    unsafe = f"\n- Wrong answers delivered, dev / held out: {b['dev']['unsafe']} / {b['heldout']['unsafe']} to {a['dev']['unsafe']} / {a['heldout']['unsafe']}" if (e.defects or e.metric == "unsafe") else ""
    return (f"### {e.name}\n\nHypothesis: {e.hypothesis}\nChange: {e.change}"
            f"{'  Compared with: ' + str(e.base) if e.base else ''}{'  Simulated defects: ' + str(e.defects) if e.defects else ''}\n\n"
            f"- Dev: {b['dev']['passed']} / {b['dev']['total']} to {a['dev']['passed']} / {a['dev']['total']}\n"
            f"- Held out: {b['heldout']['passed']} / {b['heldout']['total']} to {a['heldout']['passed']} / {a['heldout']['total']}"
            f"{unsafe}\n- Median latency per question: {b['median_latency_ms']} ms to {a['median_latency_ms']} ms (simulated model, so this is orchestration and retrieval only)\n"
            f"- Fixed: {', '.join(r['fixes']) or 'none'}. Regressed: {', '.join(r['regressions']) or 'none'}.\n"
            f"- Decision: **{r['decision']}**. {r['reason']}\n"
            f"- Adopted: {'yes' if e.adopted and r['decision'] == 'KEEP' else 'no'}.{(' ' + e.note) if e.note else ''}\n")


EXPERIMENTS = [
    Experiment("A: stemming", "Related word forms (park, parking) should match, fixing the parking question.", {"stem": True}),
    Experiment("B: top K 3 to 5", "A larger candidate set should let the newer policy version reach the agent.", {"k": 5}, adopted=False,
               note="C fixes the same case at its cause and B only makes it likelier, at the price of larger prompts. F shows B adds nothing once A and C are in."),
    Experiment("C: fetch sibling policy versions", "If one version of a policy is retrieved, fetching the others lets temporal selection choose correctly.", {"expand_versions": True}),
    Experiment("D: summary number check in the gate (planted defect: wrong number only in the summary, 50 percent of cases)",
               "The gate should also reject numbers in the summary that no verified claim backs.",
               {"check_summary": True}, defects={"hallucinate_number": 0.5}, metric="unsafe"),
    Experiment("E: everything kept so far, no planted defects", "A, C and D together should not break anything, and D should reject nothing that is correct.",
               {"stem": True, "expand_versions": True, "check_summary": True}),
    Experiment("F: top K 3 to 5 on top of A and C", "If B still adds something once A and C are in, it is worth its larger prompts.",
               {"k": 5}, base={"stem": True, "expand_versions": True}),
]
