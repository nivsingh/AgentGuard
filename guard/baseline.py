"""Golden baseline: the verdict of every case in the last accepted run, so CI can fail on any regression."""

import json
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent / "baselines"


def path_for(model):
    return BASELINE_DIR / f"{model}.json"


def snapshot(outcomes):
    return {o.case_id: {"passed": o.passed, "class": o.failure_class} for o in outcomes}


def save(model, outcomes):
    BASELINE_DIR.mkdir(exist_ok=True)
    path_for(model).write_text(json.dumps(snapshot(outcomes), indent=1, sort_keys=True) + "\n")


def check(model, outcomes):
    """Returns (regressions, fixes, problems). Any regression or missing baseline fails the check."""
    p = path_for(model)
    if not p.exists():
        return [], [], [f"no baseline for {model}; run with --update to accept the current results"]
    old, new = json.loads(p.read_text()), snapshot(outcomes)
    problems = [f"case set changed: {sorted(set(old) ^ set(new))}"] if set(old) != set(new) else []
    regressions = sorted(c for c in new if c in old and old[c]["passed"] and not new[c]["passed"])
    fixes = sorted(c for c in new if c in old and not old[c]["passed"] and new[c]["passed"])
    return regressions, fixes, problems
