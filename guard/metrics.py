"""Summaries of a run. Always raw counts (36 / 40), never a bare percentage."""

import statistics
from collections import Counter

from guard.evaluators import FAILURE_CLASSES


def _tally(outcomes):
    return {"passed": sum(o.passed for o in outcomes), "total": len(outcomes)}


def consistency(outcomes):
    """Cases that share a group are the same question in different words. A group is stable
    when every phrasing got the same verdict. Stability is separate from correctness."""
    groups = {}
    for o in outcomes:
        if o.group:
            groups.setdefault(o.group, []).append(o)
    stable = sum(1 for g in groups.values() if len({(o.passed, o.failure_class, o.status) for o in g}) == 1)
    return {"stable": stable, "groups": len(groups)}


def summarize(outcomes):
    by_split, by_category = {}, {}
    for o in outcomes:
        by_split.setdefault(o.split, []).append(o)
        by_category.setdefault(o.category, []).append(o)
    classes = Counter(o.failure_class for o in outcomes if not o.passed)
    failed = [o for o in outcomes if not o.passed]
    latencies = [o.latency_ms for o in outcomes if o.latency_ms]
    return {
        "overall": _tally(outcomes),
        "by_split": {k: _tally(v) for k, v in sorted(by_split.items())},
        "by_category": {k: _tally(v) for k, v in sorted(by_category.items())},
        "failure_classes": {c: classes[c] for c in FAILURE_CLASSES if classes[c]},
        "unsafe": sum(o.unsafe for o in outcomes),  # wrong answers delivered instead of an abstention
        "safe_failures": sum(1 for o in failed if not o.unsafe),
        "consistency": consistency(outcomes),
        "tokens": {"input": sum(o.input_tokens for o in outcomes), "output": sum(o.output_tokens for o in outcomes)},
        "median_latency_ms": statistics.median(latencies) if latencies else None,
    }


def compare(before, after):
    """Cases whose verdict changed between two runs of the same cases: (regressions, fixes)."""
    b = {o.case_id: o for o in before}
    regressions = sorted(o.case_id for o in after if not o.passed and b.get(o.case_id) and b[o.case_id].passed)
    fixes = sorted(o.case_id for o in after if o.passed and b.get(o.case_id) and not b[o.case_id].passed)
    return regressions, fixes
