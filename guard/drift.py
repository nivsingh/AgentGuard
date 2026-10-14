"""Drift scenario: the document index goes stale.

A policy changed on 2026-07-01 (D02) but the index was never refreshed. The agent keeps
answering confidently from the old version. Every claim it makes is verified against a
real document, so the evidence gate has nothing to object to. Only a check against
independently known truth notices, which is what AgentGuard does.
"""

from guard.metrics import compare
from guard.runner import run_suite

STALE_DOCS = ("D02",)


def stale_index_scenario(cases, llm, docs, db_path, options=None, categories=None):
    fresh = run_suite(cases, llm, docs, db_path, options=options, categories=categories)
    stale = run_suite(cases, llm, docs, db_path, exclude_ids=STALE_DOCS, options=options, categories=categories)
    regressions, _ = compare(fresh, stale)
    by_id = {o.case_id: o for o in stale}
    return {
        "fresh": fresh,
        "stale": stale,
        "regressions": regressions,
        # how many of the new failures looked fine to the gate: it answered, with every claim verified
        "gate_saw_nothing": [cid for cid in regressions if by_id[cid].status == "answered"],
        "missing_docs": list(STALE_DOCS),
    }
