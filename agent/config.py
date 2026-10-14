"""Agent options and one place that builds a configured agent.

BASELINE is the plain agent that Experiment 000 and 001 measured. RECOMMENDED is
BASELINE plus every change that experiments.md records as KEEP, and nothing else.
"""

from agent.graph import HealthcareAgent
from agent.retrieval import TfidfEmbedder, build_index

BASELINE = {"k": 3, "stem": False, "expand_versions": False, "check_summary": False}
# Experiments A (stemming), C (sibling versions) and D (summary number check) are KEEP; B (top K 5) is not adopted.
RECOMMENDED = {**BASELINE, "stem": True, "expand_versions": True, "check_summary": True}


def build_agent(llm, db_path, docs, options=None, exclude_ids=(), canary=None, trace_store=None):
    opts = {**RECOMMENDED, **(options or {})}
    index = build_index(docs, exclude_ids=exclude_ids, embedder=TfidfEmbedder(stemming=opts["stem"]))
    return HealthcareAgent(llm, db_path, index, docs, k=opts["k"], canary=canary, trace_store=trace_store,
                           expand_versions=opts["expand_versions"], check_summary=opts["check_summary"])
