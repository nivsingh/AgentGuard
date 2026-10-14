"""Run the AgentGuard cases against the agent and evaluate every answer."""

import json
import sqlite3
from pathlib import Path

from agent.config import build_agent
from data import verification as V
from data.docs_util import load_docs
from guard.cases import FOLLOWUP_TITLE, build_cases
from guard.evaluators import Ctx, evaluate, run_error

CANARY = "CANARY-7f3a91c2"


class Recorder:
    """Wraps any LLM and keeps the prompts of the current case, so evaluators can inspect them."""

    def __init__(self, inner):
        self.inner = inner
        self.model = getattr(inner, "model", "unknown")
        self.calls = []

    def complete(self, system, messages, max_tokens=1024, temperature=0.0):
        self.calls.append({"system": system, "messages": messages})
        return self.inner.complete(system, messages, max_tokens=max_tokens, temperature=temperature)


def _fingerprint(conn):
    try:
        return V.table_fingerprint(conn)
    except sqlite3.Error as e:  # a dropped or damaged table is itself an integrity failure
        return f"unreadable: {e}"


def load_world(data_dir):
    """Cases, documents and database path for a generated dataset directory."""
    data_dir = Path(data_dir)
    docs = load_docs(data_dir / "docs")
    conn = sqlite3.connect(f"file:{data_dir / 'agentguard.db'}?mode=ro", uri=True)
    cases = build_cases(conn, docs)
    conn.close()
    return cases, docs, data_dir / "agentguard.db"


def policy_versions(docs):
    out = []
    for d in docs:
        if d["title"] == FOLLOWUP_TITLE:
            out.append({"doc_id": d["doc_id"], "days": int(d["body"].split("within ")[1].split(" days")[0]),
                        "effective": d["effective_date"]})
    return out


def run_suite(cases, llm, docs, db_path, exclude_ids=(), only=None, options=None, progress=None, categories=None):
    """Run every case once. `exclude_ids` simulates a stale document index (those docs are missing).
    `options` overrides agent.config.RECOMMENDED (k, stem, expand_versions, check_summary). Returns a list of Outcome. A crash in one case is recorded as run_error and never stops the run."""
    rec = Recorder(llm)
    agent = build_agent(rec, db_path, docs, options=options, exclude_ids=exclude_ids, canary=CANARY)
    versions = policy_versions(docs)
    conn = V.open_db(db_path)
    outcomes = []
    for i, case in enumerate(cases):
        if (only and case.split not in only) or (categories and case.category not in categories):
            continue
        rec.calls.clear()
        before = _fingerprint(conn)
        try:
            result = agent.ask(case.question)
        except Exception as e:  # noqa: BLE001  infrastructure problems must not look like model behavior
            outcomes.append(run_error(case, e))
            continue
        ctx = Ctx(canary=CANARY, integrity_ok=_fingerprint(conn) == before, calls=list(rec.calls),
                  policy_versions=versions)
        outcomes.append(evaluate(case, result, ctx))
        if progress:
            progress(i + 1, len(cases), outcomes[-1])
    conn.close()
    return outcomes


def dump_cases(cases, path):
    Path(path).write_text(json.dumps([c.to_json() for c in cases], indent=1))
