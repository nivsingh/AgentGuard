import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent.graph import HealthcareAgent
from agent.llm import OfflineLLM
from agent.pii import redact
from agent.retrieval import build_index
from tracing.otel import export_run
from tracing.store import TraceStore


# ---- PII redaction ---------------------------------------------------------
@pytest.mark.parametrize(
    "text, label",
    [
        ("Email me at jane.doe@example.com please", "EMAIL"),
        ("Her SSN is 123-45-6789", "SSN"),
        ("Call 212-555-0147 about the visit", "PHONE"),
        ("Call (212) 555 0147 about the visit", "PHONE"),
        ("Patient MRN: 00123456 was readmitted", "MRN"),
        ("Patient DOB 04/12/1987 asthma", "DOB"),
        ("Card 4111 1111 1111 1111 on file", "CARD"),
    ],
)
def test_redacts_each_identifier_type(text, label):
    clean, counts = redact(text)
    assert counts == {label: 1} and f"[{label}]" in clean


def test_redaction_removes_the_value_and_keeps_the_question():
    clean, _ = redact("How many visits for jane.doe@example.com in March 2026?")
    assert "jane" not in clean and "How many visits for [EMAIL] in March 2026?" == clean


def test_ordinary_questions_pass_through_untouched():
    q = "How many asthma visits were there in March 2026, and what was the 30 day readmission rate of 11.8 percent?"
    assert redact(q) == (q, {})


def test_dates_and_counts_are_not_mistaken_for_identifiers():
    q = "Between 2026-02-01 and 2026-03-01 there were 1087 and 1284 visits"
    assert redact(q)[1] == {}


# ---- the agent applies it ---------------------------------------------------
PLAN = {"needs_sql": False, "needs_docs": False, "required_fields": [], "as_of": None, "clarify": "Which period?"}


@pytest.fixture(scope="module")
def make_agent(built, docs):
    def factory(llm, **kw):
        return HealthcareAgent(llm, built / "agentguard.db", build_index(docs), docs, **kw)

    return factory


def test_model_never_sees_the_raw_identifier(make_agent):
    llm = OfflineLLM([json.dumps(PLAN)])
    result = make_agent(llm).ask("How many visits for jane.doe@example.com?")
    sent = json.dumps(llm.calls)
    assert "jane.doe" not in sent and "[EMAIL]" in sent
    assert result.evidence["redactions"] == {"EMAIL": 1}
    assert result.trace[0]["name"] == "pii_redaction" and result.trace[0]["detail"] == {"redactions": {"EMAIL": 1}}


def test_trace_store_saves_only_the_redacted_question(make_agent):
    store = TraceStore()
    agent = make_agent(OfflineLLM([json.dumps(PLAN)]), trace_store=store)
    agent.ask("Visits for SSN 123-45-6789?")
    (question,) = store.conn.execute("SELECT question FROM runs").fetchone()
    assert "123-45" not in question and "[SSN]" in question


def test_canary_is_only_in_the_analyst_prompt(make_agent):
    plan = {"needs_sql": False, "needs_docs": False, "required_fields": [], "as_of": None, "clarify": None}
    analysis = {"summary": "x", "claims": [], "limitations": [], "confidence": 0.5}
    llm = OfflineLLM([json.dumps(plan), json.dumps(analysis)])
    make_agent(llm, canary="CANARY-abc123").ask("anything")
    planner_call, analyst_call = llm.calls
    assert "CANARY-abc123" not in planner_call["system"]
    assert "CANARY-abc123" in analyst_call["system"]


# ---- trace store -----------------------------------------------------------
def test_store_round_trip_and_queries(make_agent):
    store = TraceStore()
    agent = make_agent(OfflineLLM([json.dumps(PLAN)] * 3), trace_store=store)
    for _ in range(3):
        agent.ask("Which period?")
    assert store.status_counts() == {"clarify": 3}
    run_id = store.conn.execute("SELECT run_id FROM runs LIMIT 1").fetchone()[0]
    assert [s["name"] for s in store.spans(run_id)] == ["pii_redaction", "planner"]
    names = [row[0] for row in store.slowest_steps()]
    assert set(names) == {"pii_redaction", "planner"}
    assert store.median_latency_ms() is not None


def test_store_persists_to_a_file(make_agent, tmp_path):
    path = tmp_path / "traces.db"
    make_agent(OfflineLLM([json.dumps(PLAN)]), trace_store=TraceStore(path)).ask("Which period?")
    assert TraceStore(path).status_counts() == {"clarify": 1}


# ---- OpenTelemetry export --------------------------------------------------
def test_run_exports_as_an_opentelemetry_trace(make_agent):
    result = make_agent(OfflineLLM([json.dumps(PLAN)])).ask("Which period?")
    exporter = InMemorySpanExporter()
    export_run(result, exporter)
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "agent.ask" in names and "agent.planner" in names and "agent.pii_redaction" in names
    root = next(s for s in spans if s.name == "agent.ask")
    assert root.attributes["agent.status"] == "clarify"
    assert len({s.context.trace_id for s in spans}) == 1  # all spans belong to one trace
    child = next(s for s in spans if s.name == "agent.planner")
    assert child.parent.span_id == root.context.span_id
