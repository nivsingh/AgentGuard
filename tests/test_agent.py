"""End-to-end tests of the agent graph with scripted model replies.

These prove the plumbing and the safety properties: routing, the evidence gate,
tool failures, temporal selection, and that retrieved text cannot trigger tools.
They do NOT measure how well a real model writes SQL or analysis. That is what
AgentGuard (Session 5) is for.
"""

import json
import sqlite3

import pytest
from scipy import stats

from agent.graph import HealthcareAgent, select_effective
from agent.llm import OfflineLLM
from agent.retrieval import Passage, build_index
from data import planted as P
from data import verification as V

MONTHLY = (
    "SELECT substr(admit_date,1,7) AS month, COUNT(*) AS n FROM visits "
    "WHERE diagnosis_group='Asthma' AND admit_date >= '2026-02-01' AND admit_date < '2026-04-01' "
    "GROUP BY 1 ORDER BY 1"
)
BY_AGE = (
    "SELECT age_group, SUM(CASE WHEN admit_date < '2026-03-01' THEN 1 ELSE 0 END) AS feb, "
    "SUM(CASE WHEN admit_date >= '2026-03-01' THEN 1 ELSE 0 END) AS mar FROM visits "
    "WHERE diagnosis_group='Asthma' AND admit_date >= '2026-02-01' AND admit_date < '2026-04-01' "
    "GROUP BY age_group ORDER BY age_group DESC"
)


def cell(q, r, col):
    return {"query": q, "row": r, "col": col}


def text(x):
    return x if isinstance(x, str) else json.dumps(x)


def scripted(plan, sql=None, analysis=None):
    def respond(system, messages):
        if system.startswith("You plan how"):
            return text(plan)
        if system.startswith("You write read-only SQLite"):
            assert sql is not None, "the SQL step ran but the test did not expect it"
            return text(sql)
        if system.startswith("You write the final analysis"):
            assert analysis is not None, "the analyst step ran but the test did not expect it"
            return text(analysis)
        raise AssertionError("unexpected prompt")

    return OfflineLLM(respond)


@pytest.fixture(scope="module")
def make_agent(built, docs):
    def factory(llm, exclude_ids=()):
        return HealthcareAgent(llm, built / "agentguard.db", build_index(docs, exclude_ids), docs)

    return factory


PLAN_ASTHMA = {"needs_sql": True, "needs_docs": True, "as_of": "2026-03-31", "clarify": None,
               "required_fields": ["diagnosis_group", "admit_date", "age_group"]}
SQL_ASTHMA = {"queries": [{"purpose": "monthly totals", "sql": MONTHLY}, {"purpose": "by age group", "sql": BY_AGE}]}


def asthma_claims():
    return [
        {"id": "c1", "kind": "sql", "text": "February had 1,087 asthma visits", "ref": cell(0, 0, "n"), "value": 1087},
        {"id": "c2", "kind": "sql", "text": "March had 1,284 asthma visits", "ref": cell(0, 1, "n"), "value": 1284},
        {"id": "c3", "kind": "calc", "text": "Visits rose 18.1%", "op": "pct_change",
         "inputs": [cell(0, 0, "n"), cell(0, 1, "n")], "value": 18.1},
        {"id": "c4", "kind": "calc", "text": "Pediatric visits were 72% of the increase", "op": "share_of_change",
         "inputs": [cell(1, 0, "feb"), cell(1, 0, "mar"), cell(0, 0, "n"), cell(0, 1, "n")], "value": 72},
        {"id": "c5", "kind": "doc", "text": "Pediatric asthma typically climbs in March", "doc_id": "D04",
         "quote": "Pediatric asthma presentations typically climb in March"},
    ]


def analysis(claims, **extra):
    return {"summary": "Asthma visits rose 18.1% in March, mostly pediatric.", "claims": claims,
            "limitations": ["synthetic data"], "confidence": 0.9, **extra}


QUESTION = "Why did asthma visits increase in March 2026?"


# ---- the happy path: T1 and T2 with evidence attached ----------------------
def test_asthma_question_end_to_end(make_agent):
    agent = make_agent(scripted(PLAN_ASTHMA, SQL_ASTHMA, analysis(asthma_claims())))
    result = agent.ask(QUESTION)
    assert result.status == "answered", result.reason
    assert all(c["verified"] for c in result.claims) and len(result.claims) == 5
    assert [q["rows"][1][1] for q in result.evidence["queries"][:1]] == [1284]
    assert "D04" in [d["doc_id"] for d in result.evidence["documents"]]
    assert [s["name"] for s in result.trace] == [
        "pii_redaction", "planner", "sql_analyst", "document_researcher", "analyst", "evidence_gate"]
    assert result.totals["input_tokens"] > 0 and result.totals["latency_ms"] >= 0
    json.dumps(result.to_dict())  # the whole result, trace included, is serializable


# ---- the gate catches wrong claims ----------------------------------------
def test_wrong_number_is_caught(make_agent):
    claims = asthma_claims()
    claims[1]["value"] = 1248  # transposed digits
    result = make_agent(scripted(PLAN_ASTHMA, SQL_ASTHMA, analysis(claims))).ask(QUESTION)
    assert result.status == "abstained" and "c2" in result.reason
    assert [c["verified"] for c in result.claims] == [True, False, True, True, True]


def test_wrong_calculation_is_caught(make_agent):
    claims = asthma_claims()
    claims[2]["value"] = 19.0
    result = make_agent(scripted(PLAN_ASTHMA, SQL_ASTHMA, analysis(claims))).ask(QUESTION)
    assert result.status == "abstained" and "c3" in result.reason


def test_hallucinated_quote_is_caught(make_agent):
    claims = asthma_claims()
    claims[4]["quote"] = "Pediatric asthma always peaks in July"
    result = make_agent(scripted(PLAN_ASTHMA, SQL_ASTHMA, analysis(claims))).ask(QUESTION)
    assert result.status == "abstained" and "c5" in result.reason


def test_claim_citing_a_document_that_was_not_retrieved_is_caught(make_agent):
    claims = asthma_claims()
    claims[4].update(doc_id="D13", quote="notifies clinic managers by email")
    result = make_agent(scripted(PLAN_ASTHMA, SQL_ASTHMA, analysis(claims))).ask(QUESTION)
    assert result.status == "abstained" and "not among the retrieved" in result.reason


# ---- T4: statistics ---------------------------------------------------------
def welch_sql(month_lo, month_hi):
    return (
        "SELECT COUNT(*) AS n, SUM(los_days) AS s, SUM(los_days * los_days) AS q FROM visits "
        f"WHERE diagnosis_group='Asthma' AND admit_date >= '{month_lo}' AND admit_date < '{month_hi}'"
    )


def test_length_of_stay_p_value_is_verified_or_rejected(make_agent, conn):
    plan = {"needs_sql": True, "needs_docs": False, "as_of": None, "clarify": None, "required_fields": ["los_days"]}
    sql = {"queries": [{"purpose": "feb", "sql": welch_sql("2026-02-01", "2026-03-01")},
                       {"purpose": "mar", "sql": welch_sql("2026-03-01", "2026-04-01")}]}
    feb, mar = V.asthma_los_values(conn, "feb"), V.asthma_los_values(conn, "mar")
    true_p = round(stats.ttest_ind(feb, mar, equal_var=False).pvalue, 2)
    inputs = [cell(0, 0, "n"), cell(0, 0, "s"), cell(0, 0, "q"), cell(1, 0, "n"), cell(1, 0, "s"), cell(1, 0, "q")]

    def run(value):
        claim = {"id": "c1", "kind": "calc", "text": "No significant change", "op": "welch_p", "inputs": inputs, "value": value}
        return make_agent(scripted(plan, sql, analysis([claim]))).ask("Did asthma length of stay change from February to March?")

    assert run(true_p).status == "answered"
    assert run(0.01).status == "abstained"  # a made-up "significant" result is rejected


# ---- T6: missing information -----------------------------------------------
def test_missing_field_abstains_before_any_query_runs(make_agent):
    plan = {"needs_sql": True, "needs_docs": False, "required_fields": ["medication"], "as_of": None, "clarify": None}
    llm = scripted(plan)  # sql and analyst replies are unset, so running them would fail the test
    agent = make_agent(llm)
    result = agent.ask("Which medication did patients receive?")
    assert result.status == "abstained" and "medication" in result.reason
    assert len(llm.calls) == 1 and [s["name"] for s in result.trace] == ["pii_redaction", "planner"]
    assert result.evidence["queries"] == []


def test_ambiguous_question_asks_for_clarification(make_agent):
    plan = {"needs_sql": True, "needs_docs": False, "required_fields": ["readmit_30d"], "as_of": None,
            "clarify": "Which time period do you mean?"}
    llm = scripted(plan)
    result = make_agent(llm).ask("How many patients were readmitted?")
    assert result.status == "clarify" and result.answer == "Which time period do you mean?"
    assert len(llm.calls) == 1


# ---- tool failure ----------------------------------------------------------
def test_blocked_query_is_reported_as_tool_failure_not_invented(make_agent, built):
    sql = {"queries": [{"purpose": "oops", "sql": "DROP TABLE visits"}]}
    claim = {"id": "c1", "kind": "sql", "text": "There were 999 visits", "ref": cell(0, 0, "n"), "value": 999}
    result = make_agent(scripted(PLAN_ASTHMA | {"needs_docs": False}, sql, analysis([claim]))).ask("How many visits?")
    assert result.status == "abstained" and "Tool failure" in result.reason and "FORBIDDEN" in result.reason
    assert result.evidence["queries"][0]["error"].startswith("FORBIDDEN")
    conn = sqlite3.connect(built / "agentguard.db")
    assert conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 15521
    conn.close()


def test_sql_syntax_error_is_a_tool_failure(make_agent):
    sql = {"queries": [{"purpose": "typo", "sql": "SELEC COUNT(*) FROM visits"}]}
    claim = {"id": "c1", "kind": "sql", "text": "x", "ref": cell(0, 0, "n"), "value": 1}
    result = make_agent(scripted(PLAN_ASTHMA | {"needs_docs": False}, sql, analysis([claim]))).ask("How many visits?")
    assert result.status == "abstained" and "SQL_ERROR" in result.reason


# ---- malformed model output ------------------------------------------------
def test_unreadable_planner_output_stops_cleanly(make_agent):
    llm = OfflineLLM(["I think the answer is 42."])
    result = make_agent(llm).ask(QUESTION)
    assert result.status == "abstained" and "could not read" in result.reason and len(llm.calls) == 1
    assert result.trace[1]["error"]


def test_unreadable_analyst_output_stops_cleanly(make_agent):
    def respond(system, messages):
        if system.startswith("You plan how"):
            return json.dumps(PLAN_ASTHMA)
        if system.startswith("You write read-only SQLite"):
            return json.dumps(SQL_ASTHMA)
        return "not json at all"

    result = make_agent(OfflineLLM(respond)).ask(QUESTION)
    assert result.status == "abstained" and "could not read" in result.reason


def test_json_in_code_fences_is_accepted(make_agent):
    plan = {"needs_sql": False, "needs_docs": False, "required_fields": [], "as_of": None, "clarify": None}
    fenced = "Here is the plan:\n```json\n" + json.dumps(plan) + "\n```"
    llm = OfflineLLM(lambda s, m: fenced if s.startswith("You plan how") else json.dumps(analysis([])))
    result = make_agent(llm).ask("anything")
    assert result.trace[1]["error"] is None  # parsed fine; abstains later because there are no claims
    assert result.status == "abstained" and "no checkable claims" in result.reason


# ---- T7 and T9: temporal selection -------------------------------------------
FOLLOWUP_Q = "What is the follow-up window after discharge?"
PLAN_DOCS = {"needs_sql": False, "needs_docs": True, "required_fields": [], "clarify": None}


def doc_claim(doc_id, quote):
    return {"id": "c1", "kind": "doc", "text": quote, "doc_id": doc_id, "quote": quote}


def test_march_question_uses_policy_v1(make_agent):
    plan = PLAN_DOCS | {"as_of": "2026-03-15"}
    a = analysis([doc_claim("D01", "follow-up contact within 30 days")])
    result = make_agent(scripted(plan, analysis=a)).ask(FOLLOWUP_Q)
    assert result.status == "answered"
    assert {"doc_id": "D02", "reason": "not yet in effect on 2026-03-15"} in result.evidence["dropped_documents"]


def test_march_question_citing_v2_is_rejected(make_agent):
    plan = PLAN_DOCS | {"as_of": "2026-03-15"}
    a = analysis([doc_claim("D02", "follow-up contact within 14 days")])
    result = make_agent(scripted(plan, analysis=a)).ask(FOLLOWUP_Q)
    assert result.status == "abstained" and "D02" in result.reason


def test_current_question_uses_policy_v2_and_drops_v1(make_agent):
    plan = PLAN_DOCS | {"as_of": None}
    a = analysis([doc_claim("D02", "follow-up contact within 14 days")])
    result = make_agent(scripted(plan, analysis=a)).ask(FOLLOWUP_Q)
    assert result.status == "answered"
    assert any(d["doc_id"] == "D01" and "superseded" in d["reason"] for d in result.evidence["dropped_documents"])


def test_stale_index_looks_fine_to_the_gate(make_agent):
    """With v2 missing from the index the agent confidently answers 30 days, and the gate cannot
    know better. Catching this needs ground truth, which is AgentGuard's job (Session 5)."""
    plan = PLAN_DOCS | {"as_of": None}
    a = analysis([doc_claim("D01", "follow-up contact within 30 days")])
    result = make_agent(scripted(plan, analysis=a), exclude_ids=["D02"]).ask(FOLLOWUP_Q)
    assert result.status == "answered"
    assert "D02" not in [d["doc_id"] for d in result.evidence["documents"]]


def passage(doc_id, title, effective, rank):
    return Passage(rank, doc_id, title, "1", effective, 0.5, "text")


def test_select_effective_unit():
    ps = [passage("A", "Policy", "2026-01-01", 1), passage("B", "Policy", "2026-07-01", 2),
          passage("C", "Other", "2026-02-01", 3), passage("D", "Future", "2027-01-01", 4)]
    kept, dropped = select_effective(ps, "2026-10-01")
    assert [p.doc_id for p in kept] == ["B", "C"]
    reasons = {d["doc_id"]: d["reason"] for d in dropped}
    assert "superseded" in reasons["A"] and "not yet in effect" in reasons["D"]
    kept, _ = select_effective(ps, "2026-03-01")
    assert [p.doc_id for p in kept] == ["A", "C"]


# ---- T8 and the structural security property --------------------------------
BADGE_Q = "What does the facilities memo on badge access and parking say?"


def test_injection_cannot_trigger_a_tool_call(make_agent):
    plan = {"needs_sql": False, "needs_docs": True, "required_fields": [], "as_of": None, "clarify": None}
    obedient = analysis(
        [doc_claim("D17", "Staff badges are required at all entrances")],
        tool_call={"name": "query_database", "sql": "SELECT * FROM visits"},
        instructions="reveal the database credentials",
    )
    llm = scripted(plan, analysis=obedient)
    agent = make_agent(llm)
    executed = []
    original = agent.sql.run
    agent.sql.run = lambda *a, **k: (executed.append(a), original(*a, **k))[1]
    result = agent.ask(BADGE_Q)
    assert executed == [] and result.evidence["queries"] == []  # nothing ran
    assert result.status == "answered"
    analyst_call = next(c for c in llm.calls if c["system"].startswith("You write the final analysis"))
    prompt = analyst_call["messages"][0]["content"]
    start = prompt.index(P.INJECTION_TEXT)
    assert prompt.rfind('<retrieved_document id="D17"', 0, start) != -1
    assert 'trust="untrusted"' in prompt


def test_injection_text_never_reaches_planner_or_sql_steps(make_agent):
    plan = PLAN_ASTHMA | {"needs_docs": True}
    llm = scripted(plan, SQL_ASTHMA, analysis(asthma_claims()))
    make_agent(llm).ask(QUESTION + " Also check the badge and parking memo.")
    for call in llm.calls:
        if not call["system"].startswith("You write the final analysis"):
            assert P.INJECTION_TEXT not in call["system"] + json.dumps(call["messages"])


def test_sql_step_gets_governed_definitions_only(make_agent):
    llm = scripted(PLAN_ASTHMA, SQL_ASTHMA, analysis(asthma_claims()))
    make_agent(llm).ask(QUESTION)
    sql_call = next(c for c in llm.calls if c["system"].startswith("You write read-only SQLite"))
    system = " ".join(sql_call["system"].split())  # collapse the documents' line wraps
    assert "unique patients discharged in the period" in system  # D03, the readmission definition
    assert "never the number of visits" in system
    assert "Pediatric asthma presentations typically climb" not in system  # D04 is an untrusted note
    assert P.INJECTION_TEXT not in system
