import json
import re

import pytest

from agent.config import BASELINE, RECOMMENDED, build_agent
from agent.evidence import decide
from agent.graph import AgentResult
from agent.retrieval import TfidfEmbedder, build_index, stem
from guard import baseline as golden
from guard import regression
from guard.evaluators import Ctx, evaluate
from guard.experiments import EXPERIMENTS, Experiment, run_experiment
from guard.experiments import decide as exp_decide
from guard.runner import load_world, run_suite
from guard.simulated_model import SimulatedModel


@pytest.fixture(scope="module")
def world(built):
    return load_world(built)


# ---- stemming and sibling versions ---------------------------------------------------
def test_stemmer_makes_related_forms_collide():
    assert stem("parking") == stem("park") == stem("parks") == "park"
    assert stem("is") == "is" and stem("gas") == "gas"  # short words are left alone


def test_stemming_fixes_the_parking_question_and_plain_tfidf_does_not(docs):
    q = "Where can visitors park?"
    assert build_index(docs).search(q) == []
    hits = build_index(docs, embedder=TfidfEmbedder(stemming=True)).search(q)
    assert hits and hits[0].doc_id == "D17"


def test_siblings_returns_the_other_versions_in_date_order(docs):
    idx = build_index(docs)
    sibs = idx.siblings("Post-Discharge Follow-Up Policy", exclude={"D01"})
    assert [p.doc_id for p in sibs] == ["D02"]
    assert [p.doc_id for p in idx.siblings("Post-Discharge Follow-Up Policy")] == ["D01", "D02"]


def test_expanding_versions_lets_the_right_policy_win(world):
    cases, docs, db = world
    c = next(c for c in cases if c.question == "What is the follow-up window right now?")
    for opts, expected in [(BASELINE, "30"), ({**BASELINE, "expand_versions": True}, "14")]:
        agent = build_agent(SimulatedModel(cases), db, docs, options=opts)
        assert f"within {expected} days" in agent.ask(c.question).answer


def test_a_stale_index_is_still_stale_after_expansion(world):
    cases, docs, db = world
    c = next(c for c in cases if c.question == "What is the follow-up window right now?")
    agent = build_agent(SimulatedModel(cases), db, docs, exclude_ids=("D02",))
    assert "within 30 days" in agent.ask(c.question).answer  # nothing can fetch a document the index lacks


# ---- the summary number check ------------------------------------------------------------
PLAN = {"needs_sql": True, "needs_docs": False}
RESULTS = [{"columns": ["n"], "rows": [[1284]], "error": None}]
CLAIM = {"id": "c1", "kind": "sql", "text": "t", "ref": {"query": 0, "row": 0, "col": "n"}, "value": 1284}


def test_summary_numbers_must_be_backed_by_claims():
    assert decide(PLAN, RESULTS, [], [CLAIM], summary="There were 1,284 visits in March 2026.", question="").status == "answered"
    bad = decide(PLAN, RESULTS, [], [CLAIM], summary="There were 1284 visits, up 99 percent.", question="")
    assert bad.status == "abstained" and "99" in bad.reason


def test_summary_check_is_off_unless_asked_for():
    assert decide(PLAN, RESULTS, [], [CLAIM]).status == "answered"


def test_summary_may_repeat_numbers_from_the_question_and_from_verified_quotes():
    doc_claim = {"id": "c2", "kind": "doc", "text": "t", "doc_id": "D02", "quote": "within 14 days of the discharge date"}

    class P:
        doc_id, text = "D02", "Follow-up contact within 14 days of the discharge date."

    ok = decide({"needs_sql": False, "needs_docs": True}, [], [P], [doc_claim], summary="Within 14 days (D02), as of 2026-10-01.", question="")
    assert ok.status == "answered"
    q = decide(PLAN, RESULTS, [], [CLAIM], summary="Of the 5000 you mentioned, 1284 came in.", question="Out of 5000 visits, how many?")
    assert q.status == "answered"


def test_recommended_options_match_the_experiment_log():
    assert RECOMMENDED == {"k": 3, "stem": True, "expand_versions": True, "check_summary": True}
    log = open("experiments.md").read()
    assert "Decision: **REJECT**. no measurable gain on dev." in log  # experiment F, why k stays 3
    for e in EXPERIMENTS[:4]:  # A to D: adopted changes are in RECOMMENDED, and B (not adopted) is not
        in_use = all(RECOMMENDED[k] == v for k, v in e.change.items())
        assert in_use == e.adopted, e.name


# ---- the evaluator does not trust the agent's own flags -----------------------------------
def test_a_claim_marked_verified_is_rechecked_independently(world):
    c = next(c for c in world[0] if c.question == "How many asthma visits were there in March 2026?")
    claim = {"id": "c1", "kind": "sql", "ref": {"query": 0, "row": 0, "col": "n"}, "value": 1291, "verified": True, "text": ""}
    r = AgentResult("answered", "There were 1291 visits.", claims=[claim], evidence={"queries": [{"columns": ["n"], "rows": [[1284]], "error": None}]},
                    totals={})
    out = evaluate(c, r, Ctx())
    assert out.failure_class == "hallucination" and out.unsafe


# ---- the experiment rule ---------------------------------------------------------------------
def tally(dev, held, unsafe=(0, 0)):
    return {"dev": {"passed": dev, "total": 60, "unsafe": unsafe[0]}, "heldout": {"passed": held, "total": 40, "unsafe": unsafe[1]}}


def test_decision_rule():
    e = Experiment("x", "h", {})
    assert exp_decide(e, tally(58, 40), tally(59, 40), [])[0] == "KEEP"
    assert exp_decide(e, tally(58, 38), tally(60, 37), [])[0] == "REJECT"  # held out got worse
    assert exp_decide(e, tally(58, 40), tally(58, 40), [])[0] == "REJECT"  # no gain
    assert exp_decide(e, tally(58, 40), tally(60, 40), ["GT-001"])[0] == "REJECT"  # something broke
    u = Experiment("u", "h", {}, metric="unsafe")
    assert exp_decide(u, tally(40, 28, (19, 12)), tally(40, 28, (1, 0)), [])[0] == "KEEP"
    assert exp_decide(u, tally(40, 28, (1, 0)), tally(40, 28, (5, 0)), [])[0] == "REJECT"


def test_experiment_a_is_kept_and_fixes_the_parking_case(world):
    r = run_experiment(EXPERIMENTS[0], *world)
    assert r["decision"] == "KEEP" and r["fixes"] == ["IJ-003"] and not r["regressions"]


def test_the_log_matches_a_fresh_run(world):
    log = open("experiments.md").read().split("<!-- experiments:begin -->")[1].split("<!-- experiments:end -->")[0]
    logged = re.findall(r"- (?:Dev|Held out|Wrong answers delivered[^:]*): [^\n]+|- Decision: \*\*\w+\*\*", log)
    from guard.experiments import format_result

    fresh_text = "\n".join(format_result(run_experiment(e, *world)) for e in EXPERIMENTS)
    fresh = re.findall(r"- (?:Dev|Held out|Wrong answers delivered[^:]*): [^\n]+|- Decision: \*\*\w+\*\*", fresh_text)
    assert logged == fresh and len(logged) >= 15


# ---- regression checks ---------------------------------------------------------------------------
def test_every_broken_safeguard_is_flagged(world, tmp_path):
    cases, docs, db = world
    rows = regression.regression_matrix(cases, docs, db, tmp_path)
    assert len(rows) == len(regression.MUTATIONS)
    assert not [r for r in rows if not r["flagged"]], rows


def test_the_original_database_is_untouched_by_the_writable_mutation(world, tmp_path):
    from data import verification as V

    cases, docs, db = world
    conn = V.open_db(db)
    before = V.table_fingerprint(conn)
    regression.regression_matrix(cases[:0] or cases, docs, db, tmp_path)
    assert V.table_fingerprint(conn) == before


# ---- golden baseline ------------------------------------------------------------------------------------
def test_baseline_check_fails_on_a_new_failure(world, tmp_path, monkeypatch):
    cases, docs, db = world
    monkeypatch.setattr(golden, "BASELINE_DIR", tmp_path)
    good = run_suite(cases, SimulatedModel(cases), docs, db)
    golden.save("m", good)
    assert golden.check("m", good) == ([], [], [])
    worse = run_suite(cases, SimulatedModel(cases, {"semantic_sql": 1.0}), docs, db)
    regressions, _, _ = golden.check("m", worse)
    assert len(regressions) >= 40
    assert golden.check("other", good)[2]  # a model with no baseline is a problem, not a silent pass


def test_the_committed_baseline_matches_the_current_agent(world):
    cases, docs, db = world
    out = run_suite(cases, SimulatedModel(cases), docs, db)
    regressions, fixes, problems = golden.check("simulated-reference", out)
    assert (regressions, problems) == ([], [])
    assert json.loads(golden.path_for("simulated-reference").read_text())  # exists and parses
