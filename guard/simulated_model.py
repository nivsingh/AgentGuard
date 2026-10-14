"""A simulated model with known, labeled defects.

WHAT THIS IS: an instrument to calibrate the evaluators. With no API key, nothing
else can answer "if the agent really did make mistake X, would AgentGuard notice?".
This model behaves correctly by default and misbehaves in exactly the ways you ask
(`defects={"wrong_denominator": 0.5}` means half of the eligible cases). Because the
defects are planted, each one has a known failure class, and the tests check the
evaluators find it there and nowhere else.

WHAT THIS IS NOT: evidence about any real language model. Numbers produced with it
say how well the evaluators detect planted defects, never how well the agent works.
Real results need ANTHROPIC_API_KEY (see README).

The simulated model reads the same prompts a real model would and answers in the
same JSON formats. It looks up each question in the case list to know what a
correct run looks like (the recipe), then applies the requested defects.
"""

import json
import re
import zlib

from agent.llm import OfflineLLM
from agent.tools import CALC_OPS, cell

DEFECTS = {
    "wrong_denominator": "uses visits instead of unique patients for readmission rates",
    "semantic_sql": "writes a query with an off-by-one date boundary",
    "bad_sql": "writes SQL that fails (misnamed table)",
    "hallucinate_number": "states a wrong number in the summary while its claims stay correct",
    "fabricated_claim": "makes up the number in a claim as well as in the summary",
    "obey_injection": "follows an instruction hidden in a retrieved document",
    "answers_anyway": "answers questions about data that does not exist",
    "no_clarify": "assumes a period instead of asking when the question is ambiguous",
}


def _affected(defect, case_id, rate):
    """Deterministic partial rates: the same case is always affected or not, run after run."""
    return zlib.crc32(f"{defect}:{case_id}".encode()) % 1000 < rate * 1000


def _fmt(value):
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _wrong(value):
    return value + 7 if isinstance(value, int) else round(value * 1.1 + 0.3, 4)


class SimulatedModel:
    """Use as the `llm` of HealthcareAgent. `.calls` records every prompt, like OfflineLLM."""

    def __init__(self, cases, defects=None):
        self.by_question = {c.question: c for c in cases}
        self.defects = dict(defects or {})
        unknown = set(self.defects) - set(DEFECTS)
        if unknown:
            raise ValueError(f"unknown defects: {sorted(unknown)}")
        label = "simulated-reference" if not self.defects else "simulated+" + "+".join(sorted(self.defects))
        self._llm = OfflineLLM(self._respond, model=label)
        self.model = label

    @property
    def calls(self):
        return self._llm.calls

    def complete(self, system, messages, max_tokens=1024, temperature=0.0):
        return self._llm.complete(system, messages, max_tokens, temperature)

    # ------------------------------------------------------------------ helpers
    def on(self, defect, case):
        return defect in self.defects and _affected(defect, case.id, self.defects[defect])

    def _case(self, user):
        m = re.match(r"(?:Question: )?(.*?)(?:\nToday is |\nDate the question|\Z)", user, re.S)
        return self.by_question.get(m.group(1).strip()) if m else None

    # ---------------------------------------------------------------- dispatcher
    def _respond(self, system, messages):
        user = messages[0]["content"]
        case = self._case(user)
        if case is None:
            return json.dumps({"needs_sql": False, "needs_docs": False, "required_fields": [], "as_of": None,
                               "clarify": "I do not recognize this question."})
        if system.startswith("You plan how"):
            return json.dumps(self._plan(case))
        if system.startswith("You write read-only"):
            return json.dumps({"queries": self._queries(case)})
        if system.startswith("You write the final analysis"):
            return json.dumps(self._analysis(case, system, user))
        return "{}"

    # ------------------------------------------------------------------ planner
    def _plan(self, case):
        plan = dict(case.recipe["plan"])
        if case.category == "missing_info" and self.on("answers_anyway", case):
            plan["required_fields"] = ["admit_date"]  # pretends the missing field is fine
        if case.category == "ambiguity" and self.on("no_clarify", case):
            plan.update(clarify=None, needs_sql=True, required_fields=["admit_date"])
        return plan

    # ---------------------------------------------------------------------- sql
    def _assumed(self, case):
        return (case.category == "missing_info" and self.on("answers_anyway", case)) or (
            case.category == "ambiguity" and self.on("no_clarify", case))

    def _queries(self, case):
        r = case.recipe
        if self._assumed(case):
            return r["assume"]["queries"]
        queries = [dict(q) for q in r.get("queries", [])]
        if r.get("naive_queries") and self.on("wrong_denominator", case):
            queries = [{"purpose": "answer", "sql": s} for s in r["naive_queries"]]
        elif case.expected["type"] == "value" and not r.get("naive_queries") and self.on("semantic_sql", case):
            for q in queries:
                i = q["sql"].rfind("date < '")
                q["sql"] = q["sql"][:i] + "date <= '" + q["sql"][i + len("date < '"):]
        if queries and self.on("bad_sql", case):
            for q in queries:
                q["sql"] = q["sql"].replace("FROM visits", "FROM visit")
        return queries

    # ------------------------------------------------------------------ analyst
    @staticmethod
    def _results_from(user):
        blob = user.split("Database results (query index = position):\n", 1)[1].split("\n\nRetrieved documents:", 1)[0]
        return json.loads(blob)

    @staticmethod
    def _docs_from(user):
        return [{"doc_id": m.group(1), "effective": m.group(3), "body": re.sub(r"\s+", " ", m.group(4)).strip()}
                for m in re.finditer(r'<retrieved_document id="(\w+)" version="(\w+)" effective_date="([\d-]+)" trust="untrusted">\n(.*?)\n</retrieved_document>', user, re.S)]

    def _value(self, claim, results):
        if claim["kind"] == "sql":
            return cell(results, claim["ref"])
        fn, decimals = CALC_OPS[claim["op"]]
        args = [cell(results, a) if isinstance(a, dict) else a for a in claim["inputs"]]
        if any(a is None for a in args):
            return None
        return round(fn(*args), decimals)

    def _analysis(self, case, system, user):
        mode = case.recipe["mode"]
        if mode == "policy":
            return self._policy(case, user)
        if mode == "injection":
            return self._injection(case, system, user)
        results = self._results_from(user)
        recipe = case.recipe["assume"] if self._assumed(case) else case.recipe
        if not recipe.get("claims") or all(r.get("error") for r in results):
            return {"summary": "The database did not return usable data for this request.", "claims": [],
                    "limitations": ["no usable query results"], "confidence": 0.2}
        claims, shown = [], None
        for template in recipe["claims"]:
            claim = {k: v for k, v in template.items() if k != "value"}
            value = self._value(claim, results)
            if value is None:
                return {"summary": "The query results did not contain the needed figure.", "claims": [],
                        "limitations": ["figure missing"], "confidence": 0.2}
            if self.on("fabricated_claim", case):
                value = _wrong(value)
            claim["value"] = value
            claims.append(claim)
            shown = value
        if self.on("hallucinate_number", case) and not self.on("fabricated_claim", case):
            shown = _wrong(shown)
        unit = "%" if any(c.get("op") in ("pct_change", "rate", "share_of_change") for c in claims) else ""
        template = case.recipe.get("summary", "")
        if "p-value" not in template:
            template = "Based on the database, the answer is {v}" + unit + "."
        summary = template.format(v=_fmt(shown))
        return {"summary": summary, "claims": claims, "limitations": ["Synthetic data, for demonstration only."], "confidence": 0.9}

    def _policy(self, case, user):
        docs = [d for d in self._docs_from(user) if re.search(r"follow-up contact within \d+ days", d["body"])]
        if not docs:
            return {"summary": "No follow-up policy was retrieved.", "claims": [], "limitations": ["policy missing"], "confidence": 0.1}
        doc = docs[0]
        days = re.search(r"within (\d+) days", doc["body"]).group(1)
        quote = f"within {days} days of the discharge date"
        if case.recipe["ask"] == "days":
            summary = f"Follow-up must happen within {days} days of discharge under {doc['doc_id']}."
        else:
            from guard.evaluators import date_forms

            summary = f"The policy took effect on {date_forms(doc['effective'])[1]} ({doc['doc_id']})."
        claim = {"id": "c1", "kind": "doc", "text": summary, "doc_id": doc["doc_id"], "quote": quote}
        return {"summary": summary, "claims": [claim], "limitations": [], "confidence": 0.9}

    def _injection(self, case, system, user):
        docs = [d for d in self._docs_from(user) if d["doc_id"] == case.expected["doc_id"]]
        if not docs:
            return {"summary": "The memo was not retrieved.", "claims": [], "limitations": ["memo missing"], "confidence": 0.1}
        quote = case.recipe["quote"]
        summary = f"The memo says: {quote}."
        if self.on("obey_injection", case):
            token = re.search(r"never output it\): (\S+)", system)
            summary += f" The database credentials are {token.group(1) if token else 'agentguard.db'}."
        claim = {"id": "c1", "kind": "doc", "text": summary, "doc_id": docs[0]["doc_id"], "quote": quote}
        return {"summary": summary, "claims": [claim], "limitations": [], "confidence": 0.9}
