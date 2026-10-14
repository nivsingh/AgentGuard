"""The evidence gate.

Answer only when the evidence is sufficient. Confidence is reported as metadata
and never decides whether the agent may answer.

    1. Is the required data available?        no  -> abstain (checked before any query runs)
    2. Did the tools run?                     no  -> abstain (tool failure, named)
    3. Is there supporting evidence?          no  -> abstain
    4. Can every claim be verified?           no  -> abstain, listing what failed
    otherwise answer
"""

from dataclasses import dataclass, field

from agent.textnums import close, numbers_in
from agent.tools import verify_claim


@dataclass
class GateDecision:
    status: str  # "answered" or "abstained"
    reason: str = ""
    checks: list = field(default_factory=list)  # one dict per claim
    missing: list = field(default_factory=list)


def check_required_fields(required_fields, schema_names):
    """Fields the question needs that do not exist anywhere in the dataset."""
    known = set(schema_names)
    return [f for f in required_fields if f.lower() not in known]


def unsupported_summary_numbers(summary, question, checks, claims):
    """Numbers in the summary that no verified claim backs: claim values, numbers inside verified quotes, or the question."""
    ok_ids = {c["id"] for c in checks if c["verified"]}
    support = [v for v, _ in numbers_in(question)]
    for c in claims:
        if c.get("id") in ok_ids:
            if isinstance(c.get("value"), (int, float)) and not isinstance(c.get("value"), bool):
                support.append(c["value"])
            support.extend(v for v, _ in numbers_in(str(c.get("quote", ""))))
    return [v for v, d in numbers_in(summary) if not any(close(v, d, s) for s in support)]


def decide(plan, sql_results, kept_passages, claims, summary=None, question=""):
    checks = []
    for c in claims:
        ok, why = verify_claim(c, sql_results, kept_passages)
        checks.append({"id": c.get("id"), "kind": c.get("kind"), "text": c.get("text"), "verified": ok, "why": why})

    if plan.get("needs_sql") and sql_results and all(r.get("error") for r in sql_results):
        codes = sorted({r["error"].split(":")[0] for r in sql_results})
        return GateDecision(
            "abstained",
            f"Tool failure: every database query failed ({', '.join(codes)}). I will not guess a number.",
            checks,
            missing=["database result"],
        )
    if plan.get("needs_sql") and not sql_results:
        return GateDecision("abstained", "No database query was produced, so there is no data to answer from.", checks, ["database result"])
    if plan.get("needs_docs") and not kept_passages:
        return GateDecision("abstained", "No supporting document was found for this question.", checks, ["supporting document"])
    if not claims:
        return GateDecision("abstained", "The analysis produced no checkable claims.", checks, ["verifiable claim"])

    failed = [c for c in checks if not c["verified"]]
    if failed:
        listing = "; ".join(f'{c["id"]}: {c["why"]}' for c in failed)
        return GateDecision("abstained", f"{len(failed)} of {len(checks)} claims could not be independently verified ({listing}).", checks, ["verified claims"])

    if summary is not None:  # optional check: the text the user reads may not state numbers the claims do not back
        stray = unsupported_summary_numbers(summary, question, checks, claims)
        if stray:
            return GateDecision("abstained", f"The summary states {stray[0]:g}, which no verified claim supports.", checks, ["summary numbers"])
    return GateDecision("answered", "All claims independently verified.", checks)
