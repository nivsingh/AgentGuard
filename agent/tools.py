"""Deterministic helpers: calculations and claim verification.

The language model proposes claims. Nothing here trusts it: every number is
recomputed from the SQL results, and every quote is looked up in the retrieved
text. These functions are the core of the evidence gate.
"""

import json
import math
import re

from scipy import stats

# ---------------------------------------------------------------------------
# Calculations the agent may cite. Each takes plain numbers.
# ---------------------------------------------------------------------------


def pct_change(before, after):
    return (after - before) / before * 100


def share_of_change(part_before, part_after, total_before, total_after):
    return (part_after - part_before) / (total_after - total_before) * 100


def rate(numerator, denominator):
    return numerator / denominator * 100


def welch_p(n1, sum1, sumsq1, n2, sum2, sumsq2):
    """Two-sided Welch t-test p-value from aggregates (count, sum, sum of squares)."""

    def mean_sd(n, s, q):
        mean = s / n
        var = (q - n * mean * mean) / (n - 1)
        return mean, math.sqrt(max(var, 0.0))

    m1, s1 = mean_sd(n1, sum1, sumsq1)
    m2, s2 = mean_sd(n2, sum2, sumsq2)
    return float(stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, equal_var=False).pvalue)


CALC_OPS = {
    "pct_change": (pct_change, 2),
    "share_of_change": (share_of_change, 4),
    "rate": (rate, 2),
    "welch_p": (welch_p, 6),
}


# ---------------------------------------------------------------------------
# Number matching
# ---------------------------------------------------------------------------


def _decimals(stated):
    if isinstance(stated, int) and not isinstance(stated, bool):
        return 0
    text = repr(float(stated))
    return len(text.split(".")[1]) if "." in text and "e" not in text else 6


def numbers_match(stated, actual):
    """A stated number matches if it equals the true value rounded to the stated precision."""
    if stated is None or actual is None:
        return False
    if isinstance(stated, bool) or isinstance(actual, bool):
        return False
    try:
        return round(float(actual), _decimals(stated)) == float(stated)
    except (TypeError, ValueError):
        return False


def _norm(text):
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Claim verification
# ---------------------------------------------------------------------------


def cell(sql_results, ref):
    """Look up {"query": i, "row": r, "col": name_or_index} in the SQL results."""
    try:
        res = sql_results[ref["query"]]
        if res.get("error"):
            return None
        col = ref["col"]
        idx = res["columns"].index(col) if isinstance(col, str) else int(col)
        return res["rows"][int(ref["row"])][idx]
    except (KeyError, IndexError, ValueError, TypeError):
        return None


def _resolve(arg, sql_results):
    if isinstance(arg, dict):
        return cell(sql_results, arg)
    if isinstance(arg, (int, float)) and not isinstance(arg, bool):
        return arg
    return None


def verify_claim(claim, sql_results, kept_passages):
    """Return (verified: bool, reason: str). `kept_passages` are the passages that survived temporal selection."""
    kind = claim.get("kind")

    if kind == "sql":
        actual = cell(sql_results, claim.get("ref", {}))
        if actual is None:
            return False, "the referenced SQL cell does not exist"
        stated = claim.get("value")
        if isinstance(actual, str) or isinstance(stated, str):
            ok = str(actual) == str(stated)
        else:
            ok = numbers_match(stated, actual)
        return ok, "matches the SQL result" if ok else f"SQL result is {actual!r}, claim says {stated!r}"

    if kind == "calc":
        op = claim.get("op")
        if op not in CALC_OPS:
            return False, f"unknown calculation {op!r}"
        fn, _ = CALC_OPS[op]
        args = [_resolve(a, sql_results) for a in claim.get("inputs", [])]
        if any(a is None for a in args):
            return False, "a calculation input could not be resolved from SQL results"
        try:
            actual = fn(*args)
        except (TypeError, ZeroDivisionError, ValueError) as e:
            return False, f"calculation failed: {e}"
        ok = numbers_match(claim.get("value"), actual)
        return ok, "recomputed value matches" if ok else f"recomputed {op} is {actual:.6g}, claim says {claim.get('value')!r}"

    if kind == "doc":
        doc_id, quote = claim.get("doc_id"), claim.get("quote", "")
        passage = next((p for p in kept_passages if p.doc_id == doc_id), None)
        if passage is None:
            return False, f"{doc_id} is not among the retrieved documents in effect"
        if not quote or _norm(quote) not in _norm(passage.text):
            return False, f"the quoted text does not appear in {doc_id}"
        return True, f"quote found in {doc_id}"

    return False, f"unknown claim kind {kind!r}"


# ---------------------------------------------------------------------------
# Reading model output
# ---------------------------------------------------------------------------


def parse_json_object(text):
    """Pull one JSON object out of model output (tolerates code fences and a short preamble)."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in model output")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj
