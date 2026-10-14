"""Reading numbers out of free text, shared by the evidence gate and the AgentGuard evaluators."""

import re

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]
_MONTH = "|".join(MONTH_NAMES)
_DATE_PATTERNS = [
    re.compile(r"\d{4}-\d{2}-\d{2}"),
    re.compile(rf"(?:{_MONTH})\s+(?:\d{{1,2}},?\s+)?20\d\d"),
    re.compile(rf"(?:{_MONTH})\s+\d{{1,2}}\b"),
    re.compile(r"\b(?:in|of|for|during|since|from|to)\s+20\d\d\b"),
]
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?:e-?\d+)?")


def numbers_in(text):
    """Numbers stated in text as (value, decimals), skipping dates, years and ids like D02."""
    for pat in _DATE_PATTERNS:
        text = pat.sub(" ", text)
    out = []
    for tok in _NUMBER.findall(text):
        clean = tok.replace(",", "")
        if clean.endswith("."):
            continue
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean and "e" not in clean else (6 if "e" in clean else 0)
        out.append((value, decimals))
    return out


def close(stated, decimals, actual):
    """A stated number matches the truth if it is the truth rounded to the stated precision."""
    return abs(stated - float(actual)) <= 0.5 * 10 ** (-decimals) + 1e-9
