"""Read the synthetic documents and pick the right policy version for a date."""

from datetime import date
from pathlib import Path


def parse_doc(path):
    text = Path(path).read_text()
    _, header, body = text.split("---", 2)
    meta = {}
    for line in header.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    meta["body"] = body.strip()
    return meta


def load_docs(folder):
    return [parse_doc(p) for p in sorted(Path(folder).glob("*.md"))]


def policy_in_effect(docs, title, on_date):
    """Return the version of a titled policy whose effective date covers on_date.

    A version is in effect from its effective date until the next version of
    the same title takes over. Returns None if nothing covers the date.
    """
    target = date.fromisoformat(on_date) if isinstance(on_date, str) else on_date
    versions = sorted(
        (d for d in docs if d["title"] == title),
        key=lambda d: date.fromisoformat(d["effective_date"]),
    )
    chosen = None
    for doc in versions:
        if date.fromisoformat(doc["effective_date"]) <= target:
            chosen = doc
    return chosen
