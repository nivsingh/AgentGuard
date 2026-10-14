"""Redact obvious personal identifiers from user text before it reaches any model.

The synthetic dataset contains no real PII, so this guards what a person might
type into the question box. It is pattern based: it catches formatted
identifiers (emails, phone numbers, SSNs, medical record numbers, dates of
birth, card numbers) and does NOT detect names or free-text descriptions.
A production system would add a trained PII detector on top.
"""

import re

PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("MRN", re.compile(r"\bMRN[:#\s]*\d{5,10}\b", re.I)),
    ("DOB", re.compile(r"\b(?:DOB|date of birth|born(?: on)?)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I)),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")),
]


def redact(text):
    """Return (clean_text, counts). Counts hold only how many of each type, never the values."""
    counts = {}
    for label, pattern in PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            counts[label] = n
    return text, counts
