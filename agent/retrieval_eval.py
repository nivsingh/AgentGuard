"""Measure the retrieval baseline on hand-written questions.

    python -m agent.retrieval_eval

Prints recall@1 and recall@3 so the result can go straight into experiments.md.
Each question lists every document that is a correct answer.
"""

import sys
from pathlib import Path

from agent.retrieval import build_index
from data.docs_util import load_docs

DEFAULT_DOCS = Path(__file__).resolve().parent.parent / "data" / "generated" / "docs"

QUESTIONS = [
    ("What is the follow-up window after discharge?", {"D01", "D02"}),
    ("How is the 30 day readmission rate defined?", {"D03"}),
    ("Why do pediatric asthma visits increase in March?", {"D04"}),
    ("What columns are in the visits table?", {"D05"}),
    ("How are patients assigned an acuity level in the emergency department?", {"D06"}),
    ("What should happen when an air quality alert is issued?", {"D13"}),
    ("How is length of stay reported and compared between months?", {"D16"}),
    ("How do I request extra staff for a unit?", {"D25"}),
    ("When must a patient transport request be submitted?", {"D27"}),
    ("How are visits assigned a diagnosis group?", {"D23"}),
]


def evaluate(index, k=3):
    hit1 = hit_k = 0
    misses = []
    for question, expected in QUESTIONS:
        ids = [p.doc_id for p in index.search(question, k=k)]
        hit1 += bool(ids and ids[0] in expected)
        ok = any(i in expected for i in ids)
        hit_k += ok
        if not (ids and ids[0] in expected):
            misses.append((question, sorted(expected), ids))
    n = len(QUESTIONS)
    return {"n": n, "recall_at_1": hit1, f"recall_at_{k}": hit_k, "misses": misses}


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCS
    index = build_index(load_docs(folder))
    r = evaluate(index)
    print(f"Questions: {r['n']}   recall@1: {r['recall_at_1']} / {r['n']}   recall@3: {r['recall_at_3']} / {r['n']}")
    for q, expected, got in r["misses"]:
        print(f"  not top-1: {q!r} expected {expected} got {got}")
