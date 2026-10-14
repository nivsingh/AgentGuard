"""Results store: every AgentGuard run, kept in SQLite so runs can be compared later."""

import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT, model TEXT, condition TEXT,
    created REAL, notes TEXT
);
CREATE TABLE IF NOT EXISTS outcomes (
    run_id INTEGER, case_id TEXT, category TEXT, split TEXT, grp TEXT, passed INTEGER,
    status TEXT, failure_class TEXT, unsafe INTEGER, detail TEXT, answer TEXT,
    input_tokens INTEGER, output_tokens INTEGER, latency_ms REAL,
    PRIMARY KEY (run_id, case_id)
);
"""


class ResultStore:
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def save_run(self, label, model, condition, outcomes, notes=""):
        cur = self.conn.execute("INSERT INTO runs (label, model, condition, created, notes) VALUES (?,?,?,?,?)",
                                (label, model, condition, time.time(), notes))
        run_id = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [(run_id, o.case_id, o.category, o.split, o.group, int(o.passed), o.status, o.failure_class,
              int(o.unsafe), o.detail, o.answer, o.input_tokens, o.output_tokens, o.latency_ms) for o in outcomes])
        self.conn.commit()
        return run_id

    def runs(self):
        return self.conn.execute("SELECT run_id, label, model, condition, created, notes FROM runs ORDER BY run_id").fetchall()

    def outcomes(self, run_id):
        from guard.evaluators import Outcome

        rows = self.conn.execute(
            "SELECT case_id, category, split, grp, passed, status, failure_class, unsafe, detail, answer, "
            "input_tokens, output_tokens, latency_ms FROM outcomes WHERE run_id = ? ORDER BY case_id", (run_id,)).fetchall()
        return [Outcome(r[0], r[1], r[2], r[3], bool(r[4]), r[5], r[6], bool(r[7]), r[8], r[9], r[10], r[11], r[12]) for r in rows]
