"""Independent ground-truth queries, written as plain SQL.

These functions never touch the generator. They answer each planted question
straight from the database, so a test can compare the result with truths.json.
AgentGuard will reuse them later to re-run any number an agent quotes.
"""

import sqlite3

ASTHMA_MONTH = (
    "SELECT COUNT(*) FROM visits "
    "WHERE diagnosis_group = 'Asthma' AND admit_date >= :lo AND admit_date < :hi"
)

MONTH_RANGES = {
    "feb": ("2026-02-01", "2026-03-01"),
    "mar": ("2026-03-01", "2026-04-01"),
}


def _one(conn, sql, params=()):
    return conn.execute(sql, params).fetchone()[0]


def asthma_visits(conn, month):
    lo, hi = MONTH_RANGES[month]
    return _one(conn, ASTHMA_MONTH, {"lo": lo, "hi": hi})


def asthma_visits_by_age_group(conn, month):
    lo, hi = MONTH_RANGES[month]
    rows = conn.execute(
        "SELECT age_group, COUNT(*) FROM visits "
        "WHERE diagnosis_group = 'Asthma' AND admit_date >= ? AND admit_date < ? "
        "GROUP BY age_group",
        (lo, hi),
    ).fetchall()
    return dict(rows)


def asthma_visits_by_borough(conn, month):
    lo, hi = MONTH_RANGES[month]
    rows = conn.execute(
        "SELECT borough, COUNT(*) FROM visits "
        "WHERE diagnosis_group = 'Asthma' AND admit_date >= ? AND admit_date < ? "
        "GROUP BY borough",
        (lo, hi),
    ).fetchall()
    return dict(rows)


def asthma_los_values(conn, month):
    lo, hi = MONTH_RANGES[month]
    rows = conn.execute(
        "SELECT los_days FROM visits "
        "WHERE diagnosis_group = 'Asthma' AND admit_date >= ? AND admit_date < ?",
        (lo, hi),
    ).fetchall()
    return [r[0] for r in rows]


Q1 = ("2026-01-01", "2026-04-01")


def q1_unique_discharged_patients(conn):
    return _one(
        conn,
        "SELECT COUNT(DISTINCT patient_id) FROM visits WHERE discharge_date >= ? AND discharge_date < ?",
        Q1,
    )


def q1_readmitted_patients(conn):
    return _one(
        conn,
        "SELECT COUNT(DISTINCT patient_id) FROM visits "
        "WHERE readmit_30d = 1 AND discharge_date >= ? AND discharge_date < ?",
        Q1,
    )


def q1_visit_count(conn):
    return _one(
        conn,
        "SELECT COUNT(*) FROM visits WHERE discharge_date >= ? AND discharge_date < ?",
        Q1,
    )


def q1_flagged_visits(conn):
    return _one(
        conn,
        "SELECT COUNT(*) FROM visits WHERE readmit_30d = 1 AND discharge_date >= ? AND discharge_date < ?",
        Q1,
    )


def correct_readmission_rate_pct(conn):
    return round(q1_readmitted_patients(conn) / q1_unique_discharged_patients(conn) * 100, 1)


def naive_readmission_rate_pct(conn):
    """The wrong metric: flagged visits over all visits. Kept so tests can show it differs."""
    return round(q1_flagged_visits(conn) / q1_visit_count(conn) * 100, 1)


def schema_names(conn):
    """Every table and column name in the database, lower case."""
    names = []
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        names.append(table.lower())
        for col in conn.execute(f"PRAGMA table_info({table})"):
            names.append(col[1].lower())
    return names


def open_db(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    return conn


# ---------------------------------------------------------------------------
# Generic independent queries used by AgentGuard to compute expected answers.
# ---------------------------------------------------------------------------
FILTER_COLUMNS = ("diagnosis_group", "age_group", "borough", "department")


def month_range(month):
    """(first day, first day of next month) for a 2026 month number 1 to 12."""
    nxt = f"2026-{month + 1:02d}-01" if month < 12 else "2027-01-01"
    return f"2026-{month:02d}-01", nxt


def visits_count(conn, month, **filters):
    """Visits admitted in a month, optionally filtered by diagnosis_group, age_group, borough, department."""
    lo, hi = month_range(month)
    where, params = ["admit_date >= ?", "admit_date < ?"], [lo, hi]
    for col, value in filters.items():
        assert col in FILTER_COLUMNS, col
        where.append(f"{col} = ?")
        params.append(value)
    return _one(conn, "SELECT COUNT(*) FROM visits WHERE " + " AND ".join(where), params)


def los_values(conn, month, diagnosis_group="Asthma"):
    lo, hi = month_range(month)
    rows = conn.execute(
        "SELECT los_days FROM visits WHERE diagnosis_group = ? AND admit_date >= ? AND admit_date < ?",
        (diagnosis_group, lo, hi),
    ).fetchall()
    return [r[0] for r in rows]


def table_fingerprint(conn):
    """A cheap fingerprint of the visits table. If a query ever changed the data, this changes."""
    return conn.execute(
        "SELECT COUNT(*), SUM(visit_id), SUM(los_days), SUM(readmit_30d), SUM(patient_id) FROM visits"
    ).fetchone()
