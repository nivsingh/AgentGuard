import sqlite3

import pytest

from agent.sql_guard import ReadOnlySQL, SqlGuardError
from data import verification as V


@pytest.fixture(scope="module")
def sql(built):
    return ReadOnlySQL(built / "agentguard.db", max_rows=50, timeout_s=1.0)


def code_of(sql, query):
    with pytest.raises(SqlGuardError) as err:
        sql.run(query)
    return err.value.code


def test_valid_select_matches_ground_truth(sql, conn):
    result = sql.run(
        "SELECT COUNT(*) FROM visits WHERE diagnosis_group='Asthma' "
        "AND admit_date >= '2026-03-01' AND admit_date < '2026-04-01'"
    )
    assert result.scalar() == V.asthma_visits(conn, "mar") == 1284
    assert result.columns and not result.truncated


def test_group_by_and_cte_are_allowed(sql):
    result = sql.run(
        "WITH m AS (SELECT substr(admit_date,1,7) AS month, COUNT(*) AS n FROM visits GROUP BY 1) "
        "SELECT month, n FROM m ORDER BY month"
    )
    assert [r[0] for r in result.rows] == [f"2026-0{i}" for i in range(1, 7)]


def test_join_across_allowed_tables(sql):
    result = sql.run(
        "SELECT d.kind, COUNT(*) FROM visits v JOIN departments d ON d.department_name = v.department GROUP BY d.kind"
    )
    assert result.row_count > 0


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO visits (visit_id) VALUES (1)",
        "UPDATE visits SET los_days = 99",
        "DELETE FROM visits",
        "DROP TABLE visits",
        "CREATE TABLE x (a INT)",
        "ALTER TABLE visits ADD COLUMN z INT",
        "PRAGMA table_info(visits)",
        "ATTACH DATABASE ':memory:' AS other",
        "VACUUM",
    ],
)
def test_non_select_is_refused(sql, query):
    assert code_of(sql, query) in {"FORBIDDEN", "SQL_ERROR"}


def test_multiple_statements_refused(sql):
    assert code_of(sql, "SELECT 1; DROP TABLE visits") == "MULTI_STATEMENT"


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM sqlite_master",
        "SELECT name FROM sqlite_schema",
        "SELECT * FROM visits WHERE patient_id IN (SELECT rootpage FROM sqlite_master)",
        "SELECT * FROM pragma_table_info('visits')",
    ],
)
def test_non_allowlisted_tables_refused(sql, query):
    assert code_of(sql, query) == "FORBIDDEN"


def test_dangerous_function_refused(sql):
    assert code_of(sql, "SELECT load_extension('x')") == "FORBIDDEN"


def test_row_limit_truncates(sql):
    result = sql.run("SELECT visit_id FROM visits")
    assert result.row_count == 50 and result.truncated is True


def test_row_limit_not_flagged_when_under(sql):
    result = sql.run("SELECT visit_id FROM visits LIMIT 10")
    assert result.row_count == 10 and result.truncated is False


def test_timeout_interrupts_runaway_query(sql):
    runaway = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
        "SELECT COUNT(*) FROM c"
    )
    assert code_of(sql, runaway) == "TIMEOUT"


def test_syntax_error_is_reported_not_raised_raw(sql):
    assert code_of(sql, "SELEC nothing") == "SQL_ERROR"


def test_empty_sql_refused(sql):
    assert code_of(sql, "   ") == "EMPTY"


def test_database_unchanged_after_attack_attempts(sql, built):
    attempts = ["DELETE FROM visits", "DROP TABLE visits", "UPDATE visits SET los_days = 0"]
    for q in attempts:
        with pytest.raises(SqlGuardError):
            sql.run(q)
    conn = sqlite3.connect(built / "agentguard.db")
    assert conn.execute("SELECT COUNT(*) FROM visits").fetchone()[0] == 15521
    conn.close()
