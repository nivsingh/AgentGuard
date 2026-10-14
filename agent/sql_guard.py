"""Read-only SQL execution with a guard.

Rules, all enforced by SQLite itself rather than by string matching:
  * the database is opened read-only
  * only SELECT statements run (a SQLite authorizer denies every other action)
  * only allowlisted tables may be read, so sqlite_master and friends are off limits
  * only allowlisted functions may be called
  * one statement per call
  * at most max_rows rows come back (extra rows set truncated=True)
  * a wall-clock timeout interrupts long queries
"""

import sqlite3
import time
from dataclasses import dataclass, field

ALLOWED_TABLES = frozenset({"visits", "departments", "providers"})
ALLOWED_FUNCTIONS = frozenset(
    {
        "count", "sum", "avg", "min", "max", "total", "round", "abs", "coalesce", "ifnull",
        "nullif", "lower", "upper", "length", "substr", "substring", "trim", "replace",
        "date", "strftime", "julianday", "cast", "typeof", "iif", "group_concat", "instr",
        "like", "glob", "printf", "random_never",
    }
)


class SqlGuardError(Exception):
    """Raised when a query is refused or fails. `code` is stable for tests and tracing."""

    def __init__(self, code, message):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class SqlResult:
    sql: str
    columns: list
    rows: list
    truncated: bool = False
    elapsed_ms: float = 0.0
    row_count: int = field(init=False)

    def __post_init__(self):
        self.row_count = len(self.rows)

    def scalar(self):
        return self.rows[0][0] if self.rows and self.rows[0] else None


def _authorizer(allowed_tables, allowed_functions):
    allow, deny = sqlite3.SQLITE_OK, sqlite3.SQLITE_DENY
    # allowed_tables may be a set that grows between calls (see ReadOnlySQL.run)

    def check(action, arg1, arg2, db_name, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return allow
        if action == sqlite3.SQLITE_READ:
            return allow if (arg1 or "").lower() in allowed_tables else deny
        if action == sqlite3.SQLITE_FUNCTION:
            return allow if (arg2 or "").lower() in allowed_functions else deny
        return deny  # INSERT, UPDATE, DELETE, DDL, PRAGMA, ATTACH, transactions, everything else

    return check


class ReadOnlySQL:
    def __init__(
        self,
        db_path,
        allowed_tables=ALLOWED_TABLES,
        allowed_functions=ALLOWED_FUNCTIONS,
        max_rows=200,
        timeout_s=2.0,
    ):
        self.db_path = str(db_path)
        self.allowed_tables = frozenset(t.lower() for t in allowed_tables)
        self.allowed_functions = frozenset(f.lower() for f in allowed_functions)
        self.max_rows = max_rows
        self.timeout_s = timeout_s

    def _connect(self):
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def _recursive_cte_names(self, conn, sql, params):
        """Names of recursive CTEs in the query.

        SQLite reports a read of a recursive CTE under its own name, which would
        look like a non-allowlisted table. A dry run (EXPLAIN never executes)
        collects those names, and the strict pass then permits only them.
        """
        names = set()

        def collect(action, arg1, arg2, db_name, source):
            if action == sqlite3.SQLITE_RECURSIVE and source:
                names.add(source.lower())
            return sqlite3.SQLITE_OK

        conn.set_authorizer(collect)
        conn.execute("EXPLAIN " + sql, params).fetchall()
        return names

    def run(self, sql, params=()):
        if not isinstance(sql, str) or not sql.strip():
            raise SqlGuardError("EMPTY", "no SQL provided")
        conn = self._connect()
        deadline = time.monotonic() + self.timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        start = time.monotonic()
        try:
            cte_names = self._recursive_cte_names(conn, sql, params)
            conn.set_authorizer(_authorizer(self.allowed_tables | cte_names, self.allowed_functions))
            cur = conn.execute(sql, params)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.ProgrammingError as e:
            raise SqlGuardError("MULTI_STATEMENT", str(e)) from None
        except sqlite3.DatabaseError as e:
            raise _translate(e) from None
        finally:
            conn.close()
        truncated = len(rows) > self.max_rows
        return SqlResult(
            sql=sql,
            columns=columns,
            rows=[tuple(r) for r in rows[: self.max_rows]],
            truncated=truncated,
            elapsed_ms=(time.monotonic() - start) * 1000,
        )


def _translate(e):
    msg = str(e)
    low = msg.lower()
    if "not authorized" in low or "prohibited" in low:
        return SqlGuardError("FORBIDDEN", msg)
    if "interrupted" in low:
        return SqlGuardError("TIMEOUT", "query exceeded the time limit")
    if "readonly" in low or "read-only" in low:
        return SqlGuardError("FORBIDDEN", msg)
    return SqlGuardError("SQL_ERROR", msg)
