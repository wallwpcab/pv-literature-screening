import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class NullAuditRepository:
    """No-op audit repository used by default for simple local/cloud runs."""

    enabled = False

    def log(self, *args, **kwargs) -> None:
        return None

    def rows(self, *args, **kwargs) -> list[dict]:
        return []


class AuditRepository:
    enabled = True

    def __init__(self, path: str = "pv_audit.db") -> None:
        self.path = Path(path)
        self.initialize()

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pmid TEXT,
                    stage TEXT NOT NULL,
                    outcome TEXT,
                    actor TEXT NOT NULL,
                    detail TEXT,
                    query_term TEXT,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_pmid ON audit_log(pmid);
                CREATE INDEX IF NOT EXISTS idx_audit_stage ON audit_log(stage);
            """)

    def log(self, pmid: str, stage: str, outcome: str, actor: str,
            detail: str = "", query_term: str = "") -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO audit_log "
                "(pmid, stage, outcome, actor, detail, query_term, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (pmid, stage, outcome, actor, detail, query_term,
                 datetime.now(timezone.utc).isoformat()),
            )

    def rows(self, pmid: str | None = None, limit: int = 200) -> list[dict]:
        with self.connection() as conn:
            if pmid:
                rows = conn.execute(
                    "SELECT * FROM audit_log WHERE pmid = ? ORDER BY timestamp ASC",
                    (pmid,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
