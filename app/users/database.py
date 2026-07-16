import os
import sqlite3

# Points at database.db sitting right next to this file
DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def _configure_connection(conn: sqlite3.Connection) -> None:
    # Named-column row access — immune to column-order changes in the
    # schema. Every DAO reads rows via row["column_name"], never row[N].
    conn.row_factory = sqlite3.Row

    # WAL mode: readers (recognition scans) never block writers
    # (enrollment inserts) and vice versa. Does NOT make two simultaneous
    # writers lock-free — SQLite still serializes writes, WAL just shortens
    # that wait and keeps it from also blocking concurrent reads.
    conn.execute("PRAGMA journal_mode=WAL")

    # Off by default on every new connection, every time — must be
    # re-issued here, not something set once anywhere else. Enforces the
    # face_embeddings.user_id -> users.id relationship at INSERT/UPDATE/
    # DELETE time. No effect on CREATE TABLE, so schema_db.py's table
    # creation is unaffected regardless of this setting.
    conn.execute("PRAGMA foreign_keys = ON")


def get_db():
    # FastAPI dependency — same generator shape as the old SQLAlchemy
    # version, so existing `db = Depends(get_db)` signatures in routers
    # need zero changes. Only what routers/DAOs DO with `db` changes.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _configure_connection(conn)
    try:
        yield conn
    finally:
        conn.close()