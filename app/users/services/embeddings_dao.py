import json
import sqlite3


def load_face_embeds(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """
    Full-table scan, every embedding, every user. Matching/ranking is
    Python-side (vision/services.py's find_best_match) — SQLite cannot
    compute vector distance. Not a scalability concern at this project's
    size; the real CPU cost lives in extract_embedding, not this query.
    """
    return conn.execute("SELECT id, user_id, embedding FROM face_embeddings").fetchall()


def list_embeddings_for_user(conn: sqlite3.Connection, user_id: int) -> list[sqlite3.Row]:
    """
    Oldest-first. The customization endpoint's FIFO eviction reads
    index 0 to find which row to drop when a user hits the 8-photo cap.
    """
    return conn.execute(
        "SELECT id, created_at FROM face_embeddings WHERE user_id = ? ORDER BY created_at ASC",
        (user_id,),
    ).fetchall()


def count_embeddings_for_user(conn: sqlite3.Connection, user_id: int) -> int:
    row = conn.execute(
        "SELECT COUNT(*) as count FROM face_embeddings WHERE user_id = ?", (user_id,)
    ).fetchone()
    return row["count"]


def insert_embedding(conn: sqlite3.Connection, user_id: int, embedding: list[float]) -> int:
    """
    Stored as JSON text — unchanged format, carried over from the old
    SQLAlchemy JSON column. Binary-storage optimization is still
    deferred per earlier discussion, not done in this pass.
    Does NOT commit — same transaction-timing rule as insert_user.
    """
    cursor = conn.execute(
        "INSERT INTO face_embeddings (user_id, embedding) VALUES (?, ?)",
        (user_id, json.dumps(embedding)),
    )
    return cursor.lastrowid


def delete_embedding(conn: sqlite3.Connection, embedding_id: int) -> bool:
    """
    Deletes ONE row by its own id. Sole remaining caller is FIFO
    eviction on the add-embedding endpoint: routers.py resolves the
    oldest row's id via list_embeddings_for_user(...)[0]["id"] and
    passes it here right before inserting the new embedding, all
    inside one transaction. Not used for the wipe-all case — see
    delete_all_embeddings_for_user below.
    """
    cursor = conn.execute("DELETE FROM face_embeddings WHERE id = ?", (embedding_id,))
    return cursor.rowcount > 0


def delete_all_embeddings_for_user(conn: sqlite3.Connection, user_id: int) -> int:
    """
    Bulk wipe — every embedding row for this user_id, one statement.
    Used by the delete-all-embeddings endpoint (full re-enrollment
    reset). Superseded the earlier "loop delete_embedding over
    list_embeddings_for_user" design: that loop bought nothing over a
    single WHERE user_id = ? delete — same atomicity (neither commits
    on its own), N round-trips instead of one, no extra safety.
    Returns rowcount so the router can report how many rows were wiped.
    Does NOT commit — caller commits once, same convention as every
    other write in this file.
    """
    cursor = conn.execute("DELETE FROM face_embeddings WHERE user_id = ?", (user_id,))
    return cursor.rowcount