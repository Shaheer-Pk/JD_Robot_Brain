import sqlite3


def get_user_profile(conn: sqlite3.Connection, user_id: int) -> dict | None:
    """
    Feeds the recognition/brain path only. Deliberately excludes
    password_hash — this function has no business exposing credentials
    to a code path that never needed them before this session's changes.
    """
    row = conn.execute(
        "SELECT id, name, persona, memory_context FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "name": row["name"],
        "persona": row["persona"],
        "memory_context": row["memory_context"],
    }


def insert_user(
    conn: sqlite3.Connection,
    name: str,
    persona: str,
    memory_context: str,
    password_hash: str,
) -> int:
    """
    Does NOT commit. Caller (enrollment handler in routers.py) runs this
    and embeddings_dao.insert_embedding on the SAME connection, then
    commits once — the atomic-transaction pattern already agreed.
    """
    cursor = conn.execute(
        "INSERT INTO users (name, persona, memory_context, password_hash) VALUES (?, ?, ?, ?)",
        (name, persona, memory_context, password_hash),
    )
    return cursor.lastrowid


def update_user(
    conn: sqlite3.Connection,
    user_id: int,
    name: str | None = None,
    persona: str | None = None,
) -> bool:
    """
    PARTIAL update — only fields passed as non-None get changed;
    omitted fields stay untouched. This avoids forcing a caller to
    fetch-then-resubmit unchanged fields just to not null them.

    memory_context is deliberately NOT a parameter here, on purpose,
    not by oversight. It's system-managed: the future face-lost
    trigger (vision module) is the sole intended writer, serializing
    ConversationMemory to this column. If manual user-facing edit could
    also write this column, two independent writers touch the same
    field with no coordination — a real race, not hypothetical, the
    moment both mechanisms exist (e.g. user clears it manually right
    before the face-lost handler fires and overwrites it right back).
    This is an ownership boundary, not a workaround — do not add
    memory_context back as a parameter for convenience. If a manual
    memory-reset feature is ever built, it needs its own dedicated
    function (e.g. update_memory_context(conn, user_id, memory_context))
    so a caller can't cross the boundary by accident. Deferred, not
    built now — see jd-context-main.md future scope notes.
    """
    fields = {}
    if name is not None:
        fields["name"] = name
    if persona is not None:
        fields["persona"] = persona

    if not fields:
        return False

    # Column names come from this fixed whitelist above, never from raw
    # caller input — safe to interpolate into the SET clause. Values
    # remain fully parameterized.
    set_clause = ", ".join(f"{col} = ?" for col in fields)
    values = list(fields.values()) + [user_id]

    cursor = conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
    return cursor.rowcount > 0


def delete_user(conn: sqlite3.Connection, user_id: int) -> bool:
    """
    Plain delete only. Cascading removal of this user's face_embeddings
    rows happens via ON DELETE CASCADE at the schema level, enforced
    because PRAGMA foreign_keys = ON is set on every connection in
    database.py. If either piece is ever missing, this either fails
    outright (FK violation) or silently orphans embeddings — this
    function can't detect that itself, by design; DAOs don't own
    cross-table business logic.
    """
    cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return cursor.rowcount > 0

def get_users_by_name(conn: sqlite3.Connection, name: str) -> list[sqlite3.Row]:
    """
    Plural, deliberately — names are not unique (see jd-context-main.md,
    known limitation). Login checks the submitted password against every
    returned row's hash in turn; the first match wins. If two users share
    both name AND password, login succeeds as whichever row SQLite
    returns first — an accepted, documented edge case, not a bug to chase.
    Excludes password_hash from nothing — auth.py needs the real hash to
    check against, so this function intentionally returns the full row,
    unlike get_user_profile which strips it for the recognition path.
    """
    return conn.execute(
        "SELECT id, name, persona, memory_context, password_hash FROM users WHERE name = ?",
        (name,),
    ).fetchall()