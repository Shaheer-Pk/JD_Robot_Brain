"""
Run this once to create the users and face_embeddings tables.
CREATE TABLE only — no seeding. Guest fallback persona lives in
brain/robot_profile.json, not in this database.

DB_PATH is imported from database.py, NOT recomputed locally via
os.path.dirname(__file__). This file lives in app/users/schemas/,
one folder deeper than database.py in app/users/ — a locally computed
path would resolve to app/users/schemas/database.db, a different file
from the one get_db() actually connects to at runtime. That would
silently create an empty, unused database sitting next to this
script while the real app connects to a database that never got its
tables created. Importing the constant means there is exactly one
source of truth for where the database file lives, and the two files
cannot drift out of sync regardless of which folder either one is
moved to in the future.

Usage — MUST be run as a module, from the project root (same level
as main.py), not as a bare script:

    python -m app.users.schemas.schemas_db

Running `python schemas_db.py` directly will fail. This file is part
of a package (app.users.schemas) and its import of database.py relies
on that package context to resolve. Invoking it as a bare script gives
Python no package context, so the import fails with
"ImportError: attempted relative import with no known parent package"
(or an absolute-import equivalent, depending on the exact import form).
The -m invocation is what supplies that context, and it only resolves
correctly when your current working directory is the project root —
the same root main.py/uvicorn already runs from.
"""

import sqlite3

from app.users.database import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ---- users table ----
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    persona TEXT,
    memory_context TEXT,
    password_hash TEXT NOT NULL
)
""")

# ---- face_embeddings table (separate, linked via user_id) ----
cursor.execute("""
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    embedding TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
)
""")

# The original version of this script never committed or closed the
# connection — relying on the script exiting to release resources.
# CREATE TABLE statements are not guaranteed to be durable without an
# explicit commit depending on sqlite3's transaction handling; adding
# both here explicitly rather than assuming interpreter-exit cleanup
# covers it. Flagging this fix explicitly since it wasn't something
# you asked for — noticed it while relocating the file.
conn.commit()
conn.close()