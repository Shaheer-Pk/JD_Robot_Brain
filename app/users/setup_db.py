"""
Run this once to create app/users/jd_robot.db with the users and
face_embeddings tables, seeded with 2 starter rows:
  1 -> Guest (default fallback persona)
  2 -> Shahzaib (tsundere persona)

Usage:
    python setup_db.py
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ---- users table ----
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    persona TEXT,
    memory_context TEXT
)
""")

# ---- face_embeddings table (separate, linked via user_id) ----
cursor.execute("""
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    embedding TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
)
""")

# ---- seed the 2 hardcoded users (only if table is empty, so reruns are safe) ----
cursor.execute("SELECT COUNT(*) FROM users")
if cursor.fetchone()[0] == 0:
    cursor.execute(
        "INSERT INTO users (id, name, persona, memory_context) VALUES (?, ?, ?, ?)",
        (1, "Guest", "Friendly, neutral, welcoming default persona for anyone not recognized yet.", "")
    )
    cursor.execute(
        "INSERT INTO users (id, name, persona, memory_context) VALUES (?, ?, ?, ?)",
        (2, "Shahzaib", "Tsundere — outwardly blunt, sarcastic, and reluctant to show warmth, but secretly caring and helpful underneath.", "")
    )
    print("Seeded users: Guest (id=1), Shahzaib (id=2)")
else:
    print("Users table already has data — skipped seeding.")

conn.commit()
conn.close()

print(f"Database ready at: {DB_PATH}")