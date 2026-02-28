"""SQLite database layer for the Nervous System Agent.

Only stores conversation history and user profiles.
Biometric data comes from Daniel's JSON file (hackathon/data/insights.json).
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "agent.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_phone TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            message_type TEXT DEFAULT 'chat'
        );

        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT UNIQUE NOT NULL,
            name TEXT,
            goals TEXT,
            preferences TEXT,
            why_category TEXT,
            identity_segment TEXT,
            routine_anchor TEXT,
            streak_count INTEGER DEFAULT 0,
            last_test_date TEXT,
            relationship_stage TEXT DEFAULT 'translator',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_messages_phone ON messages(user_phone);
        CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(timestamp);
    """)
    conn.commit()
    conn.close()


# --- Messages ---

def save_message(phone: str, role: str, content: str, message_type: str = "chat"):
    conn = get_connection()
    conn.execute(
        "INSERT INTO messages (user_phone, role, content, message_type) VALUES (?, ?, ?, ?)",
        (phone, role, content, message_type),
    )
    conn.commit()
    conn.close()


def get_conversation_history(phone: str, limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE user_phone = ? ORDER BY timestamp ASC LIMIT ?",
        (phone, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- User Profile ---

def get_or_create_profile(phone: str) -> dict:
    conn = get_connection()
    row = conn.execute("SELECT * FROM user_profiles WHERE phone = ?", (phone,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO user_profiles (phone) VALUES (?)", (phone,))
        conn.commit()
        row = conn.execute("SELECT * FROM user_profiles WHERE phone = ?", (phone,)).fetchone()
    conn.close()
    return dict(row)


def update_profile(phone: str, **fields):
    if not fields:
        return
    conn = get_connection()
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values())
    vals.append(phone)
    conn.execute(f"UPDATE user_profiles SET {sets}, updated_at = datetime('now') WHERE phone = ?", vals)
    conn.commit()
    conn.close()


# Auto-init on import
init_db()
