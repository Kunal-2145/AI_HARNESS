from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

MAX_CONTEXT_MESSAGES = 12
MAX_CONTEXT_CHARS = 16_000


class ConversationStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    workspace_path TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS sessions_workspace_updated
                    ON sessions(workspace_path, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS messages_session_id
                    ON messages(session_id, id DESC);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
        finally:
            connection.close()

    def resume_or_create_session(self, workspace: Path) -> tuple[str, bool]:
        workspace_path = str(workspace.resolve())
        with self._connect() as connection:
            session = connection.execute(
                "SELECT id FROM sessions WHERE workspace_path = ? "
                "ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                (workspace_path,),
            ).fetchone()
        if session is not None:
            return str(session["id"]), True
        return self.create_session(workspace_path), False

    def create_session(self, workspace: Path | str) -> str:
        workspace_path = str(workspace.resolve()) if isinstance(workspace, Path) else workspace
        session_id = str(uuid.uuid4())
        with self._connect() as connection, connection:
            connection.execute(
                "INSERT INTO sessions (id, workspace_path) VALUES (?, ?)",
                (session_id, workspace_path),
            )
        return session_id

    def recent_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, MAX_CONTEXT_MESSAGES),
            ).fetchall()

        selected: list[dict[str, str]] = []
        remaining = MAX_CONTEXT_CHARS
        for row in rows:
            content = str(row["content"])
            if remaining <= 0:
                break
            if len(content) > remaining:
                marker = "...[earlier content truncated]\n"
                content = marker + content[-max(0, remaining - len(marker)):]
            selected.append({"role": str(row["role"]), "content": content})
            remaining -= len(content)
        selected.reverse()
        return selected

    def save_exchange(
        self,
        session_id: str,
        question: str,
        answer: str,
        execution_context: str | None = None,
    ) -> None:
        stored_answer = (
            f"{execution_context}\n\n{answer}"
            if execution_context
            else answer
        )
        with self._connect() as connection, connection:
            connection.executemany(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                (
                    (session_id, "user", question),
                    (session_id, "assistant", stored_answer),
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )