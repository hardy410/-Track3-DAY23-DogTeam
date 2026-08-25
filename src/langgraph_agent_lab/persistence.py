"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpointer(
    kind: str = "memory", database_url: str | None = None
) -> BaseCheckpointSaver | None:
    """Return a LangGraph checkpointer.

    SQLite persistence is supported for durable local checkpoints. Postgres remains
    an optional deployment-specific extension.

    For SQLite:
    - pip install langgraph-checkpoint-sqlite
    - Use SqliteSaver with sqlite3.connect() and WAL mode
    - See: https://langchain-ai.github.io/langgraph/how-tos/persistence/
    """
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        database_path = database_url or "outputs/checkpoints.db"
        if database_path.startswith("sqlite:///"):
            database_path = database_path.removeprefix("sqlite:///")
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, check_same_thread=False)
        connection.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn=connection)
    if kind == "postgres":
        raise RuntimeError(
            "Postgres is optional; install the postgres extra and configure DATABASE_URL"
        )
    raise ValueError(f"Unknown checkpointer kind: {kind}")
