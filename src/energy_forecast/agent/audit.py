"""Append-only Audit-Log im JSON-Lines-Format. Eine Zeile pro Ereignis."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: str, run_id: str | None = None, **data) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": datetime.now(UTC).isoformat(), "event": event, "run_id": run_id, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def executed_since(self, action: str, delta: timedelta) -> list[dict]:
        cutoff = datetime.now(UTC) - delta
        return [
            e
            for e in self.read()
            if e["event"] == "executed"
            and e.get("action") == action
            and datetime.fromisoformat(e["ts"]) >= cutoff
        ]
