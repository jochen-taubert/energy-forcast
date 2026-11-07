"""Freigabe-Queue (Human-in-the-Loop). Ausstehende Vorschläge liegen als JSON-Datei vor."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from energy_forecast.agent.audit import AuditLog
from energy_forecast.agent.executor import execute
from energy_forecast.agent.schemas import Proposal
from energy_forecast.agent.tools import Workspace


class ApprovalQueue:
    def __init__(self, path: Path, audit: AuditLog) -> None:
        self.path, self.audit = path, audit

    def _load(self) -> dict[str, dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self, items: dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, indent=2, ensure_ascii=False))

    def submit(self, run_id: str, p: Proposal) -> None:
        items = self._load()
        items[run_id] = {
            "status": "pending",
            "proposal": p.model_dump(mode="json"),
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._save(items)
        self.audit.append("approval_requested", run_id, action=p.action.value)

    def list(self, status: str | None = "pending") -> dict[str, dict]:
        return {k: v for k, v in self._load().items() if status is None or v["status"] == status}

    def _decide(self, run_id: str, status: str, by: str, note: str) -> dict:
        items = self._load()
        item = items.get(run_id)
        if item is None or item["status"] != "pending":
            raise KeyError(f"Kein offener Vorschlag {run_id}")
        item.update(
            status=status, decided_by=by, note=note, decided_at=datetime.now(UTC).isoformat()
        )
        self._save(items)
        self.audit.append(status, run_id, by=by, note=note)
        return item

    def approve(self, run_id: str, by: str, ws: Workspace, note: str = "") -> dict:
        item = self._decide(run_id, "approved", by, note)
        return execute(Proposal.model_validate(item["proposal"]), ws, self.audit, run_id)

    def reject(self, run_id: str, by: str, note: str = "") -> None:
        self._decide(run_id, "rejected", by, note)
