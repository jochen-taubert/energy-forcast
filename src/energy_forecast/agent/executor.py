"""Führt freigegebene Aktionen aus. Der einzige Ort, der die Produktion verändert."""

from __future__ import annotations

import json

from energy_forecast.agent.audit import AuditLog
from energy_forecast.agent.schemas import Action, Proposal
from energy_forecast.agent.tools import Workspace
from energy_forecast.models.registry import restore


def execute(p: Proposal, ws: Workspace, audit: AuditLog, run_id: str) -> dict:
    match p.action:
        case Action.NO_ACTION:
            result = {"status": "nothing_to_do"}
        case Action.CREATE_TICKET:
            tickets = ws.cfg.model_dir / "tickets.jsonl"
            tickets.parent.mkdir(parents=True, exist_ok=True)
            with tickets.open("a", encoding="utf-8") as f:
                ticket = {"run_id": run_id, "title": p.ticket_title, "body": p.reasoning}
                f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
            result = {"status": "ticket_created", "title": p.ticket_title}
        case Action.RETRAIN:
            from energy_forecast.models.train import train

            meta = train(ws.data, ws.cfg)
            result = {"status": "retrained", "promoted": meta["promoted"], "model": meta["model"]}
        case Action.ROLLBACK:
            restore(ws.cfg, p.version)
            result = {"status": "rolled_back", "version": p.version}
    audit.append("executed", run_id, action=p.action.value, result=result)
    return result
