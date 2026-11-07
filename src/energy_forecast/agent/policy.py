"""Policy-Schicht: harte Regeln im Code.

Der Prompt sagt dem Modell, was es tun SOLL. Die Policy legt fest, was es tun DARF.
Eine abgelehnte Aktion geht mit Begründung an den Agenten zurück; er kann neu entscheiden.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from energy_forecast.agent.audit import AuditLog
from energy_forecast.agent.schemas import REQUIRES_APPROVAL, Action, Proposal
from energy_forecast.agent.tools import Workspace, get_data_quality
from energy_forecast.models.registry import current_version, list_versions


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""
    needs_approval: bool = False


@dataclass(frozen=True)
class Policy:
    max_retrains_per_day: int = 1

    def check(
        self, p: Proposal, ws: Workspace, audit: AuditLog, tools_called: set[str]
    ) -> Decision:
        # Grounding: Belege müssen aus tatsächlich aufgerufenen Werkzeugen stammen
        if unknown := set(p.evidence) - tools_called:
            return Decision(False, f"evidence nennt nicht aufgerufene Werkzeuge: {sorted(unknown)}")

        match p.action:
            case Action.RETRAIN:
                if not get_data_quality(ws)["ok"]:
                    return Decision(False, "Retraining verboten: Datenqualität nicht in Ordnung")
                recent = audit.executed_since(Action.RETRAIN, timedelta(days=1))
                if len(recent) >= self.max_retrains_per_day:
                    return Decision(False, "Retraining-Limit erreicht (max. 1 pro 24 h)")
            case Action.ROLLBACK:
                known = {v["version"] for v in list_versions(ws.cfg)}
                if p.version not in known:
                    return Decision(False, f"Rollback-Ziel {p.version!r} existiert nicht")
                if p.version == current_version(ws.cfg):
                    return Decision(False, "Rollback-Ziel ist bereits aktiv")
            case Action.CREATE_TICKET:
                if not p.ticket_title:
                    return Decision(False, "create_ticket braucht ticket_title")

        return Decision(True, needs_approval=p.action in REQUIRES_APPROVAL)
