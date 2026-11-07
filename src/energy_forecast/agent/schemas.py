"""Strukturierte Rückgabe des Agenten. Alles, was nicht diesem Schema entspricht, wird abgelehnt."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Action(StrEnum):
    NO_ACTION = "no_action"
    CREATE_TICKET = "create_ticket"
    RETRAIN = "retrain"
    ROLLBACK = "rollback"


# Aktionen, die die Produktion verändern, brauchen eine menschliche Freigabe
REQUIRES_APPROVAL = {Action.RETRAIN, Action.ROLLBACK}


class Proposal(BaseModel):
    action: Action
    reasoning: str = Field(min_length=20, max_length=2000)
    evidence: list[str] = Field(
        min_length=1, description="Namen der Werkzeuge, deren Ergebnisse die Entscheidung stützen"
    )
    version: str | None = Field(default=None, description="Nur bei rollback: Zielversion")
    ticket_title: str | None = Field(default=None, max_length=120)


PROPOSAL_TOOL = {
    "name": "propose_action",
    "description": (
        "Schließt die Untersuchung mit genau einer Maßnahme ab. "
        "evidence muss die Namen der Werkzeuge enthalten, auf deren Ergebnisse du dich stützt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": [a.value for a in Action]},
            "reasoning": {"type": "string", "description": "Begründung mit konkreten Zahlen"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "version": {"type": "string", "description": "Nur bei rollback"},
            "ticket_title": {"type": "string", "description": "Nur bei create_ticket"},
        },
        "required": ["action", "reasoning", "evidence"],
    },
}
