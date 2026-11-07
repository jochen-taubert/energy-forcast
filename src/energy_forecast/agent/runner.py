"""Agenten-Schleife: Modell fragt Werkzeuge an, Code führt sie aus,
bis ein gültiger Vorschlag steht.

Fail-safe: Kommt nach max_steps kein gültiger Vorschlag zustande, passiert nichts.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from pydantic import ValidationError

from energy_forecast.agent.approval import ApprovalQueue
from energy_forecast.agent.audit import AuditLog
from energy_forecast.agent.executor import execute
from energy_forecast.agent.llm import LLM
from energy_forecast.agent.policy import Policy
from energy_forecast.agent.schemas import PROPOSAL_TOOL, Proposal
from energy_forecast.agent.tools import READ_TOOLS, TOOL_SPECS, Workspace

SYSTEM = """Du bist ein MLOps-Agent für einen Service, der die deutsche Stromlast
24 h im Voraus prognostiziert.

Aufgabe: Untersuche den Zustand des Systems mit den Werkzeugen und schließe mit genau einem
Aufruf von propose_action ab.

Mögliche Maßnahmen:
- no_action: Alles im Rahmen. Drift allein ohne Verschlechterung ist kein Grund zu handeln.
- retrain: Modell deutlich schlechter UND die Datenlage hat sich verändert (Drift),
  Datenqualität in Ordnung.
- rollback: Modell deutlich schlechter OHNE Daten-Drift: Dann liegt die Ursache im Modell selbst,
  nicht in den Daten. Gib als Zielversion die Version direkt vor der aktiven an.
- create_ticket: Ein Mensch muss sich kümmern, z. B. bei Datenproblemen oder wenn ein
  Retraining kürzlich schon erfolglos war.

Regeln:
- Stütze dich nur auf Werkzeug-Ergebnisse. Nenne in evidence die Werkzeuge, die du genutzt hast.
- Nenne in reasoning die entscheidenden Zahlen.
- Retrain und Rollback werden erst nach menschlicher Freigabe ausgeführt.
"""


@dataclass
class AgentRun:
    run_id: str
    status: str = "failed"  # executed | pending_approval | failed
    proposal: Proposal | None = None
    result: dict | None = None
    steps: int = 0
    tools_called: list[str] = field(default_factory=list)
    observations: dict[str, dict] = field(default_factory=dict)
    policy_denials: list[str] = field(default_factory=list)
    invalid_outputs: int = 0
    forbidden_calls: list[str] = field(default_factory=list)


def _result(tool_use_id: str, payload: dict, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(payload, ensure_ascii=False, default=str),
        "is_error": is_error,
    }


def run_agent(
    ws: Workspace,
    llm: LLM,
    audit: AuditLog,
    queue: ApprovalQueue,
    policy: Policy | None = None,
    trigger: str = "Täglicher Check des Prognosesystems.",
    max_steps: int = 8,
) -> AgentRun:
    policy = policy or Policy()
    run = AgentRun(run_id=uuid.uuid4().hex[:12])
    audit.append("run_started", run.run_id, trigger=trigger)
    messages: list[dict] = [{"role": "user", "content": trigger}]
    tools = [*TOOL_SPECS, PROPOSAL_TOOL]

    for step in range(1, max_steps + 1):
        run.steps = step
        content = llm.complete(SYSTEM, messages, tools)
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if b["type"] == "tool_use"]
        if not tool_uses:
            run.invalid_outputs += 1
            messages.append(
                {"role": "user", "content": "Schließe mit einem Aufruf von propose_action ab."}
            )
            continue

        results = []
        for tu in tool_uses:
            name, args = tu["name"], tu.get("input") or {}
            audit.append("tool_call", run.run_id, tool=name, input=args)

            if name == "propose_action":
                try:
                    proposal = Proposal.model_validate(args)
                except ValidationError as e:
                    run.invalid_outputs += 1
                    audit.append("invalid_output", run.run_id, error=str(e))
                    results.append(_result(tu["id"], {"error": f"Ungültiger Vorschlag: {e}"}, True))
                    continue
                decision = policy.check(proposal, ws, audit, set(run.tools_called))
                if not decision.allowed:
                    run.policy_denials.append(decision.reason)
                    audit.append(
                        "policy_denied",
                        run.run_id,
                        action=proposal.action.value,
                        reason=decision.reason,
                    )
                    results.append(_result(tu["id"], {"error": decision.reason}, True))
                    continue
                return _finish(run, proposal, decision.needs_approval, ws, audit, queue)

            if name not in READ_TOOLS:
                run.forbidden_calls.append(name)
                audit.append("forbidden_call", run.run_id, tool=name)
                results.append(
                    _result(tu["id"], {"error": f"Werkzeug {name!r} nicht erlaubt"}, True)
                )
                continue

            try:
                obs = READ_TOOLS[name](ws, **args)
            except Exception as e:  # Werkzeugfehler gehen als Ergebnis an das Modell zurück
                obs = {"error": f"{type(e).__name__}: {e}"}
            run.tools_called.append(name)
            run.observations[name] = obs
            results.append(_result(tu["id"], obs, "error" in obs))

        messages.append({"role": "user", "content": results})

    audit.append("agent_failed", run.run_id, reason="Kein gültiger Vorschlag", steps=run.steps)
    return run


def _finish(run, proposal, needs_approval, ws, audit, queue) -> AgentRun:
    run.proposal = proposal
    audit.append("proposal_accepted", run.run_id, **proposal.model_dump(mode="json"))
    if needs_approval:
        queue.submit(run.run_id, proposal)
        run.status = "pending_approval"
    else:
        run.result = execute(proposal, ws, audit, run.run_id)
        run.status = "executed"
    return run
