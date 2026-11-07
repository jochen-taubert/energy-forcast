"""LLM-Anbindung hinter einem schmalen Protokoll.

- ClaudeLLM:     echte Claude API mit Tool Use
- RuleBasedLLM:  regelbasierter Baseline-Agent ohne LLM, die Messlatte in den Evals
- ScriptedLLM:   feste Antworten für Unit-Tests

Inhaltsblöcke folgen dem Format der Anthropic Messages API:
  {"type": "text", "text": ...}
  {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Protocol

Block = dict
Message = dict


class LLM(Protocol):
    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> list[Block]: ...


class ClaudeLLM:
    def __init__(self, model: str | None = None, max_tokens: int = 1500) -> None:
        import anthropic

        self.client = anthropic.Anthropic()  # liest ANTHROPIC_API_KEY
        self.model = model or os.getenv("EF_AGENT_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.tokens = {"input": 0, "output": 0}

    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> list[Block]:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
        )
        self.tokens["input"] += resp.usage.input_tokens
        self.tokens["output"] += resp.usage.output_tokens
        blocks: list[Block] = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        return blocks


class ScriptedLLM:
    """Gibt vorab festgelegte Antworten der Reihe nach zurück."""

    def __init__(self, responses: list[list[Block]]) -> None:
        self.responses = iter(responses)

    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> list[Block]:
        return next(self.responses)


class RuleBasedLLM:
    """Baseline-Agent mit festen Wenn-Dann-Regeln.

    Ruft im ersten Schritt alle Werkzeuge parallel auf und entscheidet dann nach Regeln.
    Ein LLM-Agent sollte mindestens so gut sein, sonst lohnt er sich nicht.
    """

    READS = (
        "get_data_quality",
        "get_recent_performance",
        "get_drift_report",
        "get_model_versions",
        "get_action_history",
    )

    def __init__(self) -> None:
        self._ids = itertools.count()

    def _tool_use(self, name: str, payload: dict | None = None) -> Block:
        return {
            "type": "tool_use",
            "id": f"r{next(self._ids)}",
            "name": name,
            "input": payload or {},
        }

    def complete(self, system: str, messages: list[Message], tools: list[dict]) -> list[Block]:
        obs = _observations(messages)
        if not obs:
            return [self._tool_use(n) for n in self.READS]
        if obs.get("propose_action", {}).get("error"):
            # Policy hat abgelehnt: an Menschen eskalieren
            return [
                self._propose(
                    "create_ticket",
                    "Vorgeschlagene Aktion von Policy abgelehnt: " + obs["propose_action"]["error"],
                    ["get_action_history"],
                    ticket_title="Agent: Aktion blockiert, manuelle Prüfung",
                )
            ]

        dq, perf, drift = (
            obs["get_data_quality"],
            obs["get_recent_performance"],
            obs["get_drift_report"],
        )
        versions = obs["get_model_versions"]["versions"]
        history = obs["get_action_history"]["executed"]
        if not dq["ok"]:
            return [
                self._propose(
                    "create_ticket",
                    "Datenqualität fehlerhaft: " + "; ".join(dq["issues"]),
                    ["get_data_quality"],
                    ticket_title="Datenqualität prüfen",
                )
            ]
        if perf["assessment"] == "ok":
            return [
                self._propose(
                    "no_action",
                    f"Fehler im Rahmen (ratio {perf['ratio']}).",
                    ["get_recent_performance"],
                )
            ]
        current = next((v for v in versions if v["is_current"]), None)
        older = [v for v in versions if current and v["version"] < current["version"]]
        if not drift["drifted_features"] and older:
            target = max(older, key=lambda v: v["version"])
            return [
                self._propose(
                    "rollback",
                    f"Fehler ratio {perf['ratio']} ohne Daten-Drift, "
                    f"Ursache liegt im Modell. Rückkehr zu {target['version']}.",
                    ["get_recent_performance", "get_drift_report", "get_model_versions"],
                    version=target["version"],
                )
            ]
        if any(h["action"] == "retrain" for h in history):
            return [
                self._propose(
                    "create_ticket",
                    "Verschlechterung trotz Retraining in den letzten "
                    "24 h, manuelle Analyse nötig.",
                    ["get_recent_performance", "get_action_history"],
                    ticket_title="Modell bleibt nach Retraining schlecht",
                )
            ]
        return [
            self._propose(
                "retrain",
                f"Fehler ratio {perf['ratio']} und Drift in {drift['drifted_features']}.",
                ["get_recent_performance", "get_drift_report"],
            )
        ]

    def _propose(self, action: str, reasoning: str, evidence: list[str], **extra) -> Block:
        return self._tool_use(
            "propose_action",
            {"action": action, "reasoning": reasoning, "evidence": evidence, **extra},
        )


def _observations(messages: list[Message]) -> dict[str, dict]:
    """Ordnet Tool-Ergebnisse den Werkzeugnamen zu (jüngstes Ergebnis gewinnt)."""
    names: dict[str, str] = {}
    obs: dict[str, dict] = {}
    for m in messages:
        if not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b["type"] == "tool_use":
                names[b["id"]] = b["name"]
            elif b["type"] == "tool_result":
                result = json.loads(b["content"])
                obs[names[b["tool_use_id"]]] = result
    return obs
