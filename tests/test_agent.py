"""Tests der Kontrollmechanismen. Kein API-Schlüssel nötig: das LLM ist gescriptet."""

from datetime import timedelta

import pandas as pd
import pytest

from energy_forecast.agent.approval import ApprovalQueue
from energy_forecast.agent.audit import AuditLog
from energy_forecast.agent.llm import ScriptedLLM
from energy_forecast.agent.runner import run_agent
from energy_forecast.agent.tools import Workspace
from energy_forecast.models.registry import current_version, list_versions
from energy_forecast.models.train import train


def use(name, i, **inp):
    return {"type": "tool_use", "id": f"t{i}", "name": name, "input": inp}


def propose(i, action, evidence, **extra):
    reasoning = "Begründung mit ausreichend Länge für das Schema."
    return use("propose_action", i, action=action, reasoning=reasoning, evidence=evidence, **extra)


@pytest.fixture
def env(df, cfg):
    train(df, cfg)
    audit = AuditLog(cfg.model_dir / "agent" / "audit.jsonl")
    queue = ApprovalQueue(cfg.model_dir / "agent" / "approvals.json", audit)
    return Workspace(cfg, df, audit=audit), audit, queue


def test_no_action_executes_directly(env):
    ws, audit, queue = env
    llm = ScriptedLLM(
        [[use("get_recent_performance", 1)], [propose(2, "no_action", ["get_recent_performance"])]]
    )
    run = run_agent(ws, llm, audit, queue)
    assert run.status == "executed" and run.proposal.action == "no_action"
    assert {e["event"] for e in audit.read()} >= {"run_started", "tool_call", "executed"}


def test_retrain_needs_approval_and_runs_only_after_it(env):
    ws, audit, queue = env
    llm = ScriptedLLM([[use("get_data_quality", 1)], [propose(2, "retrain", ["get_data_quality"])]])
    run = run_agent(ws, llm, audit, queue)
    assert run.status == "pending_approval"
    assert not any(e["event"] == "executed" for e in audit.read())

    result = queue.approve(run.run_id, by="jost", ws=ws)
    assert result["status"] == "retrained"
    assert queue.list() == {}
    approved = [e for e in audit.read() if e["event"] == "approved"]
    assert approved[0]["by"] == "jost"


def test_evidence_must_come_from_called_tools(env):
    ws, audit, queue = env
    llm = ScriptedLLM(
        [
            [propose(1, "no_action", ["get_drift_report"])],  # nichts aufgerufen -> abgelehnt
            [use("get_drift_report", 2)],
            [propose(3, "no_action", ["get_drift_report"])],
        ]
    )
    run = run_agent(ws, llm, audit, queue)
    assert run.status == "executed"
    assert "nicht aufgerufene" in run.policy_denials[0]


def test_retrain_blocked_on_bad_data(df, cfg):
    train(df, cfg)
    audit = AuditLog(cfg.model_dir / "a.jsonl")
    queue = ApprovalQueue(cfg.model_dir / "q.json", audit)
    stale = Workspace(cfg, df, now=df.index.max() + pd.Timedelta(days=5), audit=audit)
    llm = ScriptedLLM(
        [
            [use("get_data_quality", 1)],
            [propose(2, "retrain", ["get_data_quality"])],
            [propose(3, "create_ticket", ["get_data_quality"], ticket_title="Daten veraltet")],
        ]
    )
    run = run_agent(stale, llm, audit, queue)
    assert run.proposal.action == "create_ticket"
    assert "Datenqualität" in run.policy_denials[0]


def test_retrain_rate_limit(env):
    ws, audit, queue = env
    audit.append("executed", "earlier", action="retrain")
    assert audit.executed_since("retrain", timedelta(days=1))
    llm = ScriptedLLM([[use("get_data_quality", 1)], [propose(2, "retrain", ["get_data_quality"])]])
    run = run_agent(ws, llm, audit, queue, max_steps=2)
    assert run.status == "failed" and "Limit" in run.policy_denials[0]


def test_forbidden_tool_is_refused_and_logged(env):
    ws, audit, queue = env
    llm = ScriptedLLM(
        [
            [use("delete_models", 1)],
            [use("get_model_info", 2)],
            [propose(3, "no_action", ["get_model_info"])],
        ]
    )
    run = run_agent(ws, llm, audit, queue)
    assert run.forbidden_calls == ["delete_models"]
    assert any(e["event"] == "forbidden_call" for e in audit.read())


def test_invalid_schema_is_rejected(env):
    ws, audit, queue = env
    bad = use("propose_action", 1, action="shutdown", reasoning="x", evidence=[])
    llm = ScriptedLLM(
        [[bad], [use("get_model_info", 2)], [propose(3, "no_action", ["get_model_info"])]]
    )
    run = run_agent(ws, llm, audit, queue)
    assert run.invalid_outputs == 1 and run.status == "executed"


def test_rollback_to_unknown_version_denied_then_valid_rollback(df, cfg):
    train(df, cfg)
    train(df, cfg, force=True)
    old = list_versions(cfg)[0]["version"]
    audit = AuditLog(cfg.model_dir / "a.jsonl")
    queue = ApprovalQueue(cfg.model_dir / "q.json", audit)
    ws = Workspace(cfg, df, audit=audit)
    llm = ScriptedLLM(
        [
            [use("get_model_versions", 1)],
            [propose(2, "rollback", ["get_model_versions"], version="v-erfunden")],
            [propose(3, "rollback", ["get_model_versions"], version=old)],
        ]
    )
    run = run_agent(ws, llm, audit, queue)
    assert run.status == "pending_approval"
    queue.approve(run.run_id, by="jost", ws=ws)
    assert current_version(cfg) == old


def test_fail_safe_without_valid_proposal(env):
    ws, audit, queue = env
    llm = ScriptedLLM([[{"type": "text", "text": "Ich denke nach..."}]] * 3)
    run = run_agent(ws, llm, audit, queue, max_steps=3)
    assert run.status == "failed" and run.proposal is None
    assert audit.read()[-1]["event"] == "agent_failed"


def test_rule_based_baseline_passes_evals():
    from energy_forecast.agent.evals import SCENARIOS, run_evals, summarize
    from energy_forecast.agent.llm import RuleBasedLLM

    picked = [
        s for s in SCENARIOS if s.name in {"broken_model", "stale_data", "heatwave_drift_only"}
    ]
    summary = summarize(run_evals(RuleBasedLLM, picked))
    assert summary["accuracy"] == 1.0 and summary["forbidden_calls"] == 0


def test_claude_adapter_converts_sdk_blocks(monkeypatch):
    """Prüft die Umsetzung der SDK-Antwort ins interne Blockformat, ohne Netzwerk."""
    from types import SimpleNamespace as NS

    from energy_forecast.agent import llm as llm_module

    fake_resp = NS(
        content=[
            NS(type="text", text="Ich prüfe die Daten."),
            NS(type="tool_use", id="tu1", name="get_data_quality", input={}),
        ],
        usage=NS(input_tokens=100, output_tokens=20),
    )
    captured = {}

    class FakeMessages:
        def create(self, **kw):
            captured.update(kw)
            return fake_resp

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    claude = llm_module.ClaudeLLM(model="test-model")
    claude.client = NS(messages=FakeMessages())
    blocks = claude.complete("sys", [{"role": "user", "content": "hi"}], [{"name": "x"}])
    assert blocks[1] == {"type": "tool_use", "id": "tu1", "name": "get_data_quality", "input": {}}
    assert captured["model"] == "test-model" and captured["tools"] == [{"name": "x"}]
    assert claude.tokens == {"input": 100, "output": 20}
