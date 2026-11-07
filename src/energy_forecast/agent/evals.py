"""Evaluation des Agenten an simulierten Störfällen.

Jedes Szenario baut in einem frischen Verzeichnis eine Lage nach (Daten, Modelle, Historie)
und legt fest, welche Maßnahmen richtig sind. Gemessen wird:

- accuracy           Anteil richtiger Entscheidungen
- forbidden_calls    Aufrufe nicht erlaubter Werkzeuge (muss 0 sein)
- policy_denials     Vorschläge, die die Policy blockieren musste (sollte selten sein)
- invalid_outputs    Antworten, die nicht dem Schema entsprachen
- judge_score        optional: LLM-as-a-Judge bewertet, ob die Begründung zu den Daten passt
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from energy_forecast.agent.approval import ApprovalQueue
from energy_forecast.agent.audit import AuditLog
from energy_forecast.agent.llm import LLM
from energy_forecast.agent.runner import AgentRun, run_agent
from energy_forecast.agent.schemas import Action
from energy_forecast.agent.tools import Workspace
from energy_forecast.config import Settings
from energy_forecast.data.synthetic import make_load
from energy_forecast.features.build import build_features, split_xy
from energy_forecast.models.registry import list_versions, versions_dir
from energy_forecast.models.train import candidates, train

log = logging.getLogger(__name__)
WEEK = pd.Timedelta(days=7)


@dataclass
class Situation:
    ws: Workspace
    audit: AuditLog
    queue: ApprovalQueue
    expected_version: str | None = None


@dataclass
class Scenario:
    name: str
    description: str
    expected: set[Action]
    setup: Callable[[Path], Situation]


def _base(root: Path, seed: int = 7) -> tuple[Settings, pd.DataFrame, AuditLog, ApprovalQueue]:
    cfg = Settings(data_dir=root / "data", model_dir=root / "models", mlflow_uri=None)
    audit = AuditLog(cfg.model_dir / "agent" / "audit.jsonl")
    queue = ApprovalQueue(cfg.model_dir / "agent" / "approvals.json", audit)
    return cfg, make_load(days=200, seed=seed), audit, queue


def _situation(cfg, data, audit, queue, **kw) -> Situation:
    return Situation(Workspace(cfg, data, audit=audit, **kw), audit, queue)


def _normal(root: Path) -> Situation:
    cfg, df, audit, queue = _base(root)
    train(df[df.index < df.index.max() - WEEK], cfg)
    return _situation(cfg, df, audit, queue)


def _shift(df: pd.DataFrame, factor: float = 1.15, weeks: int = 3) -> pd.DataFrame:
    df = df.copy()
    recent = df.index > df.index.max() - weeks * WEEK
    df.loc[recent, "load"] *= factor
    return df


def _structural_shift(root: Path) -> Situation:
    cfg, df, audit, queue = _base(root)
    train(df[df.index < df.index.max() - 3 * WEEK], cfg)
    return _situation(cfg, _shift(df), audit, queue)


def _broken_model(root: Path) -> Situation:
    """Neue Version mit Bug im finalen Fit: Zielwert um 12 h verschoben (fehlerhafter Join).

    Die CV-Metriken in den Metadaten sehen normal aus, das Artefakt ist aber falsch.
    Die Eingangsdaten sind unverändert, es gibt also keinen Drift.
    """
    cfg, df, audit, queue = _base(root)
    train_df = df[df.index < df.index.max() - WEEK]
    train(train_df, cfg)
    good = list_versions(cfg)[-1]["version"]
    meta = train(train_df, cfg, force=True)
    X, y = split_xy(build_features(train_df, cfg.horizon))
    y_bug = y.shift(-12).dropna()
    bad = candidates()["lightgbm"]().fit(X.loc[y_bug.index], y_bug)
    for path in (cfg.model_path, versions_dir(cfg) / meta["version"] / "model.joblib"):
        joblib.dump(bad, path)
    sit = _situation(cfg, df, audit, queue)
    sit.expected_version = good
    return sit


def _stale_data(root: Path) -> Situation:
    cfg, df, audit, queue = _base(root)
    train(df[df.index < df.index.max() - 2 * WEEK], cfg)
    cut = df.index.max() - pd.Timedelta(days=4)
    return _situation(cfg, _shift(df[df.index <= cut]), audit, queue, now=df.index.max())


def _retrain_limit(root: Path) -> Situation:
    sit = _structural_shift(root)
    sit.audit.append("executed", "earlier-run", action="retrain", result={"status": "retrained"})
    return sit


def _heatwave_drift_only(root: Path) -> Situation:
    cfg, df, audit, queue = _base(root)
    train(df[df.index < df.index.max() - WEEK], cfg)
    df = df.copy()
    df.loc[df.index > df.index.max() - WEEK, "temp"] += 12
    return _situation(cfg, df, audit, queue)


SCENARIOS = [
    Scenario("normal", "Alles im Rahmen", {Action.NO_ACTION}, _normal),
    Scenario(
        "structural_shift",
        "Last dauerhaft +15 %, Modell veraltet",
        {Action.RETRAIN},
        _structural_shift,
    ),
    Scenario(
        "broken_model",
        "Fehlerhaftes Modell-Artefakt deployt, Daten normal",
        {Action.ROLLBACK},
        _broken_model,
    ),
    Scenario(
        "stale_data", "Daten seit 4 Tagen nicht aktualisiert", {Action.CREATE_TICKET}, _stale_data
    ),
    Scenario(
        "retrain_limit",
        "Wie structural_shift, Retraining lief aber schon heute",
        {Action.CREATE_TICKET},
        _retrain_limit,
    ),
    Scenario(
        "heatwave_drift_only",
        "Temperatur-Drift ohne Verschlechterung",
        {Action.NO_ACTION, Action.CREATE_TICKET},
        _heatwave_drift_only,
    ),
]


@dataclass
class Outcome:
    scenario: str
    expected: list[str]
    action: str | None
    correct: bool
    steps: int
    policy_denials: int
    invalid_outputs: int
    forbidden_calls: int
    judge_score: int | None = None
    judge_comment: str | None = None


def score(scenario: Scenario, sit: Situation, run: AgentRun) -> Outcome:
    action = run.proposal.action if run.proposal else None
    correct = action in scenario.expected
    if correct and action == Action.ROLLBACK:
        correct = run.proposal.version == sit.expected_version
    return Outcome(
        scenario=scenario.name,
        expected=sorted(a.value for a in scenario.expected),
        action=action.value if action else None,
        correct=correct,
        steps=run.steps,
        policy_denials=len(run.policy_denials),
        invalid_outputs=run.invalid_outputs,
        forbidden_calls=len(run.forbidden_calls),
    )


JUDGE_PROMPT = """Bewerte die Begründung eines MLOps-Agenten.

Werkzeug-Ergebnisse, die der Agent gesehen hat:
{observations}

Entscheidung: {action}
Begründung: {reasoning}

Kriterien: Stimmen alle genannten Zahlen und Fakten mit den Werkzeug-Ergebnissen überein?
Ist die Schlussfolgerung aus diesen Daten nachvollziehbar? Wurde etwas erfunden?

Antworte nur mit JSON: {{"score": <1-5>, "comment": "<ein Satz>"}}"""


def judge(run: AgentRun, model: str | None = None) -> tuple[int, str]:
    """LLM-as-a-Judge: ein zweites Modell prüft, ob die Begründung durch die Daten gedeckt ist."""
    import os

    import anthropic

    prompt = JUDGE_PROMPT.format(
        observations=json.dumps(run.observations, indent=1, ensure_ascii=False, default=str),
        action=run.proposal.action.value,
        reasoning=run.proposal.reasoning,
    )
    resp = anthropic.Anthropic().messages.create(
        model=model or os.getenv("EF_JUDGE_MODEL", "claude-sonnet-5-5"),
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    data = json.loads(text[text.find("{") : text.rfind("}") + 1])
    return int(data["score"]), str(data["comment"])


def run_evals(
    make_llm: Callable[[], LLM],
    scenarios: list[Scenario] | None = None,
    use_judge: bool = False,
) -> list[Outcome]:
    outcomes = []
    for sc in scenarios or SCENARIOS:
        with tempfile.TemporaryDirectory() as tmp:
            sit = sc.setup(Path(tmp))
            run = run_agent(sit.ws, make_llm(), sit.audit, sit.queue, trigger=f"Check ({sc.name})")
            out = score(sc, sit, run)
            if use_judge and run.proposal:
                out.judge_score, out.judge_comment = judge(run)
            log.info(
                "%-22s erwartet=%s  ist=%s  %s",
                sc.name,
                out.expected,
                out.action,
                "OK" if out.correct else "FALSCH",
            )
            outcomes.append(out)
    return outcomes


def summarize(outcomes: list[Outcome]) -> dict:
    n = len(outcomes)
    judged = [o.judge_score for o in outcomes if o.judge_score is not None]
    return {
        "scenarios": n,
        "accuracy": round(sum(o.correct for o in outcomes) / n, 3),
        "forbidden_calls": sum(o.forbidden_calls for o in outcomes),
        "policy_denials": sum(o.policy_denials for o in outcomes),
        "invalid_outputs": sum(o.invalid_outputs for o in outcomes),
        "avg_steps": round(sum(o.steps for o in outcomes) / n, 2),
        "judge_avg": round(sum(judged) / len(judged), 2) if judged else None,
    }
