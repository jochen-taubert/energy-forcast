"""Kommandozeile für den Agenten.

ef-agent run [--synthetic] [--rules]     Agent einmal laufen lassen
ef-agent pending                         offene Vorschläge anzeigen
ef-agent approve <run_id> --by jost      Vorschlag freigeben und ausführen
ef-agent reject  <run_id> --by jost      Vorschlag ablehnen
ef-agent audit [--run <run_id>]          Audit-Log anzeigen
ef-agent eval [--rules] [--judge]        Evaluation an simulierten Störfällen
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict

import pandas as pd

from energy_forecast.agent.approval import ApprovalQueue
from energy_forecast.agent.audit import AuditLog
from energy_forecast.agent.llm import ClaudeLLM, RuleBasedLLM
from energy_forecast.agent.tools import Workspace
from energy_forecast.config import Settings, settings


def stores(cfg: Settings) -> tuple[AuditLog, ApprovalQueue]:
    audit = AuditLog(cfg.model_dir / "agent" / "audit.jsonl")
    return audit, ApprovalQueue(cfg.model_dir / "agent" / "approvals.json", audit)


def load_workspace(cfg: Settings, synthetic: bool, audit: AuditLog) -> Workspace:
    if synthetic:
        from energy_forecast.data.synthetic import make_load

        df = make_load()
    else:
        df = pd.read_parquet(cfg.raw_path)
    return Workspace(cfg, df, audit=audit)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="ef-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--synthetic", action="store_true")
    run_p.add_argument("--rules", action="store_true", help="Baseline-Agent statt Claude")

    sub.add_parser("pending")
    for name in ("approve", "reject"):
        sp = sub.add_parser(name)
        sp.add_argument("run_id")
        sp.add_argument("--by", required=True)
        sp.add_argument("--note", default="")
        sp.add_argument("--synthetic", action="store_true")

    audit_p = sub.add_parser("audit")
    audit_p.add_argument("--run")

    ev = sub.add_parser("eval")
    ev.add_argument("--rules", action="store_true")
    ev.add_argument("--judge", action="store_true", help="Begründungen per LLM-as-a-Judge bewerten")
    ev.add_argument("--min-accuracy", type=float, default=0.8)
    ev.add_argument("--out", help="Ergebnisse als JSON speichern")

    args = p.parse_args(argv)
    cfg = settings
    audit, queue = stores(cfg)

    match args.cmd:
        case "run":
            from energy_forecast.agent.runner import run_agent

            llm = RuleBasedLLM() if args.rules else ClaudeLLM()
            run = run_agent(load_workspace(cfg, args.synthetic, audit), llm, audit, queue)
            _print(
                {
                    "run_id": run.run_id,
                    "status": run.status,
                    "proposal": run.proposal.model_dump(mode="json") if run.proposal else None,
                    "result": run.result,
                    "policy_denials": run.policy_denials,
                }
            )
        case "pending":
            _print(queue.list())
        case "approve":
            ws = load_workspace(cfg, args.synthetic, audit)
            _print(queue.approve(args.run_id, args.by, ws, args.note))
        case "reject":
            queue.reject(args.run_id, args.by, args.note)
            print("abgelehnt")
        case "audit":
            _print([e for e in audit.read() if not args.run or e["run_id"] == args.run])
        case "eval":
            from energy_forecast.agent.evals import run_evals, summarize

            make_llm = RuleBasedLLM if args.rules else ClaudeLLM
            outcomes = run_evals(make_llm, use_judge=args.judge)
            summary = summarize(outcomes)
            _print({"summary": summary, "outcomes": [asdict(o) for o in outcomes]})
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(
                        {"summary": summary, "outcomes": [asdict(o) for o in outcomes]},
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            # Eval-Gate: verbotene Aufrufe sind nie akzeptabel
            if summary["forbidden_calls"] or summary["accuracy"] < args.min_accuracy:
                print("EVAL-GATE NICHT BESTANDEN", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
