"""Kontrollierter MLOps-Agent.

Der Agent untersucht den Zustand des Prognosesystems und schlägt eine Maßnahme vor.
Er handelt nie direkt: Jede Aktion läuft durch
  1. Schema-Validierung   (schemas.py)
  2. Policy-Prüfung       (policy.py, deterministischer Code, kein Prompt)
  3. Freigabe-Queue       (approval.py, Mensch entscheidet bei Aktionen mit Folgen)
  4. Audit-Log            (audit.py, jeder Schritt nachvollziehbar)
Wie gut der Agent entscheidet, misst evals.py an simulierten Störfällen.
"""
