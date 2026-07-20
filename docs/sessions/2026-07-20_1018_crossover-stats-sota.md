# Session 2026-07-20 10:18 — Ehrlicher Crossover, Statistik-Layer, SOTA (tabular-showdown)

## Kontext & Ziel
Überarbeitung der drei ML-Lab-Portfolio-Projekte auf Bewerbungsqualität. Grundlage:
`~/private/ml-lab/REVIEW.md` (12.07.) + SOTA-Recherche
`~/private/ml-lab/research/2026-07-19-tabular-sota.md` (Stand 2026-07-19). Plan:
`~/private/ml-lab/docs/superpowers/plans/2026-07-19-sota-upgrade.md` (Phase 2, Tasks B1–B5).
Ausführung per `subagent-driven-development` (Implementer + Spec- + Quality-Review pro Task).

Branch: **`fix/review-2026-07`**. Session durch API-Session-Limit unterbrochen
(Reset 01:20 Europe/Berlin) — laufende Subagents starben gleichzeitig.

## ⚠️ Working Tree ist NICHT clean — 4 uncommittete Dateien
Zwei getrennte, unfertige Arbeitspakete liegen uncommittet im Tree (Gate wurde durch
das Limit nie gefahren, deshalb bewusst NICHT committet):

1. **`src/tabular_showdown/stats.py`** — B2-Docstring-Korrektur (reine Doku): die
   `zero_method="pratt"`-Begründung wurde entschärft (vorher „floor stays honest" =
   Überclaim; jetzt korrekt: pratt stellt den n_pairs-Floor bei Zero-Diffs NICHT wieder
   her). Inhalt entspricht exakt der angeforderten Formulierung, wurde vom Quality-Reviewer
   inhaltlich abgesegnet. **Sicher committbar** als `docs: correct pratt zero-method rationale`
   (reiner Docstring, kann keine Tests brechen).
2. **`src/tabular_showdown/explain.py` + `scripts/make_figures.py` + `tests/test_explain.py`**
   — B3 (Kalibrierungs-Figure mit per-size getunten LGBM-Params statt Full-Train-Params +
   Log-Loss). Code wirkt vollständig (`tuned_lgbm_for_calibration` bei `explain.py:95`,
   make_figures verdrahtet, 3 neue Tests). **ABER: Gate nie gelaufen, kein Review.** Muss
   verifiziert werden, bevor es traut.

## Ergebnis (committet)
- **B1 — Seed-Spread + rauschbewusster Titel + Viz-Dedup** (`23dc571`, `101f14a`):
  `compute_seed_spread` (mean/std/min/max/n_seeds pro Größe×Modell), `learning_curve_title`
  behauptet Punkt-Crossover nur wenn Gap > max(Seed-Spread) UND vorzeichenkonsistent über
  gepaarte Seeds — sonst Range-Wortlaut. Titel sagt jetzt „tied within seed noise,
  somewhere between 2k and 5k rows" statt „overtakes at ~5,000". `plot_curve.py`
  konsolidiert nach `viz.plot_learning_curve`. **Spec + Quality beide bestanden.**
- **B2 — Paired-Stats-Modul** (`b3e1fa4`, `85b00c6`): `paired_seed_comparison` (t-CI,
  paired t, Wilcoxon, Tie-by-default = CI muss 0 ausschließen), `paired_from_curve`,
  `wilcoxon_floor_note` (n<6). `_seed_pairs` einmalig in stats.py, aus viz.py importiert.
  In `run_curve.py` verdrahtet. **Spec bestanden, Quality „with fixes" → Fixes committet
  (`85b00c6`)**; die allerletzte Docstring-Nachkorrektur liegt noch uncommittet (siehe oben).

## Entscheidungen
- Crossover-Range = **[2k, 5k]** (nicht REVIEW.mds illustratives [2k,10k]): [2k,10k] bräuchte
  einen ungepaarten Vergleich (LGBM@10k vs TabPFNs letzten Punkt@5k), genau die
  Extrapolation, die B-3 kritisiert. Nur gemessene, gepaarte Seeds.
- scipy nur als bestehende transitive Dep (via scikit-learn), keine neue Abhängigkeit.
- Verdict-Regel CI-basiert, nicht p-basiert (Position-Paper arXiv 2605.17273).

## Offene Fragen
- B3 per-size-Params: die Kurve persistiert Gewinner-Params NICHT (nur Metriken) → der
  Agent hat auf deterministisches Re-Tuning bei n=2000 gesetzt. Beim Review verifizieren,
  dass das deterministisch ist (Optuna-Sampler-Seed wie im Kurvenlauf).

## To-dos
### Nico
1. Nichts Blockierendes. Später (B4, Compute): ein Seed-Erweiterungslauf der Learning-Curve
   (~1,5–2 h CPU) — kann ein Agent autonom fahren, aber gut, wenn du weißt, dass er läuft.

### Nächste Session (Agent)
1. **Working Tree auflösen**: `uv run pytest -q && uv run ruff check .` fahren.
   - Grün → B3 committen (`fix: calibration comparison uses per-size tuned lightgbm params`)
     UND stats.py separat (`docs: correct pratt zero-method rationale`).
   - Rot → B3-Änderungen (`explain.py`, `make_figures.py`, `test_explain.py`) reparieren
     oder zurücksetzen; stats.py (reiner Docstring) ist unabhängig committbar.
2. **B3 reviewen** (Spec + Quality), sobald committet.
3. **B4** (Compute, Plan): Seed-Liste erweitern (≤2000 → 10 Seeds, n=5000 → 5), `run_curve.py`
   im Hintergrund (resumt via skip-keys, append-only), dann Meta + Stats neu.
4. **B5**: Figures regenerieren + README ehrlich umschreiben (Crossover-Range, Kalibrierung
   „within noise" + Log-Loss, LogReg-Absatz, „Where this sits in 2026": TabPFN-2.5/3
   non-commercial/login-gated → 2.0.9-Pin bewusst, TabArena zitieren).

## Einstieg für die nächste Session
Branch `fix/review-2026-07`. **Erst den Working Tree klären** (Gate laufen lassen →
committen oder zurücksetzen, s.o.) — das ist der einzige nicht anderswo persistierte Zustand.
Dann Plan Phase 2 (`~/private/ml-lab/docs/superpowers/plans/2026-07-19-sota-upgrade.md`) via
`subagent-driven-development` fortsetzen: B3-Reviews, dann B4 (Compute), dann B5 (README).
