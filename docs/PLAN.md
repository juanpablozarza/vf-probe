# vf-probe plan (v1, Nov 9 2026)

## Goal
Tell a good verifiers environment from a bad one by testing whether its grader
rewards the wrong thing. Output per env: which reward components are insensitive
to the evidence they claim to depend on, with reproducible fixtures.

Headline target: "X of Y probes blocked across N Hub envs", with FDA-Gym as the clean control.

## Definitions
- Reward insensitive to X: a reward component does not drop when we corrupt exactly
  the evidence it claims to depend on (mutator expects `drop`, component holds).
- Brittle: a component drops on a meaning-preserving change (mutator expects `hold`).
- Unattributed: the env exposes no per-component breakdown; reported, never called a hack.
- Not text-scorable: score() needs runtime/sandbox state, or every drop-mutant scores
  identically. Skipped for text mutators.
- Wording: "reward insensitive to X", never "hackable", until a real trace confirms it.

## Architecture
1. Runner: `vf-probe <env-id>` installs the env in a throwaway uv venv with its own pins,
   records the verifiers version per row.
2. Loader: tasks + Task.score (v1). Detect parser/answer field, reward components,
   runtime/judge needs.
3. Trace sources:
   - Synthetic degenerate replies (empty, prompt echo, yes, refusal, number list). $0.
   - Gold/oracle traces. $0.
   - Real traces (v2): published FDA-Gym Luna/Haiku public.jsonl traces ($0), then cheap-model
     rollouts on Hub envs only with JP's approved cap per env.
4. Mutators (each declares `target_component` and `expect: drop|hold`, returns None if N/A):
   - Answer-span only (via env parser, else final answer field, else skip):
     number perturb, yes/no or option flip, entity swap, keep format/empty content.
   - Cross-task swap: answer of task B scored on A, only when golds differ under the env's
     normalizer. Small answer spaces (yes/no, MCQ) use option flip and report chance 1/k.
   - Tool-use envs only (generalized from FDA-Gym): corrupted observation (replace tool
     outputs, keep answer; strongest), grounding (answer value appears in some tool output),
     phantom reference (cited id/URL never in any tool output), ordering (answer committed
     before first relevant tool result), delete tool calls (only if the env claims to
     measure tool use).
   - Hold (deterministic only in v1): whitespace, case, markdown strip, reorder independent
     sections, equivalent number format where the parser normalizes. No LLM paraphrase in v1.
5. Scorer: deep-copy task, call Task.score(trace, runtime=None), read reward + per-component
   metrics from the trace (confirm exact fields on verifiers 0.4.1). Rule-based graders: run
   twice to confirm determinism, then k=1, eps=0. Judge graders: skipped at $0; with an
   approved cap: temp 0, cache, k=3 paired, flag only if mutant >= base on all 3,
   eps = max(range of base runs, 0.05).
6. Analysis + report: per env x mutator raw counts k/n plus rates, bootstrap CI over tasks
   (not rollouts), Holm correction across all env x mutator tests (or labeled exploratory).
   JSON + Markdown table, one fixture (exact trace + scores) per flag.

## Category coverage (to be confirmed by the Hub survey)
- Math / short answer: answer-span mutators + cross-task swap work well.
- QA / judge-scored: answer-span + swap; judge envs skipped at $0.
- Code / sandbox-scored: usually not text-scorable with runtime=None; out of v1 scope
  (needs runtime and code mutants, milestone 2).
- Tool-use / long-horizon research (FDA-Gym style): full mutator set, incl. corrupted observation.

## Verification (how we know the probe is right)
- Unit tests on toy envs with planted holes (format-only grader, grader ignoring the task,
  grader ignoring tool outputs, state-backed grader). The probe must flag each one, and
  must not flag a clean toy env.
- FDA-Gym control on public.jsonl FG-9101..FG-9193 only: gold passes, degenerate replies 0,
  mutants drop. Numbers measured fresh, not reused.
- CI matrix: verifiers==0.4.1 (pin) and 0.4.0 job asserting the same verdicts on the
  control; any difference is flagged, not hidden.
- Human labels on 3 envs: every flagged trace + a stratified random sample of unflagged
  ones, labeled correct / hack / wrong by JP. Report precision first (we publish about
  other people's envs), then recall.

## Order
1. Repo skeleton, types (Mutator, Row, Report), CLI stub, toy envs, CI. (boilerplate PR)
2. Synthetic + gold probe on toy envs and FDA-Gym control. $0.
3. Answer-span + cross-task mutators; per-component attribution; not-text-scorable detection.
4. Hub loop over 15-20 envs, JSON/MD report, fixtures.
5. README with headline number.
6. v2: real-trace mutants (FDA-Gym public traces first; Hub rollouts with an approved cap),
   tool-use mutators on real traces, human labels on 3 envs.

## Hard rules
$0 by default; any model, judge, or rollout spend needs JP's OK and a cap. Nothing from
FDA-Gym hard sets or hard-set trajectories in fixtures, tests or CI. No posting in
discussions or issues on other people's envs without JP's go. JP merges; RL validates
each PR at URL + SHA.

## Survey adjustments
Hub survey of 60 random envs (Nov 2026): only ~30% expose a parser (none in the
verifiers v1 API), and only ~23% have two or more weighted reward components.

1. Answer locator. Fall back parser -> own extract function -> whole final reply,
   and record which method was used on each row.
2. Attribution. v1 flags on total-score sensitivity. Per-component attribution
   only when >=2 weighted components exist. Zero-weight metrics are evidence
   only, never flags.
3. Tool-use claims. The loader records whether the env claims to check grounding
   or tool use, so answer-only graders are not flagged for ignoring tool outputs.
4. Skips. ~13% need a judge and ~25% need a sandbox or external service. Both
   are marked skipped in v1.
