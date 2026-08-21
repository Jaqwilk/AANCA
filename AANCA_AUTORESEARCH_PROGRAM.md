# AANCA autoresearch program

The research loop is intentionally narrower than ordinary repository development.

## Fixed files

Do not change the frozen external evaluators, published evidence, external-test
configs, source annotations or the internal lockbox partition after the first metric
is written. `AANCA_AUTORESEARCH_PROTOCOL.md` and
`configs/aanca_autoresearch_development.yaml` define the evaluator.

## Editable research surface

Candidate definitions may vary only fields declared in the development config:
feature view, audit classifier regularisation, risk strategy, neighbour settings,
queue constraints, review budget, intervention policy and downstream classifier
regularisation. New candidate families require an explicit config amendment before
their outcomes are run.

## Loop

1. Establish the current AANCA baseline.
2. Propose one candidate or one coherent batch.
3. Run the fixed discovery evaluator without any forbidden test input.
4. Append the outcome to the experiment ledger.
5. Keep a candidate only if it improves the downstream objective without violating
   class safety; ranking-only gains are recorded but do not advance the model.
6. Use successive halving: cheap directional screening first, then full nested
   patient evaluation for finalists.
7. Freeze exactly one winner before opening the internal lockbox.
8. If the lockbox gates fail, retain the unchanged model and report the failure.

Never tune on the internal lockbox, MoNuSAC official test or frozen NuCLS outcomes.
Never convert a controlled injected-label result into a natural-error claim.

