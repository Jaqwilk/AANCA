# Automated Nucleus-Annotation Auditing

This repository is a university research prototype for ranking potentially inconsistent nucleus class annotations for expert review. It is not a diagnostic system and must never automatically modify source annotations.

## Read before changing code

1. `SPEC.md` — frozen scientific and terminology rules.
2. `PLAN.md` — milestone gates and acceptance criteria.
3. `STATUS.md` — executed commands, evidence, blockers, and next command.
4. `PRE_REGISTRATION.md` — analysis definition and freeze state.

## Mandatory invariants

- Split by `group_id` (at least source patch), never by individual nucleus.
- Keep the final reference test fold untouched, uncorrupted, and unavailable for selection or tuning.
- Primary model-based audit scores must use group-safe out-of-fold predictions.
- Permanently separate `pre_corruption_label`, `observed_label`, `is_injected_corruption`, and corruption metadata.
- Instance-dependent corruption and its evaluated auditor must use independent feature spaces; otherwise label the result `circularity_risk` and exclude it from confirmatory claims.
- Never invent metrics, citations, execution claims, expert labels, or unavailable Cleanlab values. Report failures and pilot reductions explicitly.
- Use “potentially inconsistent annotation” and “recommended for expert review”; never claim a pathologist was wrong or that model disagreement proves a medical error.

## Validation

Run `pytest`, `ruff check .`, `ruff format --check .`, and the relevant CLI functional command. Stop and fix a failed mandatory gate before advancing. Update `STATUS.md`, `DECISIONS.md`, and affected documentation after material work.

## Status vocabulary

Only use the completion stages defined in `SPEC.md`: `INITIALISED`, `PIPELINE_COMPLETE`, `PILOT_COMPLETE`, `PRE_REGISTRATION_FROZEN`, `PRIMARY_STUDY_COMPLETE`, `CONFIRMATORY_COMPLETE`, `EXTERNAL_VALIDATION_READY`, `EXTERNAL_VALIDATION_COMPLETE`, and `DEMO_COMPLETE`.
