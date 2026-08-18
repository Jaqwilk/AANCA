# Original-confirmatory sealed execution capsule v1

This document specifies the integrated original-confirmatory execution capsule.
The checked-in implementation is not by itself Q, E, a supervisor job, a
publication authorization, or scientific authority.

## Deterministic archive

- The archive is `ZIP_STORED` with `allowZip64=False`.
- Every payload member has a lowercase ASCII-safe POSIX path.
- Every regular `.py` below one final frozen `src/histo_audit/**` tree is
  mapped by stripping only `src/`; no runtime wildcard or mutable source import
  is permitted.
- Exact additional members are `__main__.py`,
  `aanca_capsule/capsule_policy.json`, and
  `aanca_capsule/entry_contract.json`.
- `AANCA_CAPSULE_MANIFEST.json` is canonical UTF-8 JSON with exactly one LF,
  is written last, and is excluded from its own `entries`.
- Entry metadata is fixed to 1980-01-01 00:00:00, Unix regular mode 100444,
  create-system 3, no compression, no comments, no extras, and no explicit
  directory entries.
- Both the builder and bootstrap verify the raw local ZIP headers. The
  bootstrap additionally parses the central directory and end record and
  rejects any bytes outside the one canonical layout.
- The manifest records-root preimage is exactly
  `relative_path\0role\0decimal_size\0lower_sha256\n` in ordinal path order.
- Deterministic construction and publication are separate APIs.
  `build_capsule_bytes` / `build_project_capsule_bytes` return an immutable
  byte build and never create a filesystem destination. Production publication
  has one entry point, `publish_capsule_create_new`, and accepts only the exact
  `artifacts/execution_capsules/<whole_sha256>/original_confirmatory.pyz`
  layout under already-existing plain ancestors.
- Production publication retains no-delete handles for the complete destination
  ancestor chain, performs native no-follow `CREATE_NEW` with the read-only
  attribute at creation, and uses the same retained leaf handle for write,
  `fsync`, byte/hash/archive readback, physical-identity, single-link,
  alternate-stream, final-path, and final readback checks. A post-create
  failure is a permanent partial STOP: the builder never cleans up, adopts, or
  retries the destination.
- Publication is content addressed as
  `<whole_capsule_sha256>/original_confirmatory.pyz`. Before importing project
  code, the bootstrap holds a native no-follow read handle that denies write
  and delete sharing, hashes the whole archive, checks the path segment, and
  retains/rechecks that identity through dispatcher completion.
- Source collection walks the lexical ancestor chain without resolving away
  symlinks or reparse points, rejects alternate streams and multiple links,
  retains each source handle through final path checks, and freezes the exact
  inventory of every regular package `.py`.

## Entry surface

The only modes are `run-confirmatory`, `verify-preterminal`, and
`verify-terminal`. There is no generic `histo_audit` CLI, callback, plugin,
capability, or caller-provided runner.

Before importing `histo_audit`, the bootstrap also requires the exact
canonical six-key capsule policy and five-key ready entry contract. It checks
all imported `histo_audit` origins immediately after importing the dispatcher
and again after the dispatcher returns. A missing capsule member therefore
cannot fall back to an installed, editable, working-directory, or
`PYTHONPATH` copy.

The scientific mode will call only
`histo_audit.experiment.original_confirmatory_runner_core:`
`_run_original_confirmatory_capsule_request` with an exact typed
`OriginalConfirmatoryCapsuleExecutionRequest`.

The terminal modes will call only the two private functions in
`histo_audit.workflows.original_confirmatory_capsule_terminal`.

Q binds both interpreter identities. The logical executable is the retained
project `.venv/Scripts/python.exe` redirector (`sys.executable`); the retained
runtime executable is the native CPython image selected consistently by
internal `GetModuleFileNameW(NULL)`, `sys._base_executable`, and
`sys.orig_argv[0]`. Neither identity substitutes for the other.

E has an exact 20-key top-level schema. Its nested 25-key
`scientific_request_projection` binds the exact Q static-runner root, job and
lineage, runs root and expected run directory, plan/control/bridge/gate/CLI
roots, nonempty checkpoint authority, exact 180-directive profile,
real-PanNuke artifact scope, outcome blindness, no selection/tuning, no
publication, and no automatic retry. Its nested 21-key
`e_consumption_contract` binds the sole early
`e_intent_consumed.json` claim, the supervisor custody receipt, bounded
READY/ACK transport, reduced duplicate-handle access, retention through ACK
and terminal custody, exact Job/process/spec checks, zero scientific input
before ACK, and no automatic retry. The bootstrap validates this contract
before importing or dispatching project code.

The bootstrap creates no E claim while validating argv, the archive, Q, E, or
the command projection. After that prevalidation it exposes one private
arming seam. The sealed handler must call that seam only after its downstream
no-data validation has rederived `run_spec.json` and `process_started.json`;
it must then take the resulting claim handle exactly once and complete the
bound READY/ACK handshake before reading scientific inputs. Synthetic tests
prove that invalid capsule/Q/E/command paths leave no claim and that a valid
prevalidated dispatch can arm and consume exactly once. The real authority,
entry, and terminal handlers implement the required run-spec/process-started
ordering. Their production-entry common-E composition and the external
supervisor's physical Q/E custody boundaries have passed separate no-science
rehearsals.

## Current STOP

The checked-in `entry_contract.json` is now the exact 305-byte ready contract
with SHA-256
`50c2796e0a3e1e06ec3fea3964c9ed1795f9552f85dbd394618529eba61bb844`.
It was promoted only after all three real handlers, the production-entry
common-E gate, and the supervisor's real pipe/process/eight-handle Q/E custody
rehearsal passed. Ready status exposes the sealed dispatcher but is not Q, E,
publication authority, or permission to execute science.

Production capsule publication remains stopped until the final source snapshot
passes the full project QA and PanNuke gates, two independent builds are
byte-identical, capacity is recomputed with the required margin, and the
CREATE_NEW publication/authority chain is independently verified. No
qualifying production capsule has yet been published.

Focused tests include byte determinism, isolated import provenance, Q/E
pre-import anchoring, dual-interpreter leases, one-use E-claim transfer, and
CREATE_NEW publication attacks (pre-existing/partial destination, missing
ancestor, symlink/junction, alternate stream, hardlink, same-byte replacement,
and exception after creation). Gate counts and hashes must be recorded from
the final source after formatting; entry-contract promotion alone does not
qualify or publish a production capsule.
