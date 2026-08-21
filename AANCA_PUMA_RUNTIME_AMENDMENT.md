# PUMA execution-only runtime amendment

**Freeze date:** 2026-08-21 (Europe/Warsaw)  
**Timing:** frozen after archive/hash/schema verification and before any corruption,
candidate scoring, queue construction, downstream fit or outcome metric  
**Scientific candidate change:** none

The internally frozen PUMA protocol fixed the candidate, split salt, seeds for controlled
corruption, endpoints and success gates, but omitted deterministic implementation
seeds for the group-fold planner, matched comparators and bootstrap. It also did not
state how exact fold-safe k-nearest-neighbour search would be executed at PUMA scale.
This amendment supplies only those execution details.

- audit group-fold seed: `26082190`
- matched-random seed start: `26082192`
- bootstrap seed: `26082191`
- maximum optimiser iterations: `400`
- neighbour search: exact fold-safe cosine top-k after fold-training standardisation,
  implemented in deterministic PyTorch float32 chunks on CUDA when available and
  CPU otherwise; query groups are absent from each fold's reference matrix
- neighbour search query chunk size: `256`
- all distances, neighbour IDs, fold assignments and convergence flags are retained
  in derived evidence or summarised with hashes

Aggregate class counts and file schema were inspected only to implement the already
frozen official mapping. No AANCA score, selected queue, corrupted-label retrieval,
downstream prediction or final-fold metric existed when this amendment was frozen.
The amendment may not change the candidate or rescue a failed result.
