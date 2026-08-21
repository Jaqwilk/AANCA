# Ethics and Limitations

This is a university research prototype for dataset annotation auditing. It is not a medical device, is not intended for diagnosis, and must not inform individual-patient care.

Model disagreement does not prove that an annotation or pathologist is wrong. PanNuke source labels are quality-controlled reference annotations, not guaranteed biological truth. The system ranks annotations with low estimated label quality for potential expert review; it does not automatically change them.

Injected label corruption enables objective software and ranking evaluation only against the injected process. Synthetic corruption may differ materially from naturally occurring ambiguity, accidental assignment, annotator variation, segmentation problems, and difficult biological boundaries. Success on injection benchmarks therefore cannot establish real-world annotation-error detection, clinical validity, safety, or patient benefit.

The accepted primary restoration result is adverse to registered hypothesis H4:
audit-guided simulated restoration did not improve downstream macro F1 relative to
equal-budget random review in that experiment. It must be retained and presented
alongside favourable ranking results. Ranking injected changes more efficiently is
not equivalent to demonstrating downstream model benefit.

The later prospectively frozen NuCLS multi-rater evaluation also did not meet its
ranking or downstream success conditions. In the primary subset, the operational
5% ranking interval crossed zero and guided correction was adverse versus leaving
labels unchanged. This is genuine external evidence, but NuCLS P-truth is inferred
pathologist consensus rather than guaranteed biological truth. The completed stage
therefore cannot be described as detection of natural errors, proof that a
pathologist was wrong, clinical validation or real-use improvement.

The frozen MoNuSAC controlled external benchmark retrieved injected changes more
efficiently than matched random review, but its downstream interval crossed zero and
its important-class safety rule failed. The frozen PUMA new-source controlled
confirmation later passed all seven prospective retrieval, downstream, direction,
convergence and every-class gates. This is stronger evidence that the current system
transfers under controlled label noise; it is still not a natural-error experiment.
PUMA publishes final expert-checked labels rather than paired natural pre/post review
decisions.

Post-confirmation PUMA stress retained positive aggregate downstream lower bounds in
all nine scenarios, while only one scenario passed every class safeguard. Opened
PUMA outcomes cannot be reused to tune and reconfirm a candidate. Natural data must
remain `retain_uncorrected` until a new blinded multi-rater programme and prospective
workflow pass the gates in [`NEXT_PHASE.md`](NEXT_PHASE.md).

For the accepted instance-dependent 10% ImageNet-context logistic cells, seeds 404,
405, and 406 produced byte-identical saved rankings and OOF predictions. Those rows
are retained because the primary result is frozen, but they represent one
deterministic realisation rather than three independent replications. No robustness
claim may count them as independent evidence.

Exploratory rankings of unmodified annotations require blinded expert validation. Expert disagreement and insufficient context must be preserved rather than forced into a single “truth.” External multi-rater evaluation requires responsible category mapping and explicit domain-shift analysis; incompatible categories must not be coerced.

Source-patch separation reduces local context leakage but does not guarantee patient- or WSI-level independence unless reliable identifiers are verified in released metadata. Near-duplicate screening is imperfect. Frozen pretrained representations, classifier probabilities, calibration, class imbalance, tissue shift, mask/crop failures, and representation circularity may all affect rankings.

Applicable university ethics review, dataset licences, access terms, governance, privacy rules, and expert-review consent/workflow requirements must be checked before real-data or human-review activity. Dataset files must not be committed. Findings cannot be extrapolated to diagnosis, prognosis, treatment, or patient outcomes.

Reports must use cautious terms such as “potentially inconsistent annotation,” “ambiguous example,” “disagreement with model evidence,” and “recommended for expert review.” They must not use claims such as “confirmed medical mistake,” “true cancer label,” “clinically validated,” or “safe for clinical use.”
