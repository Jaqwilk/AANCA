# Public independent-pathologist replication pre-outcome runtime amendment v3

Amendment ID: `public_independent_pathologist_replication_runtime_amendment_v3`.

Date: 2026-08-26.

Parent pre-outcome runtime commit:
`f9592ecf7a1f333e9244b9328eba963dfb017a98`.

Parent config SHA-256:
`1f8e1fddf3ba8ce38cc73b8769f03de8f91aad2fb0743052e6751a00b20d6321`.

Outcome inspection before amendment: **false**. All four RIVA rotations had produced
pre-reference score seals, but no RIVA reference had been opened. MIDOG++ `expert_1`
failed during crop preparation before embeddings or AANCA scores, and no MIDOG++
pre-reference score directory was created. No dataset had an AANCA/reference
association available.

## Trigger

The inherited NuCLS crop helper implements reflect padding by padding the complete
source image before slicing. That is safe for 1024 px RIVA images, but a MIDOG++ TIFF
is roughly 7,000 by 5,000 px. On the first MIDOG++ crop it attempted an additional
97.1 MiB full-image allocation and stopped with `numpy._core._exceptions._ArrayMemoryError`.

## Frozen correction

The public-replication module now performs the same fixed even-sized reflect crop by:

1. calculating the desired crop bounds;
2. mapping each requested x/y coordinate through NumPy `reflect`'s exact periodic
   boundary rule; and
3. gathering only the output-sized pixel array from the source image.

A deterministic unit test compares the new implementation byte-for-byte with the
previous frozen helper at central, corner and edge coordinates for multiple sizes.
The crop size, centre rounding, padding mode and pixel output are unchanged. The
change only removes an allocation proportional to the full TIFF dimensions.

The source data, 70-case selection, labels, groups, candidate, representation,
folds, risks, budgets, controls, bootstrap, success gates and claim boundary remain
unchanged. The config and its SHA-256 are unchanged.
