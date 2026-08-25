# Public independent-pathologist replication pre-outcome runtime amendment v4

Amendment ID: `public_independent_pathologist_replication_runtime_amendment_v4`.

Date: 2026-08-26.

Parent pre-outcome runtime commit:
`abfb89b559f86dbc55b890cd41b860c0cc866467`.

Parent config SHA-256:
`1f8e1fddf3ba8ce38cc73b8769f03de8f91aad2fb0743052e6751a00b20d6321`.

Outcome inspection before amendment: **false**. Four RIVA rotations had sealed
pre-reference scores, but no RIVA reference had been opened. MIDOG++ still had no
embedding or score artifact and no reference had been opened.

## Trigger and geometry-only audit

After the memory-bounded crop correction, MIDOG++ `expert_1` stopped at the existing
inside-centre guard before crop extraction. An outcome-blind geometry audit found
exactly one affected candidate among 3,612 selected rows:

- source annotation ID `15728`, image `309.tiff`;
- image dimensions 6,447 by 4,835 px;
- official bbox `(4186, -30, 4236, 20)` and centre `(4211, -5)`.

The official bbox intersects the top 20 pixel rows of its image, and both frozen crop
windows intersect the image. The other 3,611 candidate centres are inside their
images. No expert reference label or AANCA association was inspected.

## Frozen correction

Do not clamp, shift or exclude this released edge candidate. Permit a centre outside
the pixel rectangle only when every frozen crop window has a non-empty intersection
with the source image; use the already byte-validated periodic `reflect` mapping for
the missing pixels. Fail closed when any crop window lies entirely outside the image.

A unit test covers the released geometry class and a fully non-intersecting control.
The source row, bbox centre, pixels, crop sizes, labels, groups, candidate, model,
folds, risks, endpoints, controls, bootstrap, gates and claims remain unchanged. The
config and its SHA-256 remain unchanged.
