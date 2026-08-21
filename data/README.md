# Local data workspace

Dataset binaries are deliberately excluded from Git. Recreate the local layout as
needed; every maintained command creates its output directory before writing:

- `raw/` — immutable, lawfully obtained source releases such as PanNuke, NuCLS,
  MoNuSAC and PUMA;
- `interim/` — validated intermediate tables;
- `processed/` — derived model inputs;
- `synthetic/` — deterministic synthetic smoke data;
- `manifests/` — compact tracked metadata plus ignored large tabular manifests.

Acquisition, licence and integrity requirements are documented in
[`../DATASET_SETUP.md`](../DATASET_SETUP.md). Never commit raw images, local dataset
archives or derived patient-level data.
