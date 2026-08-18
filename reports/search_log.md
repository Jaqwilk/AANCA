# Literature search and verification log

**Search date:** 2026-07-17  
**Timezone:** Europe/Warsaw  
**Search type:** targeted starting-bibliography verification plus a bounded update for current pathology/self-supervised encoders. This was not a PRISMA/systematic review.  
**Cut-off:** sources and current software/model pages inspected on the search date.

## Verification protocol

1. Search by exact title, topic, and known author/dataset terms.
2. Prefer the publisher, official proceedings, arXiv/medRxiv record, official documentation, or the authors' official code/model repository.
3. Resolve journal/conference DOI metadata through the publisher and Crossref. Preserve arXiv identifiers for preprints; do not present an arXiv-issued resolver DOI as a peer-reviewed venue DOI.
4. Check exact author order, title, year, venue, pages/article number, DOI/arXiv identifier, dataset, task, error type, and evaluation protocol.
5. Use DBLP, search snippets, and other secondary indexes only for discovery/cross-checking. Do not create a bibliography record from a secondary source alone.
6. Where publication stages, release counts, current model versions, licences, or download routes remain ambiguous, record the ambiguity rather than choosing a convenient value.

## Databases and source sites searched

- SpringerLink and Crossref (PanNuke original; bibliographic metadata)
- arXiv (PanNuke extension, AQuA cross-check, representation/noise work, Virchow2, Phikon-v2, DINOv2, CleanPatrick)
- Journal of Artificial Intelligence Research (Confident Learning)
- Official Cleanlab stable documentation and API pages
- NeurIPS proceedings (AQuA)
- ScienceDirect/Elsevier and Crossref (imperfect annotations, CTransPath, CellViT metadata)
- Oxford Academic/GigaScience and Crossref (NuCLS)
- Nature, Nature Medicine, Nature Communications, and Crossref (UNI, CONCH, Virchow, Prov-GigaPath, current benchmark)
- CVF Open Access and Crossref (pathology SSL benchmark)
- PubMed (CellViT cross-check)
- medRxiv (Phikon)
- OpenReview/DMLR listing (CleanPatrick publication status; DINOv2 TMLR status cross-check)
- Official Tissue Image Analytics/Warwick pages and GitHub repository (PanNuke implementation/provenance)
- Official Hugging Face model cards and author repositories (UNI/UNI2-h, H-optimus-0, Virchow2, Phikon-v2 access and release checks)
- DBLP only as a secondary venue/preprint cross-check

## Queries and outcomes

The following records preserve the substantive queries used. Search-engine punctuation/capitalisation may have been normalised by the service.

| Topic | Query | Primary sources retained | Outcome |
|---|---|---|---|
| PanNuke papers | `site:arxiv.org PanNuke original dataset publication Gamper PanNuke dataset extension insights baselines` | Springer chapter; arXiv `2003.10778` | Both required papers verified; original/extension author spelling and count differences retained |
| PanNuke provenance | `site:warwick.ac.uk PanNuke dataset official 19 tissue types 200000 nuclei folds` | Warwick dataset page | Page now redirects users generally to Tissue Image Analytics; current authoritative download route unresolved |
| PanNuke repository | `site:github.com TissueImageAnalytics PanNuke` | `TissueImageAnalytics/PanNuke-metrics` | Official archived repository verifies mask convention and positive-class order; actual release files still required |
| PanNuke DOI | `"PanNuke: An Open Pan-Cancer Histology Dataset" DOI` | SpringerLink; Crossref | DOI `10.1007/978-3-030-23937-4_2`, pages 11-19, exact authors verified |
| Confident Learning | `"Confident Learning: Estimating Uncertainty in Dataset Labels" site:jair.org` | JAIR; Crossref | DOI `10.1613/jair.1.12125`, volume 70, pages 1373-1411, arXiv `1911.00068` verified |
| Cleanlab docs | `Cleanlab stable documentation find_label_issues get_label_quality_scores` | Official stable docs; filter/rank API pages | Stable docs resolved to 2.7.1; score direction and OOS-probability guidance verified; local Cleanlab 2.9.0 subsequently passed functional `get_label_quality_scores` and `find_label_issues` calls on saved group-safe OOF probabilities in the canonical synthetic runs |
| AQuA | `AQuA label quality benchmark dataset label error detection paper AQuA` | NeurIPS proceedings; arXiv `2306.09467` | 17 datasets, four modalities, seven noise settings, metrics, and synthetic-to-natural caution verified |
| Imperfect annotations | `"Impact of imperfect annotations on CNN training" digital pathology DOI` | ScienceDirect; Crossref | DOI `10.1016/j.compbiomed.2024.108586`, volume/article, exact authors verified |
| Imperfect annotations datasets | `"Impact of imperfect annotations" PanNuke MoNuSAC authors` | Publisher page; paper/arXiv cross-check | PanNuke/MoNuSAC and controlled detection/segmentation/classification perturbations verified |
| Noise-resilient pathology | `histopathology "label-noise-resilient" representations paper` | arXiv `2404.07605` | Required representation paper located |
| Noise-resilient venue | `"Contrastive-Based Deep Embeddings for Label Noise-Resilient Histopathology Image Classification" venue` | arXiv; DBLP cross-check | Preprint only; no separate peer-reviewed venue or venue DOI verified |
| CleanPatrick | `CleanPatrick synthetic real data cleaning histopathology label errors` | arXiv `2505.11034`; DMLR/OpenReview listing | Real-world cleaning benchmark located; it is dermatology, not histopathology |
| CleanPatrick status | `"CleanPatrick: A Benchmark for Image Data Cleaning" 2026 DMLR venue DOI` | arXiv v2; DMLR/OpenReview listing | arXiv v2 states accepted at DMLR; no final venue DOI/volume/pages verified |
| NuCLS | `site:academic.oup.com/gigascience NuCLS scalable crowdsourcing dataset nucleus classification segmentation breast cancer` | GigaScience; Crossref | DOI, full authors, annotation counts, and multi-rater structure verified |
| UNI | `site:nature.com "Towards a General-Purpose Foundation Model for Computational Pathology" DOI UNI` | Nature Medicine; Crossref | DOI `10.1038/s41591-024-02857-3` and full metadata verified |
| CONCH | `site:nature.com "A visual-language foundation model for computational pathology" DOI CONCH` | Nature Medicine; Crossref | DOI `10.1038/s41591-024-02856-4` and full metadata verified |
| Virchow | `site:nature.com Virchow foundation model clinical-grade computational pathology rare cancers DOI` | Nature Medicine; Crossref | DOI `10.1038/s41591-024-03141-0` and full metadata verified |
| Prov-GigaPath | `site:nature.com "A whole-slide foundation model for digital pathology" DOI GigaPath` | Nature; Crossref | DOI `10.1038/s41586-024-07441-w` and full metadata verified |
| Phikon | `"Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling" DOI Phikon` | medRxiv | DOI `10.1101/2023.07.21.23292757`; preprint status retained |
| Phikon-v2 | `"Phikon-v2" paper arXiv` | arXiv `2409.09173`; official model card | Exact authors and arXiv verified; no peer-reviewed venue DOI; non-commercial model terms noted |
| CTransPath | `"Transformer-based unsupervised contrastive learning for histopathological image classification" DOI` | Crossref/Elsevier | DOI `10.1016/j.media.2022.102559` and full authors verified |
| Pathology SSL benchmark | `"Benchmarking Self-Supervised Learning on Diverse Pathology Datasets" CVPR DOI` | CVF; Crossref | DOI `10.1109/CVPR52729.2023.00326`, pages 3344-3354 verified |
| DINOv2 | `DINOv2 TMLR 2024 OpenReview Oquab Darcet` | arXiv `2304.07193`; OpenReview status | Exact 26-author arXiv list and TMLR acceptance verified; no venue DOI |
| CellViT | `"CellViT: Vision Transformers for Precise Cell Segmentation and Classification" DOI` | PubMed; Crossref | DOI `10.1016/j.media.2024.103143`, article 103143, full authors verified |
| Current encoder update | `2025 2026 current pathology foundation model encoder UNI2-h paper` | Official UNI repository; independent benchmark | UNI2-h release verified as repository/model update; separate paper not found |
| Current encoder benchmark | `pathology foundation models 2025 encoder benchmark UNI2 Virchow2 H-optimus-0` | Nature Communications benchmark | Independent 2025 public-model benchmark verified, DOI `10.1038/s41467-025-58796-1` |
| H-optimus | `H-optimus-0 pathology foundation model paper Saillard arxiv` | Official Bioptimus/Hugging Face model card | Model release verified; no primary peer-reviewed H-optimus-0 publication located |
| Virchow2 | `site:arxiv.org "Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology"` | arXiv `2408.00738`; official model card | Exact 14-author arXiv record verified; model-card abbreviated citation was not used for authors |
| Official model cards | `site:huggingface.co/bioptimus H-optimus-0 model card`; `site:huggingface.co/owkin/phikon-v2 model card license`; `site:huggingface.co/paige-ai/Virchow2 model card license` | Official Hugging Face repositories | Access, architecture, and current terms checked only for feasibility notes; terms must be rechecked at use time |
| UNI2-h release | `site:github.com/mahmoodlab/UNI UNI2-h 2025 official release` | Official Mahmood Lab repository | Repository lists UNI2-h, January 2025, ViT-h/14-reg8; no separate publication record created |

## Crossref DOI records resolved

Crossref REST metadata were checked on 2026-07-17 for the following exact DOI strings:

- `10.1007/978-3-030-23937-4_2`
- `10.1613/jair.1.12125`
- `10.1016/j.compbiomed.2024.108586`
- `10.1093/gigascience/giac037`
- `10.1038/s41591-024-02857-3`
- `10.1038/s41591-024-02856-4`
- `10.1038/s41591-024-03141-0`
- `10.1038/s41586-024-07441-w`
- `10.1016/j.media.2022.102559`
- `10.1109/CVPR52729.2023.00326`
- `10.1016/j.media.2024.103143`
- `10.1038/s41467-025-58796-1`

Publisher/proceedings pages remained the primary content sources; Crossref supplied or cross-checked exact bibliographic fields and full author lists.

## Inclusion decisions

Included records directly cover at least one required item: PanNuke provenance, label-quality estimation, label-quality benchmarking, imperfect pathology annotations, noise-resilient pathology representations, the synthetic-to-real gap, external multi-rater nucleus annotations, or a credible current frozen encoder/comparison source. Current encoder coverage was deliberately bounded to representative general SSL, pathology SSL, nucleus-specific architecture, and independent benchmark sources.

No bibliography entry was created for UNI2-h or H-optimus-0 because a separate primary publication was not verified. They are documented as current official model releases in the review. A later 2025/2026 Nature Biomedical Engineering foundation-model benchmark was inspected during discovery but omitted from the starting bibliography because the 2025 Nature Communications clinical benchmark already supplies current independent landscape evidence and avoids publication-year ambiguity between online-first and issue assignment.

## Unresolved items and required next checks

| Item | Why unresolved | Required evidence |
|---|---|---|
| Exact PanNuke release counts and layout | Original and extension papers describe different stages/counts; current download route is unclear | Inspect acquired official files, save hashes, enumerate arrays/folds/classes/background and compare to papers |
| PanNuke licence and authoritative current URL | Old Warwick page points elsewhere without a clear current package in the page inspected | Capture licence text/download provenance at acquisition; do not rely on an unofficial mirror silently |
| Patient/WSI identifiers and independence | Publications/pages inspected do not prove usable per-instance patient/WSI IDs in the release | Inspect metadata; otherwise state source-patch-level separation only |
| Cleanlab runtime API compatibility | Local Cleanlab 2.9.0 is installed and importable, while the inspected stable web docs resolved to 2.7.1 | Execute and save a deterministic 2.9.0 functional smoke test with externally generated OOF probabilities; pin the exact calls and score direction used |
| NuCLS-to-PanNuke mapping | Taxonomies and annotation workflows differ | Define and justify a mapping or use mapping-free endpoints such as disagreement; preregister exclusions |
| UNI2-h/H-optimus publication status | Official model releases exist but no separate peer-reviewed primary paper was found | Cite model card/repository as software; update only if a publisher/arXiv primary record is located |
| Model access/licensing/hardware | Gating and terms can change; large models may not fit a 12-GB GPU efficiently | Recheck model card/terms on execution date; pin revision/checksum/preprocessing; run a small batch feasibility benchmark |
| Global novelty | Search was targeted rather than systematic | Conduct forward/backward citation chasing and database-wide structured searches before a final publication novelty claim |

## Integrity notes

- “Potentially inconsistent annotation” and “recommended for expert review” are the appropriate terms; none of the methods proves that a pathologist was wrong.
- Reported source-label or consensus labels are quality-controlled references, not guaranteed biological truth.
- No source supports substituting nucleus-level random cross-validation for source-group-safe OOF predictions.
- No source supports treating controlled injected corruption as evidence of natural-error prevalence or clinical utility.
