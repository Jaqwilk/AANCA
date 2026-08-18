# Verified starting literature review

**Project:** Automated nucleus-annotation auditing  
**Search and verification date:** 2026-07-17 (Europe/Warsaw)  
**Scope:** targeted verification of the required starting bibliography, not a systematic review or a claim of exhaustive coverage  
**Machine-readable companion:** `reports/literature_matrix.csv`  
**Bibliography:** `references/references.bib`

## Executive assessment

The verified literature supports every major ingredient of the proposed study, but the targeted search did not identify a publication that combines all of them in the same protocol: nucleus-level PanNuke class-label auditing, source-group-safe out-of-fold (OOF) probabilities, multiple risk rankings including current Cleanlab, controlled corruption with an independent generator representation, fixed-budget review metrics, equal-budget restoration, and a downstream untouched-fold test. This is a bounded novelty assessment of the sources searched, not evidence that no similar work exists anywhere.

The closest prior work divides into four groups. Confident Learning and Cleanlab provide general label-quality estimators; AQuA provides a cross-domain benchmark design; the imperfect-annotation study directly tests annotation perturbations on PanNuke and MoNuSAC but studies model robustness rather than review ranking; and CleanPatrick evaluates real-world image-cleaning rankings but on dermatology photographs rather than nuclei. NuCLS provides the strongest identified route to external, multi-rater nucleus validation. Pathology-specific and general self-supervised encoders are plausible feature spaces, yet current benchmarks show that model rankings vary by downstream task. Therefore hypothesis H6 must remain empirical.

## 1. PanNuke provenance and what is not yet safe to assume

The original PanNuke conference chapter reports a semi-automatically constructed pan-cancer nucleus dataset and describes 455 visual fields, 216.4K labelled nuclei, 19 tissue types, and expert validation ([Gamper et al., 2019](https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2), DOI `10.1007/978-3-030-23937-4_2`). The later extension preprint describes approximately 200K nuclei in five positive classes and, in the full text, reports 481 visual fields and 189,744 exhaustively annotated nuclei ([Gamper et al., 2020](https://arxiv.org/abs/2003.10778), arXiv `2003.10778`). These figures refer to different stages/descriptions and should not be collapsed into one supposedly definitive release count.

The archived official [Tissue Image Analytics PanNuke metrics repository](https://github.com/TissueImageAnalytics/PanNuke-metrics) documents the mask shape as `N × 256 × 256 × C` and lists positive-class indices 0–4 as neoplastic, inflammatory, connective tissue, dead, and non-neoplastic epithelial. This is strong implementation evidence, but the project should still read the actual downloaded arrays, inspect whether and how background is represented, verify the fold files, and save hashes before freezing any loader assumption. The old [University of Warwick PanNuke page](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/) now points generally to Tissue Image Analytics and does not expose a clearly current download route in the page inspected on the search date.

Consequences for this project:

- Treat exact counts, channel count/background, class order, file naming, fold composition, licence, and available metadata as unresolved until the acquired release is inspected.
- Keep source patch as the minimum `group_id`. Claim patient- or WSI-level independence only if identifiers can be verified in the downloaded metadata.
- Record the exact source URL, download date, file hashes, extraction process, and any deviations from the publications.
- Preserve the spelling of authors as printed in each primary source: the 2019 chapter has “Ksenija Benet” and “Ali Khuram”; the 2020 arXiv record has “Ksenija Benes” and “Syed Ali Khurram.”

## 2. Label-quality estimation and benchmark design

### Confident Learning and current Cleanlab

Confident Learning estimates the joint distribution between noisy and latent labels under assumptions including class-conditional label noise, then supports counting, pruning, and ranking likely label issues ([Northcutt, Jiang, and Chuang, 2021](https://www.jair.org/index.php/jair/article/view/12125), DOI `10.1613/jair.1.12125`, arXiv `1911.00068`). It is modality-agnostic and is prior art for the estimator, not for the proposed histopathology workflow.

The [stable Cleanlab documentation](https://docs.cleanlab.ai/stable/) resolved to version **2.7.1** on 2026-07-17. The documented `cleanlab.filter.find_label_issues` interface consumes labels plus predicted class probabilities and can return a Boolean mask or indices ranked by `self_confidence`, `normalized_margin`, or `confidence_weighted_entropy`. The [ranking API](https://docs.cleanlab.ai/stable/cleanlab/rank.html) states that lower label-quality scores are more suspicious. The project-wide risk interface instead requires larger values to mean more suspicious; the lossless convention is therefore:

```text
cleanlab_quality = get_label_quality_scores(...)
cleanlab_risk = 1 - cleanlab_quality
```

Both values and the exact method must be saved. Class columns must use the fixed 0…K−1 order. Cleanlab strongly recommends out-of-sample predicted probabilities. Its ordinary cross-validation defaults do not establish source-patch separation, so the project should generate its own `StratifiedGroupKFold`/`GroupKFold` OOF probabilities and pass those probabilities into the functional API. The local environment has Cleanlab **2.9.0 installed and importable**, whereas the stable web documentation inspected resolved to 2.7.1. A deterministic functional smoke test against the 2.9.0 runtime remains required before relying on the documented call signatures and score semantics.

### AQuA

AQuA is the strongest verified benchmark-design precedent ([Goswami et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc20ea8d104cab737a5561096f9bde9b-Abstract.html), NeurIPS 2023, DOI `10.52202/075280-3494`, arXiv `2306.09467`). It covers 17 datasets across image, text, time-series, and tabular modalities; formalises a design space; evaluates seven injected-noise settings; and reports ranking/classification measures including average precision, ROC-AUC, precision, recall, and F1. Its paper explicitly cautions that injected noise may not represent natural annotation errors and that manual validation usually measures only precision at limited scale.

AQuA does not establish a nucleus-specific or patch-group-safe protocol. Its aggregate result that method ordering changes with modality and noise mechanism—including weak aggregate performance for Confident Learning in its tested configuration—must not be transplanted as a PanNuke conclusion. It supports the planned multi-mechanism, multi-metric benchmark and the decision to compare methods rather than assume one winner.

## 3. Histopathology annotation imperfection and representation robustness

Gálvez Jiménez and Decaestecker directly study imperfect annotations in simplified PanNuke and MoNuSAC settings using HoVer-Net ([2024](https://www.sciencedirect.com/science/article/pii/S0010482524006711), DOI `10.1016/j.compbiomed.2024.108586`, arXiv `2410.14365`). They perturb detection, contour, and class annotations and examine training/performance effects, including the role of a small accurately annotated validation set and pretraining. This is the most dataset-adjacent source found. It is not an annotation-risk ranking study: it does not evaluate review-budget AP/lift or the proposed equal-budget restoration comparison, and it mixes error types that this project deliberately scopes apart. Its “accurate” reference annotations are operational reference labels, not guaranteed biological truth.

Dedieu et al. test frozen deep embeddings under label noise across six histopathology patch-classification datasets ([2024](https://arxiv.org/abs/2404.07605), arXiv `2404.07605`). Their experiments support a hypothesis that contrastive/self-supervised pathology representations can improve robustness relative to non-contrastive representations under uniform and asymmetric synthetic noise. The study does not rank individual nucleus labels, use PanNuke instances, enforce source-patch-safe OOF auditing, or test restoration at a review budget. Its authors explicitly identify synthetic-noise and real-world-validation limitations. It therefore supports H6 as a preregistered comparison, not as a guaranteed advantage.

## 4. Synthetic-to-real cleaning gap

CleanPatrick is a large, real-world image-data-cleaning benchmark built from Fitzpatrick17k dermatology images ([Gröger et al., accepted at DMLR, arXiv v2 2026](https://arxiv.org/abs/2505.11034), arXiv `2505.11034`). It collects 496,377 binary judgements from 933 medical crowd workers and covers off-topic images, near-duplicates, and label issues followed by expert review. It treats detection as ranking and reports audit-oriented measures such as average precision, AUROC, precision/recall at `k`, and review-budget behaviour. The primary record directly motivates the gap between convenient synthetic-noise benchmarks and naturally occurring data problems.

This evidence justifies keeping the proposed controlled and exploratory/external analyses separate. CleanPatrick is dermatology photography, not H&E histology; its issue ontology and binary judgements differ from PanNuke’s five-class nucleus labels. Its numerical results should not be used as expected PanNuke performance. Controlled injected-corruption results measure recovery of the injected process only; original PanNuke flags must remain “potentially inconsistent annotations recommended for expert review.”

## 5. NuCLS as external multi-rater evidence

NuCLS contains more than 220,000 breast-cancer nucleus annotations and includes both single-rater and multi-rater regions from non-pathologists and pathologists ([Amgad et al., 2022](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac037/6586817), DOI `10.1093/gigascience/giac037`). The publication reports 222,396 annotations, with over 125,000 single-rater and roughly 97,000 multi-rater annotations, collected with 32 non-pathologists and 7 pathologists. This provides a plausible external test of whether high audit risk correlates with disagreement, adjudication, or reduced consensus.

NuCLS is breast-only, has its own taxonomy and annotation workflow, and includes algorithmic suggestions and truth-inference procedures. A mapping to PanNuke classes has not been verified and must not be forced. Rater disagreement is a measurable external signal, not proof that a particular annotator is wrong or that consensus equals biological truth. Any external analysis should report the mapping, exclusions, rater subsets, and unit of resampling separately from the controlled PanNuke benchmark.

## 6. Encoder landscape and practical selection

The literature establishes several relevant feature families:

| Family | Verified primary source | Relevance | Main caveat for this project |
|---|---|---|---|
| General self-supervised | DINOv2 ([Oquab et al., TMLR 2024](https://arxiv.org/abs/2304.07193), arXiv `2304.07193`) | Ungated general-purpose frozen comparison with several model sizes | Natural-image pretraining; not nucleus- or stain-specific |
| Pathology contrastive | CTransPath ([Wang et al., 2022](https://doi.org/10.1016/j.media.2022.102559), DOI `10.1016/j.media.2022.102559`) | Smaller pathology-aligned encoder; practical candidate | Weight/preprocessing/version terms require a reproducibility check |
| Pathology masked-image modelling | Phikon ([Filiot et al., medRxiv 2023](https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v2), DOI `10.1101/2023.07.21.23292757`) | ViT-B-scale frozen candidate trained on histology | Preprint; exact model-card terms and preprocessing must be pinned |
| Scaled pathology SSL | Phikon-v2 ([Filiot et al., 2024](https://arxiv.org/abs/2409.09173), arXiv `2409.09173`) | ViT-L/DINOv2 pathology comparison | Non-commercial model terms; larger memory/throughput cost |
| General-purpose pathology | UNI ([Chen et al., 2024](https://www.nature.com/articles/s41591-024-02857-3), DOI `10.1038/s41591-024-02857-3`) | Strong tile representation benchmark | Gated weights and non-commercial terms; UNI2-h is larger |
| Pathology vision-language | CONCH ([Lu et al., 2024](https://www.nature.com/articles/s41591-024-02856-4), DOI `10.1038/s41591-024-02856-4`) | Image encoder can provide pathology features | Language-aligned objective is not clearly necessary for nucleus crops |
| Large clinical pathology SSL | Virchow ([Vorontsov et al., 2024](https://www.nature.com/articles/s41591-024-03141-0), DOI `10.1038/s41591-024-03141-0`) and Virchow2 ([Zimmermann et al., 2024](https://arxiv.org/abs/2408.00738), arXiv `2408.00738`) | Current large-scale mixed-tissue comparator | Gated access, large weights, and non-commercial terms for Virchow2; high 12-GB-GPU cost |
| Whole-slide pathology | Prov-GigaPath ([Xu et al., 2024](https://www.nature.com/articles/s41586-024-07441-w), DOI `10.1038/s41586-024-07441-w`) | Large tile encoder plus slide model; broad pathology pretraining | More compute than needed for nucleus crops; slide encoder is out of scope |
| Nucleus architecture | CellViT ([Hörst et al., 2024](https://pubmed.ncbi.nlm.nih.gov/38507894/), DOI `10.1016/j.media.2024.103143`) | Direct PanNuke cell segmentation/classification evidence and target delineation context | End-to-end segmenter/classifier, not a drop-in label auditor; training from scratch is outside scope |

Kang et al. provide a pathology-specific SSL benchmark across multiple downstream tasks, including CoNSeP nucleus instance segmentation ([CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Kang_Benchmarking_Self-Supervised_Learning_on_Diverse_Pathology_Datasets_CVPR_2023_paper.html), DOI `10.1109/CVPR52729.2023.00326`). A later independent clinical benchmark compares public pathology foundation models and emphasizes accuracy–throughput trade-offs and task-dependent rankings ([Campanella et al., 2025](https://www.nature.com/articles/s41467-025-58796-1), DOI `10.1038/s41467-025-58796-1`). Neither benchmark answers which representation best ranks inconsistent PanNuke nucleus labels.

Two current releases require special handling. The official [UNI repository](https://github.com/mahmoodlab/UNI) lists UNI2-h (released January 2025, ViT-h/14-reg8), but no separate peer-reviewed UNI2-h publication was verified; the repository points to the original UNI paper. The official [H-optimus-0 model card](https://huggingface.co/bioptimus/H-optimus-0) describes a gated Apache-2.0 release, but no primary peer-reviewed H-optimus-0 paper was located in this targeted search. These are model releases, not invented publication records. Their current terms, access, checksums, preprocessing, and hardware fit must be checked at execution time.

Practical implication: retain the frozen ImageNet baseline required by the specification, then pilot one reproducible pathology encoder that fits the 12-GB GPU. CTransPath, Phikon, or a smaller DINOv2 variant are lower-friction starting points than a 600M–1B-parameter foundation model. The pathology encoder should be selected by a predeclared availability rule and small feasibility benchmark, not by inspecting primary outcome rankings. The corruption generator and evaluated auditor must use independent representation spaces; otherwise record `circularity_risk` and exclude that cell from confirmatory claims.

## 7. What is supported, novel, and unresolved

### Supported by prior work

- Label-quality ranking from out-of-sample probabilities is established general prior art.
- Benchmark results depend on noise mechanism, modality, metric, and representation; no universal winner is supported.
- Imperfect nucleus annotations can affect segmentation/classification training and evaluation.
- Pathology/self-supervised representations are plausible robustness features, but their benefit is task-dependent.
- Synthetic injected errors do not substitute for natural, expert-reviewed or multi-rater validation.

### Project-level contribution that remains defensible after this targeted review

- A controlled, source-group-safe OOF comparison of complementary annotation-risk rankings on already segmented PanNuke nuclei.
- An explicit independence matrix between corruption-generator and auditor representations.
- Fixed review budgets with AP/lift and group-resampled uncertainty, followed by equal-budget random-versus-guided restoration on an untouched reference fold.
- Exact target indication and neighbouring-nucleus shortcut tests.
- Clear separation of injected-corruption recovery, exploratory original-label ranking, and genuine external/expert validation.

These are contribution candidates, not a guarantee of publication novelty. A broader systematic search, forward/backward citation chasing, and venue-specific search should precede a final novelty statement.

### Unresolved claims that must not be filled by inference

1. **PanNuke release:** exact downloaded counts, arrays, background representation, official-fold files, licence, current authoritative download URL, and patient/WSI identifiers.
2. **Cleanlab runtime compatibility:** version 2.9.0 is installed and passed functional `get_label_quality_scores` and `find_label_issues` calls on group-safe OOF probabilities in the canonical synthetic runs; the stable web docs inspected resolved to 2.7.1, so the documentation/runtime version difference remains recorded.
3. **NuCLS mapping:** a scientifically defensible correspondence to PanNuke classes and the exact external-validation endpoint.
4. **Current model releases:** separate peer-reviewed publications were not verified for UNI2-h or H-optimus-0; access and licence terms can change.
5. **Preprints:** PanNuke extension, Dedieu et al., Virchow2, and Phikon-v2 have no separately verified peer-reviewed venue/venue DOI; Phikon is a medRxiv preprint. CleanPatrick arXiv v2 states acceptance at DMLR, but volume/pages/venue DOI were not verified.
6. **Exhaustiveness:** the search was targeted to the required starting bibliography and selected current encoders. Absence of an almost identical study from this set is not proof of global absence.

## 8. Citation verification policy

Every row in the companion matrix is marked with a verification status. Exact journal/conference metadata were checked against publisher or proceedings pages and Crossref where applicable; arXiv metadata were taken from the corresponding abstract records; software/API claims were taken from official documentation or model repositories. Secondary indices were used only for discovery or cross-checking, never as the sole basis for a record. No performance number is treated as transferable to the proposed PanNuke experiment.
