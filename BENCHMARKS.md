# CellVault Benchmarks

## Current MVP Evidence

The current delivery evidence separates three questions that older aggregate
speedup numbers could not distinguish: how much work is avoided by sharing a
scan, how much depends on the H5AD versus Zarr read path, and what overhead is
added by the general CellVault executor. All recurring timings exclude one-time
CellVault conversion unless explicitly stated.

| MVP | Dataset and workload | Validated result | Evidence |
|---|---|---|---|
| R1–R2 | Synthetic 357,000 × 2,000 sparse matrix; three full-matrix groupings | CellVault joint `2.370 s` versus backed H5AD independent `3.568 s` (`1.51×`); lightweight Zarr shared `2.096 s` exposes `0.274 s` of remaining general-executor overhead | 5 randomized fresh-process runs; 216 array comparisons passed |
| R3 | Wu 2021, 100,064 × 28,468; five lineages, fine-label write-back, multi-level summaries, TNBC versus ER+ marker ranking | Complete measured workflow `7.920 s`; local inputs and aggregation each used one source scan; zero intermediate H5AD bytes | Stable-ID write-back verified; lineage counts and fingerprints recorded |
| R4 | McFarland 2020, 182,875 × 32,738; five matched drug-control tasks over 23 response genes | Median access `5.716 s → 3.727 s` (`1.53×`); scans `5 → 1`; reads `26 → 18` | Cohorts, aggregate values, and response rankings matched exactly |
| R5 | MIBI-TOF, 3,309 × 36; author FOV/donor summaries plus two local FOV analyses | One budgeted run `0.236 s → 0.192 s`; scans `3 → 2`; reads `13 → 10`; two waves under a 0.9 MiB managed-buffer budget | Aggregate, local-analysis, membership, and context checks passed |

The corresponding records are
`benchmark_results/synthetic_357k_execution_attribution.json`,
`benchmark_results/wu2021_complete_workflow.json`,
`benchmark_results/mcfarland_2020_response_workflow.json`, and
`benchmark_results/mibitof_complete_spatial_workflow.json`.

## Execution Strategy Decision Matrix

| Strategy | Task relationship and input semantics | Sharing mechanism | Result type | Choose it when | Boundary |
|---|---|---|---|---|---|
| Independent execution | Tasks are unrelated, use incompatible matrix sources/features, or must be isolated as controls | No sharing; one scan per task | Any task-specific output | There is one task, little row overlap, ample I/O bandwidth, or a clean baseline is required | Repeats metadata and matrix work; may still minimize simultaneously retained outputs |
| Shared aggregation | Compatible tasks consume the same source/features and can be expressed as grouped reductions; cohorts may overlap | Scan the union of source rows once and update multiple accumulators | Group×feature `sum`, `mean`, `count_nonzero`, and `n_cells` | Pseudobulk, abundance/expression summaries, matched comparisons, and marker summaries do not need individual-cell matrices | Cannot replace downstream methods that require cells, neighborhoods, embeddings, or per-cell state |
| Shared materialization | Compatible views need separate cell-level matrices and their outputs fit together | Deduplicate source-row reads, then dispatch rows to each materialized view | Task-level `AnnData` or matrix inputs | Several annotation branches or local analyses reuse the same source and retaining all outputs is acceptable | Retaining every subset can dominate peak memory; the Wu fan-out used about 2.18× the RSS of sequential materialization |
| Budgeted mixed execution | A workload combines reducible summaries with a small number of cell-level consumers | Share bounded decoded batches across aggregate and materialization tasks; split consumers into deterministic waves when necessary | Group summaries plus consumed/local cell-level results | Only selected scopes need PCA, clustering, communication input, or another cell-level calculation, especially under a memory cap | Extra waves can reread overlaps and be slower; the budget covers CellVault-managed buffers, not total process RSS |

Decision rule: first ask whether every output is reducible. If yes, use shared
aggregation. If any output needs individual cells, use shared materialization
when all required matrices fit, otherwise use budgeted mixed execution. Use
independent execution when tasks cannot share source/feature semantics or when
an isolated reference is more important than reuse. SQL cohort selection and
shared execution are separate ideas: SQL defines membership; the executor only
shares compatible matrix access.

## PBMC 68k End-to-End Validation

The public 10x Genomics `fresh_68k_pbmc_donor_a` dataset was downloaded from:

```text
https://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz
```

Download and reproduce preprocessing with:

```bash
python scripts/prepare_pbmc68k.py
```

Preprocessing used Scanpy with `min_genes=200`, `min_cells=3`, mitochondrial
fraction below 20%, total-count normalization to 10,000, `log1p`, and the top
2,000 highly variable genes using the Seurat flavor.

| Stage | Result |
|---|---:|
| Raw matrix | 68,579 cells × 32,738 genes |
| Processed matrix | 68,548 cells × 2,000 genes |
| Processed nonzero values | 2,802,943 |
| Selected Leiden cluster 2 | 5,284 cells |

The complete CellVault workflow finished successfully:

| Stage | Seconds |
|---|---:|
| Backed H5AD → CellVault | 3.676 |
| PCA, 30 components | 8.944 |
| 15-neighbor graph | 39.637 |
| UMAP | 61.339 |
| Leiden | 10.422 |
| H5AD export | 1.107 |
| Complete measured workflow | 125.818 |

## Subset Extraction

Each method ran in a fresh process with one computational thread. Method order
was randomized across five repetitions. The operating-system page cache was not
cleared, so these are warm-cache measurements. All methods materialized exactly
`X`, `obs`, and `var` for the same 5,284 cells.

| Method | Process wall median | Data access subtotal | Peak RSS median |
|---|---:|---:|---:|
| AnnData in memory | 0.956 s | 0.238 s | 241.9 MB |
| AnnData backed | 0.981 s | 0.257 s | 231.1 MB |
| CellVault SQL view | 0.862 s | 0.116 s | 259.6 MB |

The data-access subtotal is `open + select + materialize`, excluding common
Python import and cleanup costs. On this dataset, CellVault data access was
approximately 2.05× faster than full AnnData loading and 2.21× faster than
backed AnnData. End-to-end fresh-process improvement was smaller because package
imports dominate this short workload.

CellVault peak RSS remained 7–12% higher for subset extraction. This is a known
optimization target and should not be presented as a memory advantage at this
dataset size.

This single-subset benchmark does not save or reload an intermediate H5AD. Use
the multi-lineage benchmark below when the traditional workflow persists each
lineage before the next annotation stage.

## Identical Downstream Analysis

A separate run applied the same Scanpy PCA, neighbors, UMAP, and Leiden calls to
each extracted subset with one thread and the same random seed.

| Method | Process wall | Analysis stage | Peak RSS |
|---|---:|---:|---:|
| AnnData in memory | 15.935 s | 14.431 s | 562.4 MB |
| AnnData backed | 15.888 s | 14.424 s | 563.7 MB |
| CellVault SQL view | 15.891 s | 14.491 s | 625.1 MB |

All three methods produced 11 clusters and the same SHA-256 hash for ordered
Leiden labels. This supports a data-access and workflow claim, not an algorithmic
speedup claim.

## SQL Metadata Update

Selecting 5,284 cells by SQL took 0.024 seconds, and updating their metadata in
place with `update_obs_where` took 0.070 seconds. This path does not deserialize
or rewrite the complete observation table.

## Synthetic 357k Stress Test

A deterministic sparse dataset was generated with 357,000 cells, 2,000 genes,
35.7 million nonzero values, and a randomly distributed 14,000-cell target
cluster. The H5AD file was 159.2 MB and the CellVault representation was 40.5 MB.
Backed H5AD conversion took 3.994 seconds and is excluded from query timings.

| Method | Process wall median | Data access subtotal | Peak RSS median |
|---|---:|---:|---:|
| AnnData in memory | 1.849 s | 1.116 s | 545.8 MB |
| AnnData backed | 1.953 s | 1.221 s | 269.9 MB |
| CellVault SQL view | 1.158 s | 0.386 s | 323.7 MB |

CellVault was 1.60× faster end to end than eager AnnData and 1.69× faster than
backed AnnData. For data access alone, the speedups were 2.89× and 3.16×. Peak
RSS was 41% below eager AnnData but 20% above backed AnnData.

Two changes were required for stable behavior on the 256-core benchmark host:
DuckDB connections use one worker, and sparse range reads cap intermediate Zarr
blocks at 16 MB. The CellVault wall-time IQR fell from 0.429 to 0.019 seconds,
and peak-RSS IQR fell from 210.2 to 5.4 MB.

A one-run validation then applied 30-component PCA, neighbors, UMAP, and Leiden
to the extracted 14,000 cells:

| Method | Process wall | Analysis stage | Peak RSS |
|---|---:|---:|---:|
| AnnData in memory | 43.420 s | 40.612 s | 833.1 MB |
| AnnData backed | 43.358 s | 40.514 s | 807.2 MB |
| CellVault SQL view | 42.747 s | 40.728 s | 895.1 MB |

All three paths produced 62 clusters and the same ordered Leiden-label SHA-256.
This single run validates equivalence; it is not a downstream speed estimate.

## Multi-Lineage Annotation Fan-Out

`scripts/benchmark_lineage_workflow.py` models a second annotation level after
the parent object already contains a main-lineage column. It compares:

1. AnnData direct: subset and continue in memory without intermediate files.
2. AnnData saved: subset every lineage, write LZF-compressed H5AD files, release
   the parent, then reload every subset.
3. CellVault SQL: query every lineage and materialize `X`, `obs`, and `var`
   without intermediate subset files.
4. CellVault batch: partition all lineages with one metadata scan and use shared
   matrix reads to materialize them together.

### Wu 2021 Breast Cancer Atlas

The real-data benchmark uses Wu et al., *Nature Genetics* 2021, "A
single-cell and spatially resolved atlas of human breast cancers" (DOI
`10.1038/s41588-021-00911-1`; GEO `GSE176078`). The CELLxGENE-curated H5AD is
843,892,052 bytes with SHA-256
`ef9e792e3e25a811af7778e38913fe0758fb887284899b47bdd54c8af0e91281` and
contains 100,064 cells, 28,468 genes, and three author annotations:
`celltype_major`, `celltype_minor`, and `celltype_subset`.
CELLxGENE exposes the file for public download but does not provide a
dataset-specific SPDX or Creative Commons license in its collection metadata;
reuse should cite the publication and follow CELLxGENE terms of use.

The nine author major labels were combined into five exhaustive branches:

| Branch | Source `celltype_major` labels | Cells |
|---|---|---:|
| T/NK | T-cells | 35,214 |
| B | B-cells, Plasmablasts | 6,730 |
| Stromal | CAFs, Endothelial, PVL | 19,601 |
| Epithelial | Cancer Epithelial, Normal Epithelial | 28,844 |
| Myeloid | Myeloid | 9,675 |

Five randomized fresh-process repetitions on the shared research filesystem
produced the following baseline median (IQR) values:

| Method | Fan-out, source already open | Including source open | Peak RSS | Intermediate files |
|---|---:|---:|---:|---:|
| AnnData direct | 76.673 s (65.278) | 117.822 s (38.249) | 5,593.1 MB | 0 MB |
| AnnData saved + reloaded | 96.493 s (26.623) | 139.885 s (102.287) | 5,602.2 MB | 856.6 MB |
| CellVault SQL views | 9.139 s (9.351) | 9.163 s (9.354) | 2,237.1 MB | 0 MB |

The saved AnnData path spent a median 8.176 seconds writing and 4.727 seconds
reloading five intermediate H5AD files. Relative to that complete fan-out, the
baseline CellVault path was 10.56× faster with an already-open source and
15.27× faster when source-open time was included. It avoided 856.6 MB of
intermediate files.

All methods selected the same ordered cell IDs and passed exact SHA-256 matrix
fingerprint checks for every lineage. Timing variance was large on the shared
node, particularly for eager H5AD reads and SciPy CSR row materialization; the
IQR and raw runs must therefore accompany the median speedups.

The compact, versionable baseline snapshot is stored at
`benchmark_results/wu2021_breast_cancer_lineage_fanout.json`; the complete raw
per-repeat output remains under `benchmark_outputs/` and is intentionally
ignored by Git. This snapshot predates the P0–P1 storage and read-path changes
described next.

### Current Optimized CellVault Measurements

The current optimized store was ingested from the same 100,064 × 28,468 Wu
2021 H5AD in **31.99 s**, with **502 MB** peak RSS and a **651 MB** CellVault
store. The store includes AnnData `raw` and layers; their availability is not a
feature-scope limitation for CellVault.

The current CellVault-only comparison used five randomized fresh-process
repetitions, one thread, and a warm filesystem cache. Values are median (IQR):

| Materialization path | Select | Materialize | Fan-out | Including open | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Sequential SQL views | 0.826 s (0.008) | 4.142 s (0.257) | 4.960 s (0.266) | 4.979 s (0.271) | 1,006.7 MiB (20.4) |
| Shared batch views | 0.735 s (0.004) | 1.064 s (0.017) | 1.797 s (0.020) | 1.816 s (0.020) | 2,195.8 MiB (25.6) |

The shared-batch path uses `partition_obs` plus `materialize_many`, whose
`read_X_many` operation shares source-row reads across branches. It was 2.76×
faster than the current sequential path and 5.08× faster than the **9.139 s**
CellVault median in the historical baseline. The batch path retains all five
matrices concurrently, so its lower runtime comes with roughly 2.18× the peak
RSS of sequential materialization.

For the originally requested save-inclusive comparison, dividing the historical
AnnData saved/reloaded medians by the current batch medians gives 53.69× for
fan-out and 77.02× including source open. Because those values come from two
separate benchmark sessions, they are directional cross-run comparisons rather
than a new paired head-to-head result.

The compact optimized snapshot is stored at
`benchmark_results/wu2021_breast_cancer_cellvault_batch.json`; complete raw
per-repeat output remains under `benchmark_outputs/` and is ignored by Git.

These P0–P1 measurements include compact `int32` CSR indexes when safe, cached
CSR row pointers, a sorted-row read fast path, and shared multi-selection
reads. Sequential materialization uses less peak memory; shared batching trades
additional RSS for substantially less materialization time.

The historical benchmark payload intentionally materialized the common
`X`/`obs`/`var` slots for all methods. The source H5AD also contains `raw` and a
layer, both of which current CellVault imports, exports, and row-subsets.
Source-open measurements still include the cost of opening the complete H5AD,
so the source-already-open fan-out comparison remains the cleaner measure of
subset workflow cost.

### Synthetic Sanity Check

On the synthetic 357,000-cell dataset, four branches covered all cells: T/NK,
B, Stromal, and Myeloid. Five fresh-process repetitions on local temporary
storage produced these medians:

| Method | Fan-out, source already open | Including source open | Peak RSS | Intermediate files |
|---|---:|---:|---:|---:|
| AnnData direct | 0.210 s | 1.180 s | 665.9 MB | 0 MB |
| AnnData saved + reloaded | 2.578 s | 3.555 s | 666.8 MB | 152.8 MB |
| CellVault SQL views | 2.582 s | 2.592 s | 525.6 MB | 0 MB |

The traditional persistence path spent a median 1.372 seconds writing and
0.967 seconds reloading the four H5AD files. With the source already open,
CellVault and saved AnnData were effectively tied in this run: the avoided H5AD
I/O was offset by materializing discontiguous Zarr rows. Including source-open
cost, CellVault was 1.37× faster because opening the database is lazy while
`read_h5ad` loads the parent matrix. CellVault also avoided 152.8 MB of
intermediate artifacts and reduced median peak RSS by about 21%.

AnnData direct remained the fastest option when the parent was already in
memory and no subset file was required. The performance claim is therefore
workflow-specific: CellVault removes intermediate-file management and can
improve reopen-heavy workflows, but SQL is not itself an algorithmic speedup.
The shared research filesystem also favored CellVault over saved AnnData but
showed high matrix-I/O variance, so target-dataset runs should report median and
IQR rather than a single timing.

## Joint Multi-Group Aggregation

The aggregation benchmark executes three real groupings over the complete
100,064 × 28,468 Wu 2021 matrix:

- donor × major cell type;
- donor × minor cell type;
- breast-cancer subtype × major cell type.

Every task computes `sum`, `mean`, and `count_nonzero`. The comparison includes
independent backed-AnnData scans, a hand-written single-scan implementation,
and `CellDB.aggregate_many()`. Each method ran in a fresh process, with random
method order, one thread, a 4,096-row batch size, and five repeats. All 126
cross-method metric-array comparisons passed (`rtol=1e-6`, `atol=1e-8`; integer
statistics exact).

| Method | Aggregation median | Including open | Logical batch reads | Decoded matrix bytes | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Independent tasks | 19.235 s | 19.615 s | 75 | 4,258,298,724 | 1,299.8 MiB |
| Manual single scan | 9.064 s | 9.428 s | 25 | 1,419,432,908 | 1,380.8 MiB |
| CellVault joint | 7.062 s | 7.079 s | 25 | 1,419,432,908 | 1,468.5 MiB |

CellVault joint aggregation was `2.72×` faster than independent execution,
reduced logical matrix reads and decoded bytes by `3×`, and took `0.78×` the
time of the hand-written single-scan control. Peak RSS was 13.0% above the
independent path and 6.4% above the manual single scan because all final dense
group-level metric matrices remain resident. No task-level H5AD intermediates
were produced. Logical reads and decoded bytes are API-level measurements, not
physical disk reads or decompression counts.

The summary and all raw run records are saved in
`benchmark_results/wu2021_breast_cancer_joint_aggregation.json` and its sibling
`_raw` directory.

### Complete Wu Research Workflow

The complete workflow connects the previously separate capabilities: five
main-lineage SQL views, lineage-local marker inputs, stable-ID fine-label
write-back, donor-level aggregation, and a biologically interpretable subtype
summary. This is one measured end-to-end execution, not a repeated timing
distribution.

| Stage | Time | Shared access | Output |
|---|---:|---|---|
| Five lineage-local marker inputs | 1.592 s | 1 source scan, 25 logical reads | 23 available marker features for all 100,064 cells |
| Main- and fine-label write-back | 2.081 s | Metadata update by stable cell ID | `workflow_main_lineage` and `workflow_fine_label`, both verified |
| Multi-level aggregation | 2.491 s | 1 source scan, 25 logical reads | 122 donor×main-lineage, 611 donor×fine-label, and 20 cancer-epithelial subtype groups |
| Complete measured workflow | 7.920 s | Includes orchestration and validation | 0 intermediate H5AD bytes |

The lineage counts exactly match the five-branch reference shown earlier.
`RGS5` was absent from `var` and was recorded as missing; the other 23 requested
markers were consumed in memory and released rather than written as lineage
files. The fine labels come from the author's `celltype_minor` annotation, so
this validates deterministic refinement inputs and write-back rather than a new
classification model.

The downstream example ranks donor-level Cancer Epithelial expression for TNBC
minus ER+: 8 TNBC donors (10,836 cells) versus 9 ER+ donors (11,878 cells).
`YBX1`, `VIM`, and `CD24` lead the positive descriptive ranking; `XBP1`, `AGR2`,
and `BTG2` lead the negative ranking. Source `X` contains normalized expression,
so these are descriptive donor-level differences, not raw-count or
covariate-aware differential-expression results. The complete fingerprints and
stage report are in `benchmark_results/wu2021_complete_workflow.json`.

## Shared Matched Controls

The McFarland MIX-Seq validation used 182,875 cells and five drug tasks. Each
task included only `cell_quality == "normal"` cells and matched treatment and
control membership by `cell_line × time`. The `channel` field was audited but
not used as a universal batch key because it is missing for several arms and
has disjoint values for others. Controls were therefore never pooled globally
just because their perturbation label was `control`.

The complete response workflow uses a curated 23-gene panel and separates data
access from the downstream matched subtraction. Five randomized repeats gave:

| Method | Access median | Downstream median | Total median | Source scans | Logical reads | Decoded bytes |
|---|---:|---:|---:|---:|---:|---:|
| Independent comparisons | 5.716 s | 0.0354 s | 5.752 s | 5 | 26 | 9,086,108 |
| Joint overlapping cohorts | 3.727 s | 0.0359 s | 3.763 s | 1 | 18 | 6,392,800 |

Joint access was `1.53×` faster while downstream computation remained about
0.035 seconds. Independent and joint execution produced identical cohort
memberships, aggregate values, and final rankings for BRD3379, Dabrafenib,
Navitoclax, AZD5591, and JQ1. Each response is the unweighted mean of
`cell_line × time` treated-minus-control mean raw-count expression. It is a
descriptive matched summary, not a fitted model that corrects all experimental
confounders. The current result is
`benchmark_results/mcfarland_2020_response_workflow.json`; the earlier
512-feature executor-only record remains at
`benchmark_results/mcfarland_2020_shared_control.json`.

## Spatial Membership and Mixed Execution

Two public spatial datasets serve different validation purposes.

The complete MIBI-TOF workflow uses author-provided `library_id` fields of view
nested inside donor specimens. Every one of 3,309 cells belongs to one FOV
scope and one donor scope, producing 6,618 membership edges and 40 regional
cell-type abundance/expression rows. The two largest FOVs, `point23` (1,241
cells) and `point8` (1,045 cells), additionally produce local PCA variance and
cluster-marker profiles while preserving their FOV and donor context.

| MIBI-TOF path | Time | Source scans | Logical reads | Decoded bytes | Managed peak buffer |
|---|---:|---:|---:|---:|---:|
| Sequential aggregation and local analyses | 0.236 s | 3 | 13 | 1,633,792 | 724,768 bytes |
| Budgeted joint execution | 0.192 s | 2 | 10 | 1,271,408 | 747,808 bytes |

This is one measured execution. The joint planner stayed below its 943,718-byte
(0.9 MiB) managed-buffer budget by using two waves; the second local FOV was
therefore reread rather than retained with the first. Runtime improved by
`1.23×` and decoded backend output fell 22%, but this small run is primarily a
semantic validation. Aggregate fingerprints, local-analysis outputs, scope
membership, and context IDs matched the sequential reference. See
`benchmark_results/mibitof_complete_spatial_workflow.json`; the earlier
aggregation-only record remains at
`benchmark_results/squidpy_mibitof_predefined_roi.json`.

The IMC validation uses four fixed rectangles on normalized coordinates as an
explicitly controlled overlap experiment, not as curated biological regions.
It contains 10,376 membership edges over 4,668 unique cells.

| IMC workload | Baseline median | Joint median | Speedup | Reads |
|---|---:|---:|---:|---:|
| Four ROI summaries | 0.148 s | 0.065 s | 2.28× | 22 → 10 |
| Sequential summaries/local analyses vs budgeted mixed | 0.192 s | 0.209 s | 0.92× | 20 → 14 |

The budgeted mixed run used a 0.9 MiB hard limit and two deterministic waves.
Its 839,664-byte peak buffer was 40% below the 1,389,899 bytes required when all
local matrices were retained simultaneously. The extra wave made it slower
than both all-at-once (`0.157 s`) and sequential execution on this small input,
but it still reduced sequential reads from 20 to 14. Aggregate outputs and
local-analysis fingerprints matched all paths. Full reports are in
`benchmark_results/squidpy_imc_spatial_roi.json`.

## MVP-5 Adapter Validation

The extension layer is implemented with public task declarations rather than
new executor branches:

- On the Wu atlas, 26 complete donor inputs over 128 selected features were
  prepared in one source scan. Median time was 4.371 s. Donor×cell-type
  pseudobulk took 2.320 s, after which all 26 leave-one-donor objects were
  generated from the small group-level result in 0.076 s without rereading X.
- On the public 10x 5k PBMC CITE-seq data, the same controlled QC cohort was
  applied to separate 33,538-feature RNA and 32-feature ADT stores. Both direct
  references passed; the two necessary source scans took a combined median
  0.332 s.

Machine-readable records are in `benchmark_results/wu2021_mvp5_workflows.json`
and `benchmark_results/pbmc5k_citeseq_multimodal.json`.

## 357k Execution Attribution and Optimization

The same frozen three-task aggregation workload was repeated on the
deterministic 357,000 × 2,000 sparse dataset after adding contiguous-CSR slicing,
vectorized multi-column group encoding, and no-copy reuse for tasks covering a
complete batch. A 32,768-row batch was used for every method. Five methods ran
in randomized fresh processes, one thread each, across five repeats.

| Method | Aggregation median (IQR) | Logical reads | Peak RSS median |
|---|---:|---:|---:|
| Backed H5AD, independent | 3.568 s (0.018) | 33 | 342.8 MiB |
| Backed H5AD, manual single scan | 2.241 s (0.015) | 11 | 327.6 MiB |
| Direct Zarr, independent | 2.818 s (0.102) | 33 | 491.8 MiB |
| Direct Zarr, lightweight single scan | 2.096 s (0.012) | 11 | 468.1 MiB |
| CellVault public joint executor | 2.370 s (0.001) | 11 | 442.2 MiB |

Phase medians expose where the time went:

| Method | Grouping | Matrix read / CSR construction | Dispatch / aggregation | Finalization | Framework overhead |
|---|---:|---:|---:|---:|---:|
| Backed H5AD, independent | 0.305 s | 2.128 s | 1.138 s | 0.001 s | 0 s |
| Backed H5AD, single scan | 0.307 s | 0.786 s | 1.139 s | 0.001 s | 0 s |
| Direct Zarr, independent | 0.689 s | 0.907 s | 1.152 s | 0.003 s | 0 s |
| Direct Zarr, single scan | 0.689 s | 0.271 s | 1.132 s | 0.003 s | 0 s |
| CellVault joint | 0.361 s | 0.284 s | 1.137 s | 0.223 s | 0.363 s |

The attribution supports three separate conclusions:

- Sharing the scan improved backed H5AD by `1.59×` and direct Zarr by `1.34×`.
- For the lightweight shared implementations, Zarr took `0.935×` the H5AD time;
  the storage backend was not the remaining bottleneck in this run.
- CellVault's public executor took `0.274 s` (`1.13×`) more than the lightweight
  Zarr shared control, while still beating independent backed H5AD by `1.51×`.

The CellVault median improved from the frozen `4.043 s` negative baseline to
`2.370 s`, a `1.71×` improvement. All 216 cross-method array comparisons passed
(`rtol=1e-6`, `atol=1e-8`; integer sums and `count_nonzero` exact). The current
result and raw records are in
`benchmark_results/synthetic_357k_execution_attribution.json` and its `_raw`
directory. `benchmark_results/synthetic_357k_joint_aggregation.json` is retained
unchanged as the historical negative baseline. Phase values are medians of each
component and need not sum to the displayed median run.

## Reproduction

Generate a deterministic 357k-cell synthetic dataset if a public or private
dataset is unavailable:

```bash
python scripts/generate_synthetic_357k.py
```

Run extraction-only measurements:

```bash
python scripts/benchmark_subset_workflow.py \
  --input-h5ad benchmark_data/synthetic_357k.h5ad \
  --column leiden \
  --value 2 \
  --repeats 5 \
  --threads 1 \
  --rebuild-cellvault
```

Add `--run-analysis` for the matched downstream workflow. The result JSON stores
raw runs, median/IQR summaries, package versions, parameters, fingerprints, and
system information.

Run a complete main-lineage fan-out, including traditional save and reload:

```bash
python scripts/benchmark_lineage_workflow.py \
  --input-h5ad your_annotated_data.h5ad \
  --column main_lineage \
  --lineage 'T/NK=T/NK' \
  --lineage 'B=B' \
  --lineage 'Stromal=Stromal' \
  --lineage 'Epithelial=Epithelial' \
  --lineage 'Myeloid=Myeloid' \
  --repeats 5 \
  --threads 1 \
  --rebuild-cellvault
```

To isolate the optimized CellVault fan-out paths after preparing a store, use
`--methods cellvault-sql cellvault-batch` and omit `--rebuild-cellvault`.

If T and NK are separate source labels but one branch, use
`--lineage 'T/NK=T|NK'`. Add `--run-analysis` to apply the same PCA, neighbors,
UMAP, and Leiden calls after each lineage is acquired. Analysis output and
fine-label writeback are not persisted by this benchmark.

Reproduce the real breast-cancer benchmark:

```bash
curl -L --fail --continue-at - \
  -o benchmark_data/wu2021_breast_cancer.h5ad \
  https://datasets.cellxgene.cziscience.com/22a27631-aecf-463b-86c6-a8334a2f2cf2.h5ad

python scripts/benchmark_lineage_workflow.py \
  --input-h5ad benchmark_data/wu2021_breast_cancer.h5ad \
  --cellvault-path benchmark_outputs/wu2021_breast_cancer/wu2021.cvdb \
  --output-json benchmark_outputs/wu2021_breast_cancer/lineage_workflow_results.json \
  --column celltype_major \
  --lineage 'T/NK=T-cells' \
  --lineage 'B=B-cells|Plasmablasts' \
  --lineage 'Stromal=CAFs|Endothelial|PVL' \
  --lineage 'Epithelial=Cancer Epithelial|Normal Epithelial' \
  --lineage 'Myeloid=Myeloid' \
  --repeats 5 \
  --threads 1 \
  --rebuild-cellvault
```

Run the joint aggregation benchmark against the prepared Wu CellVault store:

```bash
python scripts/benchmark_aggregate_workflow.py \
  --input-h5ad benchmark_data/wu2021_breast_cancer.h5ad \
  --cellvault-path benchmark_outputs/wu2021_breast_cancer/wu2021.cvdb \
  --output-json benchmark_results/wu2021_breast_cancer_joint_aggregation.json \
  --groupby donor_major=donor_id,celltype_major \
  --groupby donor_minor=donor_id,celltype_minor \
  --groupby subtype_major=subtype,celltype_major \
  --metrics sum mean count_nonzero \
  --batch-size 4096 \
  --repeats 5 \
  --threads 1
```

Run the complete Wu workflow against the optimized store:

```bash
python scripts/run_wu_research_workflow.py
```

Reproduce the complete matched-control response workflow:

```bash
python scripts/run_mcfarland_response_workflow.py
```

The spatial scripts download and verify their small public inputs when absent:

```bash
python scripts/validate_predefined_spatial_rois.py --rebuild-cellvault
python scripts/benchmark_spatial_roi.py --rebuild-cellvault
python scripts/run_spatial_scope_workflow.py
```

Reproduce the public CITE-seq multi-modal validation and the Wu MVP-5 adapter
validation:

```bash
python scripts/benchmark_multimodal_cohort.py --rebuild-cellvault
python scripts/benchmark_mvp5_workflows.py
```

Run the current five-way 357k attribution after generating its input:

```bash
python scripts/benchmark_aggregate_workflow.py \
  --input-h5ad benchmark_outputs/synthetic_357k/input.h5ad \
  --cellvault-path benchmark_outputs/synthetic_357k/input.cvdb \
  --output-json benchmark_results/synthetic_357k_execution_attribution.json \
  --groupby sample_cell_type=sample,cell_type \
  --groupby donor_cell_type=donor,cell_type \
  --groupby sample_cluster=sample,leiden \
  --metrics sum mean count_nonzero \
  --batch-size 32768 \
  --repeats 5 \
  --methods independent manual-single-scan zarr-independent zarr-single-scan cellvault-joint \
  --threads 1
```

## Interpretation Limits

- Repeat on the intended deployment storage and biological dataset before making performance claims.
- Report both warm-cache and controlled cold-cache experiments.
- Compare against both in-memory and backed AnnData, not only standard eager loading.
- Separate one-time conversion cost, SQL filtering, matrix materialization, and algorithms.
- Report peak memory even when runtime improves.
- Treat logical batch reads and decoded bytes as API-level access measures, not physical storage operations.
- Treat the single-run Wu and MIBI-TOF workflow times as execution records and correctness evidence, not timing distributions.
- Remember that `memory_budget_bytes` limits CellVault-managed decoded batches, accumulators, and consumer buffers, not total process RSS.
