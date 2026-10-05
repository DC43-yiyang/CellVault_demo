# CellVault

SQL-driven, reproducibility-oriented data system for single-cell analysis.

CellVault provides an AnnData-compatible interface backed by DuckDB (for obs
metadata) and Zarr (for array data), with lazy row-subset views, provenance
tracking, canonical naming, and pipeline state validation.

## Highlights

- Stores dense data and sparse CSR matrices. CSR `indices` and `indptr` use
  `int32` when the matrix dimensions and nonzero count permit it.
- Preserves AnnData `X`, `obs`, `var`, `layers`, `obsm`, `obsp`, `varm`,
  `varp`, `raw`, and JSON-serializable `uns`; categorical metadata and index
  names round-trip.
- Reads sparse rows with cached CSR pointers and a sorted-row fast path.
  `read_X_many` shares source reads across several selections.
- Supports recursive and optionally named `CellView` objects, partitioned
  observation groups, shared multi-view materialization, bounded matrix
  batches, and view-to-parent metadata write-back.
- Executes compatible multi-group aggregations with one shared matrix scan,
  returning group-level `sum`, `mean`, `count_nonzero`, and cell counts without
  materializing task-level cell matrices.
- Runs the bundled PCA, neighbors, UMAP, and Leiden wrappers on either a
  `CellDB` or `CellView`; view-local analysis artifacts stay local unless a
  write-back option is requested.

## Installation

Python 3.11 or newer is required.

```bash
pip install -e ".[scanpy,dev]"
```

## Quick Start

```python
import anndata
import numpy as np
import pandas as pd
from cellvault import CellDB

# Create from AnnData; replacement always requires explicit opt-in
adata = anndata.AnnData(
    X=np.random.rand(100, 50),
    obs=pd.DataFrame(
        {"cell_type": ["T cell"] * 40 + ["other"] * 60},
        index=[f"cell_{index}" for index in range(100)],
    ),
)
cdb = CellDB.from_anndata(adata, "my_data.cvdb")

# Select cells in DuckDB, then load only their matrix rows
t_cells = cdb.query_obs('"cell_type" = ?', ["T cell"])
t_cell_adata = t_cells.to_anndata(slots={"X", "obs", "var"})

cdb.close()
```

`CellView` keeps stable row positions rather than copying the full expression
matrix. Its `X`, `layers`, `obsm`, `raw`, and induced `obsp` data are read only
when requested.

## Joint Aggregation

Multiple pseudobulk and marker-summary requests can share one bounded-memory
matrix scan:

```python
from cellvault import AggregateTask

run = cdb.aggregate_many(
    [
        AggregateTask(
            "sample_lineage",
            ("donor_id", "celltype_major"),
            metrics=("sum",),
        ),
        AggregateTask(
            "subtype_markers",
            ("subtype", "celltype_major"),
            metrics=("mean", "count_nonzero"),
            features=("CD3D", "MS4A1", "EPCAM"),
        ),
    ],
    batch_size=4096,
)

pseudobulk = run.results["sample_lineage"]
print(run.report.to_dict())
```

Each result is a group×gene `AnnData` with `X=None`, group keys and `n_cells`
in `obs`, and requested statistics in equally named layers. Use
`source="layers:counts"` for a counts layer. Tasks share a scan only when their
source and feature selection are identical.

Each task can additionally define a parameterized SQL `where`/`params` cohort.
Overlapping cohorts share their union read. For spatial or other externally
defined many-to-many groups, pass a long `membership` DataFrame containing
`cell_id` plus grouping columns; overlapping membership contributions remain
independent even though physical matrix rows are deduplicated.

`cdb.execute_tasks()` additionally accepts `MaterializeTask` objects so a shared
batch can update aggregate summaries and feed the few analyses that genuinely
need cell-level matrices. `memory_budget_bytes` provides a hard upper guard for
decoded batches, accumulators, and materialization buffers.

## Choosing an Execution Mode

| Mode | Use it when | Shared work | Result and main tradeoff |
|---|---|---|---|
| Independent | There is one task, tasks use incompatible sources/features, or an isolated baseline is required | None; each task scans separately | Any task-specific result; simplest and often lowest retained memory, but repeated reads are not reused |
| Shared aggregation | Several tasks can be reduced to group statistics from the same matrix source and feature selection | One union scan updates all compatible accumulators | Group×feature `sum`, `mean`, `count_nonzero`, and cell counts; no cell-level matrix is retained |
| Shared materialization | Several compatible subsets need cell-level matrices and all outputs fit comfortably in memory | Source rows are deduplicated and dispatched to multiple views | One matrix per subset; fastest fan-out can require substantially more peak memory |
| Budgeted mixed | A workload combines summaries with a few cell-level analyses, or all materialized outputs do not fit together | Compatible aggregate and materialization tasks share bounded batches; the planner creates deterministic waves when needed | Both group summaries and local cell-level outputs; extra waves can reread overlaps and cost time |

Use shared aggregation whenever the downstream calculation is reducible. Use
shared materialization only when a consumer genuinely needs individual cells.
Set a memory budget for mixed workloads, and use independent execution when
there is no reusable overlap or when measuring a control. A shared scan requires
compatible matrix sources and feature selections. Its budget covers
CellVault-managed buffers, not total process RSS.

The optional workflow adapters reuse these primitives without adding
application-specific branches to the executor:

```python
from cellvault import (
    aggregate_modalities,
    leave_one_out_pseudobulk,
    prepare_communication_inputs,
)

sample_inputs = prepare_communication_inputs(
    cdb,
    sample_column="donor_id",
    cell_type_column="cell_type",
    consumer=run_communication_method,
    memory_budget_bytes=4 * 1024**3,
)

leave_one_donor_out = leave_one_out_pseudobulk(
    pseudobulk,
    donor_column="donor_id",
)

modalities = aggregate_modalities(
    {"rna": rna_cdb, "adt": adt_cdb},
    groupby="cohort",
    metrics=("sum", "mean", "count_nonzero"),
)
```

RNA and ADT stay in separate stores and are scanned independently; only the
cohort declaration is reused. Likewise, communication inputs retain complete
per-sample cell matrices unless a specific downstream method explicitly accepts
summaries.

## 357k-Cell Subset Benchmark

Compare full in-memory AnnData, backed AnnData, and CellVault in isolated
processes with the same selected cells:

```bash
# Optional: create a reproducible 357k x 2k sparse input with 14k target cells
python scripts/generate_synthetic_357k.py

python scripts/benchmark_subset_workflow.py \
  --input-h5ad benchmark_data/synthetic_357k.h5ad \
  --column leiden \
  --value 2 \
  --repeats 5 \
  --threads 1 \
  --rebuild-cellvault
```

Add `--run-analysis` to run identical PCA, neighbors, UMAP, and Leiden steps on
each materialized subset. The JSON output contains every raw run plus
median/IQR wall-time and peak-RSS summaries.

For hierarchical annotation, benchmark all main-lineage branches together and
include the traditional intermediate-file cost:

```bash
python scripts/benchmark_lineage_workflow.py \
  --input-h5ad annotated.h5ad \
  --column main_lineage \
  --lineage 'T/NK=T/NK' \
  --lineage 'B=B' \
  --lineage 'Stromal=Stromal' \
  --lineage 'Epithelial=Epithelial' \
  --lineage 'Myeloid=Myeloid' \
  --repeats 5 \
  --rebuild-cellvault
```

This reports AnnData direct, AnnData subset-save-reload, sequential CellVault
SQL views, and shared CellVault batch materialization separately. CellVault
avoids intermediate lineage files, but analysis still materializes selected
rows into memory.

On the 100,064 × 28,468 Wu 2021 breast-cancer atlas, optimized CellVault
ingestion took 31.99 seconds (502 MB peak RSS; 651 MiB store). In five
randomized, one-thread warm-cache runs, sequential CellVault fan-out took a
median 4.960 seconds and shared batch materialization took 1.797 seconds: 2.76×
faster, with higher peak RSS (2,195.8 versus 1,006.7 MiB). See
[`BENCHMARKS.md`](BENCHMARKS.md) for the phase timings, RSS, methodology, data
provenance, and interpretation limits. The earlier save-and-reload baseline took
96.493 seconds and wrote 856.6 MiB of intermediate files; its 53.69× ratio to
the current batch result is directional because the runs were not paired.

## Validated Research Workflows

- **357k execution attribution and sparse optimization:** across five
  randomized, one-thread runs, aggregation medians were `3.568 s` for backed
  H5AD independent scans, `2.241 s` for a backed H5AD single scan, `2.818 s`
  for direct Zarr independent scans, `2.096 s` for a direct Zarr single scan,
  and `2.370 s` for the CellVault public joint executor. CellVault was `1.51×`
  faster than H5AD independent execution, but retained `0.274 s` of overhead
  over the lightweight Zarr shared control. Its prior `4.043 s` result remains
  as a frozen negative baseline; contiguous CSR slicing, vectorized grouping,
  and no-copy full-batch reuse improved it by `1.71×`. All 216 cross-method
  array comparisons passed.
- **Wu 2021 hierarchical annotation:** one complete measured workflow built
  five main-lineage views, consumed lineage-local marker matrices, wrote the
  author `celltype_minor` labels back by stable cell ID, generated donor×lineage
  and donor×fine-label summaries, and ranked a descriptive TNBC-minus-ER+
  cancer-epithelial contrast in `7.920 s`. Local preparation (`1.592 s`) and
  multi-level aggregation (`2.491 s`) each used one source scan, and no
  intermediate H5AD was written. The author labels are a deterministic
  reference, not predictions from a new classifier.
- **McFarland matched responses:** for five treatments and a curated 23-gene
  panel, median access time fell from `5.716 s` to `3.727 s` (`1.53×`), source
  scans from `5` to `1`, and logical reads from `26` to `18`. Independent and
  joint cohort membership, aggregate values, and response rankings matched.
  The treated-minus-control values are descriptive matched summaries, not a
  fitted response model.
- **MIBI-TOF spatial workflow:** author-provided FOV and donor memberships
  produced 40 regional cell-type summaries, while the two largest FOVs produced
  local PCA variance and cluster-marker profiles with their context IDs intact.
  One budgeted run took `0.192 s` versus `0.236 s` sequentially, reduced source
  scans `3 → 2` and logical reads `13 → 10`, and stayed below its 0.9 MiB
  managed-buffer budget in two waves. Aggregate, local-analysis, membership,
  and context validation all passed.

Machine-readable evidence is stored in
`benchmark_results/synthetic_357k_execution_attribution.json`,
`benchmark_results/wu2021_complete_workflow.json`,
`benchmark_results/mcfarland_2020_response_workflow.json`, and
`benchmark_results/mibitof_complete_spatial_workflow.json`. The earlier Wu
joint-aggregation, controlled IMC overlap, CITE-seq, and adapter validations
remain available in the same directory. Logical reads are API-level batch-read
counts, not physical disk operations.

## Interface Guide

See [`INTERFACE.md`](INTERFACE.md) for a full API and usage guide.
See [`BENCHMARKS.md`](BENCHMARKS.md) for methodology and validated benchmark results.

## License

MIT
