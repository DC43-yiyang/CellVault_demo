# CellVault Interface Guide

This document describes the current user-facing interfaces in `cellvault` and how to use them in practice.

CellVault requires Python 3.11 or newer.

## Public API Surface

```python
from cellvault import CellDB, CellView, NameRegistry, PipelineStateValidator
from cellvault.validator import CellVaultStateError
from cellvault.tools import pca, neighbors, umap, leiden
```

Core idea:
- `CellDB` is the main data container (AnnData-like interface).
- `CellView` is a read-only SQL-filtered view over selected cells.
- `NameRegistry` enforces canonical key naming.
- `PipelineStateValidator` checks pipeline preconditions.
- `cellvault.tools` provides Scanpy wrappers with validation + provenance logging.

## 1. `CellDB` (Main Interface)

### 1.1 Create and Open

```python
from cellvault import CellDB

# Create empty database directory
cdb = CellDB.create("data.cvdb")

# Open existing database
cdb2 = CellDB.open("data.cvdb")
```

### 1.2 Import / Export

```python
import anndata as ad
from cellvault import CellDB

adata = ad.read_h5ad("input.h5ad")

# Import from AnnData object
cdb = CellDB.from_anndata(adata, "my.cvdb")

# Import directly from h5ad
cdb = CellDB.from_h5ad("input.h5ad", "my.cvdb")

# Export to AnnData
adata2 = cdb.to_anndata()

# Export to h5ad
cdb.to_h5ad("output.h5ad")
```

Important:
- Existing paths are never replaced implicitly. Pass `overwrite=True` to
  `create`, `from_anndata`, or `from_h5ad` when replacement is intentional.

### 1.3 AnnData-like Slots

`CellDB` exposes these properties:
- `cdb.obs` (`pandas.DataFrame`)
- `cdb.var` (`pandas.DataFrame`)
- `cdb.X` (`numpy.ndarray` or `scipy.sparse` matrix)
- `cdb.layers` (dict-like accessor)
- `cdb.obsm` (dict-like accessor)
- `cdb.obsp` (dict-like accessor)
- `cdb.varm` (dict-like accessor)
- `cdb.varp` (dict-like accessor)
- `cdb.raw` (when present)
- `cdb.uns` (`dict`)
- `cdb.n_obs`, `cdb.n_vars`, `cdb.shape`

AnnData round-trips preserve categorical columns (including category ordering)
and `obs`/`var` index names, as well as the supported data slots above.

Example:

```python
obs = cdb.obs
obs["batch"] = "A"
cdb.obs = obs

print(cdb.shape)   # (n_obs, n_vars)
print(cdb.n_obs)   # row count
print(cdb.n_vars)  # feature count
```

### 1.4 Partial Metadata Update

```python
cell_ids = ["cell_1", "cell_7", "cell_20"]
new_values = ["T", "B", "T"]

cdb.update_obs("cell_type", cell_ids, new_values)

# Add a column without rewriting the complete obs table
cdb.add_obs_column("reviewed", [False] * cdb.n_obs)

# Update matching rows entirely inside DuckDB
updated = cdb.update_obs_where(
    "reviewed",
    True,
    '"cell_type" = ? AND "donor" = ?',
    ["T cell", "donor_1"],
)
```

### 1.5 SQL Cell Views

Use a parameterized DuckDB predicate to select cells. SQL values belong in
`params`; do not interpolate them into the predicate.

```python
t_cells = cdb.query_obs(
    '"cell_type" = ? AND "sample" IN (?, ?)',
    ["T cell", "sample_1", "sample_2"],
)

print(t_cells.shape)
print(t_cells.obs.head())

# These reads touch selected rows rather than materializing the parent X.
X_t = t_cells.X
pca_t = t_cells.obsm["X_pca"]
graph_t = t_cells.obsp["connectivities"]

# Produce an analysis-ready AnnData or an independent CellVault database.
adata_t = t_cells.to_anndata(slots={"X", "obs", "var"})
t_cell_db = t_cells.materialize("t_cells.cvdb")
```

Project only needed metadata columns when `obs` is wide:

```python
t_cells = cdb.query_obs(
    '"cell_type" = ?',
    ["T cell"],
    columns=["cell_type", "sample", "donor"],
)
```

The view preserves original row order and maps `obsp` to the corresponding
induced subgraph. An empty query remains a valid `(0, n_vars)` view.

### 1.6 Selective Materialization (`to_anndata`)

Use `slots` to avoid loading everything:

```python
# Only what PCA needs
adata_pca = cdb.to_anndata(slots={"X", "obs", "var"})

# Only neighbors graph + metadata
adata_graph = cdb.to_anndata(
    slots={"obs", "var", "obsp", "uns"},
    obsp_keys=["connectivities", "distances"],
)

# Select layers or feature-level matrices without loading X.
adata_counts = cdb.to_anndata(
    slots={"obs", "var", "layers", "raw", "varm", "varp"},
    layer_keys=["counts"],
    varm_keys=["PCs"],
    varp_keys=["correlations"],
)
```

Valid `slots` are `X`, `obs`, `var`, `layers`, `obsm`, `obsp`, `varm`, `varp`,
`raw`, and `uns`. When requested, `raw` and each layer are row-subset along
with `X`; `varm` and `varp` remain feature-aligned.

### 1.7 Resource Management

```python
with CellDB.open("data.cvdb") as cdb:
    print(cdb.shape)
```

`CellDB` holds a DuckDB connection. A context manager or explicit `close()` is
recommended.

### 1.8 Hierarchical Lineage Annotation

A main-lineage label can fan out into repeated fine-annotation runs without
writing one H5AD per lineage:

```python
lineage_queries = {
    "T/NK": ('"main_lineage" = ?', ["T/NK"]),
    "B": ('"main_lineage" = ?', ["B"]),
    "Stromal": ('"main_lineage" = ?', ["Stromal"]),
    "Epithelial": ('"main_lineage" = ?', ["Epithelial"]),
    "Myeloid": ('"main_lineage" = ?', ["Myeloid"]),
}

if "fine_label" not in cdb.obs_columns:
    cdb.add_obs_column("fine_label", [""] * cdb.n_obs)

for lineage, (predicate, params) in lineage_queries.items():
    view = cdb.query_obs(predicate, params)
    subset = view.to_anndata(slots={"X", "obs", "var"})

    # Run the lineage-specific Scanpy or annotation workflow here.
    subset.obs["fine_label"] = annotate_lineage(subset, lineage)

    # Write labels back by stable cell identifier; no parent-matrix rewrite.
    cdb.update_obs("fine_label", subset.obs_names, subset.obs["fine_label"])
```

The SQL query itself returns a lightweight, read-only `CellView`. Matrix rows
are loaded when `X`, `layers`, or `raw` is read, or when `to_anndata()` is
called. `CellView` supports the same row-subset behavior for these slots.

### 1.9 Partition, Reuse, and Batch Views

Create non-overlapping lineage views from one observation scan. Set
`require_complete=True` when every observation must belong to exactly one
group, and `persist=True` to save the resulting definitions by name.

```python
views = cdb.partition_obs(
    "main_lineage",
    {
        "T/NK": ["T", "NK"],
        "B": ["B"],
        "Myeloid": ["Myeloid"],
    },
    require_complete=True,
    persist=True,
)

# Read overlapping source rows once while materializing all requested groups.
subsets = cdb.materialize_many(
    views,
    slots={"X", "obs", "var", "layers", "raw"},
    layer_keys=["counts"],
)
```

`materialize_many` uses the backend's shared `read_X_many` path, which reads
source rows once while preserving each view's requested order and duplicates.
For streaming workflows, use bounded batches instead of materializing a whole
view:

```python
for batch_obs, batch_X in views["T/NK"].iter_X_batches(
    batch_size=4096,
    layer="counts",
):
    consume(batch_obs, batch_X)
```

`cdb.iter_X_batches(...)` provides the same interface for all observations.

### 1.10 Joint Group Aggregation

Use `AggregateTask` when an analysis needs group-level expression summaries
rather than one materialized cell matrix per subset:

```python
from cellvault import AggregateTask

run = cdb.aggregate_many(
    [
        AggregateTask(
            name="sample_lineage",
            groupby=("sample", "main_lineage"),
            source="layers:counts",
            metrics=("sum",),
        ),
        AggregateTask(
            name="condition_cell_type",
            groupby=("condition", "cell_type"),
            source="X",
            metrics=("mean", "count_nonzero"),
        ),
    ],
    batch_size=4096,
)
```

`run.results` maps task names to group×gene AnnData objects. Their `X` is
unset, each requested statistic is stored in a same-named layer, group keys and
`n_cells` are stored in `obs`, and the selected feature table is stored in
`var`. A task with `metrics=()` computes cell composition without reading an
expression matrix.

Tasks with identical `source` and `features` share one batch scan. Supported
sources are `"X"` and `"layers:<name>"`; supported metrics are `sum`, `mean`,
and `count_nonzero`. `features` accepts ordered var names. Missing grouping
values form an explicit `"<NA>"` group, and only observed categorical groups
are emitted.

`run.report` records the number of compatible source scans and matrix batch
reads, logical requested and unique rows, decoded matrix bytes returned by the
backend, reuse ratio, elapsed time, and result sizes. The byte counter is not a
physical storage-read or decompression counter.

The same method is available on a `CellView`; in that case only cells in the
view contribute to the aggregation.

Tasks may select different, overlapping cohorts with parameterized SQL. The
matrix reader scans the union of compatible task rows, while each task keeps
its own membership:

```python
shared_control = cdb.aggregate_many(
    [
        AggregateTask(
            "drug_a",
            ("cell_line", "treatment"),
            where='"treatment" IN (?, ?)',
            params=("control", "drug_a"),
        ),
        AggregateTask(
            "drug_b",
            ("cell_line", "treatment"),
            where='"treatment" IN (?, ?)',
            params=("control", "drug_b"),
        ),
    ]
)
```

For externally defined overlapping groups such as spatial ROIs, attach a long
membership table. Grouping columns can come from either the membership table
or `obs`:

```python
membership = pandas.DataFrame(
    {
        "cell_id": ["cell_1", "cell_1", "cell_2"],
        "roi": ["tumor_edge", "immune_niche", "tumor_edge"],
    }
)

roi_run = cdb.aggregate_many(
    [
        AggregateTask(
            "roi_cell_type",
            ("roi", "cell_type"),
            membership=membership,
            metrics=("mean", "count_nonzero"),
        )
    ]
)
```

Duplicate cell/group edges, unknown cell IDs, and weighted memberships are
rejected. An optional binary `membership` column filters inactive rows. A cell
in multiple groups is read once but contributes once to every declared group.

Both concrete task types implement the structural `Task` protocol. When some
tasks require cell-level matrices, combine `AggregateTask` and
`MaterializeTask` through `execute_tasks()`:

```python
from cellvault import AggregateTask, MaterializeTask

run = cdb.execute_tasks(
    [
        AggregateTask("sample_summary", "sample", metrics=("sum",)),
        MaterializeTask(
            "t_cells",
            where='"main_lineage" = ?',
            params=("T/NK",),
            features=marker_genes,
        ),
    ],
    batch_size=4096,
    memory_budget_bytes=4 * 1024**3,
)
```

Compatible aggregate and materialize consumers receive the same decoded
batches. A materialized result places the requested source in `X`; its original
source is recorded in `uns["cellvault_materialization"]`. `cell_ids` may be
supplied for externally defined, overlapping views, and output row order follows
the parent matrix.

The memory budget is a hard safety limit over decoded batches, accumulators,
derived aggregate buffers such as `mean`, materialization buffers, and retained
consumer outputs. It does not attempt to meter allocations made internally by
an arbitrary consumer. If CellVault-managed buffers cannot fit, execution
raises `MemoryError` with a deterministic instruction to reduce the batch,
feature, cohort, metric, or concurrent-task size instead of silently exceeding
the budget.
The planner preserves task order and packs as many compatible materializers as
the budget estimate permits; it does not split tasks that fit in one wave.
`run.report.scan_plan` records each source/feature wave, its consumers, row and
batch counts, repeated rows, and peak batch, accumulator, consumer, and total
buffer bytes. `run.report.degradation_reasons` explains when a configured
budget serializes otherwise compatible materialization tasks.

### 1.11 Workflow Adapters

Three thin adapters cover MVP extension scenarios without changing the core
execution loop:

```python
from cellvault import (
    aggregate_modalities,
    leave_one_out_pseudobulk,
    prepare_communication_inputs,
)

communication = prepare_communication_inputs(
    cdb,
    sample_column="sample",
    cell_type_column="cell_type",
    consumer=run_one_sample,
    memory_budget_bytes=4 * 1024**3,
)

leave_one_out = leave_one_out_pseudobulk(
    pseudobulk,
    donor_column="donor_id",
)

multimodal = aggregate_modalities(
    {"rna": rna_cdb, "adt": adt_cdb},
    groupby=("condition", "cell_type"),
    where='"condition" IN (?, ?)',
    params=("control", "treated"),
    metrics=("sum", "mean", "count_nonzero"),
)
```

`prepare_communication_inputs()` materializes every selected cell for each
sample; it does not assume that a communication method can use pseudobulk.
Providing a consumer lets the input be processed and released before a later
memory-budget wave. `leave_one_out_pseudobulk()` slices the already aggregated
group-level object and never rereads the cell matrix. `aggregate_modalities()`
applies one SQL cohort definition to separate stores and feature axes; its
reports remain separate because it does not claim cross-modality physical I/O
sharing.

### 1.12 Recursive Named Views and Write-Back

Refine a view with another parameterized predicate. Passing `name=` persists a
JSON-serializable query definition; `load_view` re-evaluates it against the
current database without copying matrix data.

```python
t_cells = cdb.query_obs('"main_lineage" = ?', ["T"], name="t_cells")
activated_t = t_cells.query_obs('"score" > ?', [2], name="activated_t")

reopened = cdb.load_view("activated_t")
print(cdb.named_views)
```

Views are read-only with respect to parent array slots, but selected metadata
can be written back to the parent by stable cell identifier:

```python
activated_t.update_obs(
    "fine_label",
    ["T1"] * activated_t.n_obs,
    create=True,
    fill_value="",
)
```

### 1.13 Sparse Read Behavior

Sparse arrays are stored as CSR Zarr groups. `indices` and `indptr` use
`int32` when the shape and nonzero count fit, otherwise `int64`. CellVault
caches CSR row pointers, uses a sorted-row read fast path, and restores the
caller order for unsorted selections. These implementation details reduce
repeated and discontiguous row-read overhead without changing the returned
matrix semantics.

## 2. `NameRegistry` (Canonical Naming)

`NameRegistry` maps analysis operation + integration mode to canonical keys.

```python
from cellvault import NameRegistry

NameRegistry.get("pca")                 # "X_pca"
NameRegistry.get("umap", "harmony")     # "X_umap_harmony"
NameRegistry.lookup("X_pca")            # ("pca", None)
NameRegistry.is_canonical("X_pca")      # True
```

Register custom key:

```python
reg = NameRegistry()
reg.register("custom_embedding", None, "X_custom_embedding")
```

Important:
- Writing through `cdb.obsm[key] = ...` and `cdb.obsp[key] = ...` requires `key` to be canonical.
- Use `NameRegistry.register(...)` before writing custom keys.

## 3. `PipelineStateValidator` (Precondition Check)

```python
from cellvault import PipelineStateValidator
from cellvault.validator import CellVaultStateError

state = cdb.get_state()
try:
    PipelineStateValidator.validate("neighbors", state)
except CellVaultStateError as e:
    print(e.operation, e.missing, e.available)
```

Helper methods:
- `PipelineStateValidator.get_preconditions(operation)`
- `PipelineStateValidator.list_operations()`

## 4. `cellvault.tools` (Analysis Wrappers)

Provided wrappers:
- `pca(cdb, n_comps=50, integration=None, **kwargs)`
- `neighbors(cdb, n_neighbors=15, integration=None, use_rep=None, **kwargs)`
- `umap(cdb, integration=None, **kwargs)`
- `leiden(cdb, resolution=1.0, integration=None, **kwargs)`

These wrappers:
- validate state before execution,
- use canonical output names,
- update `cdb.uns` metadata,
- append provenance logs.

All four wrappers accept a `CellDB` or `CellView`. On a view, PCA, neighbors,
and UMAP outputs are stored on that view and do not add parent-wide matrix or
graph artifacts. `leiden(view, write_back=True, output_column="fine_cluster")`
optionally writes the resulting labels to the selected parent `obs` rows.

### End-to-End Example

```python
from cellvault import CellDB
from cellvault.tools import pca, neighbors, umap, leiden

cdb = CellDB.from_h5ad("pbmc.h5ad", "pbmc.cvdb")

pca_key = pca(cdb, n_comps=30)              # "X_pca"
neighbors_key = neighbors(cdb, n_neighbors=15)
umap_key = umap(cdb)                        # "X_umap"
cluster_key = leiden(cdb, resolution=1.0)   # "leiden"

print(pca_key, neighbors_key, umap_key, cluster_key)
print(cdb.obsm[umap_key].shape)
print(cdb.obs[cluster_key].head())

cdb.close()
```

Run the same tools against one selected branch:

```python
view = cdb.query_obs('"main_lineage" = ?', ["T"])
pca(view, n_comps=30)
neighbors(view, n_neighbors=15)
umap(view)
leiden(view, write_back=True, output_column="fine_cluster")
```

## 5. Provenance

All major writes append JSONL entries:

```python
entries = cdb.provenance.read_log()
latest_pca = cdb.provenance.query(operation="pca")
```

Each entry includes:
- timestamp,
- operation,
- target,
- key,
- params,
- old/new hash (when available).

## 6. Practical Notes

- Install with Scanpy extras if you use `cellvault.tools`:

```bash
pip install -e ".[scanpy]"
```

- `uns` is JSON-backed; non-serializable values may be skipped during import with warnings.
- Storage is split by slot: DuckDB (`obs`), Parquet (`var` and raw `var`),
  Zarr (`X`, `layers`, `obsm`, `obsp`, `varm`, `varp`, and raw arrays), JSON
  (`uns` and dataframe schema), JSONL (`provenance`).
- `from_h5ad` opens `X` in backed mode and streams dense or CSR values into
  Zarr. CSR indexes are compacted to `int32` where safe.

## 7. Large Subset Benchmark

`scripts/benchmark_subset_workflow.py` compares three independent-process
paths: full in-memory AnnData, backed AnnData, and a CellVault SQL view. It
checks that cell IDs and expression-matrix fingerprints match before reporting
median/IQR timing and peak RSS.

```bash
python scripts/generate_synthetic_357k.py

python scripts/benchmark_subset_workflow.py \
  --input-h5ad benchmark_data/synthetic_357k.h5ad \
  --column leiden \
  --value 2 \
  --repeats 5 \
  --threads 1 \
  --rebuild-cellvault
```

Use `--run-analysis` to apply the same PCA, neighbors, UMAP, and Leiden calls to
each subset. Conversion to CellVault is reported separately and excluded from
the repeated query timings. Method order is randomized, but the script does not
clear the operating-system file cache.

For a full main-lineage fan-out that includes traditional subset save and
reload time, use:

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

The JSON separates fan-out time with an already-open source from the total that
also includes opening the parent H5AD or CellVault database. Its
`cellvault-batch` method benchmarks `partition_obs()` plus `materialize_many()`
against sequential `cellvault-sql` views.
