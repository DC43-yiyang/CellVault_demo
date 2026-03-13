# CellVault: A Reproducibility-Oriented Storage Layer for Single-Cell Analysis
## Slide Outline (~24 slides)

---

## PART 1 · Introduction (2 slides)

### Slide 1 — Title
- Title: **CellVault: Reproducibility-Oriented Storage for Single-Cell Analysis**
- Subtitle: An AnnData-compatible data system powered by DuckDB + Zarr
- Author / Date

---

### Slide 2 — The Scale Reality of Single-Cell Analysis
- Scale: cell counts are growing from tens of thousands to millions (this benchmark: 357,093 cells × 20,522 genes)
- Typical pipeline:
  ```
  Raw data → QC → Normalization → Highly-Variable Gene Selection
           → Scaling → PCA → Neighbors → UMAP → Leiden Clustering
  ```
- Key pain points (preview):
  - Large datasets = extremely slow computation, very high memory demand
  - Storage format = full load required, huge disk footprint
  - Result keys = ad-hoc naming, easy to mix up
  - Analysis process = no provenance, hard to reproduce

---

## PART 2 · Limitations of Existing Tools (4 slides)

### Slide 3 — Problem 1: The Full-Load Memory Wall
- **AnnData + HDF5 (.h5ad)** is the current standard format
- Reading any data → the entire file must be loaded into memory
- Example scenario: updating the annotation of 14,039 cells in Cluster 2
  - Scanpy approach: `sc.read_h5ad(...)` loads everything (including the X matrix) → modify → write everything back
  - Peak memory ≈ 2× file size
- For large datasets (>1 million cells), even a single metadata edit demands enormous server memory

---

### Slide 4 — Problem 2: No Traceability of the Analysis Process
- After the script finishes, the result is buried in `adata.obsm['X_pca_v3_final2']`
- No machine-readable record: which parameters? at what time? was it overwritten?
- Deleting an intermediate result → must rerun the entire pipeline at high cost
- The reproducibility crisis is just as severe in single-cell biology; others cannot reproduce results after publication

---

### Slide 5 — Problem 3: No Naming Convention
- obsm key names are completely free-form: `X_pca` / `PCA` / `pca_harmony` / `pca_integrated` used interchangeably
- Different team members writing the same pipeline produce different key names → `KeyError` in downstream tools
- No standard naming scheme when multiple integration methods coexist (Harmony, Scanorama, etc.)
- No way to programmatically verify whether a given step's result exists

---

### Slide 6 — Problem 4: No Pipeline State Validation
- Common mistake: calling `sc.tl.leiden()` right after PCA, skipping `sc.pp.neighbors()`
  - Scanpy gives no clear error; it raises `KeyError: 'neighbors'`, leaving the user to figure out which step was missed
- Another failure mode: running `sc.tl.umap()` without `connectivities` present — error message is opaque
- Scanpy does not validate upstream dependencies at runtime; errors only surface when computation fails
- In production, results from a broken pipeline may be used directly in downstream analysis or publication

---

## PART 3 · Motivation and Design Goals (2 slides)

### Slide 7 — Core Design Principles
1. **Selective Materialization** — load only the data slots needed by the current step
2. **Canonical Naming** — validate on write; non-canonical keys are rejected at the storage boundary
3. **Immutable Provenance** — every write automatically appends a content-fingerprinted log entry
4. **Pipeline State Validation** — intercept missing-dependency errors before a step runs
5. **Bidirectional AnnData Compatibility** — import from h5ad, export back to h5ad at any time
   ```python
   cdb = CellDB.from_h5ad("input.h5ad", "my.cvdb")   # h5ad → cvdb
   cdb.to_h5ad("output.h5ad")                         # cvdb → h5ad, export at any time
   ```

---

### Slide 8 — Design Constraints and Trade-offs
- **Does not replace Scanpy** — wraps it as a storage layer, reusing its mature algorithm implementations
- **Introduces no new file formats** — backed by well-established open-source formats (DuckDB / Zarr / Parquet)
- **Minimal public API**: only three names are exported — `CellDB`, `NameRegistry`, `PipelineStateValidator`
- Current limitation: tool coverage is narrow (PCA / Neighbors / UMAP / Leiden); expansion is ongoing

---

## PART 4 · Implementation Details (9 slides)

### Slide 9 — Layered Architecture
```
tools.py            ← Scanpy analysis entry point (the only public analysis API)
  └─ validator.py   ← Pre-execution pipeline state validation
  └─ registry.py    ← Canonical name table (operation + integration → key)
  └─ celldb.py      ← AnnData-compatible user-facing object
  └─ backend.py     ← Physical storage (DuckDB + Zarr + Parquet + JSON)
  └─ provenance.py  ← Append-only JSONL audit log
```
- Dependency direction is strictly one-way; upper layers cannot be referenced by lower layers
- Users only interact with the top-level API; underlying formats are completely transparent

---

### Slide 10 — Storage Backend Design: Why These Three Formats?
| Slot | Format | Rationale |
|------|--------|-----------|
| obs (cell metadata) | DuckDB | In-place SQL updates, millisecond-range queries, columnar compression |
| var (gene metadata) | Parquet | Columnar compression (RLE + dictionary encoding), cross-language interop |
| X / obsm / obsp | Zarr (one store per key) | Chunked + Blosc compression, sparse storage support, per-key independent reads |
| uns | JSON | Lightweight config, human-readable |
| provenance | JSONL (append-only) | Immutable audit log, append-only guarantees integrity |

- Each obsm key gets its own Zarr store → **no blocking between keys; each can be read independently**
- All formats collectively contribute to compression: the same processed dataset fits in 1.1 GB (vs 6.1 GB as h5ad)

---

### Slide 11 — Technology Background: DuckDB
- **DuckDB** is an embedded columnar OLAP database — no separate server process required
- Key characteristics:
  - Single-file storage (`obs.duckdb`); queryable as soon as the process starts
  - Supports SQL `UPDATE` / `INSERT`, enabling **in-place modification** of arbitrary rows and columns
  - On reads, only the required columns are scanned (columnar storage), making wide-table queries very fast
- Role in CellVault:
  - Cell metadata (obs) is stored in DuckDB
  - Updating cluster labels for 5,000 cells → one SQL `UPDATE`, no need to load the X matrix

---

### Slide 12 — Technology Background: Parquet
- **Apache Parquet** is a columnar file format designed for analytical workloads
- Key characteristics:
  - High compression ratio (RLE + dictionary encoding), far smaller than CSV
  - Columnar layout: when only a few columns are needed, I/O scales with column width, not row width
  - Cross-language support: natively readable by Python, R, Spark, and DuckDB
- Role in CellVault:
  - Gene metadata (var) is stored in Parquet
  - Gene lists are typically read-only; Parquet's compression and efficient reads are a natural fit

---

### Slide 13 — Technology Background: Zarr
- **Zarr** is a chunked N-dimensional array format designed for scientific computing
- Key characteristics:
  - Arrays are split into independent chunk files
  - **On-demand reads**: only the required chunks are fetched; the full array is never loaded
  - Native support for sparse formats (COO / CSR)
  - Works on both local storage and cloud storage (S3, GCS)
- Role in CellVault:
  - The X matrix and each obsm key (e.g., X_pca, X_umap) are stored in separate Zarr stores
  - The Leiden step reads only obsp (the graph Zarr store), **completely skipping X**

---

### Slide 14 — Selective Materialization
- **Core idea**: adopt a database perspective — not "load the entire AnnData" but "declare which slots you need"
- `cdb.to_anndata(slots={"X", "obs", "var"})` — PCA needs only these three
- `cdb.to_anndata(slots={"obs", "var", "obsp", "uns"}, obsp_keys=["connectivities"])` — Leiden skips X and obsm entirely
- Implementation: `to_anndata` decides whether to trigger each backend's I/O based on the slot set
- Effect: avoids loading high-dimensional matrices unnecessarily; this is the direct reason for the 5.7× Leiden speedup

---

### Slide 15 — Canonical Naming System (NameRegistry)
- Static lookup table: `(operation, integration) → canonical_key`
  - `pca` → `"X_pca"`
  - `umap + harmony` → `"X_umap_harmony"`
- Write interception: `_ObsmAccessor.__setitem__` calls `NameRegistry.is_canonical(key)` on every assignment
- Users can register custom keys: `reg.register("custom", None, "X_custom")`
- Effect: non-canonical keys are rejected at write time; naming chaos is eliminated at the source

---

### Slide 16 — Pipeline State Validator (PipelineStateValidator)
- Declarative precondition table:
  - `neighbors` requires `X_pca` to exist in obsm
  - `umap` requires `connectivities` to exist in obsp
  - `leiden` requires `connectivities` to exist in obsp
- `validate(operation, state)` is called automatically before every tool invocation
- State snapshot is obtained via `cdb.get_state()` (reads DuckDB + scans Zarr directories) — pure I/O, no array data loaded
- Violations raise `CellVaultStateError` carrying `operation`, `missing`, and `available` fields for clear diagnostics

---

### Slide 17 — Immutable Provenance Log
- Every backend write appends one JSONL record containing:
  - `timestamp`, `operation`, `target` (slot), `key`, `params`
  - `old_hash` / `new_hash` (content fingerprints)
- Hashing strategy (naïve MD5 over a large sparse matrix is not viable):
  - Dense arrays / DataFrames → truncated MD5
  - Sparse matrices (CSR) → hash `.data`, `.indices`, and `.indptr` separately, then combine
    - **Why**: densifying a 357k × 20k sparse matrix requires ~50 GB of memory — infeasible
    - Structural hashing detects data changes while completely avoiding densification
- Query interface: `cdb.provenance.query(operation="pca")`

---

## PART 5 · Benchmark Results (4 slides)

### Slide 18 — Experimental Setup
- **Dataset**: 357,093 cells × 20,522 genes (real integrated dataset)
- **Pipeline task**: from_h5ad → PCA(50) → Neighbors(15) → UMAP → Leiden(1.0) → export h5ad
- **Metadata edit task**: modify two columns for cells in Leiden cluster 2 (14,039 cells) — re-annotation + re-typing
- **Cold-read test**: a 40 GB cache buster clears the OS page cache before each metadata edit, simulating a true cold start
- Baseline: **native Scanpy + h5ad** (same parameters, same machine)

---

### Slide 19 — Pipeline Timing Comparison
| Step | Scanpy (s) | CellVault (s) | Notes |
|------|-----------|---------------|-------|
| Data load / conversion | 1.14 | 7.77 | CellVault includes format-conversion write overhead |
| PCA | 103.4 | 106.6 | Same computation, ≈ parity |
| Neighbors | 25.6 | 25.6 | Same computation, ≈ parity |
| UMAP | 285.0 | 285.3 | Same computation, ≈ parity |
| **Leiden** | **633.7** | **110.9** | **5.7× speedup: X matrix load skipped** |
| Export h5ad | 1.6 | 4.5 | CellVault includes format-conversion read overhead |
| **Total** | **1052.5** | **540.9** | **~1.94× overall** |

- **Key insight**: Scanpy's Leiden step requires loading the full X matrix into dense format first; CellVault skips this entirely

---

### Slide 20 — Cold Metadata Edit Comparison
- Scenario: OS page cache cleared (cache buster), then full cycle: open → edit cluster labels → write back

| Step | Scanpy (cold read) | CellVault (cold read) |
|------|-------------------|----------------------|
| Open / load file | 1.71 s (full h5ad, includes X matrix) | ~0.01 s (open DuckDB connection only) |
| Locate cluster subset | 0.25 s | ~0.03 s (SQL `WHERE` query) |
| Modify metadata | 0.0003 s | ~0.08 s (SQL `UPDATE`) |
| Write back / close | 1.63 s (re-read full h5ad) + 2.66 s (write h5ad) | ~0.002 s (commit DuckDB transaction) |
| **Total** | **~6.30 s** | **< 0.15 s** |

- **Core advantage**: CellVault's metadata edits never load any array data; I/O is proportional only to the obs metadata file size
- Scanpy's h5ad model requires two full reads: once to extract cluster IDs, and again to write the modifications back to the complete file

---

### Slide 21 — Disk Usage Comparison (same dataset, after full pipeline)

| Format | Size | Notes |
|--------|------|-------|
| Raw input `integrated.h5ad` | 5.9 GB | Input file, includes X matrix |
| Scanpy processed `scanpy_processed.h5ad` | 6.1 GB | Includes X + PCA + UMAP + Leiden + graph matrices |
| **CellVault `integrated.cvdb`** | **1.1 GB** | **Same content, per-slot compressed storage** |

- **~5.5× smaller** — the most tangible storage efficiency gain
- Why:
  - Zarr independently chunks and Blosc-compresses each array key (highly effective for sparse structures)
  - DuckDB columnar compression for obs metadata (dictionary encoding for deduplication)
  - Sparse X matrix stored in CSR sparse Zarr format — never densified
- **Zero interoperability loss**: `cdb.to_h5ad("output.h5ad")` exports back to standard h5ad at any time

---

## PART 6 · Future Directions (3 slides)

### Slide 22 — Near-Term Roadmap
- **Expand tool coverage**: Harmony / Scanorama integration, doublet detection, differential expression analysis
- **Lazy X loading**: implement Zarr deferred reads for X in `to_anndata`, rather than eagerly densifying
- **Incremental obs updates**: support SQL `UPDATE`-level granularity instead of replacing the entire DataFrame
- **CLI tools**: `cvdb inspect`, `cvdb provenance show`, and other interactive utilities

---

### Slide 23 — Long-Term Vision: GPU Acceleration with rapids-singlecell
- **rapids-singlecell** is a GPU-accelerated single-cell analysis library built on NVIDIA RAPIDS (cuML)
  - PCA / Neighbors / UMAP / Leiden all have GPU implementations, roughly 100× faster than CPU Scanpy
- **CellVault + rapids-singlecell integration concept**:
  - CellVault owns the storage layer: load on demand, pass only the required data to the GPU
  - rapids-singlecell owns the compute layer: replaces Scanpy algorithm kernels with GPU-accelerated equivalents
  - Clean separation of concerns; neither component intrudes on the other
- Potential outcome: million-cell pipelines compressed from hours to minutes, while retaining CellVault's provenance and validation guarantees

---

### Slide 24 — Summary
- **One sentence**: CellVault is Scanpy's storage companion — it doesn't replace Scanpy, it makes Scanpy faster, more trustworthy, and more reproducible
- **Three core contributions**:
  1. **Selective materialization + per-slot compressed storage** → database-inspired on-demand reads break the full-load memory wall; disk footprint reduced 5.5×; Leiden 5.7× faster; metadata cold edit from 6.3 s → < 0.15 s; full round-trip export to h5ad at any time
  2. **Canonical naming + pipeline validation** → errors are intercepted at write time and before execution, not discovered after computation fails
  3. **Automatic provenance** → every write leaves an immutable, content-fingerprinted log entry, making the analysis process auditable and trustworthy
- **Open source**: `pip install -e ".[scanpy]"`
- Q & A

---
