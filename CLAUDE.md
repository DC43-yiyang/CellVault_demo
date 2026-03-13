# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with scanpy and dev deps)
pip install -e ".[scanpy,dev]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_celldb.py

# Run a single test
pytest tests/test_celldb.py::test_function_name

# Run with coverage
pytest --cov=cellvault tests/
```

No linter is configured — none should be assumed.

## Architecture

CellVault is a reproducibility-oriented storage layer for single-cell analysis. It provides an `AnnData`-compatible interface (`CellDB`) backed by **DuckDB** (obs metadata) and **Zarr** (array data), with built-in provenance, canonical naming, and pipeline validation.

### Layered Design

```
tools.py          ← scanpy wrappers; the only public analysis API
  └─ validator.py ← declarative precondition checks before any tool runs
  └─ registry.py  ← static (operation, integration) → canonical key lookup
  └─ celldb.py    ← AnnData-compatible user-facing object
       └─ backend.py   ← physical storage (DuckDB + Zarr + Parquet + JSON)
            └─ provenance.py ← append-only JSONL audit trail
```

**Public surface** (`__init__.py`): only three names are exported — `CellDB`, `NameRegistry`, `PipelineStateValidator`.

### Key Design Decisions

1. **Selective materialization** — `CellDB.to_anndata(slots=...)` lets tools load only what they need (e.g., `leiden` skips `X` entirely, loading only the graph obsm key). This is the primary performance mechanism for large datasets.

2. **Canonical naming enforced at the write boundary** — `_ObsmAccessor` and `_ObspAccessor` call `NameRegistry.is_canonical()` on every write, preventing ad-hoc key names from entering the store.

3. **Immutable provenance** — `ProvenanceLogger` appends to `provenance.jsonl` with content hashes (truncated MD5 for dense arrays/DataFrames; structural hash via `.data`/`.indices`/`.indptr` for sparse matrices to avoid densification).

### Storage Layout (on disk, inside a `.cvdb` directory)

| Slot | Format |
|------|--------|
| `obs` | DuckDB (`obs.duckdb`) |
| `var` | Parquet |
| `X`, `obsm[key]`, `obsp[key]` | Zarr stores (one per key) |
| `uns` | JSON |
| registry state | JSON |
| provenance | JSONL |

### Adding a New Tool

Follow the pattern in `tools.py`: (1) call `PipelineStateValidator.validate()`, (2) call `cdb.to_anndata(slots=...)` with only the needed slots, (3) run the scanpy function, (4) write results back using canonical keys from `NameRegistry`, (5) provenance is logged automatically by the backend.

### Test Fixtures (`tests/conftest.py`)

| Fixture | Description |
|---------|-------------|
| `small_adata` | 100 cells × 50 genes, sparse CSR, categorical `batch` |
| `celldb_small` | `CellDB` loaded from `small_adata` |
| `celldb_with_pipeline` | `CellDB` after full PCA → neighbors → UMAP → Leiden run |
