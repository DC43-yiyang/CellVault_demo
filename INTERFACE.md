# CellVault Interface Guide

This document describes the current user-facing interfaces in `cellvault` and how to use them in practice.

## Public API Surface

```python
from cellvault import CellDB, NameRegistry, PipelineStateValidator
from cellvault.validator import CellVaultStateError
from cellvault.tools import pca, neighbors, umap, leiden
```

Core idea:
- `CellDB` is the main data container (AnnData-like interface).
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
- `CellDB.from_anndata(..., cvdb_path)` overwrites `cvdb_path` if it already exists.

### 1.3 AnnData-like Slots

`CellDB` exposes these properties:
- `cdb.obs` (`pandas.DataFrame`)
- `cdb.var` (`pandas.DataFrame`)
- `cdb.X` (`numpy.ndarray` or `scipy.sparse` matrix)
- `cdb.obsm` (dict-like accessor)
- `cdb.obsp` (dict-like accessor)
- `cdb.uns` (`dict`)
- `cdb.n_obs`, `cdb.n_vars`, `cdb.shape`

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
```

### 1.5 Selective Materialization (`to_anndata`)

Use `slots` to avoid loading everything:

```python
# Only what PCA needs
adata_pca = cdb.to_anndata(slots={"X", "obs", "var"})

# Only neighbors graph + metadata
adata_graph = cdb.to_anndata(
    slots={"obs", "var", "obsp", "uns"},
    obsp_keys=["connectivities", "distances"],
)
```

### 1.6 Resource Management

```python
cdb.close()
```

`CellDB` holds a DuckDB connection; explicit `close()` is recommended.

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
- Storage is split by slot: DuckDB (`obs`), Parquet (`var`), Zarr (`X`/`obsm`/`obsp`), JSON (`uns`), JSONL (`provenance`).
