# CellVault Project Guide

## Project Overview

CellVault is a reproducibility-oriented data system for single-cell analysis. It provides an AnnData-compatible interface with:

- **DuckDB backend** for obs/var metadata (SQL queryable)
- **Zarr backend** for array data (X, obsm, obsp, varm, varp, layers)
- **Built-in provenance tracking** - automatic logging of all operations
- **Canonical naming system** - consistent naming across analyses
- **Pipeline state validation** - ensures operations run in correct order

### Architecture

```
CellDB (main API)
├─ DuckDBZarrBackend - storage layer (26 methods)
├─ ProvenanceLogger - operation tracking
├─ NameRegistry - canonical naming
└─ PipelineStateValidator - state checking
```

### Key Features

1. **Reproducibility**: Every operation is logged with timestamps and data hashes
2. **Performance**: DuckDB for fast metadata queries, Zarr for efficient array storage
3. **Compatibility**: Full AnnData round-trip support
4. **Validation**: Prevents invalid operation sequences (e.g., clustering before PCA)

## Development Environment

### Setup with uv (Recommended)

```bash
# Create virtual environment
uv venv

# Install with all dependencies
uv pip install -e ".[dev,scanpy]"

# For LSP/type checking support
uv pip install pyright
```

### Environment Details

- **Python**: >=3.10
- **Virtual environment**: `.venv/` (created with uv)
- **LSP**: Pyright 1.1.408 configured via `pyrightconfig.json`
- **Package manager**: uv (faster than pip)

### Type Checking

```bash
# Run type checking
.venv/bin/pyright src/cellvault

# With statistics
.venv/bin/pyright --stats src/cellvault
```

Current type health: 72/100 (28 errors, mainly in backend.py)

## Code Quality Status

### File Health

- ✅ `validator.py` - 100% (0 errors)
- ✅ `provenance.py` - 95% (1 error)
- ⚠️ `tools.py` - 90% (2 errors)
- ⚠️ `celldb.py` - 80% (3 errors)
- 🔴 `backend.py` - 20% (22 errors)

### Known Issues

1. **backend.py**: Optional[Connection] access without guards (12 occurrences)
2. **backend.py**: zarr 3.x API migration needed (7 occurrences)
3. **celldb.py/tools.py**: DataFrame/ndarray type mismatches (5 occurrences)

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=cellvault
```

## Common Tasks

### Creating a CellDB

```python
from cellvault import CellDB
import anndata

# From AnnData
adata = anndata.read_h5ad("data.h5ad")
cdb = CellDB.from_anndata(adata, "output.cvdb")

# From scratch
cdb = CellDB.create("new.cvdb", n_obs=1000, n_vars=2000)
```

### Running Analysis

```python
from cellvault import tools

# Standard workflow
tools.pca(cdb)
tools.neighbors(cdb)
tools.umap(cdb)
tools.leiden(cdb)

# All operations are validated and logged
```

### Checking Provenance

```python
# View operation history
log = cdb.provenance.read_log()
print(log)

# Query specific operations
pca_ops = cdb.provenance.query(operation="pca")
```

## Dependencies

### Core
- anndata >=0.10
- duckdb >=1.0
- zarr >=3.0
- numpy >=1.24
- pandas >=2.0
- pyarrow >=14.0
- scipy >=1.10

### Optional
- scanpy >=1.9 (for analysis tools)

### Development
- pytest >=7.0
- pytest-cov
- pyright (for type checking)

## Notes for AI Assistants

- Always activate `.venv` before running commands
- Use `uv pip` for package management (faster than pip)
- Type checking is configured but has known issues in backend.py
- All file operations should go through the CellDB API, not direct file access
- Provenance logging is automatic - don't bypass it
