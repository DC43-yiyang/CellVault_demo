# CellVault

Reproducibility-oriented data system for single-cell analysis.

CellVault provides an AnnData-compatible interface backed by DuckDB (for obs metadata) and Zarr (for array data), with built-in provenance tracking, canonical naming, and pipeline state validation.

## Installation

```bash
pip install -e ".[scanpy,dev]"
```

## Quick Start

```python
import anndata
import numpy as np
from cellvault import CellDB

# Create from AnnData
adata = anndata.AnnData(X=np.random.rand(100, 50))
cdb = CellDB.from_anndata(adata, "my_data.cvdb")

# Round-trip back to AnnData
adata2 = cdb.to_anndata()

# Enable debug mode
import cellvault
cellvault.set_debug(True)

cdb.close()
```

## Debug Mode

```python
import cellvault
cellvault.set_debug(True)  # or set CELLVAULT_DEBUG=1 env var
```

## License

MIT
