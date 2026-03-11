"""Scanpy wrappers with NameRegistry and StateValidator integration."""

import time
import numpy as np
from typing import Optional

from .registry import NameRegistry
from .validator import PipelineStateValidator, CellVaultStateError
from .celldb import CellDB
from ._debug import logger


def _import_scanpy():
    """Lazy import of scanpy."""
    try:
        import scanpy as sc
        return sc
    except ImportError:
        raise ImportError(
            "scanpy is required for cellvault.tools. "
            "Install it with: pip install cellvault[scanpy]"
        )


def pca(cdb: CellDB, n_comps: int = 50, integration: Optional[str] = None, **kwargs):
    """Run PCA with canonical naming and state validation."""
    sc = _import_scanpy()
    t0 = time.perf_counter()

    state = cdb.get_state()
    PipelineStateValidator.validate("pca", state)

    adata = cdb.to_anndata()
    sc.tl.pca(adata, n_comps=n_comps, **kwargs)

    key = NameRegistry.get("pca", integration)
    cdb.obsm[key] = adata.obsm["X_pca"]

    # Store PCA variance info in uns
    uns = cdb.uns
    uns[f"{key}_variance_ratio"] = adata.uns.get("pca", {}).get("variance_ratio", np.array([])).tolist()
    cdb.uns = uns

    cdb.provenance.log("pca", "obsm", key=key, params={"n_comps": n_comps, "integration": integration})
    logger.debug("pca: n_comps=%d, elapsed=%.3fs, output_key=%s", n_comps, time.perf_counter() - t0, key)
    return key


def neighbors(cdb: CellDB, n_neighbors: int = 15, integration: Optional[str] = None,
              use_rep: Optional[str] = None, **kwargs):
    """Run neighbors with canonical naming and state validation."""
    sc = _import_scanpy()
    t0 = time.perf_counter()

    state = cdb.get_state()
    PipelineStateValidator.validate("neighbors", state)

    adata = cdb.to_anndata()

    if use_rep is None:
        pca_key = NameRegistry.get("pca", integration)
        if pca_key in state["obsm_keys"]:
            use_rep = pca_key
        else:
            use_rep = "X_pca"

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, **kwargs)

    conn_key = NameRegistry.get("neighbors", integration)
    # Store connectivities and distances
    cdb.obsp["connectivities"] = adata.obsp["connectivities"]
    cdb.obsp["distances"] = adata.obsp["distances"]

    # Store neighbor params in uns (include 'method' for Scanpy compat)
    uns = cdb.uns
    uns["neighbors"] = {"connectivities_key": "connectivities", "distances_key": "distances",
                        "params": {"n_neighbors": n_neighbors, "use_rep": use_rep, "method": "umap"}}
    cdb.uns = uns

    cdb.provenance.log("neighbors", "obsp", key=conn_key,
                       params={"n_neighbors": n_neighbors, "use_rep": use_rep})
    logger.debug("neighbors: n_neighbors=%d, use_rep=%s, elapsed=%.3fs", n_neighbors, use_rep, time.perf_counter() - t0)
    return conn_key


def umap(cdb: CellDB, integration: Optional[str] = None, **kwargs):
    """Run UMAP with canonical naming and state validation."""
    sc = _import_scanpy()
    t0 = time.perf_counter()

    state = cdb.get_state()
    PipelineStateValidator.validate("umap", state)

    adata = cdb.to_anndata()
    # Restore neighbors connectivity
    sc.tl.umap(adata, **kwargs)

    key = NameRegistry.get("umap", integration)
    cdb.obsm[key] = adata.obsm["X_umap"]

    cdb.provenance.log("umap", "obsm", key=key, params={"integration": integration})
    logger.debug("umap: elapsed=%.3fs, output_key=%s", time.perf_counter() - t0, key)
    return key


def leiden(cdb: CellDB, resolution: float = 1.0, integration: Optional[str] = None, **kwargs):
    """Run Leiden clustering with canonical naming and state validation."""
    sc = _import_scanpy()
    t0 = time.perf_counter()

    state = cdb.get_state()
    PipelineStateValidator.validate("leiden", state)

    adata = cdb.to_anndata()
    leiden_kwargs = {"resolution": resolution}
    leiden_kwargs.update(kwargs)
    # scanpy >= 1.10 changed API: try with flavor, fallback without
    for attempt_kwargs in [
        {**leiden_kwargs, "flavor": "igraph"},
        {**leiden_kwargs, "flavor": "igraph", "n_iterations": 2},
        leiden_kwargs,
    ]:
        try:
            sc.tl.leiden(adata, **attempt_kwargs)
            break
        except (TypeError, ValueError):
            continue

    key = NameRegistry.get("leiden", integration)

    # Update obs with clustering result
    obs = cdb.obs
    obs[key] = adata.obs["leiden"].values
    cdb.obs = obs

    cdb.provenance.log("leiden", "obs", key=key, params={"resolution": resolution})
    logger.debug("leiden: resolution=%.2f, elapsed=%.3fs, output_key=%s", resolution, time.perf_counter() - t0, key)
    return key
