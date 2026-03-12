"""Scanpy wrappers with NameRegistry and StateValidator integration.

Performance-optimized: each tool loads ONLY the data it needs via
selective to_anndata(slots=...), avoiding full database materialization.

Before (4-step pipeline at 1M cells): 4x full materialization = ~32GB I/O
After: each step loads only its required slots = ~70-90% I/O reduction
"""

import scanpy as sc
import numpy as np
from typing import Optional

from .registry import NameRegistry
from .validator import PipelineStateValidator, CellVaultStateError
from .celldb import CellDB


def pca(cdb: CellDB, n_comps: int = 50, integration: Optional[str] = None, **kwargs):
    """Run PCA with canonical naming and state validation.

    Loads: X, obs, var (skips obsm, obsp, uns)
    """
    state = cdb.get_state()
    PipelineStateValidator.validate("pca", state)

    # Only load X + obs + var — skip all embeddings, graphs, and uns
    adata = cdb.to_anndata(slots={"X", "obs", "var"})
    sc.tl.pca(adata, n_comps=n_comps, **kwargs)

    key = NameRegistry.get("pca", integration)
    cdb.obsm[key] = adata.obsm["X_pca"]

    # Store PCA variance info in uns
    uns = cdb.uns
    uns[f"{key}_variance_ratio"] = adata.uns.get("pca", {}).get("variance_ratio", np.array([])).tolist()
    cdb.uns = uns

    cdb.provenance.log("pca", "obsm", key=key, params={"n_comps": n_comps, "integration": integration})
    return key


def neighbors(cdb: CellDB, n_neighbors: int = 15, integration: Optional[str] = None,
              use_rep: Optional[str] = None, **kwargs):
    """Run neighbors with canonical naming and state validation.

    Loads: obs + one obsm key (PCA embedding) + uns
    Skips: X (the biggest data), all other obsm/obsp
    """
    state = cdb.get_state()
    PipelineStateValidator.validate("neighbors", state)

    if use_rep is None:
        pca_key = NameRegistry.get("pca", integration)
        if pca_key in state["obsm_keys"]:
            use_rep = pca_key
        else:
            use_rep = "X_pca"

    # Only load the specific obsm key needed — skip X entirely
    adata = cdb.to_anndata(
        slots={"obs", "var", "obsm"},
        obsm_keys=[use_rep],
    )

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, **kwargs)

    conn_key = NameRegistry.get("neighbors", integration)
    connectivities_key = NameRegistry.get("connectivities", integration)
    distances_key = NameRegistry.get("distances", integration)
    # Store connectivities and distances with namespaced keys
    cdb.obsp[connectivities_key] = adata.obsp["connectivities"]
    cdb.obsp[distances_key] = adata.obsp["distances"]

    # Store neighbor params in uns (include 'method' for Scanpy compat)
    uns = cdb.uns
    uns["neighbors"] = {"connectivities_key": connectivities_key, "distances_key": distances_key,
                        "params": {"n_neighbors": n_neighbors, "use_rep": use_rep, "method": "umap"}}
    cdb.uns = uns

    cdb.provenance.log("neighbors", "obsp", key=conn_key,
                       params={"n_neighbors": n_neighbors, "use_rep": use_rep})
    return conn_key


def umap(cdb: CellDB, integration: Optional[str] = None, **kwargs):
    """Run UMAP with canonical naming and state validation.

    Loads: obs + var + obsp (connectivities/distances) + uns + use_rep obsm key
    Skips: X (the biggest slot)
    Note: scanpy's UMAP internally needs the representation (X_pca) for
          initialization via _choose_representation, so we must load it.
    """
    state = cdb.get_state()
    PipelineStateValidator.validate("umap", state)

    # Get neighbor keys from uns to load only required obsp
    uns = cdb.uns
    neighbor_info = uns.get("neighbors", {})
    conn_key = neighbor_info.get("connectivities_key", "connectivities")
    dist_key = neighbor_info.get("distances_key", "distances")
    # scanpy UMAP needs the representation used for neighbors (e.g. X_pca)
    use_rep = neighbor_info.get("params", {}).get("use_rep", "X_pca")
    needed_obsm = [use_rep] if use_rep in state["obsm_keys"] else []

    # Load obsp graphs + uns + the representation obsm key — skip X
    adata = cdb.to_anndata(
        slots={"obs", "var", "obsm", "obsp", "uns"},
        obsm_keys=needed_obsm,
        obsp_keys=[conn_key, dist_key],
    )
    sc.tl.umap(adata, **kwargs)

    key = NameRegistry.get("umap", integration)
    cdb.obsm[key] = adata.obsm["X_umap"]

    cdb.provenance.log("umap", "obsm", key=key, params={"integration": integration})
    return key


def leiden(cdb: CellDB, resolution: float = 1.0, integration: Optional[str] = None, **kwargs):
    """Run Leiden clustering with canonical naming and state validation.

    Loads: obs + obsp (connectivities/distances) + uns (neighbor params)
    Skips: X, var, all obsm
    Writes: single column via add_obs_column (not full obs rewrite)
    """
    state = cdb.get_state()
    PipelineStateValidator.validate("leiden", state)

    # Get neighbor keys from uns to load only required obsp
    uns = cdb.uns
    neighbor_info = uns.get("neighbors", {})
    conn_key = neighbor_info.get("connectivities_key", "connectivities")
    dist_key = neighbor_info.get("distances_key", "distances")

    # Only load obsp graphs + uns — skip X and all obsm
    adata = cdb.to_anndata(
        slots={"obs", "var", "obsp", "uns"},
        obsp_keys=[conn_key, dist_key],
    )
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

    # Use add_obs_column for single-column write instead of full obs rewrite.
    # This avoids: read full obs → add column → DROP + CREATE entire table.
    values = adata.obs["leiden"].values
    if key in cdb._backend.obs_columns:
        # Column exists: update in-place
        obs = cdb.obs
        obs[key] = values
        cdb.obs = obs
    else:
        # New column: efficient ALTER TABLE + UPDATE
        cdb._backend.add_obs_column(key, values)

    cdb.provenance.log("leiden", "obs", key=key, params={"resolution": resolution})
    return key
