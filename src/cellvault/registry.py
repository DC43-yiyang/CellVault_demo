"""NameRegistry: Canonical naming resolution for single-cell analysis outputs."""

from typing import Optional


# Canonical name table: (operation, integration) -> canonical_key
_CANONICAL_TABLE = {
    # PCA
    ("pca", None): "X_pca",
    ("pca", "harmony"): "X_pca_harmony",
    ("pca", "scanorama"): "X_pca_scanorama",
    ("pca", "bbknn"): "X_pca_bbknn",
    ("pca", "scvi"): "X_pca_scvi",
    # Neighbors
    ("neighbors", None): "neighbors",
    ("neighbors", "harmony"): "neighbors_harmony",
    ("neighbors", "scanorama"): "neighbors_scanorama",
    # Connectivities (output of neighbors)
    ("connectivities", None): "connectivities",
    ("connectivities", "harmony"): "connectivities_harmony",
    ("connectivities", "scanorama"): "connectivities_scanorama",
    # Distances (output of neighbors)
    ("distances", None): "distances",
    ("distances", "harmony"): "distances_harmony",
    ("distances", "scanorama"): "distances_scanorama",
    # UMAP
    ("umap", None): "X_umap",
    ("umap", "harmony"): "X_umap_harmony",
    ("umap", "scanorama"): "X_umap_scanorama",
    # TSNE
    ("tsne", None): "X_tsne",
    # Clustering
    ("leiden", None): "leiden",
    ("leiden", "harmony"): "leiden_harmony",
    ("louvain", None): "louvain",
    ("louvain", "harmony"): "louvain_harmony",
    # DE
    ("rank_genes_groups", None): "rank_genes_groups",
    # Diffmap
    ("diffmap", None): "X_diffmap",
    # Draw graph
    ("draw_graph", None): "X_draw_graph_fa",
}

# Reverse lookup: canonical_key -> (operation, integration)
_REVERSE_TABLE = {v: k for k, v in _CANONICAL_TABLE.items()}


class NameRegistry:
    """Resolves analysis operations to canonical key names.

    Prevents naming drift by requiring all outputs to go through
    deterministic name resolution.
    """

    def __init__(self):
        self._custom: dict[tuple[str, Optional[str]], str] = {}

    @staticmethod
    def get(operation: str, integration: Optional[str] = None) -> str:
        """Resolve an operation + integration to its canonical key name."""
        key = (operation.lower(), integration.lower() if integration else None)
        result = _CANONICAL_TABLE.get(key)
        if result is None:
            raise KeyError(
                f"No canonical name registered for operation='{operation}', "
                f"integration='{integration}'. "
                f"Available operations: {sorted(set(k[0] for k in _CANONICAL_TABLE))}"
            )
        return result

    @staticmethod
    def lookup(canonical_key: str) -> tuple[str, Optional[str]]:
        """Reverse lookup: from canonical key to (operation, integration)."""
        result = _REVERSE_TABLE.get(canonical_key)
        if result is None:
            raise KeyError(f"Key '{canonical_key}' is not a registered canonical name.")
        return result

    def register(self, operation: str, integration: Optional[str], canonical_key: str):
        """Register a custom canonical name mapping."""
        key = (operation.lower(), integration.lower() if integration else None)
        self._custom[key] = canonical_key
        _CANONICAL_TABLE[key] = canonical_key
        _REVERSE_TABLE[canonical_key] = key

    @staticmethod
    def list_all() -> dict[tuple[str, Optional[str]], str]:
        """List all registered canonical names."""
        return dict(_CANONICAL_TABLE)

    @staticmethod
    def is_canonical(key: str) -> bool:
        """Check if a key is a registered canonical name."""
        return key in _REVERSE_TABLE
