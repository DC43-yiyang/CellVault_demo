"""PipelineStateValidator: Precondition checks before analysis operations."""

import re
from typing import Optional

from ._debug import logger


# Precondition rules: operation -> required state
_PRECONDITIONS = {
    "pca": {
        "X": True,  # expression matrix must exist
    },
    "neighbors": {
        "obsm": [r"X_pca.*"],  # any PCA embedding required
    },
    "umap": {
        "obsp": [r"(connectivities|distances)"],  # neighbors graph required
    },
    "tsne": {
        "obsm": [r"X_pca.*"],
    },
    "leiden": {
        "obsp": [r"(connectivities|distances)"],
    },
    "louvain": {
        "obsp": [r"(connectivities|distances)"],
    },
    "rank_genes_groups": {
        "obs_columns": [r"(leiden.*|louvain.*)"],  # clustering result needed
    },
    "diffmap": {
        "obsp": [r"(connectivities|distances)"],
    },
}


class CellVaultStateError(Exception):
    """Raised when pipeline state preconditions are not met."""

    def __init__(self, operation: str, missing: str, available: list[str]):
        self.operation = operation
        self.missing = missing
        self.available = available
        msg = (
            f"Cannot run '{operation}': requires {missing}. "
            f"Available: {available}"
        )
        super().__init__(msg)


class PipelineStateValidator:
    """Validates that required preconditions are met before analysis operations."""

    @staticmethod
    def validate(operation: str, state: dict, groupby: Optional[str] = None):
        """Validate preconditions for an operation.

        Args:
            operation: Name of the operation (e.g., 'pca', 'umap')
            state: Current state dict with keys: 'X_exists', 'obsm_keys',
                   'obsp_keys', 'obs_columns', 'uns_keys'
            groupby: For rank_genes_groups, the column to group by
        """
        rules = _PRECONDITIONS.get(operation.lower())
        if rules is None:
            logger.debug("validate(%s): no preconditions registered", operation)
            return  # no preconditions registered

        if rules.get("X") and not state.get("X_exists", False):
            raise CellVaultStateError(operation, "expression matrix (X)", [])

        for slot in ("obsm", "obsp"):
            patterns = rules.get(slot, [])
            for pattern in patterns:
                available = state.get(f"{slot}_keys", [])
                if not any(re.match(pattern, k) for k in available):
                    raise CellVaultStateError(operation, f"{slot} matching '{pattern}'", available)

        col_patterns = rules.get("obs_columns", [])
        for pattern in col_patterns:
            available = state.get("obs_columns", [])
            # If groupby is specified, check it directly
            if groupby and any(re.match(pattern, groupby) for _ in [1]):
                continue
            if not any(re.match(pattern, c) for c in available):
                raise CellVaultStateError(operation, f"obs column matching '{pattern}'", available)

        logger.debug("validate(%s): all preconditions met ✓", operation)

    @staticmethod
    def get_preconditions(operation: str) -> dict:
        """Get precondition rules for an operation."""
        return _PRECONDITIONS.get(operation.lower(), {})

    @staticmethod
    def list_operations() -> list[str]:
        """List all operations with registered preconditions."""
        return list(_PRECONDITIONS.keys())
