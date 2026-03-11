"""CellVault: Reproducibility-oriented data system for single-cell analysis."""

__version__ = "0.1.0"

from .registry import NameRegistry
from .validator import PipelineStateValidator, CellVaultStateError
from .celldb import CellDB
from .backend import DuckDBZarrBackend
from .provenance import ProvenanceLogger
from ._debug import set_debug, is_debug

__all__ = [
    "NameRegistry",
    "PipelineStateValidator",
    "CellVaultStateError",
    "CellDB",
    "DuckDBZarrBackend",
    "ProvenanceLogger",
    "set_debug",
    "is_debug",
    "__version__",
]
