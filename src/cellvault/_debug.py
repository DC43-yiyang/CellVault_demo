"""CellVault debug mode: configurable logging for all operations."""

import logging
import os

logger = logging.getLogger("cellvault")

_DEBUG = False


def set_debug(enabled: bool = True):
    """Enable or disable debug mode globally."""
    global _DEBUG
    _DEBUG = enabled
    level = logging.DEBUG if enabled else logging.WARNING
    logger.setLevel(level)
    if enabled and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return _DEBUG


# Initialize from environment variable
if os.environ.get("CELLVAULT_DEBUG", "").lower() in ("1", "true", "yes"):
    set_debug(True)
