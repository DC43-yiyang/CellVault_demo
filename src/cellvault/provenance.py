"""Provenance Logger: Append-only audit trail for all data mutations."""

import json
import time
import hashlib
import os
from typing import Any, Optional

from ._debug import logger


def _hash_data(data) -> str:
    """Compute a fast hash of data for change detection."""
    import numpy as np

    if data is None:
        return "null"
    if isinstance(data, np.ndarray):
        return hashlib.md5(data.tobytes()[:4096]).hexdigest()[:12]
    if hasattr(data, "toarray"):  # sparse
        return hashlib.md5(data.toarray().tobytes()[:4096]).hexdigest()[:12]
    return hashlib.md5(str(data).encode()[:4096]).hexdigest()[:12]


class ProvenanceLogger:
    """Append-only provenance log for CellVault operations."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(
        self,
        operation: str,
        target: str,
        key: Optional[str] = None,
        params: Optional[dict] = None,
        old_hash: Optional[str] = None,
        new_hash: Optional[str] = None,
        actor: str = "cellvault",
    ):
        """Log a provenance entry."""
        entry = {
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "actor": actor,
            "operation": operation,
            "target": target,
            "key": key,
            "old_hash": old_hash,
            "new_hash": new_hash,
            "params": params or {},
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.debug(
            "provenance: %s %s key=%s old=%s new=%s params=%s",
            operation, target, key, old_hash, new_hash, params,
        )

    def read_log(self) -> list[dict]:
        """Read all provenance entries."""
        if not os.path.exists(self.log_path):
            return []
        entries = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def query(self, key: Optional[str] = None, operation: Optional[str] = None) -> list[dict]:
        """Query provenance entries by key or operation."""
        entries = self.read_log()
        if key:
            entries = [e for e in entries if e.get("key") == key]
        if operation:
            entries = [e for e in entries if e.get("operation") == operation]
        return entries
