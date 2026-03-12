"""Tests for provenance logging and data hashing."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellvault.provenance import ProvenanceLogger, _hash_data


# ── _hash_data ──────────────────────────────────────────────────────


class TestHashData:
    def test_none(self):
        assert _hash_data(None) == "null"

    def test_dense_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        h = _hash_data(arr)
        assert isinstance(h, str) and len(h) == 12

    def test_dense_deterministic(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert _hash_data(arr) == _hash_data(arr)

    def test_dense_different(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert _hash_data(a) != _hash_data(b)

    def test_sparse_no_densification(self):
        """Sparse hashing must NOT call .toarray() — the old code allocated 12GB."""
        mat = sparse.random(10000, 5000, density=0.001, format="csr")
        # If this calls .toarray(), it would allocate ~400MB even at this size.
        # We just verify it completes fast and returns a valid hash.
        h = _hash_data(mat)
        assert isinstance(h, str) and len(h) == 12

    def test_sparse_deterministic(self):
        mat = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]]))
        assert _hash_data(mat) == _hash_data(mat)

    def test_sparse_different(self):
        a = sparse.csr_matrix(np.array([[1, 0], [0, 1]]))
        b = sparse.csr_matrix(np.array([[0, 1], [1, 0]]))
        assert _hash_data(a) != _hash_data(b)

    def test_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        h = _hash_data(df)
        assert isinstance(h, str) and len(h) == 12

    def test_dataframe_deterministic(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert _hash_data(df) == _hash_data(df)

    def test_string(self):
        h = _hash_data("hello world")
        assert isinstance(h, str) and len(h) == 12


# ── ProvenanceLogger ────────────────────────────────────────────────


class TestProvenanceLogger:
    def test_log_and_read(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        logger = ProvenanceLogger(log_path)

        logger.log("write_obs", "obs", key="batch", new_hash="abc123")
        entries = logger.read_log()

        assert len(entries) == 1
        assert entries[0]["operation"] == "write_obs"
        assert entries[0]["target"] == "obs"
        assert entries[0]["key"] == "batch"
        assert entries[0]["new_hash"] == "abc123"
        assert entries[0]["actor"] == "cellvault"
        assert "timestamp" in entries[0]
        assert "timestamp_iso" in entries[0]

    def test_append_only(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        logger = ProvenanceLogger(log_path)

        logger.log("op1", "t1")
        logger.log("op2", "t2")
        logger.log("op3", "t3")

        assert len(logger.read_log()) == 3

    def test_query_by_key(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        logger = ProvenanceLogger(log_path)

        logger.log("write_obsm", "obsm", key="X_pca")
        logger.log("write_obsm", "obsm", key="X_umap")
        logger.log("write_obs", "obs", key="leiden")

        results = logger.query(key="X_pca")
        assert len(results) == 1
        assert results[0]["key"] == "X_pca"

    def test_query_by_operation(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        logger = ProvenanceLogger(log_path)

        logger.log("write_obs", "obs")
        logger.log("write_obsm", "obsm")
        logger.log("write_obs", "obs")

        results = logger.query(operation="write_obs")
        assert len(results) == 2

    def test_empty_log(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        logger = ProvenanceLogger(log_path)
        assert logger.read_log() == []

    def test_custom_actor(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        logger = ProvenanceLogger(log_path)
        logger.log("op", "target", actor="user_alice")
        assert logger.read_log()[0]["actor"] == "user_alice"

    def test_params_logged(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        logger = ProvenanceLogger(log_path)
        logger.log("pca", "obsm", params={"n_comps": 50})
        assert logger.read_log()[0]["params"] == {"n_comps": 50}
