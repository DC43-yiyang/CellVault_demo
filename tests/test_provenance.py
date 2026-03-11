"""Tests for ProvenanceLogger."""

import json
import os
import tempfile

import pytest

from cellvault.provenance import ProvenanceLogger, _hash_data


class TestHashData:
    def test_hash_none(self):
        assert _hash_data(None) == "null"

    def test_hash_ndarray(self):
        import numpy as np
        arr = np.array([1, 2, 3])
        h = _hash_data(arr)
        assert isinstance(h, str)
        assert len(h) == 12

    def test_hash_sparse(self):
        from scipy import sparse
        import numpy as np
        mat = sparse.csr_matrix(np.eye(3))
        h = _hash_data(mat)
        assert isinstance(h, str)
        assert len(h) == 12

    def test_hash_string(self):
        h = _hash_data("hello")
        assert isinstance(h, str)
        assert len(h) == 12

    def test_hash_deterministic(self):
        import numpy as np
        arr = np.array([1.0, 2.0, 3.0])
        assert _hash_data(arr) == _hash_data(arr)


class TestProvenanceLogger:
    def test_log_creates_file(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        prov = ProvenanceLogger(log_path)
        prov.log("test_op", "test_target")
        assert os.path.exists(log_path)

    def test_log_entry_structure(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        prov = ProvenanceLogger(log_path)
        prov.log("write_obs", "obs", key="col1", params={"n_rows": 10})

        entries = prov.read_log()
        assert len(entries) == 1
        entry = entries[0]
        assert entry["operation"] == "write_obs"
        assert entry["target"] == "obs"
        assert entry["key"] == "col1"
        assert entry["params"] == {"n_rows": 10}
        assert "timestamp" in entry
        assert "timestamp_iso" in entry

    def test_log_appends(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        prov = ProvenanceLogger(log_path)
        prov.log("op1", "t1")
        prov.log("op2", "t2")
        entries = prov.read_log()
        assert len(entries) == 2

    def test_read_log_empty(self, tmp_path):
        log_path = str(tmp_path / "nonexistent.jsonl")
        prov = ProvenanceLogger(log_path)
        assert prov.read_log() == []

    def test_query_by_key(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        prov = ProvenanceLogger(log_path)
        prov.log("op1", "t1", key="k1")
        prov.log("op2", "t2", key="k2")
        prov.log("op3", "t3", key="k1")

        results = prov.query(key="k1")
        assert len(results) == 2
        assert all(e["key"] == "k1" for e in results)

    def test_query_by_operation(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        prov = ProvenanceLogger(log_path)
        prov.log("write_obs", "obs")
        prov.log("write_X", "X")
        prov.log("write_obs", "obs")

        results = prov.query(operation="write_obs")
        assert len(results) == 2

    def test_query_combined(self, tmp_path):
        log_path = str(tmp_path / "prov.jsonl")
        prov = ProvenanceLogger(log_path)
        prov.log("write_obs", "obs", key="k1")
        prov.log("write_obs", "obs", key="k2")
        prov.log("write_X", "X", key="k1")

        results = prov.query(key="k1", operation="write_obs")
        assert len(results) == 1
