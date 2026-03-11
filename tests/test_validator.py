"""Tests for PipelineStateValidator."""

import pytest

from cellvault.validator import PipelineStateValidator, CellVaultStateError


class TestValidatePCA:
    def test_pca_passes_with_X(self):
        state = {"X_exists": True, "obsm_keys": [], "obsp_keys": [], "obs_columns": []}
        PipelineStateValidator.validate("pca", state)  # should not raise

    def test_pca_fails_without_X(self):
        state = {"X_exists": False, "obsm_keys": [], "obsp_keys": [], "obs_columns": []}
        with pytest.raises(CellVaultStateError, match="expression matrix"):
            PipelineStateValidator.validate("pca", state)


class TestValidateNeighbors:
    def test_neighbors_passes_with_pca(self):
        state = {"X_exists": True, "obsm_keys": ["X_pca"], "obsp_keys": [], "obs_columns": []}
        PipelineStateValidator.validate("neighbors", state)

    def test_neighbors_fails_without_pca(self):
        state = {"X_exists": True, "obsm_keys": [], "obsp_keys": [], "obs_columns": []}
        with pytest.raises(CellVaultStateError, match="obsm"):
            PipelineStateValidator.validate("neighbors", state)


class TestValidateUMAP:
    def test_umap_passes_with_neighbors(self):
        state = {"X_exists": True, "obsm_keys": ["X_pca"], "obsp_keys": ["connectivities", "distances"], "obs_columns": []}
        PipelineStateValidator.validate("umap", state)

    def test_umap_fails_without_neighbors(self):
        state = {"X_exists": True, "obsm_keys": ["X_pca"], "obsp_keys": [], "obs_columns": []}
        with pytest.raises(CellVaultStateError):
            PipelineStateValidator.validate("umap", state)


class TestValidateLeiden:
    def test_leiden_passes_with_neighbors(self):
        state = {"X_exists": True, "obsm_keys": [], "obsp_keys": ["connectivities"], "obs_columns": []}
        PipelineStateValidator.validate("leiden", state)

    def test_leiden_fails_without_neighbors(self):
        state = {"X_exists": True, "obsm_keys": [], "obsp_keys": [], "obs_columns": []}
        with pytest.raises(CellVaultStateError):
            PipelineStateValidator.validate("leiden", state)


class TestValidateUnknownOp:
    def test_unknown_op_passes(self):
        state = {"X_exists": False}
        PipelineStateValidator.validate("unknown_op", state)  # no preconditions


class TestStateErrorAttributes:
    def test_error_has_attributes(self):
        err = CellVaultStateError("pca", "X", ["a", "b"])
        assert err.operation == "pca"
        assert err.missing == "X"
        assert err.available == ["a", "b"]


class TestHelperMethods:
    def test_get_preconditions(self):
        prec = PipelineStateValidator.get_preconditions("pca")
        assert prec.get("X") is True

    def test_get_preconditions_unknown(self):
        assert PipelineStateValidator.get_preconditions("unknown") == {}

    def test_list_operations(self):
        ops = PipelineStateValidator.list_operations()
        assert "pca" in ops
        assert "umap" in ops
        assert "leiden" in ops
