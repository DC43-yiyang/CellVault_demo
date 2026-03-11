"""Tests for NameRegistry."""

import pytest

from cellvault.registry import NameRegistry


class TestNameRegistryGet:
    def test_pca_default(self):
        assert NameRegistry.get("pca") == "X_pca"

    def test_pca_harmony(self):
        assert NameRegistry.get("pca", "harmony") == "X_pca_harmony"

    def test_umap_default(self):
        assert NameRegistry.get("umap") == "X_umap"

    def test_leiden_default(self):
        assert NameRegistry.get("leiden") == "leiden"

    def test_neighbors_default(self):
        assert NameRegistry.get("neighbors") == "neighbors"

    def test_case_insensitive(self):
        assert NameRegistry.get("PCA") == "X_pca"
        assert NameRegistry.get("pca", "Harmony") == "X_pca_harmony"

    def test_unknown_operation_raises(self):
        with pytest.raises(KeyError, match="No canonical name"):
            NameRegistry.get("unknown_op")

    def test_unknown_integration_raises(self):
        with pytest.raises(KeyError, match="No canonical name"):
            NameRegistry.get("pca", "unknown_integration")


class TestNameRegistryLookup:
    def test_lookup_x_pca(self):
        op, integ = NameRegistry.lookup("X_pca")
        assert op == "pca"
        assert integ is None

    def test_lookup_x_pca_harmony(self):
        op, integ = NameRegistry.lookup("X_pca_harmony")
        assert op == "pca"
        assert integ == "harmony"

    def test_lookup_unknown_raises(self):
        with pytest.raises(KeyError, match="not a registered"):
            NameRegistry.lookup("not_a_key")


class TestNameRegistryRegister:
    def test_register_custom(self):
        reg = NameRegistry()
        reg.register("custom_op", None, "X_custom")
        assert NameRegistry.get("custom_op") == "X_custom"
        assert NameRegistry.lookup("X_custom") == ("custom_op", None)

    def test_is_canonical(self):
        assert NameRegistry.is_canonical("X_pca")
        assert not NameRegistry.is_canonical("not_canonical")

    def test_list_all_returns_dict(self):
        result = NameRegistry.list_all()
        assert isinstance(result, dict)
        assert ("pca", None) in result
