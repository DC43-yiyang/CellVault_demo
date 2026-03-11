"""Tests for CellVault debug mode."""

import logging
import os

import pytest


def test_set_debug_enables_logging():
    """Test that set_debug(True) enables DEBUG level logging."""
    from cellvault._debug import set_debug, is_debug, logger

    set_debug(False)
    assert not is_debug()
    assert logger.level == logging.WARNING

    set_debug(True)
    assert is_debug()
    assert logger.level == logging.DEBUG

    # Cleanup
    set_debug(False)


def test_set_debug_adds_handler():
    """Test that set_debug(True) adds a StreamHandler."""
    from cellvault._debug import set_debug, logger

    # Remove existing handlers for clean test
    logger.handlers.clear()

    set_debug(True)
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

    # Calling again should not add duplicate handlers
    handler_count = len(logger.handlers)
    set_debug(True)
    assert len(logger.handlers) == handler_count

    # Cleanup
    set_debug(False)


def test_is_debug_returns_bool():
    """Test that is_debug returns a boolean."""
    from cellvault._debug import set_debug, is_debug

    set_debug(False)
    assert is_debug() is False

    set_debug(True)
    assert is_debug() is True

    set_debug(False)


def test_public_api_exports():
    """Test that set_debug and is_debug are exported from cellvault."""
    import cellvault

    assert hasattr(cellvault, "set_debug")
    assert hasattr(cellvault, "is_debug")
    assert callable(cellvault.set_debug)
    assert callable(cellvault.is_debug)


def test_debug_log_output(capsys):
    """Test that debug mode produces log output."""
    from cellvault._debug import set_debug, logger

    logger.handlers.clear()
    set_debug(True)

    logger.debug("test message %s", "hello")

    # The handler writes to stderr
    captured = capsys.readouterr()
    assert "test message hello" in captured.err

    set_debug(False)
