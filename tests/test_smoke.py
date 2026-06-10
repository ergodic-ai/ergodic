# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the package skeleton."""

from __future__ import annotations

import importlib

import ergodic


def test_version_is_nonempty_string() -> None:
    assert isinstance(ergodic.__version__, str)
    assert ergodic.__version__


def test_subpackages_import() -> None:
    for name in ("discovery", "inference", "process"):
        module = importlib.import_module(f"ergodic.{name}")
        assert module is not None
