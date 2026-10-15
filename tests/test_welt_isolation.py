# SPDX-License-Identifier: Apache-2.0
"""ergodic core never imports welt or httpx; only ergodic.welt does (DEC-078)."""

from __future__ import annotations

import subprocess
import sys

_PROBE = """
import importlib, pkgutil, sys
import ergodic
names = []
def walk(package):
    # pkgutil.walk_packages would import ergodic.welt to recurse into it; skip it.
    for info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        if info.name == "ergodic.welt":
            continue
        names.append(info.name)
        module = importlib.import_module(info.name)
        if info.ispkg:
            walk(module)
walk(ergodic)
leaked = sorted(k for k in sys.modules if k.split(".")[0] in ("welt", "httpx"))
assert not leaked, leaked
assert "ergodic.welt" not in sys.modules
print(len(names))
"""


def test_core_and_every_non_welt_module_never_import_welt_or_httpx() -> None:
    # A fresh interpreter, so modules another test imported cannot hide a leak.
    done = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert int(done.stdout.strip()) > 100  # every subpackage and module was imported
