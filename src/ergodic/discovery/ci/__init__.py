# SPDX-License-Identifier: Apache-2.0
"""The conditional-independence tests, one module per family."""

from __future__ import annotations

from ._discrete import Discrete
from ._gcm import GCM
from ._linear import Linear
from ._mixed import Mixed

__all__ = ["GCM", "Discrete", "Linear", "Mixed"]
