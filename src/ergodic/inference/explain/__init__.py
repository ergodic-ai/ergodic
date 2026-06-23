# SPDX-License-Identifier: Apache-2.0
"""Tier-B explainability: characterize treatment-effect heterogeneity, honestly.

The inference-valid tools downstream of any CATE estimator, all reading one shared
doubly-robust score object so the cross-fitting and orthogonality stay consistent.
:class:`Heterogeneity` (from :meth:`~ergodic.inference.CATEResult.explain`) is the
entry point; the per-tool result records carry the estimates with honest intervals.

This is Tier-B *characterization* (BLP, GATES, CLAN, the calibration test, RATE).
Tier-A drivers (SHAP, partial dependence, variable importance) and policy trees are
later phases.
"""

from __future__ import annotations

from ._blp import BLPResult
from ._calibration import CalibrationResult
from ._clan import CLANResult
from ._gates import GATESResult
from ._rate import RATEResult
from ._record import Heterogeneity

__all__ = [
    "BLPResult",
    "CLANResult",
    "CalibrationResult",
    "GATESResult",
    "Heterogeneity",
    "RATEResult",
]
