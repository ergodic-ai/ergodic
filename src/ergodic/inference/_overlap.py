# SPDX-License-Identifier: Apache-2.0
"""Propensity diagnostics and explicit overlap guardrails, without trimming rows."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def _validate_overlap(clip: float, overlap: str, threshold: float) -> None:
    if not 0.0 < clip < 0.5:
        raise ValueError("clip must be in (0, 0.5)")
    if overlap not in {"warn", "raise", "ignore"}:
        raise ValueError("overlap must be 'warn', 'raise', or 'ignore'")
    if not 0.0 < threshold < 0.5:
        raise ValueError("overlap_threshold must be in (0, 0.5)")


def _propensity_diagnostics(
    a: npt.NDArray[np.float64],
    raw: npt.NDArray[np.float64],
    *,
    clip: float,
    overlap: str = "warn",
    threshold: float = 0.01,
) -> dict[str, float]:
    """Summarize raw predictions and ATE weights; guard extreme fitted overlap.

    At least 10% outside [threshold, 1-threshold], or arm ESS below 10% of
    its count, warns. At least 50% requiring clipping refuses, unless the caller
    explicitly chooses ``overlap='ignore'``. These are diagnostic thresholds,
    not a positivity test or a promise that smaller violations are harmless.
    """
    _validate_overlap(clip, overlap, threshold)
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != a.shape or not np.all(np.isfinite(raw)) or np.any((raw < 0) | (raw > 1)):
        raise ValueError("propensity predictions must be finite probabilities in [0, 1]")
    if not np.any(a == 0) or not np.any(a == 1):
        raise ValueError("overlap diagnostics require both treatment arms")
    e = np.clip(raw, clip, 1.0 - clip)
    clipped = float(np.mean((raw < clip) | (raw > 1.0 - clip)))
    tail = float(np.mean((raw < threshold) | (raw > 1.0 - threshold)))
    diagnostics = {
        "propensity_min": float(np.min(raw)),
        "propensity_q01": float(np.quantile(raw, 0.01)),
        "propensity_q05": float(np.quantile(raw, 0.05)),
        "propensity_median": float(np.median(raw)),
        "propensity_q95": float(np.quantile(raw, 0.95)),
        "propensity_q99": float(np.quantile(raw, 0.99)),
        "propensity_max": float(np.max(raw)),
        "propensity_clipped_fraction": clipped,
        "propensity_tail_fraction": tail,
        "overlap_threshold": threshold,
        "propensity_clip": clip,
    }
    weak_ess = False
    for arm, weights in (("treated", a / e), ("control", (1.0 - a) / (1.0 - e))):
        positive = weights[weights > 0]
        ess = float(np.sum(positive) ** 2 / np.sum(positive**2))
        diagnostics[f"{arm}_weight_max"] = float(np.max(positive))
        diagnostics[f"{arm}_weight_q99"] = float(np.quantile(positive, 0.99))
        diagnostics[f"{arm}_effective_n"] = ess
        weak_ess = weak_ess or ess < 0.1 * positive.size
    att_weights = ((1.0 - a) * e / (1.0 - e))[a == 0]
    diagnostics["att_control_effective_n"] = float(
        np.sum(att_weights) ** 2 / np.sum(att_weights**2)
    )
    diagnostics["att_control_weight_max"] = float(np.max(att_weights))
    severe = clipped >= 0.5
    poor = tail >= 0.1 or weak_ess or severe
    diagnostics["overlap_warning"] = float(poor)
    diagnostics["overlap_severe"] = float(severe)
    if poor and overlap != "ignore":
        message = (
            f"poor estimated overlap: {tail:.1%} of raw propensities outside "
            f"[{threshold}, {1 - threshold}], {clipped:.1%} clipped; "
            "inspect propensity quantiles, arm effective sizes and weight extremes. "
            "Clipping does not restore counterfactual support. No rows were trimmed. "
            "Reconsider the nuisance model or study design; overlap='ignore' is an "
            "explicit diagnostic override, not valid-inference certification."
        )
        if severe or overlap == "raise":
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=3)
    return diagnostics
