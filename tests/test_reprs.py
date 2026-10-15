# SPDX-License-Identifier: Apache-2.0
"""Result reprs stay short: one line, no arrays, enum values not enum members.

Regression (API-149): the dataclass subclasses of ``EffectEstimate`` and
``PerUnitEffect`` regenerated a field-by-field repr that dumped every array, and
``DiscoveryResult`` printed ``GraphKind.CPDAG`` where its time-series sibling
printed ``cpdag``.
"""

from __future__ import annotations

import numpy as np

from ergodic.discovery import DiscoveryResult
from ergodic.graph import cpdag
from ergodic.inference import CATEResult, EffectTarget, ITEResult
from ergodic.inference.estimators._did import DiDEffect
from ergodic.inference.estimators._its import ITSEffect
from ergodic.inference.estimators._synthetic_control import SyntheticControlEffect


def _short(text: str, name: str) -> None:
    assert text.startswith(f"{name}(")
    assert "\n" not in text
    assert "array(" not in text
    assert len(text) < 200


def test_discovery_result_repr_prints_the_kind_value() -> None:
    result = DiscoveryResult(graph=cpdag(["a -- b", "b -- c"]), method="pc", test="fisherz")
    text = repr(result)
    assert "pc/fisherz: cpdag, 3 nodes, 2 edges" in text
    assert "GraphKind" not in text


def test_per_unit_reprs_are_short() -> None:
    values = np.linspace(-1.0, 1.0, 500)
    cate_res = CATEResult(
        treatment="A",
        outcome="Y",
        covariates=("Z",),
        n=500,
        kind="analytic",
        source="dr_learner",
        _values=values,
        _std_error=np.full(500, 0.1),
    )
    _short(repr(cate_res), "CATEResult")
    assert "ate=" in repr(cate_res)
    ite_res = ITEResult(
        treatment="A",
        outcome="Y",
        covariates=(),
        n=500,
        kind="posterior",
        source="scm",
        _draws=np.tile(values, (20, 1)),
        _factual=values,
    )
    _short(repr(ite_res), "ITEResult")


def test_design_effect_reprs_are_short() -> None:
    long = tuple(float(v) for v in np.arange(300))
    att = EffectTarget.ATT
    did = DiDEffect(
        target=att,
        estimator="did",
        learners={},
        n=300,
        kind="bootstrap",
        _draws=np.linspace(1.0, 2.0, 400),
    )
    _short(repr(did), "DiDEffect")
    its = ITSEffect(
        target=att,
        estimator="its",
        learners={},
        n=300,
        kind="analytic",
        _point=1.5,
        _std_error=0.2,
        times=long,
        observed=long,
        counterfactual=long,
        fitted=long,
    )
    _short(repr(its), "ITSEffect")
    assert "se=0.2" in repr(its)
    synth = SyntheticControlEffect(
        target=att,
        estimator="synthetic_control",
        learners={},
        n=300,
        kind="placebo",
        _point=1.5,
        periods=long,
        observed=long,
        synthetic=long,
    )
    _short(repr(synth), "SyntheticControlEffect")
