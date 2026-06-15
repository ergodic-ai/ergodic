# SPDX-License-Identifier: Apache-2.0
"""Real-world datasets with their causal question, in the style of sklearn loaders.

Each loader returns a :class:`Bunch`: the typed dataset under ``data``, plus the
treatment, the outcome, the confounders or the instrument, and the ``target`` the
literature already measured. So an example can run the analysis and check itself
against a known number.

``load_*`` reads bundled or generated data and runs offline. The bundled CSVs are
public-domain or permissively licensed; see the NOTICE for provenance.

    from ergodic.datasets import load_birthweight

    ds = load_birthweight()
    est = estimate_effect(graph, ds.data, ds.treatment, ds.outcome, treated=ds.treated)
    assert abs(est.value - ds.target["value"]) < ds.target["tol"]
"""

from __future__ import annotations

from ._bayesnet import load_asia, load_earthquake
from ._bunch import Bunch
from ._bundled import load_birthweight, load_germany, load_pension401k, load_smoking
from ._tables import load_berkeley

__all__ = [
    "Bunch",
    "load_asia",
    "load_berkeley",
    "load_birthweight",
    "load_earthquake",
    "load_germany",
    "load_pension401k",
    "load_smoking",
]
