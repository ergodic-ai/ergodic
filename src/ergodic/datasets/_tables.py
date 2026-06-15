# SPDX-License-Identifier: Apache-2.0
"""Small published tables, hardcoded as facts. No file, no license question."""

from __future__ import annotations

import pandas as pd

from ..data import tabular
from ._bunch import Bunch

# UC Berkeley graduate admissions, fall 1973 (Bickel, Hammel and O'Connell, 1975).
# Per department: (male admitted, male rejected, female admitted, female rejected).
_BERKELEY: dict[str, tuple[int, int, int, int]] = {
    "A": (512, 313, 89, 19),
    "B": (353, 207, 17, 8),
    "C": (120, 205, 202, 391),
    "D": (138, 279, 131, 244),
    "E": (53, 138, 94, 299),
    "F": (22, 351, 24, 317),
}

_BERKELEY_DESCR = """\
UC Berkeley graduate admissions, fall 1973 (Bickel, Hammel and O'Connell, 1975).

One row per applicant: gender, department (the six largest, A through F), and
whether admitted. The textbook confounding lesson. Pooled, men were admitted at
44.5 percent and women at 30.4 percent, which reads as bias against women.
Department is the confounder: women applied more often to the competitive
departments, and within almost every department the admission rates are close or
favor women.

Task: the effect of gender on admission. Estimate it naively, with the graph
dag(["gender -> admit"]), and then adjusted for department, with
dag(["dept -> gender", "dept -> admit", "gender -> admit"]). The naive gap is
about -0.14 (women minus men); conditioning on department moves it to about zero.
The auto-tier story is [dept] < [gender] < [admit].
"""


def load_berkeley() -> Bunch:
    """The Berkeley admissions table, expanded to one row per applicant.

    Returns a :class:`Bunch` with ``data`` (columns ``gender``, ``dept``,
    ``admit``), ``treatment`` (``"gender"``), ``outcome`` (``"admit"``),
    ``treated`` (``"Female"``), ``confounders`` (``("dept",)``), and ``target``
    carrying both the naive and the department-adjusted risk difference.
    """
    genders: list[str] = []
    depts: list[str] = []
    admits: list[int] = []
    for dept, (ma, mr, fa, fr) in _BERKELEY.items():
        for gender, admitted, rejected in (("Male", ma, mr), ("Female", fa, fr)):
            genders.extend([gender] * (admitted + rejected))
            depts.extend([dept] * (admitted + rejected))
            admits.extend([1] * admitted + [0] * rejected)
    df = pd.DataFrame({"gender": genders, "dept": depts, "admit": admits})
    data = tabular(
        df, types={"gender": "categorical", "dept": "categorical", "admit": "categorical"}
    )
    return Bunch(
        data=data,
        task="confounding",
        treatment="gender",
        outcome="admit",
        treated="Female",
        confounders=("dept",),
        target={
            "estimand": "risk difference (Female minus Male)",
            "value": 0.0,
            "tol": 0.08,
            "naive": -0.141,
            "source": "Bickel, Hammel and O'Connell (1975)",
        },
        DESCR=_BERKELEY_DESCR,
    )
