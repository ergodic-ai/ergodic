# SPDX-License-Identifier: Apache-2.0
"""Loaders for the bundled CSVs: birthweight, pension, smoking, and reunification.

The data files ship inside the wheel and load offline. The birthweight and
pension files derive from public-domain US government data (natality records and
the Survey of Income and Program Participation); the smoking and germany files
are replication data the authors of the cited studies publish without an explicit
licence. See the NOTICE for each source and its terms.
"""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

from ..data import panel, tabular
from ._bunch import Bunch


def _frame(name: str) -> pd.DataFrame:
    """Read a bundled CSV out of the installed package."""
    resource = files("ergodic.datasets").joinpath("data", name)
    with resource.open("rb") as handle:
        return pd.read_csv(handle)


_BIRTHWEIGHT_CONFOUNDERS = (
    "mmarried",
    "mage",
    "medu",
    "fage",
    "fedu",
    "mrace",
    "frace",
    "mhisp",
    "fhisp",
    "foreign",
    "alcohol",
    "deadkids",
    "nprenatal",
    "monthslb",
    "order",
    "fbaby",
    "prenatal1",
)

_BIRTHWEIGHT_DESCR = """\
Maternal smoking and infant birthweight (Cattaneo, 2010; Almond, Chay and Lee,
2005). 4,642 singleton births in Pennsylvania, drawn from public-domain US
natality records.

Treatment mbsmoke (the mother smoked during pregnancy), outcome bweight (grams),
and a set of parental and prenatal-care confounders. The lesson is the gap
between the naive difference (about -275 g) and the adjusted effect (about
-231 g): smoking lowers birthweight, and adjustment moves the estimate toward the
published figure. That figure is the doubly robust (AIPW) estimate on this same
4,642-row extract in the Stata treatment-effects manual (teffects aipw, example
1: -231 g, standard error 26 g); other adjustments there land between about
-210 g (nearest-neighbour matching) and -240 g (regression adjustment).
Build the graph from ds.confounders and run
estimate_effect(graph, ds.data, "mbsmoke", "bweight", treated="smoker",
strategy="aipw").
"""


def load_birthweight() -> Bunch:
    """Maternal smoking and infant birthweight, the textbook backdoor-adjustment set.

    Returns a :class:`Bunch` with ``data`` (a :class:`TabularData`),
    ``treatment`` (``"mbsmoke"``), ``outcome`` (``"bweight"``), ``treated``
    (``"smoker"``), ``confounders``, and ``target``.
    """
    df = _frame("birthweight.csv")
    continuous = (
        "bweight",
        "mage",
        "medu",
        "fage",
        "fedu",
        "nprenatal",
        "monthslb",
        "order",
        "deadkids",
    )
    categorical = (
        "mbsmoke",
        "mmarried",
        "mrace",
        "frace",
        "mhisp",
        "fhisp",
        "foreign",
        "alcohol",
        "fbaby",
        "prenatal1",
    )
    types: dict[str, str] = dict.fromkeys(continuous, "continuous")
    types.update(dict.fromkeys(categorical, "categorical"))
    data = tabular(df, types=types)
    return Bunch(
        data=data,
        task="adjustment",
        treatment="mbsmoke",
        outcome="bweight",
        treated="smoker",
        confounders=_BIRTHWEIGHT_CONFOUNDERS,
        target={
            "estimand": "ATE",
            "value": -231.0,
            "tol": 30.0,
            "naive": -275.3,
            "units": "grams",
            "source": (
                "StataCorp, Stata Treatment-Effects Reference Manual (Release 13), "
                "teffects aipw, example 1, on this Cattaneo (2010) extract: "
                "ATE -230.99 g (SE 26.2); data from US natality records as used by "
                "Almond, Chay and Lee (2005) and Cattaneo (2010)"
            ),
        },
        DESCR=_BIRTHWEIGHT_DESCR,
    )


_PENSION_CONFOUNDERS = ("age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown")

_PENSION_DESCR = """\
401(k) eligibility and net financial assets (Chernozhukov et al., 2018; SIPP
1991). 9,915 households. Eligibility for a 401(k) is taken as good as random once
you condition on income, which makes this the standard double-machine-learning
demonstration.

Treatment e401 (eligibility), outcome net_tfa (net total financial assets,
dollars), confounders led by income. The naive gap is about $19,500, inflated
because higher earners are both more often eligible and wealthier; adjusting with
double machine learning and a flexible learner pulls it to about $9,000 (a linear
nuisance gives a smaller adjustment, near $6,000, because the income effect is
nonlinear). The same table also supports an instrumental-variables reading, with
eligibility (e401) as the instrument for actual participation (p401);
ds.instrument and ds.participation name those columns. Build the graph from
ds.confounders and run estimate_effect(graph, ds.data, "e401", "net_tfa",
strategy="dml", learner="knn").
"""


def load_pension401k() -> Bunch:
    """401(k) eligibility and wealth: adjustment, double ML, and an instrument in one table.

    Returns a :class:`Bunch` with ``data``, ``treatment`` (``"e401"``),
    ``outcome`` (``"net_tfa"``), ``confounders``, ``instrument`` (``"e401"``),
    ``participation`` (``"p401"``, the endogenous treatment for the IV reading),
    and ``target``.
    """
    df = _frame("pension401k.csv")
    continuous = ("net_tfa", "age", "inc", "educ", "fsize")
    categorical = ("e401", "p401", "marr", "twoearn", "db", "pira", "hown")
    types: dict[str, str] = dict.fromkeys(continuous, "continuous")
    types.update(dict.fromkeys(categorical, "categorical"))
    data = tabular(df, types=types)
    return Bunch(
        data=data,
        task="adjustment",
        treatment="e401",
        outcome="net_tfa",
        instrument="e401",
        participation="p401",
        confounders=_PENSION_CONFOUNDERS,
        target={
            "estimand": "ATE (eligibility)",
            "value": 9000.0,
            "tol": 2500.0,
            "naive": 19559.0,
            "units": "USD",
            "source": "Chernozhukov et al. (2018)",
        },
        DESCR=_PENSION_DESCR,
    )


_SMOKING_DESCR = """\
California Proposition 99 and cigarette sales (Abadie, Diamond and Hainmueller,
2010). Per-capita cigarette sales for 39 states, 1970 to 2000. California passed
Proposition 99 in 1989; the other states are the donor pool.

Task: synthetic control. Build a weighted combination of donor states that tracks
California before 1989, then read the post-1989 gap. By 2000 the gap reaches about
-26 packs per capita. The headline value is the mean gap over the post period. Run
synthetic_control(ds.data, ds.outcome, treated="California", at=1989); p_value is
the Abadie placebo rank and interval() is the placebo band. The same series also
makes an interrupted-time-series example on California alone.
"""


def load_smoking() -> Bunch:
    """California Proposition 99: the canonical synthetic-control panel.

    Returns a :class:`Bunch` with ``data`` (a :class:`PanelData` keyed by state
    and year), ``outcome`` (``"cigsale"``), ``treated`` (``"California"``),
    ``at`` (1989), and ``target``.
    """
    df = _frame("smoking.csv")
    measures = ("cigsale", "lnincome", "beer", "age15to24", "retprice")
    data = panel(df, entity="state", time="year", types=dict.fromkeys(measures, "continuous"))
    return Bunch(
        data=data,
        task="synthetic_control",
        outcome="cigsale",
        treated="California",
        at=1989,
        target={
            "estimand": "ATT (mean post-period gap)",
            "value": -20.0,
            "tol": 12.0,
            "endpoint": -26.0,
            "units": "packs per capita",
            "source": "Abadie, Diamond and Hainmueller (2010)",
        },
        DESCR=_SMOKING_DESCR,
    )


_GERMANY_DESCR = """\
German reunification and GDP per capita (Abadie, Diamond and Hainmueller, 2015).
GDP per capita for West Germany and 16 OECD countries, 1960 to 2003. West Germany
is treated by reunification in 1990; the other countries are donors.

Task: synthetic control. Track West Germany before 1990 with a weighted donor
combination, then read the gap. The post-1990 gap is negative and widens, on the
order of -1,600 dollars per capita per year. Run
synthetic_control(ds.data, ds.outcome, treated="West Germany", at=1990). It pairs
with the smoking panel as a second synthetic-control case.
"""


def load_germany() -> Bunch:
    """German reunification: a second synthetic-control panel.

    Returns a :class:`Bunch` with ``data`` (a :class:`PanelData` keyed by country
    and year), ``outcome`` (``"gdp"``), ``treated`` (``"West Germany"``), ``at``
    (1990), and ``target``.
    """
    df = _frame("germany.csv")
    measures = (
        "gdp",
        "infrate",
        "trade",
        "schooling",
        "invest60",
        "invest70",
        "invest80",
        "industry",
    )
    data = panel(
        df,
        entity="country",
        time="year",
        types=dict.fromkeys(measures, "continuous"),
        exclude=["code"],
    )
    return Bunch(
        data=data,
        task="synthetic_control",
        outcome="gdp",
        treated="West Germany",
        at=1990,
        target={
            "estimand": "ATT (mean post-period gap)",
            "value": -1600.0,
            "tol": 1500.0,
            "units": "USD per capita",
            "source": "Abadie, Diamond and Hainmueller (2015)",
        },
        DESCR=_GERMANY_DESCR,
    )
