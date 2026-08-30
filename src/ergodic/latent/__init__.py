# SPDX-License-Identifier: Apache-2.0
"""Hidden-regime inference: recovering a switch nobody logged.

One observed series; a hidden two-state process (call it the switch); a
mechanism that changes how the series behaves while the switch is on. This
pillar asks three questions and keeps them apart on purpose. :func:`recover`
answers WHEN the switch was on and WHICH named mechanism block changed: it
returns the posterior state path and a delta interval per block (level,
spread, rhythm, amplitude, in the built-in vocabularies). :func:`certify`
answers WHETHER a regime exists at all, vocabulary-free: it fits the
likelihood's own one-state null, simulates control worlds from it, and
places the observed two-state improvement in that measured null
distribution. :func:`calibrate` measures a configured detector's false-fire
rate on the same kind of control worlds, with a Wilson interval on the rate.
Conflating the three is the standard failure of switching models: a fitted
path always explains the data it was fitted to, so "which block moved" must
never double as "a regime exists". :func:`consensus` lays several
likelihoods' verdicts on one series side by side, absences (a vocabulary
that cannot express a block) kept distinct from non-fires.

The model underneath is one hidden Markov chain whose path is summed out
exactly by the forward algorithm, so NUTS samples only continuous parameters.
What varies is the emission: a :class:`RegimeLikelihood` says how the regime
shows up in the data, declares its delta blocks with shrinkage priors, and
everything else (chain priors, marginalization, sampling, decoding, label
alignment, verdicts) is shared engine. Two likelihoods ship:
:func:`~ergodic.latent.likelihoods.spectral_whittle`, the fast flagship that
reads level, spread, rhythm, and amplitude off framed local spectra, and
:func:`~ergodic.latent.likelihoods.switching_ar`, the autoregressive second
implementation that proves the contract generalizes and is documented blind
to amplitude (a sinusoid's AR coefficients do not depend on its amplitude).

Label switching is handled by alignment, never by ordering constraints:
decoded state trajectories are correlated against an iterated reference and
flipped draws have their deltas sign-corrected, so the posterior stays
untruncated and a delta near zero is not pushed off it. Verdicts are
equal-tailed credible intervals at a familywise-corrected level, gated by the
likelihood's own sanity checks (the spectral wave gate). The measured
sentinels behind those choices ride in the module docstrings: the spectral
model fires on zero of twenty control worlds at the 99% familywise level and
recovers a planted noise tripling as a log-spread delta near 1.1, and the
switching AR fires six of six on level, spread, and rhythm arms while its
amplitude blindness stands.

pymc is imported inside the functions that sample, so the bare install never
loads it; ``pip install 'ergodic[latent]'`` adds it, and jax plus numpyro get
the fast sampler path.
"""

from __future__ import annotations

from ._facade import LatentModel, calibrate, certify, consensus, latent_model, recover
from ._likelihood import (
    DeltaBlock,
    EmissionSpec,
    Gate,
    MarkovChain,
    PreparedData,
    RegimeLikelihood,
    semi_markov,
)
from ._result import (
    CalibrationReport,
    CertifyResult,
    Consensus,
    GateOutcome,
    LatentResult,
)

__all__ = [
    "CalibrationReport",
    "CertifyResult",
    "Consensus",
    "DeltaBlock",
    "EmissionSpec",
    "Gate",
    "GateOutcome",
    "LatentModel",
    "LatentResult",
    "MarkovChain",
    "PreparedData",
    "RegimeLikelihood",
    "calibrate",
    "certify",
    "consensus",
    "latent_model",
    "recover",
    "semi_markov",
]
