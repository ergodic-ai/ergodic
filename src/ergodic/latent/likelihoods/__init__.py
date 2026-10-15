# SPDX-License-Identifier: Apache-2.0
"""The built-in regime likelihoods.

Two ship, chosen for fitting speed because calibration reruns the whole
pipeline over many control worlds. :func:`spectral_whittle` is the flagship:
frame spectra with level, spread, freq, and amp deltas, seconds per fit.
:func:`switching_ar` is the reference second implementation proving the
contract generalizes, with level, rhythm, and spread deltas and a documented
structural blindness to amplitude.

Not available: a volatility (GARCH) likelihood, whose amp/spread consensus
rule the ``consensus()`` helper reserves a slot for, and a harmonic emission.
Either would be a new :class:`~ergodic.latent.RegimeLikelihood` subclass, with
no engine changes.
"""

from __future__ import annotations

from ._ar import SwitchingAR, switching_ar
from ._spectral import SpectralWhittle, spectral_whittle

__all__ = [
    "SpectralWhittle",
    "SwitchingAR",
    "spectral_whittle",
    "switching_ar",
]
