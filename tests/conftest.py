# SPDX-License-Identifier: Apache-2.0
"""Test configuration.

Thread pinning, set before anything imports numpy or pymc. Multiprocess NUTS
over several nodes oversubscribes the cores against multi-threaded BLAS, and the
OS then SIGKILLs the sampler workers. Pinning the BLAS and OMP thread counts to
one keeps the SCM's per-node fits (and the pymc-gated golden tests) robust under
CI. This must run at import time, before numpy is imported, so it lives at the
very top of conftest with no preceding imports.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
