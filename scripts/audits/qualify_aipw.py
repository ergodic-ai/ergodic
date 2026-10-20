# SPDX-License-Identifier: Apache-2.0
"""API-180 frozen synthetic diagnostics, not general inference certification.

Run ``python -m scripts.audits.qualify_aipw`` against the candidate source. Each
fixture uses seeds 0..99, n=400, p=5 unless explicitly named small_arm (n=80).
Original trees matches the 0.3.0 audit diagnostic exactly; supported fixtures use
native OLS/logistic nuisances. No data, fitted models or outputs are written.
"""

from __future__ import annotations

import json
import warnings

import numpy as np

import ergodic
from ergodic import dag, estimate_effect, tabular


def main() -> None:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    print(json.dumps({"source": ergodic.__file__, "version": ergodic.__version__}))
    graph = dag([f"Z{i} -> A" for i in range(5)] + [f"Z{i} -> Y" for i in range(5)] + ["A -> Y"])
    for fixture in (
        "legacy_trees_in_sample",
        "original_trees",
        "linear",
        "nonlinear",
        "heterogeneous",
        "poor_overlap",
        "small_arm",
    ):
        points: list[float] = []
        errors: list[float] = []
        refused = hits = warned = 0
        for seed in range(100):
            rng = np.random.default_rng(seed)
            n = 80 if fixture == "small_arm" else 400
            z = rng.normal(size=(n, 5))
            logit = (3 if fixture == "poor_overlap" else 1) * z[:, 0]
            probability = np.full(n, 0.05) if fixture == "small_arm" else 1 / (1 + np.exp(-logit))
            a = rng.binomial(1, probability).astype(float)
            tau = 2 + z[:, 0] if fixture == "heterogeneous" else np.full(n, 2.0)
            baseline = z[:, 0] ** 2 + z[:, 1] if fixture == "nonlinear" else z[:, 0]
            y = tau * a + baseline + rng.normal(size=n)
            data = tabular({**{f"Z{i}": z[:, i] for i in range(5)}, "A": a, "Y": y})
            learner = (
                {
                    "outcome": DecisionTreeRegressor(random_state=seed),
                    "propensity": DecisionTreeClassifier(random_state=seed),
                }
                if fixture in {"original_trees", "legacy_trees_in_sample"}
                else None
            )
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                try:
                    # Fixed split seed is separate from the fixture / learner seed.
                    estimate = estimate_effect(
                        graph,
                        data,
                        "A",
                        "Y",
                        learner=learner,
                        random_state=0,
                        mode="in_sample" if fixture == "legacy_trees_in_sample" else "auto",
                        overlap="ignore" if fixture == "legacy_trees_in_sample" else "warn",
                    )
                except ValueError:
                    refused += 1
                    continue
            warned += bool(captured)
            points.append(estimate.value)
            assert estimate.std_error is not None
            errors.append(estimate.std_error)
            lo, hi = estimate.interval()
            hits += lo <= 2 <= hi
        print(
            json.dumps(
                {
                    "fixture": fixture,
                    "seeds": "0..99",
                    "n": 80 if fixture == "small_arm" else 400,
                    "p": 5,
                    "split_seed": 0,
                    "truth": 2,
                    "returned": len(points),
                    "refused": refused,
                    "warned": warned,
                    "covered": hits,
                    "coverage_among_returned": hits / len(points) if points else None,
                    "mean": float(np.mean(points)) if points else None,
                    "empirical_sd": float(np.std(points, ddof=1)) if len(points) > 1 else None,
                    "mean_se": float(np.mean(errors)) if errors else None,
                }
            )
        )


if __name__ == "__main__":
    main()
