"""Frozen API-167 diagnostic; see Welt API DEC-104. JSON output goes outside Git."""

from __future__ import annotations

import json

import numpy as np

from ergodic.discovery import GCM


def main() -> None:
    records = []
    for family in ("linear", "quadratic", "sine"):
        for n in (200, 1000):
            for learner in ("knn", "ols"):
                rejected = [0, 0]
                refused = [0, 0]
                for seed in range(200):
                    rng = np.random.default_rng(seed)
                    z, ex, ey = rng.normal(size=(3, n))
                    f = (
                        z
                        if family == "linear"
                        else z**2
                        if family == "quadratic"
                        else np.sin(2 * z)
                    )
                    x = (f + ex)[:, None]
                    for alternative in (0, 1):
                        y = (f + 0.5 * alternative * ex + ey)[:, None]
                        try:
                            _, p, _ = GCM(learner=learner).statistic(x, y, z[:, None])
                            rejected[alternative] += int(p <= 0.05)
                        except ValueError:
                            refused[alternative] += 1
                records.append(
                    {
                        "family": family,
                        "n": n,
                        "learner": learner,
                        "rejected": rejected,
                        "refused": refused,
                        "trials": 200,
                    }
                )
                print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
