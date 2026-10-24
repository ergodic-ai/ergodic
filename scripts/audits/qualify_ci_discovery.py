"""Frozen DEC-104 paired discovery diagnostic. Run separately on baseline and candidate."""

from __future__ import annotations

import json

import numpy as np

from ergodic import discover, tabular


def main() -> None:
    for seed in range(150):
        rng = np.random.default_rng(seed)
        names = tuple("ABCDEFG")
        coefficients = np.zeros((7, 7))
        for i in range(7):
            for j in range(i + 1, 7):
                if rng.random() < 0.35:
                    coefficients[i, j] = rng.choice([-1, 1]) * rng.uniform(0.8, 1.5)
        values = rng.normal(size=(2000, 7))
        for j in range(7):
            values[:, j] += values[:, :j] @ coefficients[:j, j]
        data = tabular(dict(zip(names, values.T, strict=True)))
        reverse = tabular(dict(zip(names[::-1], values.T[::-1], strict=True)))
        truth = {
            tuple(sorted((names[i], names[j])))
            for i, j in zip(*np.nonzero(coefficients), strict=True)
        }
        for method in ("mmpc", "pc", "fci", "rfci"):
            graph = discover(data, method=method, test="linear").graph
            pairs = {tuple(sorted((e.u, e.v))) for e in graph.edges}
            permuted = discover(reverse, method=method, test="linear").graph
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "method": method,
                        "edges": sorted(pairs),
                        "errors": len(pairs ^ truth),
                        "permutation_equal": graph == permuted,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
