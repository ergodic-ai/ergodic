"""The ergodic documentation builder on the Welt "Graphite" core (API-143, DEC-078).

The rendering core (``graphite.py``) and the theme assets (``docsite/assets``) are
vendored from ``welt-python``; ``docsite/provenance.json`` records the source commit
and file hashes. Everything product-specific comes from ``docsite/site.toml``.

Run ``python -m scripts.docsite --help`` from the repository root. Commands:

- ``build``: render the versioned site (``site/v<version>/`` plus the current root).
- ``check``: run every ``file="…"`` program, check excerpts and recorded outputs.
- ``record``: rerun programs and write their output into the ``output`` panels.
- ``links``: check every internal link and anchor of a built site.
- ``notebooks``: execute the learn and case-study notebooks (release CI).
"""
