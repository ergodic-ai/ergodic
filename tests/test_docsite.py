"""The Graphite documentation builder (API-143): rendering, examples, reference, build."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest
from scripts.docsite import examples, notebooks
from scripts.docsite.build import build
from scripts.docsite.config import ROOT, Site, load_site
from scripts.docsite.graphite import DocsError, fence_info, render_markdown
from scripts.docsite.links import check_links
from scripts.docsite.reference import build_reference, own_doc

PROGRAM = """```python file="demo.py"
# 1. Add.
total = 1 + 2
print("total", total)
```

```output from="demo.py"
total 3
```
"""


def site_with_pages(tmp_path: Path, pages: dict[str, str]) -> Site:
    folder = tmp_path / "docsite" / "pages"
    folder.mkdir(parents=True)
    for name, text in pages.items():
        (folder / f"{name}.md").write_text(text)
    return dataclasses.replace(load_site(), root=tmp_path, example_timeout=60)


def test_render_markdown_components() -> None:
    text = (
        '# Title\n\nShort lead.\n\n!!! need "You\'ll need"\n    - ergodic\n\n'
        "## First {.step}\n\nOne.\n\n## Second {.step}\n\nTwo.\n\n" + PROGRAM
    )
    body, toc = render_markdown(text)
    assert '<p class="lead">Short lead.</p>' in body
    assert '<div class="callout need" role="note">' in body
    assert body.count('<section class="step">') == 2
    assert 'href="downloads/demo.py"' in body
    assert '<div class="output-head">Recorded output</div>' in body
    assert toc[0]["name"] == "Title"


def test_fence_attributes_are_checked() -> None:
    assert fence_info('output from="a.py" tol="0.01"') == (
        "output",
        {"from": "a.py", "tol": "0.01"},
    )
    with pytest.raises(DocsError):
        fence_info('python from="a.py"')
    with pytest.raises(DocsError):
        render_markdown("!!! danger\n    Unknown callout.\n")
    assert "callout note" in render_markdown("!!! danger\n    Body.\n", lenient=True)[0]


def test_excerpts_must_come_from_a_program(tmp_path: Path) -> None:
    good = "# Page\n\n" + PROGRAM + '\n```python\ntotal = 1 + 2\nprint("total", total)\n```\n'
    drifted = "# Page\n\n" + PROGRAM + "\n```python\ntotal = 2 + 2\n```\n"
    unlinked = "# Page\n\n" + PROGRAM + "\n```output\n3\n```\n"
    site = site_with_pages(tmp_path, {"good": good, "drifted": drifted, "unlinked": unlinked})
    pages = site.pages_dir
    assert examples.excerpt_problems(pages / "good.md", site) == []
    assert "not a verbatim" in examples.excerpt_problems(pages / "drifted.md", site)[0]
    assert "needs from=" in examples.excerpt_problems(pages / "unlinked.md", site)[0]


def test_record_then_check_outputs(tmp_path: Path) -> None:
    page = "# Page\n\n" + PROGRAM.replace("total 3", "stale")
    site = site_with_pages(tmp_path, {"page": page})
    stale = examples.check(site)
    assert len(stale.problems) == 1 and "output of demo.py changed" in stale.problems[0]
    recorded = examples.check(site, record=True)
    assert recorded.recorded == ["page.md: demo.py"]
    assert '```output from="demo.py"\ntotal 3\n```' in (site.pages_dir / "page.md").read_text()
    assert examples.check(site).problems == []


def test_numeric_tolerance() -> None:
    assert examples.same("ate=0.7982 [0.733, 0.863]", "ate=0.7982 [0.733, 0.863]")
    assert not examples.same("ate=0.7982", "ate=0.8010")
    assert examples.same("ate=0.7982", "ate=0.8010", tol=0.01)
    assert examples.same("ate=100.0", "ate=104.0", rtol=0.05)
    assert not examples.same("ate=0.7982", "att=0.7982", tol=0.01)


def test_programs_run_without_network(tmp_path: Path) -> None:
    code = (
        "import socket\ntry:\n    socket.create_connection(('example.com', 80), timeout=1)\n"
        "except OSError as error:\n    print('blocked:', error)\n"
    )
    result = examples.run(examples.Program(tmp_path / "x.md", "net.py", code, 60))
    assert result.returncode == 0
    assert result.stdout.startswith("blocked: network access is disabled")


def test_reference_documents_every_public_name() -> None:
    reference = build_reference(load_site())
    assert reference.missing == [], reference.missing
    modules = [m for m, _ in load_site().reference_modules]
    assert {f"reference-{m}" for m in modules} | {"reference"} == set(reference.pages)
    graph = reference.pages["reference-graph"][1]
    assert "## MixedGraph {#MixedGraph}" in graph and "### MixedGraph.to_dict" in graph


def test_generated_dataclass_doc_is_not_a_docstring() -> None:
    @dataclasses.dataclass
    class Bare:
        x: int

    @dataclasses.dataclass
    class Documented:
        """Has a docstring."""

        x: int

    assert own_doc(Bare) is None
    assert own_doc(Documented) == "Has a docstring."


def test_percent_notebooks_publish_cleared() -> None:
    cells = notebooks.parse_percent("# %% [markdown]\n# # Title\n#\n# Text.\n\n# %%\nx = 1\nx\n")
    assert cells == [("markdown", "# Title\n\nText."), ("code", "x = 1\nx")]
    nb = notebooks.Notebook("Learn", Path("demo.py"), tuple(cells), "Title")
    document = json.loads(notebooks.to_ipynb(nb))
    code = document["cells"][1]
    assert code["outputs"] == [] and code["execution_count"] is None


def test_vendored_assets_match_provenance() -> None:
    record = json.loads((ROOT / "docsite/provenance.json").read_text())
    for asset in record["assets"]:
        if "sha256" not in asset:
            continue
        data = (ROOT / "docsite/assets" / asset["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == asset["sha256"], asset["file"]
        if asset["transformation"].startswith("None"):
            assert asset["sha256"] == asset["source_sha256"], asset["file"]


def test_build_is_versioned_and_linked(tmp_path: Path) -> None:
    site = load_site()
    output = tmp_path / "site"
    (output / "v0.0.1").mkdir(parents=True)  # An older version that CI placed first.
    (output / "v0.0.1" / "index.html").write_text("<!doctype html>")
    (output / "stale.html").write_text("left over from another builder")
    summary = build(site, output)
    version = summary["version"]
    assert summary["seconds"] < 60
    assert (output / "v0.0.1" / "index.html").exists() and not (output / "stale.html").exists()
    for name in ("index.html", "start.html", "start.md", "reference.html", "llms.txt"):
        assert (output / f"v{version}" / name).exists() and (output / name).exists()
    for name in ("sitemap.xml", "robots.txt", ".nojekyll", "search-index.json"):
        assert (output / name).exists()
    assert (output / "CNAME").read_text().strip() == site.cname
    assert "https://welt.ergodic.dev/llms.txt" in (output / "llms.txt").read_text()
    root_index = (output / "index.html").read_text()
    assert f'<link rel="canonical" href="{site.public_url}/">' in root_index
    assert 'data-theme-key="ergodic-docs-theme"' in root_index
    assert "https://ergodic-ai.github.io/welt-python/" in root_index
    assert not re.search(r"welt-docs-theme|Welt Python SDK", root_index)
    assert check_links(output) == []
