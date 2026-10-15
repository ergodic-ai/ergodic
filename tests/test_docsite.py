"""The Graphite documentation builder (API-143): rendering, examples, reference, build."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path

import pytest
from scripts.docsite import examples, notebooks
from scripts.docsite.build import build, expand_includes, reference_site
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
    relabelled = "# Page\n\n" + PROGRAM + "\n```py3\ntotal = 2 + 2\n```\n"
    site = site_with_pages(
        tmp_path,
        {"good": good, "drifted": drifted, "unlinked": unlinked, "relabelled": relabelled},
    )
    pages = site.pages_dir
    assert examples.excerpt_problems(pages / "good.md", site) == []
    assert "not a verbatim" in examples.excerpt_problems(pages / "drifted.md", site)[0]
    assert "needs from=" in examples.excerpt_problems(pages / "unlinked.md", site)[0]
    assert "fence label 'py3'" in examples.excerpt_problems(pages / "relabelled.md", site)[0]


def test_record_then_check_outputs(tmp_path: Path) -> None:
    page = "# Page\n\n" + PROGRAM.replace("total 3", "stale")
    site = site_with_pages(tmp_path, {"page": page})
    stale = examples.check(site)
    assert len(stale.problems) == 1 and "output of demo.py changed" in stale.problems[0]
    recorded = examples.check(site, record=True)
    assert recorded.recorded == ["page.md: demo.py"]
    assert '```output from="demo.py"\ntotal 3\n```' in (site.pages_dir / "page.md").read_text()
    assert examples.check(site).problems == []


def test_programs_that_need_a_service_are_checked_but_not_run(tmp_path: Path) -> None:
    def hosted(name: str) -> str:
        return (
            f'```python file="{name}.py" requires="a Welt API key"\n'
            'raise SystemExit("never run")\nprint("done")\n```\n'
        )

    page = "# Page\n\n" + hosted("hosted") + '\n```python\nprint("done")\n```\n'
    drifted = "# Page\n\n" + hosted("drifted") + '\n```python\nprint("other")\n```\n'
    panel = "# Page\n\n" + hosted("panel") + '\n```output from="panel.py"\ndone\n```\n'
    site = site_with_pages(tmp_path, {"page": page, "drifted": drifted, "panel": panel})
    report = examples.check(site, ["page"])
    assert report.runs == [] and report.problems == []
    assert [(p.name, p.requires) for p in report.skipped] == [("hosted.py", "a Welt API key")]
    assert "not a verbatim" in examples.excerpt_problems(site.pages_dir / "drifted.md", site)[0]
    assert "never run" in examples.excerpt_problems(site.pages_dir / "panel.md", site)[0]
    body, _ = render_markdown(page)
    assert '<span class="code-requires">Needs a Welt API key</span>' in body
    assert 'href="downloads/hosted.py"' in body


def test_pages_include_repository_files_verbatim(tmp_path: Path) -> None:
    (tmp_path / "NOTES.md").write_text("# Notes\n\nText.\n")
    (tmp_path / "CODE.md").write_text("# Code\n\n```python\nx = 1\n```\n")
    assert expand_includes("{{include:NOTES.md}}\n", tmp_path) == "# Notes\n\nText.\n"
    inline = "Inline {{include:NOTES.md}} stays."
    assert expand_includes(inline, tmp_path) == inline
    for bad in ("../NOTES.md", "MISSING.md", "CODE.md"):
        with pytest.raises(DocsError):
            expand_includes(f"{{{{include:{bad}}}}}\n", tmp_path)


def test_numeric_tolerance() -> None:
    assert examples.same("ate=0.7982 [0.733, 0.863]", "ate=0.7982 [0.733, 0.863]")
    assert not examples.same("ate=0.7982", "ate=0.8010")
    assert examples.same("ate=0.7982", "ate=0.8010", tol=0.01)
    assert examples.same("ate=100.0", "ate=104.0", rtol=0.05)
    assert not examples.same("ate=0.7982", "att=0.7982", tol=0.01)


def test_numbers_vary_compares_text_only() -> None:
    assert examples.same("min_ess=25.8 ok=True", "min_ess=5.4 ok=True", numbers_vary=True)
    assert examples.same("x  25.8", "x   5.4", numbers_vary=True)
    assert not examples.same("min_ess=25.8 ok=True", "min_ess=5.4 ok=False", numbers_vary=True)
    assert not examples.same("a=1 b=2", "a=1", numbers_vary=True)
    assert not examples.same("min_ess=25.8", "min_ess=5.4")


def test_vary_lines_scope_the_looser_rule() -> None:
    expected = "coverage 1.00\nmin_ess 25.8\n"
    assert examples.same(expected, "coverage 1.00\nmin_ess 5.4\n", vary_lines={2})
    assert not examples.same(expected, "coverage 0.50\nmin_ess 5.4\n", vary_lines={2})
    assert examples.same(expected, "coverage 1.02\nmin_ess 5.4\n", tol=0.05, vary_lines={2})
    assert not examples.same(expected, "coverage 1.00\n", vary_lines={2})
    assert not examples.same(expected, "coverage 1.00\nmax_ess 5.4\n", vary_lines={2})


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
    for retained in site.previous_versions:  # CI rebuilds these from their tags first.
        (output / f"v{retained}").mkdir()
        (output / f"v{retained}" / "index.html").write_text("<!doctype html>")
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
    assert '<summary><span>Products</span><span class="sr-only">' in root_index
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["reference"]["public_names"] == summary["public_names"]
    assert not re.search(r"welt-docs-theme|Welt Python SDK", root_index)
    changelog = (output / "changelog.html").read_text()
    release = re.search(r"^## \[(\d+\.\d+\.\d+)\]", (ROOT / "CHANGELOG.md").read_text(), re.M)
    assert release is not None and f">{release[1]}</a> - " in changelog
    assert check_links(output) == []


def test_build_fails_on_an_undocumented_public_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ergodic.knowledge

    def probe(x: int) -> int:
        return x

    probe.__module__ = "ergodic.knowledge"
    monkeypatch.setattr(ergodic.knowledge, "probe", probe, raising=False)
    with pytest.raises(
        DocsError,
        match=r"1 public name\(s\) have no docstring:\n  - "
        r"ergodic\.knowledge\.probe",
    ):
        build(load_site(), tmp_path / "site")
    summary = build(load_site(), tmp_path / "site", strict_reference=False)
    assert summary["undocumented"] == 1


def test_every_public_module_is_listed() -> None:
    site = load_site()
    unlisted = dataclasses.replace(site, reference_modules=site.reference_modules[:-1])
    with pytest.raises(DocsError, match=r"not in \[reference\].*'datasets'"):
        reference_site(unlisted)
    excluded = dataclasses.replace(unlisted, reference_exclude=(("datasets", "covered elsewhere"),))
    assert "datasets" not in {m for m, _ in reference_site(excluded).reference_modules}


def test_optional_modules_need_their_requirement(monkeypatch: pytest.MonkeyPatch) -> None:
    site = load_site()
    # The real optional modules (ergodic.welt) are excluded here, so the result doesn't
    # depend on whether their requirement happens to be installed.
    optional = dataclasses.replace(
        site,
        reference_modules=site.reference_modules[:-1],
        reference_optional=(("datasets", "Datasets", "no_such_requirement_api143"),),
        reference_exclude=(
            *site.reference_exclude,
            *((m, "not under test") for m, _, _ in site.reference_optional),
        ),
    )
    monkeypatch.delenv("DOCS_REQUIRE_OPTIONAL", raising=False)
    assert "datasets" not in {m for m, _ in reference_site(optional).reference_modules}
    monkeypatch.setenv("DOCS_REQUIRE_OPTIONAL", "1")
    with pytest.raises(DocsError, match="no_such_requirement_api143"):
        reference_site(optional)
    present = dataclasses.replace(optional, reference_optional=(("datasets", "Datasets", "json"),))
    assert ("datasets", "Datasets") in reference_site(present).reference_modules


def test_programs_get_thread_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in examples.THREAD_CAPS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("PYTENSOR_FLAGS", "compiledir=/tmp/x")
    code = (
        "import os\n"
        "for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',\n"
        "            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTENSOR_FLAGS'):\n"
        "    print(key, os.environ.get(key))\n"
    )
    result = examples.run(examples.Program(tmp_path / "x.md", "env.py", code, 60))
    assert result.stdout.split("\n")[:6] == [
        "OMP_NUM_THREADS 1",
        "OPENBLAS_NUM_THREADS 2",
        "MKL_NUM_THREADS 2",
        "VECLIB_MAXIMUM_THREADS 2",
        "NUMEXPR_NUM_THREADS 2",
        "PYTENSOR_FLAGS compiledir=/tmp/x",
    ]


def test_slow_programs_run_only_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    slow = PROGRAM.replace('file="demo.py"', 'file="demo.py" slow="true"')
    long = PROGRAM.replace("demo.py", "long.py").replace(
        'file="long.py"', 'file="long.py" timeout="300"'
    )
    site = site_with_pages(tmp_path, {"slow": "# Slow\n\n" + slow, "long": "# Long\n\n" + long})
    monkeypatch.delenv("DOCS_SLOW", raising=False)
    skipped = examples.check(site)
    assert skipped.runs == [] and skipped.problems == []
    assert [f"{p.page.name}: {p.name}" for p in skipped.skipped] == [
        "long.md: long.py",
        "slow.md: demo.py",
    ]
    monkeypatch.setenv("DOCS_SLOW", "1")
    included = examples.check(site, jobs=2)
    assert len(included.runs) == 2 and included.problems == [] and included.skipped == []
