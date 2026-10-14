"""Notebooks: py:percent sources as cleared downloads, static previews and a release run.

The learn series and case studies are py:percent scripts. The build never executes
them: it publishes each as a ``.py`` and a cleared ``.ipynb`` download plus a static
preview page. ``execute`` runs them all (fresh process each, no network) for release CI.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Site
from .examples import Program, Run, run_all
from .graphite import DocsError, render_markdown


@dataclass(frozen=True)
class Notebook:
    group: str
    path: Path
    cells: tuple[tuple[str, str], ...]  # (kind, source); kind is "markdown" or "code"
    title: str

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def page(self) -> str:
        return "notebook-" + self.stem


def parse_percent(text: str) -> list[tuple[str, str]]:
    """Split a py:percent script into markdown and code cells."""
    cells: list[tuple[str, list[str]]] = []
    for line in text.split("\n"):
        marker = re.match(r"^# %%(.*)$", line)
        if marker:
            kind = "markdown" if "[markdown]" in marker[1] else "code"
            cells.append((kind, []))
        elif cells:
            cells[-1][1].append(line)
        elif line.strip():
            cells.append(("code", [line]))
    result = []
    for kind, lines in cells:
        if kind == "markdown":
            lines = [re.sub(r"^# ?", "", line) if line.startswith("#") else line for line in lines]
        source = "\n".join(lines).strip("\n")
        if source.strip():
            result.append((kind, source))
    return result


def collect(site: Site) -> list[Notebook]:
    notebooks = []
    seen: set[str] = set()
    for group in site.notebook_groups:
        for path in sorted(group.source.glob("*.py")):
            cells = parse_percent(path.read_text())
            title = next(
                (
                    m[1]
                    for kind, source in cells
                    if kind == "markdown"
                    for m in [re.search(r"^# (.+)$", source, re.M)]
                    if m
                ),
                None,
            )
            if title is None:
                raise DocsError(f"Notebook needs a '# Title' markdown cell: {path}")
            if path.stem in seen:
                raise DocsError(f"Two notebooks share the name {path.stem}")
            seen.add(path.stem)
            notebooks.append(Notebook(group.label, path, tuple(cells), title))
    return notebooks


def to_ipynb(notebook: Notebook) -> str:
    """A cleared notebook (no outputs, no execution counts)."""
    cells = []
    for kind, source in notebook.cells:
        lines = source.split("\n")
        cell: dict[str, object] = {
            "cell_type": kind,
            "metadata": {},
            "source": [line + "\n" for line in lines[:-1]] + lines[-1:],
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(document, indent=1, ensure_ascii=False) + "\n"


def actions(notebook: Notebook) -> str:
    label = html.escape(notebook.title, quote=True)
    return (
        f'<div class="notebook-actions" role="group" aria-label="{label} notebook actions">'
        f'<a href="downloads/{notebook.stem}.ipynb" download="{notebook.stem}.ipynb">'
        f'Download notebook<span class="sr-only">: {label}</span></a>'
        f'<a href="downloads/{notebook.stem}.py" download="{notebook.stem}.py">'
        f'Download script<span class="sr-only">: {label}</span></a>'
        f'<a href="{notebook.page}.html">View<span class="sr-only">: {label}</span></a></div>'
    )


def relink(text: str, stems: set[str], pages: set[str]) -> str:
    """Point notebook links at previews and site pages; drop links to retired pages."""

    def link(match: re.Match[str]) -> str:
        label, target = match[1], match[2]
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            return match[0]
        path, _, fragment = target.partition("#")
        stem = Path(path).stem
        anchor = "#" + fragment if fragment else ""
        if path.endswith(".py") and stem in stems:
            return f"[{label}](notebook-{stem}.html{anchor})"
        if path.endswith(".md") and stem in pages:
            return f"[{label}]({stem}.html{anchor})"
        return label

    return re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", link, text)


def preview(notebook: Notebook, stems: set[str], pages: set[str], index_page: str) -> str:
    """Render cleared cells without executing them."""
    parts = [
        f'<p><a href="{index_page}.html">← All notebooks</a></p>',
        actions(notebook),
        '<p class="notebook-preview-note">Static notebook preview · outputs are cleared. '
        "Download the notebook to run the cells.</p>",
    ]
    count = 0
    for kind, source in notebook.cells:
        if kind == "markdown":
            parts.append(render_markdown(relink(source, stems, pages), lenient=True)[0])
            continue
        count += 1
        longest = max((len(m[0]) for m in re.finditer(r"`+", source)), default=0)
        fence = "`" * max(3, 1 + longest)
        code = render_markdown(f"{fence}python\n{source}\n{fence}")[0]
        parts.append(
            f'<section class="notebook-cell" aria-label="Code cell {count}">'
            '<span class="cell-prompt" aria-hidden="true">In [ ]</span>' + code + "</section>"
        )
    return '<div class="notebook-view">' + "".join(parts) + "</div>"


def index_markdown(site: Site, notebooks: list[Notebook]) -> str:
    parts = [f"# {site.notebooks_title}", site.notebooks_lead]
    for group in site.notebook_groups:
        members = [n for n in notebooks if n.group == group.label]
        if not members:
            continue
        parts.append(f"## {group.label}")
        for notebook in members:
            parts.append(f"### {notebook.title}\n\n{{{{actions:{notebook.stem}}}}}")
    return "\n\n".join(parts) + "\n"


def execute(site: Site, timeout: int = 1800, workers: int = 2) -> list[Run]:
    """Run every notebook as a script (release CI). Returns one run per notebook."""
    programs = [Program(n.path, n.path.name, n.path.read_text(), timeout) for n in collect(site)]
    return run_all(programs, workers)
