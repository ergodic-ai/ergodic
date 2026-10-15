"""Build the versioned static site: ``<output>/v<version>/`` plus the current root.

Product strings, navigation and links come from ``docsite/site.toml``; rendering comes
from the vendored Graphite core. The build reads the installed package for the
reference and never executes page programs or notebooks.
"""

from __future__ import annotations

import html
import importlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import notebooks as nb
from .config import Site
from .examples import all_programs
from .graphite import (
    DocsError,
    code_block,
    fences,
    icon,
    render_markdown,
    search_records,
)
from .reference import Reference, build_reference, page_name


@dataclass
class Page:
    name: str
    title: str
    markdown: str  # the published Markdown copy
    body: str
    toc: list[dict[str, Any]]
    source: bool  # True for hand-written pages under pages_source


def release_tuple(version: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", version)
    if not match:
        raise DocsError(f"Unrecognised version: {version!r}")
    return tuple(int(part or 0) for part in match.groups())


def git_revision(root: Path) -> str:
    revision = os.environ.get("DOCS_SOURCE_REF")
    if revision:
        return revision
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class SiteBuilder:
    def __init__(self, site: Site, version: str) -> None:
        self.site = site
        self.version = version
        self.reference: Reference | None = None
        self.notebooks: list[nb.Notebook] = []

    # Navigation ----------------------------------------------------------------------
    def groups(self) -> list[tuple[str, list[tuple[str, str]]]]:
        groups = []
        for group in self.site.groups:
            pages = list(group.pages)
            if group.label == self.site.reference_group:
                pages += [(label, page_name(m)) for m, label in self.site.reference_modules]
            groups.append((group.label, pages))
        return groups

    def current(self, name: str) -> str:
        if name.startswith("notebook-"):
            return self.site.notebooks_page
        return name

    def section(self, name: str) -> str:
        name = self.current(name)
        if name.startswith("reference-"):
            name = "reference"
        return self.site.section_of(name)

    def pager(self, name: str) -> str:
        order = [entry for _, pages in self.groups() for entry in pages]
        pages = [page for _, page in order]
        current = self.current(name)
        if current not in pages or name.startswith("notebook-"):
            return ""
        index = pages.index(current)
        previous = order[index - 1] if index else ("Overview", "index")
        following = order[index + 1] if index + 1 < len(order) else None
        items = [
            f'<a class="pager-prev" href="{self.link(previous[1])}"><small>{icon("left")}'
            f"Previous</small><strong>{html.escape(previous[0])}</strong></a>"
        ]
        if following:
            items.append(
                f'<a class="pager-next" href="{self.link(following[1])}"><small>Next'
                f"{icon('right')}</small><strong>{html.escape(following[0])}</strong></a>"
            )
        return '<nav class="pager" aria-label="Previous and next">' + "".join(items) + "</nav>"

    @staticmethod
    def link(page: str) -> str:
        return "index.html" if page == "index" else page + ".html"

    def sidebar(self, name: str) -> str:
        current = self.current(name)
        parts = []
        for label, pages in self.groups():
            items = "".join(
                f'<a class="nav-item" href="{self.link(page)}"'
                + (' aria-current="page"' if page == current else "")
                + f">{html.escape(text)}</a>"
                for text, page in pages
            )
            parts.append(
                f'<div class="nav-group"><p class="nav-group-title">{html.escape(label)}</p>'
                f"{items}</div>"
            )
        if self.reference and name.startswith("reference"):
            anchors = self.reference.index.get(name, [])
            if anchors:
                items = "".join(
                    f'<a class="nav-item" href="#{html.escape(a, quote=True)}"><code>'
                    f"{html.escape(a)}</code></a>"
                    for a in anchors
                )
                parts.append(
                    '<div class="nav-group page-index"><p class="nav-group-title">On this page'
                    f"</p>{items}</div>"
                )
        return "".join(parts)

    def switcher(self) -> str:
        products = self.site.products
        if not products:
            return ""
        current = next(p for p in products if p.current)
        items = []
        for product in products:
            href = "index.html" if product.current else product.url or "#"
            mark = ' aria-current="true"' if product.current else ""
            external = "" if product.current else icon("external")
            items.append(
                f'<a href="{html.escape(href, quote=True)}"{mark}><strong>'
                f"{html.escape(product.name)}{external}</strong>"
                f"<small>{html.escape(product.description)}</small></a>"
            )
        return (
            '<details class="product-switcher"><summary aria-label="Switch product">'
            f"<span>{html.escape(current.name)}</span>{icon('switch')}</summary>"
            '<nav class="product-menu" aria-label="Ergodic products">'
            + "".join(items)
            + "</nav></details>"
        )

    # Page frame ----------------------------------------------------------------------
    def shell(self, page: Page | None, body: str, name: str, title: str) -> str:
        site, version = self.site, self.version
        home = name == "index"
        section = self.section(name)
        nav = "".join(
            f'<a href="{self.link(s.page)}"'
            + (' aria-current="page"' if s.label == section and not home else "")
            + f">{html.escape(s.label)}</a>"
            for s in site.nav
        )
        sidebar = self.sidebar(name)
        versions = f'<option value="index.html">v{version}</option>' + "".join(
            f'<option value="../v{old}/">v{old}</option>'
            for old in sorted(site.previous_versions, key=release_tuple, reverse=True)
            if release_tuple(old) < release_tuple(version)
        )
        page_url = f"{site.public_url}/v{version}/" + self.link(name)
        issue = (
            f"{site.repository}/issues/new?title="
            + quote(f"Docs: {title}")
            + "&body="
            + quote(f"Page: {page_url}\n\nWhat should change?\n")
        )
        actions = []
        if page is not None:
            actions.append(
                f'<button type="button" data-markdown="{name}.md">{icon("file")}'
                "<span>Copy page as Markdown</span></button>"
            )
            if page.source:
                actions.append(
                    f'<a href="{site.repository}/blob/v{version}/{site.pages_source}/{name}.md">'
                    f"{icon('source')}<span>View source at v{version}</span></a>"
                )
        actions.append(f'<a href="{issue}">{icon("edit")}<span>Suggest an edit</span></a>')
        outline = ""
        if page is not None and page.toc and page.toc[0].get("children"):
            links = "".join(
                f'<a href="#{item["id"]}">{html.escape(item["name"])}</a>'
                for item in page.toc[0]["children"]
                if item["level"] == 2
            )
            if links:
                outline = '<p class="outline-title">On this page</p>' + links
        rail = (f'<div class="outline-inner">{outline}</div>' if outline else "") + (
            '<div class="page-actions">' + "".join(actions) + "</div>"
        )
        inline_actions = (
            '<div class="page-actions page-actions-inline">' + "".join(actions) + "</div>"
        )
        crumbs = (
            ""
            if home
            else f'<p class="crumbs">{html.escape(section)}{icon("chevron")}'
            f"<span>{html.escape(title)}</span></p>"
        )
        if home:
            main = f'<main id="main" tabindex="-1" class="home">{body}</main>'
        else:
            main = (
                f'<div class="layout"><nav class="sidebar" aria-label="Documentation">{sidebar}'
                f'</nav><main id="main" tabindex="-1" class="article">{crumbs}{body}'
                f"{self.pager(name)}{inline_actions}</main>"
                f'<aside class="outline" aria-label="On this page">{rail}</aside></div>'
            )
        key = json.dumps(site.theme_key)
        theme_boot = (
            f"try{{const t=localStorage.getItem({key});"
            "if(t==='light'||t==='dark')document.documentElement.dataset.theme=t}catch(e){}"
        )
        cta_label, cta_page = site.cta
        cta = (
            f'<a class="btn btn-primary btn-sm open-console" href="{self.link(cta_page)}">'
            f"{html.escape(cta_label)}</a>"
        )
        footer_links = "".join(
            f'<a href="{html.escape(link.url, quote=True)}">{html.escape(link.label)}</a>'
            for link in site.footer_links
        ) + (
            f'<a href="{site.repository}/tree/v{version}">Release source</a>'
            f'<a href="https://pypi.org/project/{site.distribution}/{version}/">PyPI</a>'
            '<a href="llms.txt">llms.txt</a>'
        )
        canonical = f"{site.public_url}/v{version}/" + ("" if home else self.link(name))
        name_html = html.escape(site.name)
        subtitle = html.escape(site.subtitle)
        doc_title = html.escape(title) + f" · {name_html} {version}"
        description = html.escape(site.description, quote=True)
        return (
            '<!doctype html><html lang="en" data-theme-key="'
            + html.escape(site.theme_key, quote=True)
            + '"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width,initial-scale=1">'
            f'<meta name="description" content="{description}">'
            '<meta name="color-scheme" content="light dark">'
            f'<title>{doc_title}</title><link rel="canonical" href="{canonical}">'
            '<link rel="icon" type="image/svg+xml" href="assets/ergodic-mark-dark.svg">'
            '<link rel="preload" href="assets/fonts/geist-latin-wght-normal.woff2" as="font" '
            'type="font/woff2" crossorigin><link rel="stylesheet" href="assets/tokens.css">'
            '<link rel="stylesheet" href="assets/docs.css">'
            '<link rel="stylesheet" href="assets/site.css">'
            f'<script>{theme_boot}</script><script defer src="assets/docs.js"></script></head>'
            '<body data-search="search-index.json" data-search-prompt="'
            + html.escape(site.search_prompt, quote=True)
            + '" data-search-hint="'
            + html.escape(site.search_hint, quote=True)
            + '"><a class="skip" href="#main">Skip to content</a><header class="topbar">'
            '<button id="menu" class="icon-button menu-button" type="button" '
            f'aria-haspopup="dialog" aria-label="Open navigation">{icon("menu")}</button>'
            '<a class="brand" href="index.html"><img src="assets/ergodic-mark-dark.svg" alt="" '
            f'width="28" height="28"><strong>{name_html}</strong><span class="brand-sep" '
            f'aria-hidden="true"></span><span class="brand-sub">{subtitle}</span></a>'
            f"{self.switcher()}"
            f'<nav class="topnav" aria-label="Main navigation">{nav}</nav>'
            '<span class="spacer"></span><button id="search" class="search-trigger" '
            f'type="button" aria-haspopup="dialog">{icon("search")}<span class="search-label">'
            'Search docs</span><kbd class="shortcut" aria-hidden="true">⌘K</kbd>'
            '<span class="sr-only">Search documentation</span></button><select id="version" '
            'class="version" data-version aria-label="Documentation version">'
            f'{versions}</select><button id="theme" class="icon-button" type="button" '
            f'aria-label="Switch theme">{icon("theme")}</button>{cta}</header>{main}'
            f'<footer class="footer"><span>{html.escape(site.footer)} · {version}</span>'
            f'<span class="footer-links">{footer_links}</span></footer>'
            '<dialog id="navigation" class="nav-dialog" aria-labelledby="navigation-title">'
            '<header><strong id="navigation-title">Documentation</strong><button type="button" '
            f'class="icon-button" data-close aria-label="Close navigation">{icon("close")}'
            '</button></header><div class="nav-dialog-tools"><select class="version" '
            f'data-version aria-label="Documentation version">{versions}</select>{cta}</div>'
            f'<nav class="nav-dialog-sections" aria-label="Sections">{nav}</nav>'
            f'<nav aria-label="Mobile guide navigation">{sidebar}</nav></dialog>'
            '<dialog id="search-dialog" class="palette" aria-labelledby="search-title">'
            f'<h2 id="search-title" class="sr-only">Search {name_html} {version}</h2>'
            f'<div class="palette-input">{icon("search")}<label for="search-input" '
            'class="sr-only">Task, function or error</label><input id="search-input" '
            'type="search" autocomplete="off" placeholder="Search guides, functions and '
            'errors" role="combobox" aria-expanded="false" aria-controls="search-results" '
            'aria-autocomplete="list"><button type="button" data-close class="palette-close" '
            'aria-label="Close search"><kbd>Esc</kbd></button></div><ul id="search-results" '
            'role="listbox" aria-label="Search results"></ul><p id="search-empty">'
            f"{html.escape(site.search_prompt)}</p>"
            '<div class="palette-footer" aria-hidden="true"><span><kbd>↑</kbd><kbd>↓</kbd> to '
            f"move</span><span><kbd>↵</kbd> to open</span><span>Searching v{version}</span>"
            '</div></dialog><div id="announcement" class="sr-only" role="status" '
            'aria-live="polite"></div></body></html>'
        )

    # Landing -------------------------------------------------------------------------
    def landing_program(self, pages: dict[str, Page]) -> tuple[str, str, str]:
        """The landing's program and its recorded output, rendered from the source page."""
        page_ref, _, file = self.site.landing["program"].partition("#")
        page = pages.get(Path(page_ref).stem)
        if page is None:
            raise DocsError(f"Landing program page not found: {page_ref}")
        found = fences(page.markdown)
        program = next((f for f in found if f.attributes.get("file") == file), None)
        if program is None:
            raise DocsError(f"Landing program {file} not found on {page_ref}")
        output = next(
            (
                f
                for f in found
                if f.language == "output"
                and f.attributes.get("from") == file
                and "lines" not in f.attributes
            ),
            None,
        )
        rendered = code_block([(program.language, program.attributes, program.code)])
        if output is not None:
            rendered += code_block([(output.language, output.attributes, output.code)])
        markdown = f"```python\n{program.code}```"
        if output is not None:
            markdown += f"\n\nOutput:\n\n```text\n{output.code}```"
        return page.name, rendered, markdown

    def landing(self, pages: dict[str, Page]) -> tuple[str, str]:
        site, landing = self.site, self.site.landing
        tabs_buttons, tabs_panels = [], []
        for index, (label, command) in enumerate(site.install):
            key = f"install-{index}"
            selected = index == 0
            tabs_buttons.append(
                f'<button id="{key}-tab" type="button" role="tab" '
                f'aria-selected="{str(selected).lower()}" aria-controls="{key}-panel" '
                f'tabindex="{0 if selected else -1}">{html.escape(label)}</button>'
            )
            tabs_panels.append(
                f'<div id="{key}-panel" role="tabpanel" aria-labelledby="{key}-tab" '
                'tabindex="0"'
                + ("" if selected else " hidden")
                + ">"
                + code_block([("sh", {}, command + "\n")])
                + "</div>"
            )
        install = (
            '<div class="task-tabs" role="tablist" aria-label="Install command">'
            + "".join(tabs_buttons)
            + "</div>"
            + "".join(tabs_panels)
        )
        start, program, program_md = self.landing_program(pages)
        secondary = landing.get("secondary")
        cta_label, cta_page = site.cta
        hero = (
            '<section class="home-hero" aria-labelledby="home-title"><p class="eyebrow">'
            f'<span class="badge mono">v{self.version}</span>'
            f"{html.escape(landing.get('eyebrow', ''))}</p>"
            f'<h1 id="home-title">{html.escape(landing["title"])}<br><span>'
            f"{html.escape(landing.get('title_muted', ''))}</span></h1>"
            f'<p class="lead">{html.escape(landing["lead"])}</p><div class="actions">'
            f'<a class="btn btn-primary btn-lg" href="{self.link(cta_page)}">'
            f"{html.escape(cta_label)}{icon('right')}</a>"
            + (
                f'<a class="btn btn-lg" href="{self.link(secondary["page"])}">'
                f"{html.escape(secondary['label'])}</a>"
                if secondary
                else ""
            )
            + "</div></section>"
        )
        quickstart = (
            '<section class="home-band" aria-labelledby="setup-title"><div class="band-head">'
            f'<h2 id="setup-title">{html.escape(pages[start].title)}</h2>'
            f'<a href="{self.link(start)}">Full walkthrough{icon("right")}</a></div>'
            '<div class="quickstart-cards"><section class="start-card notebook-card" '
            'aria-labelledby="install-title"><header class="card-heading"><span '
            'class="card-number" aria-hidden="true">1</span><h3 id="install-title">Install</h3>'
            f"</header>{install}</section>"
            '<section class="start-card notebook-card" aria-labelledby="run-title">'
            '<header class="card-heading"><span class="card-number" aria-hidden="true">2</span>'
            f'<h3 id="run-title">Run</h3><a href="{self.link(start)}">Step by step</a></header>'
            f"{program}</section></div></section>"
        )
        tasks = "".join(
            f'<a class="task-card" href="{self.link(t.page)}"><strong>{html.escape(t.title)}'
            f"</strong><span>{html.escape(t.text)}</span></a>"
            for t in site.tasks
        )
        bands = (
            '<section class="home-band" aria-labelledby="tasks-title"><div class="band-head">'
            '<h2 id="tasks-title">What do you want to do?</h2></div>'
            f'<div class="task-cards">{tasks}</div></section>'
        )
        others = [p for p in site.products if not p.current and p.url]
        if others:
            cards = "".join(
                f'<a class="task-card" href="{html.escape(p.url or "", quote=True)}"><strong>'
                f"{html.escape(p.name)}</strong><span>{html.escape(p.description)}</span></a>"
                for p in others
            )
            bands += (
                '<section class="home-band" aria-labelledby="products-title">'
                '<div class="band-head"><h2 id="products-title">Also from Ergodic</h2></div>'
                f'<div class="task-cards">{cards}</div></section>'
            )
        markdown_lines = [
            f"# {site.name}: {site.tagline}",
            "",
            landing["lead"],
            "",
            "## Install",
            "",
            *(f"```sh\n{command}\n```" for _, command in site.install),
            "",
            f"## {pages[start].title}",
            "",
            program_md,
            "",
            "## What do you want to do?",
            "",
            *(f"- [{t.title}]({t.page}.md): {t.text}" for t in site.tasks),
        ]
        if others:
            markdown_lines += ["", "## Also from Ergodic", ""]
            markdown_lines += [f"- [{p.name}]({p.url}): {p.description}" for p in others]
        return hero + quickstart + bands, "\n".join(markdown_lines) + "\n"

    # Outputs -------------------------------------------------------------------------
    def llms(self, pages: dict[str, Page]) -> str:
        site, version = self.site, self.version
        base = f"{site.public_url}/v{version}"
        lines = [
            f"# {site.name} {version}",
            "",
            f"> {site.tagline}. {site.description} Each link is this release's Markdown source.",
            "",
        ]
        for link in site.related_llms:
            lines += [f"For {link.label}, see {link.url}", ""]
        lines += [f"- [Overview]({base}/index.md)", ""]
        for label, entries in self.groups():
            lines += [f"## {label}", ""]
            lines += [f"- [{pages[p].title}]({base}/{p}.md)" for _, p in entries if p in pages]
            lines.append("")
        return "\n".join(lines)


INCLUDE = re.compile(r"^\{\{include:([^}\n]+)\}\}[ \t]*$", re.M)


def expand_includes(text: str, root: Path) -> str:
    """Replace each ``{{include:path}}`` line with that repository file, verbatim.

    A page can publish a file kept elsewhere in the repository (the changelog) without
    a copy that drifts. Paths are relative to the repository root; includes do not
    nest, and an included file may not hold programs (the checker reads page sources).
    """

    def insert(match: re.Match[str]) -> str:
        base = root.resolve()
        target = (base / match[1].strip()).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise DocsError(f"Cannot include {match[1]!r}: not a file in the repository")
        included = target.read_text().rstrip("\n")
        if INCLUDE.search(included) or any(
            fence.language in ("python", "py") for fence in fences(included)
        ):
            raise DocsError(f"Included file {match[1]!r} may not hold includes or programs")
        return included

    return INCLUDE.sub(insert, text)


def page_title(text: str, name: str) -> str:
    match = re.search(r"^# (.+)$", text, re.M)
    if not match:
        raise DocsError(f"Page {name} needs a '# Title' first heading")
    return match[1].strip()


def clean_output(output: Path, version: str) -> None:
    """Empty the output root but keep other versions' folders (CI places them first)."""
    output.mkdir(parents=True, exist_ok=True)
    for entry in output.iterdir():
        if entry.is_dir() and re.fullmatch(r"v\d+\.\d+.*", entry.name):
            if entry.name == f"v{version}":
                shutil.rmtree(entry)
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def build(
    site: Site,
    output: Path,
    *,
    strict_reference: bool = True,
    report: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    package = importlib.import_module(site.package)
    version = str(getattr(package, "__version__", ""))
    if installed_version(site.distribution) != version:
        raise DocsError(
            f"Installed {site.distribution} {installed_version(site.distribution)} does not "
            f"match {site.package}.__version__ {version}; reinstall the package."
        )
    builder = SiteBuilder(site, version)
    sources = {
        p.stem: expand_includes(p.read_text(), site.root)
        for p in sorted(site.pages_dir.glob("*.md"))
    }
    listed = {page for group in site.groups for _, page in group.pages}
    generated = {"reference", site.notebooks_page}
    if missing := sorted(listed - set(sources) - generated):
        raise DocsError(f"Navigation lists missing pages: {missing}")
    if orphans := sorted(set(sources) - listed):
        raise DocsError(f"Pages missing from the navigation in site.toml: {orphans}")
    mutable = re.compile(re.escape(site.repository) + r"/(?:blob|tree)/(?:main|dev)\b")
    for name, text in sources.items():
        if mutable.search(text):
            raise DocsError(f"{name}.md links to mutable source; link a release tag instead")
    programs = all_programs(site)

    reference = build_reference(site)
    builder.reference = reference
    coverage = {
        "version": version,
        "public_names": reference.names,
        "undocumented": len(reference.missing),
        "undocumented_names": reference.missing,
        "undocumented_members": len(reference.missing_members),
        "undocumented_member_names": reference.missing_members,
        "exempt_names": reference.exempt,
    }
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(coverage, indent=2) + "\n")
    if reference.missing and strict_reference:
        listing = "\n".join(f"  - {name}" for name in reference.missing)
        raise DocsError(
            f"{len(reference.missing)} public name(s) have no docstring:\n{listing}\n"
            "Add a docstring (or a #: comment above a constant), or build with "
            "--allow-undocumented to report without failing."
        )

    builder.notebooks = nb.collect(site)
    stems = {n.stem for n in builder.notebooks}

    clean_output(output, version)
    out = output / f"v{version}"
    out.mkdir(parents=True)
    shutil.copytree(site.assets_dir, out / "assets", ignore=shutil.ignore_patterns("*.json"))
    downloads = out / "downloads"
    downloads.mkdir()
    for program in programs.values():
        (downloads / program.name).write_text(program.code)
    for notebook in builder.notebooks:
        shutil.copy2(notebook.path, downloads / notebook.path.name)
        (downloads / f"{notebook.stem}.ipynb").write_text(nb.to_ipynb(notebook))

    pages: dict[str, Page] = {}
    markdown_sources: dict[str, tuple[str, bool]] = {
        name: (text, True) for name, text in sources.items()
    }
    for name, (_title, text) in reference.pages.items():
        markdown_sources[name] = (text, False)
    markdown_sources[site.notebooks_page] = (nb.index_markdown(site, builder.notebooks), False)
    page_names = set(markdown_sources)
    search: list[dict[str, str]] = []
    for name, (text, source) in markdown_sources.items():
        body, toc = render_markdown(text)
        if name == site.notebooks_page:
            for notebook in builder.notebooks:
                body = body.replace(f"<p>{{{{actions:{notebook.stem}}}}}</p>", nb.actions(notebook))
            body = '<div class="notebook-index">' + body + "</div>"
            text = re.sub(
                r"\{\{actions:([a-z0-9-]+)\}\}",
                lambda m: (
                    f"[Download notebook](downloads/{m[1]}.ipynb) · "
                    f"[Download script](downloads/{m[1]}.py) · [View](notebook-{m[1]}.html)"
                ),
                text,
            )
        title = page_title(text, name)
        pages[name] = Page(name, title, text, body, toc, source)
    for page in pages.values():
        (out / f"{page.name}.md").write_text(page.markdown)
        (out / f"{page.name}.html").write_text(
            builder.shell(page, page.body, page.name, page.title)
        )
        search.extend(
            search_records(page.body, page.name, page.title, deep=page.name.startswith("reference"))
        )
    for notebook in builder.notebooks:
        body = nb.preview(notebook, stems, page_names, site.notebooks_page)
        (out / f"{notebook.page}.html").write_text(
            builder.shell(None, body, notebook.page, notebook.title)
        )
        search.append(
            {
                "title": notebook.title,
                "text": f"{notebook.title} · Notebook preview · {notebook.group}",
                "href": notebook.page + ".html",
            }
        )
    home, home_markdown = builder.landing(pages)
    (out / "index.html").write_text(builder.shell(None, home, "index", site.tagline))
    (out / "index.md").write_text(home_markdown)
    search.insert(0, {"title": site.name, "text": site.landing["lead"], "href": "index.html"})
    (out / "llms.txt").write_text(builder.llms(pages))
    (out / "search-index.json").write_text(json.dumps(search, ensure_ascii=False, indent=2) + "\n")
    manifest = {
        "version": version,
        "pages": sorted(pages),
        "notebooks": sorted(n.stem for n in builder.notebooks),
        "programs": sorted(programs),
        "previous_versions": sorted(site.previous_versions, key=release_tuple),
        "reference": {k: v for k, v in coverage.items() if not k.endswith("_names")},
        "documentation_ref": git_revision(site.root),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "reference-coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")

    # The current, unversioned routes share the same bytes.
    shutil.copytree(out, output, dirs_exist_ok=True)
    for path in output.glob("*.html"):
        text = path.read_text()
        text = text.replace(f"{site.public_url}/v{version}/", f"{site.public_url}/")
        text = text.replace('value="../v', 'value="v')
        path.write_text(text)
    urls = [f"{site.public_url}/", f"{site.public_url}/v{version}/"]
    names = [*pages, *(n.page for n in builder.notebooks)]
    urls += [f"{site.public_url}/{n}.html" for n in names]
    urls += [f"{site.public_url}/v{version}/{n}.html" for n in names]
    for old in sorted(site.previous_versions, key=release_tuple):
        if (output / f"v{old}" / "index.html").exists():
            urls += [f"{site.public_url}/v{old}/"]
    (output / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><urlset '
        'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(f"<url><loc>{html.escape(url)}</loc></url>" for url in urls)
        + "</urlset>\n"
    )
    (output / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\nSitemap: {site.public_url}/sitemap.xml\n"
    )
    (output / ".nojekyll").write_text("")
    if site.cname:
        (output / "CNAME").write_text(site.cname + "\n")
    seconds = time.perf_counter() - started
    return {
        "version": version,
        "pages": len(pages),
        "notebooks": len(builder.notebooks),
        "programs": len(programs),
        "undocumented": len(reference.missing),
        "public_names": reference.names,
        "seconds": round(seconds, 2),
        "output": str(output),
    }
