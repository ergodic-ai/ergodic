"""Per-site configuration (``docsite/site.toml``): everything product-specific."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graphite import DocsError

ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = ROOT / "docsite"
CONFIG = SITE_DIR / "site.toml"


@dataclass(frozen=True)
class Link:
    label: str
    url: str


@dataclass(frozen=True)
class Product:
    name: str
    description: str
    url: str | None
    current: bool


@dataclass(frozen=True)
class Section:
    label: str
    page: str
    groups: tuple[str, ...]


@dataclass(frozen=True)
class Group:
    label: str
    pages: tuple[tuple[str, str], ...]  # (sidebar text, page name)


@dataclass(frozen=True)
class Task:
    title: str
    page: str
    text: str


@dataclass(frozen=True)
class NotebookGroup:
    label: str
    source: Path


@dataclass(frozen=True)
class Site:
    root: Path
    name: str
    subtitle: str
    tagline: str
    description: str
    package: str
    distribution: str
    public_url: str
    cname: str | None
    repository: str
    pages_source: str
    theme_key: str
    footer: str
    cta: tuple[str, str]
    search_prompt: str
    search_hint: str
    related_llms: tuple[Link, ...]
    install: tuple[tuple[str, str], ...]
    previous_versions: tuple[str, ...]
    products: tuple[Product, ...]
    footer_links: tuple[Link, ...]
    nav: tuple[Section, ...]
    groups: tuple[Group, ...]
    landing: dict[str, Any]
    tasks: tuple[Task, ...]
    reference_group: str
    reference_modules: tuple[tuple[str, str], ...]
    notebooks_page: str
    notebooks_title: str
    notebooks_lead: str
    notebook_groups: tuple[NotebookGroup, ...]
    example_timeout: int

    @property
    def pages_dir(self) -> Path:
        return self.root / self.pages_source

    @property
    def assets_dir(self) -> Path:
        return self.root / "docsite" / "assets"

    def page_order(self) -> list[tuple[str, str]]:
        return [entry for group in self.groups for entry in group.pages]

    def group_of(self, page: str) -> str | None:
        for group in self.groups:
            if any(name == page for _, name in group.pages):
                return group.label
        return None

    def section_of(self, page: str) -> str:
        group = self.group_of(page)
        for section in self.nav:
            if group in section.groups:
                return section.label
        return self.nav[0].label


def _require(table: dict[str, Any], key: str, where: str) -> Any:
    if key not in table:
        raise DocsError(f"{CONFIG.name}: [{where}] needs {key!r}")
    return table[key]


def load_site(path: Path = CONFIG, root: Path = ROOT) -> Site:
    data = tomllib.loads(path.read_text())
    site = _require(data, "site", "")
    landing = dict(_require(data, "landing", ""))
    tasks = tuple(Task(t["title"], t["page"], t["text"]) for t in landing.pop("tasks", []))
    reference = _require(data, "reference", "")
    notebooks = data.get("notebooks", {})
    nav = tuple(Section(s["label"], s["page"], tuple(s["groups"])) for s in data["nav"])
    groups = tuple(
        Group(g["label"], tuple((text, page) for text, page in g["pages"])) for g in data["groups"]
    )
    known_groups = {g.label for g in groups}
    for section in nav:
        if set(section.groups) - known_groups:
            raise DocsError(f"{path.name}: nav {section.label!r} names an unknown group")
    products = tuple(
        Product(p["name"], p.get("description", ""), p.get("url"), bool(p.get("current")))
        for p in data.get("products", [])
    )
    if products and sum(p.current for p in products) != 1:
        raise DocsError(f"{path.name}: exactly one product must be current")
    cta = _require(site, "cta", "site")
    return Site(
        root=root,
        name=_require(site, "name", "site"),
        subtitle=site.get("subtitle", "Docs"),
        tagline=_require(site, "tagline", "site"),
        description=_require(site, "description", "site"),
        package=_require(site, "package", "site"),
        distribution=_require(site, "distribution", "site"),
        public_url=_require(site, "public_url", "site").rstrip("/"),
        cname=site.get("cname"),
        repository=_require(site, "repository", "site").rstrip("/"),
        pages_source=_require(site, "pages_source", "site"),
        theme_key=_require(site, "theme_key", "site"),
        footer=_require(site, "footer", "site"),
        cta=(cta["label"], cta["page"]),
        search_prompt=site.get("search_prompt", "Search this version."),
        search_hint=site.get("search_hint", "No matches. Try a shorter query."),
        related_llms=tuple(Link(label, url) for label, url in site.get("related_llms", [])),
        install=tuple((label, command) for label, command in data["install"]["commands"]),
        previous_versions=tuple(data.get("versions", {}).get("previous", [])),
        products=products,
        footer_links=tuple(Link(x["label"], x["url"]) for x in data.get("footer_links", [])),
        nav=nav,
        groups=groups,
        landing=landing,
        tasks=tasks,
        reference_group=reference.get("group", "Reference"),
        reference_modules=tuple((m, label) for m, label in reference["modules"]),
        notebooks_page=notebooks.get("page", "notebooks"),
        notebooks_title=notebooks.get("title", "Notebooks"),
        notebooks_lead=notebooks.get("lead", ""),
        notebook_groups=tuple(
            NotebookGroup(g["label"], root / g["source"]) for g in notebooks.get("groups", [])
        ),
        example_timeout=int(data.get("examples", {}).get("timeout", 180)),
    )
