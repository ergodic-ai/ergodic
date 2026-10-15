"""Check every internal link, asset and anchor of a built site (no network)."""

from __future__ import annotations

import html
import re
from functools import cache
from pathlib import Path
from urllib.parse import unquote, urlsplit

ATTRIBUTE = re.compile(r'(?:href|src|value|data-markdown|data-search)="([^"]*)"')
EXTERNAL = ("http:", "https:", "mailto:", "data:", "javascript:")


@cache
def ids(path: Path) -> frozenset[str]:
    return frozenset(html.unescape(x) for x in re.findall(r'\sid="([^"]+)"', path.read_text()))


def target_file(base: Path, reference: str) -> Path:
    target = (base / unquote(reference)).resolve()
    return target / "index.html" if target.is_dir() else target


def check_links(output: Path) -> list[str]:
    problems = []
    root = output.resolve()
    for page in sorted(root.rglob("*.html")):
        text = page.read_text()
        for match in ATTRIBUTE.finditer(text):
            reference = html.unescape(match[1])
            if not reference or reference.startswith(EXTERNAL):
                continue
            if match[0].startswith("value=") and not reference.endswith(("/", ".html")):
                continue  # Only version-picker options carry paths in value="…".
            parts = urlsplit(reference)
            target = page if not parts.path else target_file(page.parent, parts.path)
            where = page.relative_to(root)
            if not target.is_relative_to(root):
                problems.append(f"{where}: {reference} leaves the site")
                continue
            if not target.exists():
                problems.append(f"{where}: missing {reference}")
                continue
            fragment = unquote(parts.fragment)
            if fragment and target.suffix == ".html" and fragment not in ids(target):
                problems.append(f"{where}: missing anchor {reference}")
    return problems
