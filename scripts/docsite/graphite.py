"""Generic Graphite rendering core, vendored from welt-python ``scripts/build_docs.py``.

Source: ergodic-ai/welt-python at the commit recorded in ``docsite/provenance.json``.
Ported functions: ``icon``, ``format_signature``, ``highlight``, ``mark_lines``,
``fence_info``, ``code_block``, ``extract_code``, ``wrap_steps``, ``render_markdown``
and ``search_records``. Changes from the source, all product-neutral:

- typed for mypy strict and formatted for ruff; errors raise :class:`DocsError`;
- fence attributes add ``from``/``lines``/``tol``/``rtol``/``numbers`` (recorded output
  panels),
  ``timeout`` (programs) and ``requires`` (a program that needs an outside service,
  shown as a "Needs …" note and never run by the checker); a ``file="…"`` header
  links its download;
- ``render_markdown(lenient=True)`` maps unknown callouts to notes (notebook previews);
- ``search_records(deep=True)`` indexes ``###`` headings (reference pages).

Welt-only features (the models table, method contract tables, error tables, the Iris
notebook landing and the 0.8 redirects) are not vendored.
"""

from __future__ import annotations

import hashlib
import html
import inspect
import json
import re
import secrets
from dataclasses import dataclass
from typing import Any

import markdown


class DocsError(Exception):
    """A documentation source or build problem; the CLI prints it and exits 1."""


ICONS = {
    "menu": '<path d="M4 6h16M4 12h16M4 18h16"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
    "theme": (
        '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 '
        '1.4M2 12h2M20 12h2M6.3 17.7l-1.4 1.4M19.1 4.9l-1.4 1.4"/>'
    ),
    "chevron": '<path d="m9 18 6-6-6-6"/>',
    "left": '<path d="M19 12H5M11 18l-6-6 6-6"/>',
    "right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "file": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v5h6M16 13H8M16 17H8"/>'
    ),
    "edit": ('<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/>'),
    "source": '<path d="m16 18 6-6-6-6M8 6l-6 6 6 6"/>',
    "close": '<path d="M18 6 6 18M6 6l12 12"/>',
    "external": '<path d="M7 17 17 7M8 7h9v9"/>',
    "switch": '<path d="m7 15 5 5 5-5M7 9l5-5 5 5"/>',
}


def icon(name: str) -> str:
    return (
        '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
        f"{ICONS[name]}</svg>"
    )


def format_signature(label: str, signature: inspect.Signature) -> str:
    """One line when short; otherwise one parameter per line so nothing scrolls sideways."""
    text = f"{label}{signature}"
    if len(text) <= 72 or not signature.parameters:
        return text
    items: list[str] = []
    keyword_only = False
    parameters = list(signature.parameters.values())
    for index, parameter in enumerate(parameters):
        if (
            items
            and parameters[index - 1].kind is parameter.POSITIONAL_ONLY
            and parameter.kind is not parameter.POSITIONAL_ONLY
        ):
            items.append("/")
        if parameter.kind is parameter.VAR_POSITIONAL:
            keyword_only = True
        elif parameter.kind is parameter.KEYWORD_ONLY and not keyword_only:
            items.append("*")
            keyword_only = True
        items.append(str(parameter))
    if parameters[-1].kind is parameters[-1].POSITIONAL_ONLY:
        items.append("/")
    tail = str(signature)[str(signature).rindex(")") + 1 :]
    return f"{label}(\n" + "".join(f"    {item},\n" for item in items) + ")" + tail


FENCE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>[^\n`~]*)\n(?P<code>.*?)\n?"
    r"^(?P=indent)(?P=fence)[ \t]*$",
    re.M | re.S,
)
PYTHON_TOKEN = re.compile(
    r'("[^"\n]*"|\x27[^\x27\n]*\x27|#[^\n]*|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]\w*\b)'
)
PYTHON_KEYWORDS = frozenset(
    [
        "from",
        "import",
        "with",
        "as",
        "if",
        "elif",
        "else",
        "for",
        "while",
        "in",
        "not",
        "and",
        "or",
        "is",
        "None",
        "True",
        "False",
        "await",
        "async",
        "try",
        "except",
        "finally",
        "raise",
        "assert",
        "return",
        "def",
        "class",
        "lambda",
        "yield",
        "pass",
        "break",
        "continue",
    ]
)
CALLOUTS = {
    "note": "Note",
    "tip": "Tip",
    "caution": "Caution",
    "warning": "Caution",
    "need": "You'll need",
}
# Attributes each fence kind accepts. ``output`` panels name the program they record.
FENCE_ATTRIBUTES = {
    "code": frozenset({"file", "tab", "hl", "title", "timeout", "slow", "requires"}),
    "output": frozenset({"title", "from", "lines", "tol", "rtol", "numbers"}),
}


def highlight(raw: str, language: str) -> str:
    """Highlight literal code without executing it or adding a runtime dependency."""
    if language not in ("python", "py", "sh", "bash", "shell"):
        return html.escape(raw)
    if language in ("sh", "bash", "shell"):
        lines = []
        for line in raw.splitlines(keepends=True):
            content = line.rstrip("\n")
            ending = line[len(content) :]
            escaped = html.escape(content)
            if content.lstrip().startswith("#"):
                escaped = f'<span class="co">{escaped}</span>'
            lines.append(escaped + ending)
        return "".join(lines)
    pieces: list[str] = []
    cursor = 0
    for item in PYTHON_TOKEN.finditer(raw):
        value = item[0]
        if value.startswith("#"):
            kind = "co"
        elif value.startswith(('"', "'")):
            kind = "st"
        elif value[0].isdigit():
            kind = "nu"
        elif value in PYTHON_KEYWORDS:
            kind = "kw"
        elif raw[item.end() : item.end() + 1] == "(":
            kind = "fn"
        else:
            continue
        pieces += [
            html.escape(raw[cursor : item.start()]),
            f'<span class="{kind}">{html.escape(value)}</span>',
        ]
        cursor = item.end()
    pieces.append(html.escape(raw[cursor:]))
    return "".join(pieces)


def line_numbers(spec: str, count: int) -> set[int]:
    """Parse 1-based line ranges such as "2-3,7" against ``count`` lines."""
    selected: set[int] = set()
    for part in spec.split(","):
        start, _, stop = part.strip().partition("-")
        if (
            not start.isdigit()
            or (stop and not stop.isdigit())
            or not 1 <= int(start) <= int(stop or start) <= count
        ):
            raise DocsError(f"Invalid line range: {spec!r} (for {count} lines)")
        selected.update(range(int(start), int(stop or start) + 1))
    return selected


def mark_lines(highlighted: str, spec: str | None, count: int) -> str:
    """Wrap 1-based highlighted line ranges such as "2-3,7"; other lines stay unchanged."""
    if not spec:
        return highlighted
    selected = line_numbers(spec, count)
    return "\n".join(
        f'<span class="hl">{line or " "}</span>' if number in selected else line
        for number, line in enumerate(highlighted.split("\n"), 1)
    )


def fence_info(info: str) -> tuple[str, dict[str, str]]:
    language, _, rest = info.strip().partition(" ")
    if not re.fullmatch(r'(?:\s*[a-z]+="[^"]*")*\s*', rest):
        raise DocsError(f"Malformed code fence attributes in: {info!r}")
    attributes = dict(re.findall(r'([a-z]+)="([^"]*)"', rest))
    allowed = FENCE_ATTRIBUTES["output" if language == "output" else "code"]
    if set(attributes) - allowed:
        raise DocsError(f"Unsupported code fence attribute in: {info!r}")
    return (language or "text"), attributes


@dataclass(frozen=True)
class Fence:
    """One fenced block of a Markdown page, with its source span."""

    language: str
    attributes: dict[str, str]
    code: str  # dedented, with its final newline
    indent: str
    marker: str
    info: str
    start: int
    end: int
    code_start: int
    code_end: int


def fences(text: str) -> list[Fence]:
    """Every fenced block of ``text`` in order (the excerpt and output checks use this)."""
    found = []
    for match in FENCE.finditer(text):
        language, attributes = fence_info(match["info"])
        indent = match["indent"]
        raw = "\n".join(
            line[len(indent) :] if line.startswith(indent) else line
            for line in match["code"].split("\n")
        )
        found.append(
            Fence(
                language,
                attributes,
                raw + "\n" if raw else raw,
                indent,
                match["fence"],
                match["info"],
                match.start(),
                match.end(),
                match.start("code"),
                match.end("code"),
            )
        )
    return found


def code_block(blocks: list[tuple[str, dict[str, str], str]], position: int = 0) -> str:
    """Render one fence, or adjacent tab="…" fences as one accessible tab group."""
    if len(blocks) == 1 and blocks[0][0] == "output":
        _, attributes, raw = blocks[0]
        default = "Recorded output" if "from" in attributes else "Output"
        label = html.escape(attributes.get("title", default))
        return (
            f'<div class="output"><div class="output-head">{label}</div>'
            f'<pre tabindex="0" aria-label="{label}"><code>{html.escape(raw)}</code></pre></div>'
        )
    panes = []
    for language, attributes, raw in blocks:
        name = html.escape(attributes.get("tab") or language, quote=True)
        marked = mark_lines(highlight(raw, language), attributes.get("hl"), len(raw.splitlines()))
        panes.append(
            (
                language,
                attributes,
                f'<pre tabindex="0" aria-label="{name} code"><code>{marked}</code></pre>',
            )
        )
    if len(panes) == 1:
        language, attributes, pane = panes[0]
        label = attributes.get("file") or language
        file_class = " code-file" if "file" in attributes else ""
        download = ""
        if "requires" in attributes:
            download += (
                f'<span class="code-requires">Needs {html.escape(attributes["requires"])}</span>'
            )
        if "file" in attributes:
            target = html.escape(attributes["file"], quote=True)
            download += (
                f'<a class="code-download" href="downloads/{target}" download="{target}">'
                "Download</a>"
            )
        return (
            f'<div class="code-wrap"><div class="code-head"><span class="code-language'
            f'{file_class}">{html.escape(label)}</span>{download}</div>{pane}</div>'
        )
    key = "tabs-" + hashlib.sha256(json.dumps([position, blocks]).encode()).hexdigest()[:10]
    buttons, panels = [], []
    for index, (_, attributes, pane) in enumerate(panes):
        selected = index == 0
        buttons.append(
            f'<button id="{key}-{index}-tab" type="button" role="tab" '
            f'aria-selected="{str(selected).lower()}" aria-controls="{key}-{index}-panel" '
            f'tabindex="{0 if selected else -1}">{html.escape(attributes["tab"])}</button>'
        )
        panels.append(
            f'<div id="{key}-{index}-panel" role="tabpanel" aria-labelledby="{key}-{index}-tab"'
            + ("" if selected else " hidden")
            + f">{pane}</div>"
        )
    return (
        '<div class="code-wrap code-tabbed"><div class="code-head"><div class="code-tabs" '
        'role="tablist" aria-label="Code variants">'
        + "".join(buttons)
        + "</div></div>"
        + "".join(panels)
        + "</div>"
    )


def extract_code(text: str, nonce: str) -> tuple[str, list[str]]:
    """Replace fences with placeholders; adjacent tab fences at one indent form one group."""
    rendered: list[str] = []
    output: list[str] = []
    cursor = 0
    matches = list(FENCE.finditer(text))
    index = 0
    while index < len(matches):
        match = matches[index]
        indent = match["indent"]
        if len(indent.expandtabs(4)) % 4:
            raise DocsError(
                "Indent fenced code by a multiple of four spaces (inside lists and callouts)."
            )
        group = [match]
        if 'tab="' in match["info"]:
            while (
                index + 1 < len(matches)
                and 'tab="' in matches[index + 1]["info"]
                and matches[index + 1]["indent"] == indent
                and not text[group[-1].end() : matches[index + 1].start()].strip()
            ):
                index += 1
                group.append(matches[index])
        blocks = []
        for item in group:
            language, attributes = fence_info(item["info"])
            raw = "\n".join(
                line[len(indent) :] if line.startswith(indent) else line
                for line in item["code"].split("\n")
            )
            blocks.append((language, attributes, raw + "\n" if raw else raw))
        if len(blocks) > 1 and any(language == "output" for language, _, _ in blocks):
            raise DocsError("Output panels cannot be tabbed.")
        token = f"DOCSCODE{nonce}N{len(rendered)}X"
        rendered.append(code_block(blocks, len(rendered)))
        output += [text[cursor : group[0].start()], f"\n{indent}{token}\n"]
        cursor = group[-1].end()
        index += 1
    output.append(text[cursor:])
    return "".join(output), rendered


def wrap_steps(body: str) -> str:
    """Group consecutive `## Title {.step}` sections into a numbered steps list."""
    starts: list[int] = []
    depth = 0
    pattern = r"<(/?)(div|details|section|blockquote|ul|ol|li|table)\b[^>]*>|<h2 [^>]*>"
    for match in re.finditer(pattern, body):
        if match[0].startswith("<h2 "):
            if depth == 0:
                starts.append(match.start())
            elif re.search(r'class="[^"]*\bstep\b', match[0]):
                raise DocsError(
                    "Step headings must be top-level, not inside a callout, list or details block."
                )
        else:
            depth += -1 if match[1] else 1
    bounds = zip([0, *starts], [*starts, len(body)], strict=True)
    parts = [body[a:b] for a, b in bounds if body[a:b]]
    result: list[str] = []
    open_steps = False

    def drop_step(match: re.Match[str]) -> str:
        classes = " ".join(x for x in match[2].split() if x != "step")
        return match[1] + (f' class="{classes}"' if classes else "")

    for part in parts:
        is_step = re.match(r'<h2 [^>]*class="[^"]*\bstep\b', part) is not None
        if is_step and not open_steps:
            result.append('<div class="steps">')
            open_steps = True
        if not is_step and open_steps and part.startswith("<h2 "):
            result.append("</div>")
            open_steps = False
        if is_step:  # The section carries the step; the heading keeps any other classes.
            part = re.sub(r'^(<h2 [^>]*?) class="([^"]*)"', drop_step, part, count=1)
        result.append(f'<section class="step">{part}</section>' if is_step else part)
    if open_steps:
        result.append("</div>")
    return "".join(result)


def render_markdown(text: str, *, lenient: bool = False) -> tuple[str, list[dict[str, Any]]]:
    """Render a page; returns the HTML body and the table-of-contents tokens."""
    text = re.sub(
        r"\]\(([^():#]+)\.md(#[^)]*)?\)",
        lambda m: "](" + m[1] + ".html" + (m[2] or "") + ")",
        text,
    )
    nonce = secrets.token_hex(8)  # Placeholders cannot collide with page text or code.
    text, blocks = extract_code(text, nonce)
    renderer = markdown.Markdown(
        extensions=["tables", "toc", "admonition", "attr_list", "md_in_html"],
        extension_configs={"toc": {"permalink": False}},
    )
    body = renderer.convert(text)
    token = f"DOCSCODE{nonce}N(\\d+)X"
    # One pass, so inserted code is never rescanned; an indented placeholder may arrive as code.
    body = re.sub(
        rf"<pre><code>\s*{token}\s*</code></pre>|<p>{token}</p>|{token}",
        lambda m: blocks[int(next(group for group in m.groups() if group is not None))],
        body,
    )
    # Card lists read "[Title](page.md): description"; the link becomes the card title.
    body = re.sub(
        r'<div class="cards">.*?</div>',
        lambda m: re.sub(r"(</a>):\s*", r"\1 ", m[0]),
        body,
        flags=re.S,
    )
    body = body.replace(
        "<table>",
        '<div class="table-wrap" tabindex="0" role="region" aria-label="Scrollable table"><table>',
    ).replace("</table>", "</table></div>")

    def heading(match: re.Match[str]) -> str:
        level, attributes, content = match[1], match[2], match[3]
        identifier = re.search(r' id="([^"]+)"', attributes)
        if not identifier:
            return match[0]
        rest = re.sub(r' id="[^"]+"', "", attributes)
        label = html.escape(re.sub("<[^>]+>", "", content), quote=True)
        return (
            f'<h{level} id="{identifier[1]}"{rest}>{content}<a class="anchor" '
            f'href="#{identifier[1]}" aria-label="Link to {label}">#</a></h{level}>'
        )

    body = re.sub(r'<h([23])((?: [\w-]+="[^"]*")*)>(.*?)</h\1>', heading, body, flags=re.S)

    def callout(match: re.Match[str]) -> str:
        kinds = match[1].split()
        if len(kinds) != 1 or kinds[0] not in CALLOUTS:
            if not lenient:
                raise DocsError(f"Unsupported callout type: {match[1]}")
            kinds = ["note"]
        kind = kinds[0]
        title = match[2]
        if title is not None and title == match[1].split()[0].capitalize():
            title = html.escape(CALLOUTS[kind])  # No explicit title: use the component's label.
        heading_html = "" if title is None else f'<p class="callout-title">{title}</p>'
        css = "caution" if kind == "warning" else kind
        return f'<div class="callout {css}" role="note">\n{heading_html}'

    body = re.sub(
        r'<div class="admonition ([^"]*)">\s*(?:<p class="admonition-title">(.*?)</p>)?',
        callout,
        body,
        flags=re.S,
    )
    # A short opening paragraph reads as the page lead; long ones stay body text.
    body = re.sub(
        r"(</h1>\s*)<p>((?:(?!</p>).){1,320}?</p>)",
        r'\1<p class="lead">\2',
        body,
        count=1,
        flags=re.S,
    )
    toc: list[dict[str, Any]] = renderer.toc_tokens
    return wrap_steps(body), toc


def plain_text(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", value))
    return " ".join(value.split())


def search_records(body: str, name: str, title: str, *, deep: bool = False) -> list[dict[str, str]]:
    """Index rendered heading IDs; ``deep`` pages also index each ``###`` anchor."""
    heading_pattern = r'<h([23]) id="([^"]+)"[^>]*>(.*?)</h\1>'
    headings = [item for item in re.finditer(heading_pattern, body, re.S) if item[1] == "2" or deep]
    intro_end = headings[0].start() if headings else len(body)
    records = [{"title": title, "text": plain_text(body[:intro_end]), "href": name + ".html"}]
    for index, item in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        label = plain_text(item[3].split('<a class="anchor"', 1)[0])
        record = {
            "title": f"{title} · {label}",
            "text": plain_text(body[item.end() : end]),
            "href": name + ".html#" + item[2],
        }
        if deep:
            record["method"] = label
        records.append(record)
    return records
