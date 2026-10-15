"""Generate the Python reference from the live package: one Markdown page per module.

Signatures come from :mod:`inspect`; prose comes from docstrings, with Google-style
sections (``Args:``, ``Returns:``, ``Raises:``, ``Attributes:``, ``Examples:``,
``Note:``, ``Warning:``) and the common reStructuredText roles converted to Markdown.
Every public name (a module's ``__all__``, or the package's top-level exports) and
every public method or property of a documented class must carry a docstring;
:attr:`Reference.missing` lists those that do not (members are also listed on their
own in :attr:`Reference.missing_members`), so a strict build fails on either.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import inspect
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import Site

SECTION = re.compile(
    r"^(Args|Arguments|Parameters|Params|Keyword Args|Returns|Return|Yields|Raises|Attributes|"
    r"Examples?|Notes?|Warnings?|See Also|References):\s*$"
)
ROLE = re.compile(r":(?:py:)?(class|func|meth|attr|data|mod|obj|exc|const|math):`~?([^`]+)`")
RST_LINK = re.compile(r"`([^`<]+?)\s*<(https?://[^>]+)>`__?")


class _Annotation:
    """An annotation string shown bare in a signature (``x: int``, not ``x: 'int'``)."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def page_name(module: str) -> str:
    return f"reference-{module}"


def kind_of(obj: object) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        if issubclass(obj, enum.Enum):
            return "enum"
        if issubclass(obj, BaseException):
            return "exception"
        return "dataclass" if dataclasses.is_dataclass(obj) else "class"
    if callable(obj):
        return "function"
    return "data"


def bare_signature(obj: Callable[..., Any]) -> inspect.Signature | None:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None

    def bare(annotation: Any) -> Any:
        return _Annotation(annotation) if isinstance(annotation, str) else annotation

    parameters = [
        parameter.replace(annotation=bare(parameter.annotation))
        for name, parameter in signature.parameters.items()
        if name not in ("self", "cls")
    ]
    return signature.replace(
        parameters=parameters, return_annotation=bare(signature.return_annotation)
    )


def own_doc(obj: object) -> str | None:
    """The object's own docstring; generated dataclass and enum defaults do not count."""
    if inspect.isclass(obj):
        doc = obj.__dict__.get("__doc__")
        if not doc or (dataclasses.is_dataclass(obj) and doc.startswith(obj.__name__ + "(")):
            return None
        return inspect.cleandoc(doc)
    if inspect.ismodule(obj) or callable(obj):
        doc = getattr(obj, "__doc__", None)
        return inspect.cleandoc(doc) if doc else None
    return None


def data_docs(module: ModuleType) -> dict[str, str]:
    """``#:`` comments or a string literal after a top-level assignment, per name."""
    found: dict[str, str] = {}
    files: Iterable[Path]
    source = getattr(module, "__file__", None)
    if not source:
        return found
    path = Path(source)
    files = sorted(path.parent.rglob("*.py")) if path.name == "__init__.py" else [path]
    for file in files:
        text = file.read_text()
        lines = text.splitlines()
        body = ast.parse(text).body
        for index, node in enumerate(body):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not names:
                continue
            comments: list[str] = []
            line = node.lineno - 2
            while line >= 0 and lines[line].lstrip().startswith("#:"):
                comments.insert(0, lines[line].lstrip()[2:].strip())
                line -= 1
            doc = " ".join(comments)
            following = body[index + 1] if index + 1 < len(body) else None
            if (
                not doc
                and isinstance(following, ast.Expr)
                and isinstance(following.value, ast.Constant)
                and isinstance(following.value.value, str)
            ):
                doc = inspect.cleandoc(following.value.value)
            for name in names:
                if doc and name not in found:
                    found[name] = doc
    return found


def public_names(module: ModuleType) -> list[str]:
    names = getattr(module, "__all__", None)
    if names is not None:
        return list(names)
    return [
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and getattr(value, "__module__", None) == module.__name__
        and (inspect.isclass(value) or inspect.isfunction(value))
    ]


@dataclass
class Item:
    name: str
    module: str  # documented on this module's page
    kind: str
    obj: Any
    doc: str | None


@dataclass
class Reference:
    """The generated reference: pages, sidebar indexes and the docstring gaps."""

    pages: dict[str, tuple[str, str]] = field(default_factory=dict)  # name -> (title, md)
    index: dict[str, list[str]] = field(default_factory=dict)  # page -> anchors in order
    missing: list[str] = field(default_factory=list)
    missing_members: list[str] = field(default_factory=list)
    exempt: list[str] = field(default_factory=list)  # dunder metadata such as __version__
    names: int = 0


class _Builder:
    def __init__(self, site: Site) -> None:
        self.site = site
        self.package = importlib.import_module(site.package)
        self.items: dict[str, list[Item]] = {}
        self.targets: dict[str, tuple[str, str]] = {}  # symbol -> (page, anchor)
        self.ref = Reference()

    # Collection ---------------------------------------------------------------------
    def collect(self) -> None:
        documented: dict[int, tuple[str, str]] = {}
        for module_name, _ in self.site.reference_modules:
            module = importlib.import_module(f"{self.site.package}.{module_name}")
            docs = data_docs(module)
            items = []
            for name in public_names(module):
                obj = getattr(module, name)
                kind = kind_of(obj)
                doc = docs.get(name) if kind == "data" else own_doc(obj)
                items.append(Item(name, module_name, kind, obj, doc))
                self.targets.setdefault(name, (page_name(module_name), name))
                if kind != "data":
                    documented.setdefault(id(obj), (page_name(module_name), name))
                if doc is None:
                    self.ref.missing.append(f"{self.site.package}.{module_name}.{name}")
            self.items[module_name] = items
        top: list[Item] = []
        modules = {m for m, _ in self.site.reference_modules}
        for name in getattr(self.package, "__all__", []):
            obj = getattr(self.package, name)
            if inspect.ismodule(obj) and obj.__name__.rsplit(".", 1)[-1] in modules:
                continue
            if id(obj) in documented and kind_of(obj) != "data":
                self.targets.setdefault(name, documented[id(obj)])
                continue
            kind = kind_of(obj)
            doc = data_docs(self.package).get(name) if kind == "data" else own_doc(obj)
            if re.fullmatch(r"__\w+__", name):
                self.ref.exempt.append(f"{self.site.package}.{name}")
                continue  # Module metadata, not an API object.
            top.append(Item(name, "", kind, obj, doc))
            self.targets.setdefault(name, ("reference", name))
            if doc is None:
                self.ref.missing.append(f"{self.site.package}.{name}")
        self.top_only = top
        self.top_level = list(getattr(self.package, "__all__", []))
        self.documented = documented
        self.ref.names = sum(len(v) for v in self.items.values()) + len(top)

    # Docstring conversion ----------------------------------------------------------
    def link(self, symbol: str, page: str) -> str | None:
        parts = symbol.split(".")
        for size in (2, 1):
            key = ".".join(parts[-size:])
            if key in self.targets:
                target_page, anchor = self.targets[key]
                return f"#{anchor}" if target_page == page else f"{target_page}.md#{anchor}"
        return None

    def inline(self, text: str, page: str, *, table: bool = False) -> str:
        """Convert RST roles, links and escapes in prose outside code spans."""

        def role(match: re.Match[str]) -> str:
            kind, symbol = match[1], match[2].strip()
            label = symbol.split(".")[-1] if not symbol.startswith(".") else symbol[1:]
            if kind in ("meth", "func") and not label.endswith(")"):
                label += "()"
            if kind == "math":
                return f"`{symbol}`"
            target = self.link(symbol.rstrip("()"), page)
            return f"[`{label}`]({target})" if target else f"`{label}`"

        pieces = re.split(r"(``.+?``|`[^`\n]+`(?!_))", text)
        out = []
        for index, piece in enumerate(pieces):
            if index % 2:
                out.append("`" + piece[2:-2] + "`" if piece.startswith("``") else piece)
                continue
            piece = RST_LINK.sub(lambda m: f"[{m[1]}]({m[2]})", piece)
            piece = piece.replace("<", "&lt;")  # Docstring text is never raw HTML.
            out.append(piece.replace("|", "\\|") if table else piece)
        return ROLE.sub(role, "".join(out))

    def prose(self, text: str, page: str) -> str:
        """Prose with literal blocks (``::``), doctests and RST headings made Markdown."""
        lines = text.split("\n")
        out: list[str] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            nxt = lines[index + 1] if index + 1 < len(lines) else ""
            if line.strip() and re.fullmatch(r"\s*([-=~^])\1{2,}\s*", nxt):
                out.append(f"**{self.inline(line.strip(), page)}**")
                index += 2
                continue
            if line.lstrip().startswith(">>>"):
                block = []
                while index < len(lines) and lines[index].strip():
                    block.append(lines[index].strip())
                    index += 1
                out += ["", "```python", *block, "```", ""]
                continue
            literal = line.rstrip().endswith("::")
            if literal:
                head = line.rstrip()[:-2].rstrip()
                if head:
                    out.append(self.inline(head + ":", page))
            elif re.match(r"^\s{4,}\S", line) and (not out or not out[-1].strip()):
                literal = True  # An indented block after a blank line is code.
                index -= 1
            else:
                out.append(self.inline(line, page))
            if literal:
                index += 1
                while index < len(lines) and not lines[index].strip():
                    index += 1
                block = []
                while index < len(lines) and (
                    not lines[index].strip() or re.match(r"^\s", lines[index])
                ):
                    block.append(lines[index])
                    index += 1
                while block and not block[-1].strip():
                    block.pop()
                width = min(len(b) - len(b.lstrip()) for b in block if b.strip()) if block else 0
                out += ["", "```python", *(b[width:] for b in block), "```", ""]
                continue
            index += 1
        return "\n".join(out)

    def sections(self, doc: str) -> list[tuple[str, str]]:
        """Split a docstring into (title, text) in order; prose has the title ""."""
        found: list[tuple[str, list[str]]] = [("", [])]
        for line in doc.split("\n"):
            match = SECTION.match(line)
            if match:
                found.append((match[1], []))
            elif found[-1][0] and line.strip() and not line.startswith((" ", "\t")):
                found.append(("", [line]))  # An unindented line ends a section.
            else:
                found[-1][1].append(line)
        return [
            (title, inspect.cleandoc("\n".join(lines)) if title else "\n".join(lines).strip())
            for title, lines in found
            if "\n".join(lines).strip()
        ]

    def entries(self, text: str, page: str) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        for line in text.split("\n"):
            match = re.match(r"^(\*{0,2}[\w.]+)(?:\s*\(([^)]*)\))?\s*:\s*(.*)$", line)
            if match and not line.startswith((" ", "\t")):
                name = match[1] + (f" ({match[2]})" if match[2] else "")
                rows.append((name, match[3]))
            elif rows:
                rows[-1] = (rows[-1][0], (rows[-1][1] + " " + line.strip()).strip())
        return [(name, self.inline(desc, page, table=True)) for name, desc in rows]

    def render_doc(self, doc: str, page: str) -> list[str]:
        parts = []
        for title, text in self.sections(doc):
            if not title:
                parts.append(self.prose(text, page))
            elif title in ("Args", "Arguments", "Parameters", "Params", "Keyword Args"):
                rows = self.entries(text, page)
                table = ["| Parameter | Description |", "| --- | --- |"]
                table += [f"| `{name}` | {desc} |" for name, desc in rows]
                parts.append("\n".join(table))
            elif title in ("Raises", "Attributes"):
                label = "Raises" if title == "Raises" else "Attribute"
                rows = self.entries(text, page)
                table = [f"| {label} | Description |", "| --- | --- |"]
                table += [f"| `{name}` | {desc} |" for name, desc in rows]
                parts.append("\n".join(table))
            elif title in ("Returns", "Return", "Yields"):
                word = "Yields" if title == "Yields" else "Returns"
                parts.append(f"**{word}:** " + self.inline(" ".join(text.split()), page))
            elif title.startswith(("Note", "Warning")):
                kind = "note" if title.startswith("Note") else "caution"
                indented = "\n".join("    " + x for x in self.prose(text, page).split("\n"))
                parts.append(f'!!! {kind} "{title}"\n{indented}')
            else:
                parts.append(f"**{title}**\n\n" + self.prose(text, page))
        return parts

    # Pages -------------------------------------------------------------------------
    def signature_block(self, label: str, obj: Any, keyword: str) -> list[str]:
        from .graphite import format_signature

        signature = bare_signature(obj)
        if signature is None:
            return []
        text = format_signature(label, signature)
        return [f"```python\n{keyword} {text}\n```"]

    def members(self, cls: type, page: str) -> list[str]:
        parts: list[str] = []
        seen: set[str] = set()
        for klass in cls.__mro__:
            if not klass.__module__.startswith(self.site.package):
                continue
            for name, value in vars(klass).items():
                if name.startswith("_") or name in seen:
                    continue
                function: Callable[..., Any] | None
                if isinstance(value, staticmethod | classmethod):
                    function = value.__func__
                elif isinstance(value, property):
                    function = value.fget
                elif inspect.isfunction(value):
                    function = value
                else:
                    continue
                seen.add(name)
                anchor = f"{cls.__name__}.{name}"
                parts.append(f"### {cls.__name__}.{name} {{#{anchor}}}")
                if isinstance(value, property):
                    returns = bare_signature(function) if function else None
                    annotation = (
                        returns.return_annotation
                        if returns and returns.return_annotation is not inspect.Signature.empty
                        else None
                    )
                    shown = f": {annotation!r}" if annotation is not None else ""
                    parts.append(f"```python\nproperty {cls.__name__}.{name}{shown}\n```")
                elif function is not None:
                    keyword = "classmethod" if isinstance(value, classmethod) else "def"
                    if isinstance(value, staticmethod):
                        keyword = "staticmethod"
                    parts += self.signature_block(f"{cls.__name__}.{name}", function, keyword)
                doc = inspect.getdoc(value if isinstance(value, property) else function)
                if doc and doc == inspect.getdoc(property):
                    doc = None
                if doc:
                    parts += self.render_doc(doc, page)
                else:
                    self.ref.missing_members.append(
                        f"{self.site.package}.{cls.__module__.split('.')[1]}.{cls.__name__}.{name}"
                    )
        return parts

    def fields(self, cls: type) -> list[str]:
        if not dataclasses.is_dataclass(cls):
            return []
        rows = []
        for item in dataclasses.fields(cls):
            if item.name.startswith("_") or not item.init:
                continue
            annotation = (
                item.type
                if isinstance(item.type, str)
                else getattr(item.type, "__name__", str(item.type))
            )
            if item.default is not dataclasses.MISSING:
                default = f"`{item.default!r}`"
            elif item.default_factory is not dataclasses.MISSING:
                default = "factory"
            else:
                default = "required"
            rows.append(f"| `{item.name}` | `{annotation}` | {default} |")
        if not rows:
            return []
        return ["| Field | Type | Default |\n| --- | --- | --- |\n" + "\n".join(rows)]

    def item(self, item: Item, page: str) -> list[str]:
        self.ref.index.setdefault(page, []).append(item.name)
        parts = [f"## {item.name} {{#{item.name}}}"]
        obj = item.obj
        if item.kind in ("class", "dataclass", "exception", "enum"):
            if item.kind != "enum":
                parts += self.signature_block(item.name, obj, "class")
            else:
                parts.append(f"```python\nclass {item.name}(enum.Enum)\n```")
        elif item.kind == "function":
            parts += self.signature_block(item.name, obj, "def")
        elif item.kind == "data":
            parts.append(f"```python\n{item.name}: {type(obj).__name__} = {obj!r}\n```")
        elif item.kind == "module":
            parts.append(f"```python\nimport {obj.__name__}\n```")
        if item.doc:
            parts += self.render_doc(item.doc, page)
        else:
            parts.append("*No docstring yet.*")
        if item.kind == "enum":
            rows = [f"| `{m.name}` | `{m.value!r}` |" for m in obj]
            parts.append("| Member | Value |\n| --- | --- |\n" + "\n".join(rows))
        if item.kind in ("class", "dataclass"):
            parts += self.fields(obj)
        if item.kind in ("class", "dataclass", "enum", "exception"):
            parts += self.members(obj, page)
        return parts

    def module_page(self, module_name: str, label: str) -> None:
        page = page_name(module_name)
        module = importlib.import_module(f"{self.site.package}.{module_name}")
        doc = own_doc(module) or ""
        summary, _, rest = doc.partition("\n\n")
        parts = [
            f"# {label}",
            self.inline(" ".join(summary.split()), page)
            + f" Import from `{self.site.package}.{module_name}`.",
        ]
        if rest.strip():
            parts.append(self.prose(rest.strip(), page))
        names = self.items[module_name]
        exported = [i.name for i in names if i.name in self.top_level]
        if exported:
            parts.append(
                f"Also exported from `{self.site.package}`: "
                + ", ".join(f"[`{n}`](#{n})" for n in exported)
                + "."
            )
        for item in names:
            parts += self.item(item, page)
        self.ref.pages[page] = (label, "\n\n".join(parts) + "\n")

    def index_page(self) -> None:
        version = getattr(self.package, "__version__", "")
        parts = [
            "# Python reference",
            f"Every public name of `{self.site.package}` {version}, generated from the "
            "installed package's signatures and docstrings. One page per module.",
            "## Modules",
            "| Module | Contents |",
        ]
        rows = ["| --- | --- |"]
        for module_name, label in self.site.reference_modules:
            module = importlib.import_module(f"{self.site.package}.{module_name}")
            summary = (own_doc(module) or "").split("\n\n")[0]
            rows.append(
                f"| [{label}]({page_name(module_name)}.md) | "
                f"{self.inline(' '.join(summary.split()), 'reference')} |"
            )
        parts[-1] += "\n" + "\n".join(rows)
        parts.append("## Top-level exports")
        parts.append(
            f"`from {self.site.package} import …` reaches these names. Each links to its "
            "module page."
        )
        links = []
        for name in self.top_level:
            if re.fullmatch(r"__\w+__", name):
                continue
            target = self.link(name, "reference")
            if target:
                links.append(f"[`{name}`]({target})")
            else:
                links.append(f"`{name}`")
        parts.append(" · ".join(links))
        for item in self.top_only:
            parts += self.item(item, "reference")
        self.ref.pages["reference"] = ("Python reference", "\n\n".join(parts) + "\n")


def build_reference(site: Site) -> Reference:
    builder = _Builder(site)
    builder.collect()
    for module_name, label in site.reference_modules:
        builder.module_page(module_name, label)
    builder.index_page()
    # Undocumented members fail the strict build too; the report keeps them separate.
    builder.ref.missing.extend(builder.ref.missing_members)
    return builder.ref
