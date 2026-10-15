"""Run, check and record the programs published on documentation pages.

The rules (the Welt next-pass rules in welt-api ``docs/sdk-docs-handoff.md``):

- A ``python`` fence with ``file="name.py"`` is a complete, runnable program. It runs
  in a fresh process with a timeout, from an empty working directory, with network
  access disabled, so it can read only the package's bundled datasets.
- Every other ``python`` fence on that page must be a verbatim, contiguous excerpt of
  one of the page's programs, so fragments cannot drift from the executed code.
- A program with ``requires="…"`` needs an outside service (an account or a key), so
  the checker never runs it; its excerpts are still checked, it cannot have an output
  panel, and a test of its own should run it against a fake service.
- An ``output`` panel names its program with ``from="name.py"`` (optionally a
  ``lines="a-b"`` slice of its standard output). ``record`` writes the program's output
  into the panel; ``check`` reruns the program and compares, exactly by default or with
  ``tol``/``rtol`` numeric tolerances for sampling-based pages. ``numbers="vary"`` is
  for MCMC pages whose draws differ between platforms (the recording Mac and Linux
  CI): the text, and how many numbers it holds, must still match exactly, but the
  numbers themselves are not compared.
"""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .config import Site
from .graphite import DocsError, Fence, fences, line_numbers

PYTHON = ("python", "py")
# Fence labels a page may use; anything else (py3, pycon, an empty label) could hide
# Python from the excerpt check, so it is refused.
LANGUAGES = frozenset({*PYTHON, "sh", "bash", "shell", "text", "output"})
# Runs the program as __main__ after disabling sockets: documentation examples use
# bundled data only and never reach the network.
BOOTSTRAP = """
import runpy, socket, sys
def _blocked(*args, **kwargs):
    raise OSError("network access is disabled while running documentation examples")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
# Thread caps reach every program; unset ones default to 2 so a docs check cannot
# saturate the machine. PYTENSOR_FLAGS passes through unchanged when set.
THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PASS_THROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "PYTENSOR_FLAGS")
# A program is slow when marked slow="true" or given a timeout above this many seconds.
SLOW_TIMEOUT = 60
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class Program:
    page: Path
    name: str
    code: str
    timeout: int
    slow: bool = False
    requires: str | None = None  # an outside service the program needs; never run here


@dataclass(frozen=True)
class Run:
    program: Program
    returncode: int
    stdout: str
    stderr: str
    seconds: float


def page_paths(site: Site, only: list[str] | None = None) -> list[Path]:
    paths = sorted(site.pages_dir.glob("*.md"))
    if only:
        wanted = {Path(name).stem for name in only}
        paths = [p for p in paths if p.stem in wanted]
        if {p.stem for p in paths} != wanted:
            raise DocsError(f"Unknown page(s): {sorted(wanted - {p.stem for p in paths})}")
    return paths


def programs_of(path: Path, site: Site) -> list[Program]:
    found = []
    for fence in fences(path.read_text()):
        if fence.language in PYTHON and "file" in fence.attributes:
            name = fence.attributes["file"]
            if not re.fullmatch(r"[a-z0-9_]+\.py", name):
                raise DocsError(f"{path.name}: program names are snake_case .py files: {name!r}")
            timeout = int(fence.attributes.get("timeout", site.example_timeout))
            slow = fence.attributes.get("slow") == "true" or (
                "timeout" in fence.attributes and timeout > SLOW_TIMEOUT
            )
            requires = fence.attributes.get("requires")
            found.append(Program(path, name, fence.code, timeout, slow, requires))
    return found


def all_programs(site: Site) -> dict[str, Program]:
    """Every program on the site, by file name (names are unique: they share downloads/)."""
    programs: dict[str, Program] = {}
    for path in page_paths(site):
        for program in programs_of(path, site):
            if program.name in programs:
                other = programs[program.name].page.name
                raise DocsError(f"Program {program.name} appears on {other} and {path.name}")
            programs[program.name] = program
    return programs


def excerpt_problems(path: Path, site: Site) -> list[str]:
    """Python fences that are not a contiguous excerpt of one of the page's programs."""
    text = path.read_text()
    programs = [p.code.rstrip("\n").split("\n") for p in programs_of(path, site)]
    names = {p.name for p in programs_of(path, site)}
    unrun = {p.name for p in programs_of(path, site) if p.requires}
    problems = []
    for fence in fences(text):
        line = text.count("\n", 0, fence.start) + 1
        if not fence.info.strip() or fence.language not in LANGUAGES:
            label = fence.info.strip() or "no label"
            problems.append(
                f"{path.name}:{line}: fence label {label!r} is not one of "
                f"{', '.join(sorted(LANGUAGES))}"
            )
            continue
        if fence.language == "output":
            source = fence.attributes.get("from")
            if source is None:
                problems.append(f'{path.name}:{line}: output panel needs from="<program>.py"')
            elif source not in names:
                problems.append(f"{path.name}:{line}: output from unknown program {source!r}")
            elif source in unrun:
                problems.append(
                    f"{path.name}:{line}: {source} has requires=… and is never run, so it "
                    "cannot have a recorded output panel; describe its output in prose"
                )
            continue
        if fence.language not in PYTHON or "file" in fence.attributes:
            continue
        lines = fence.code.rstrip("\n").split("\n")
        if not any(
            program[i : i + len(lines)] == lines
            for program in programs
            for i in range(len(program) - len(lines) + 1)
        ):
            problems.append(
                f"{path.name}:{line}: python fence is not a verbatim, contiguous excerpt of a "
                'file="…" program on this page'
            )
    return problems


def run(program: Program) -> Run:
    """Run one program in a fresh interpreter, from an empty directory, without network."""
    with tempfile.TemporaryDirectory(prefix="docs-example-") as folder:
        script = Path(folder) / program.name
        script.write_text(program.code)
        env = program_env()
        started = time.perf_counter()
        try:
            done = subprocess.run(
                [sys.executable, "-s", "-P", "-c", BOOTSTRAP, str(script)],
                cwd=folder,
                env=env,
                capture_output=True,
                text=True,
                timeout=program.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Run(program, -1, "", f"timed out after {program.timeout} s", program.timeout)
        return Run(
            program, done.returncode, done.stdout, done.stderr, time.perf_counter() - started
        )


def program_env() -> dict[str, str]:
    """The environment of a program: a short allowlist plus capped thread pools."""
    env = {key: value for key, value in os.environ.items() if key in PASS_THROUGH}
    env.update({key: os.environ.get(key) or "2" for key in THREAD_CAPS})
    env.update(PYTHONHASHSEED="0", MPLBACKEND="Agg", PYTHONWARNINGS="ignore")
    return env


def run_all(programs: list[Program], workers: int = 1) -> list[Run]:
    """Run programs one at a time by default; ``workers`` > 1 runs them in parallel (CI)."""
    if workers <= 1:
        return [run(program) for program in programs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, programs))


def slow_enabled() -> bool:
    return os.environ.get("DOCS_SLOW") == "1"


def clean(text: str) -> str:
    """Recorded output: trailing spaces and trailing blank lines removed."""
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def select(stdout: str, spec: str | None) -> str:
    text = clean(stdout)
    if not spec:
        return text
    lines = text.rstrip("\n").split("\n")
    chosen = sorted(line_numbers(spec, len(lines)))
    return "\n".join(lines[i - 1] for i in chosen) + "\n"


def same(
    expected: str,
    actual: str,
    tol: float = 0.0,
    rtol: float = 0.0,
    *,
    numbers_vary: bool = False,
) -> bool:
    """Exact match, or numbers within ``tol + rtol * |expected|`` and the same text.

    With ``numbers_vary`` only the text and the count of numbers must match.
    """
    if expected == actual:
        return True
    if not tol and not rtol and not numbers_vary:
        return False
    left, right = NUMBER.split(expected), NUMBER.split(actual)
    numbers_left, numbers_right = NUMBER.findall(expected), NUMBER.findall(actual)
    if len(numbers_left) != len(numbers_right):
        return False
    if [" ".join(x.split()) for x in left] != [" ".join(x.split()) for x in right]:
        return False
    if numbers_vary:
        return True
    return all(
        abs(float(a) - float(b)) <= tol + rtol * abs(float(a))
        for a, b in zip(numbers_left, numbers_right, strict=True)
    )


def diff(expected: str, actual: str, label: str) -> str:
    lines = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        f"{label} (recorded)",
        f"{label} (now)",
        n=1,
        lineterm="",
    )
    return "\n".join(list(lines)[:40])


def output_panels(text: str) -> list[Fence]:
    return [f for f in fences(text) if f.language == "output" and "from" in f.attributes]


def replace_fence(text: str, fence: Fence, content: str) -> str:
    body = "".join(fence.indent + line if line else line for line in content.splitlines(True))
    head = f"{fence.indent}{fence.marker}{fence.info}\n"
    tail = f"{fence.indent}{fence.marker}"
    return text[: fence.start] + head + body + tail + text[fence.end :]


@dataclass
class Report:
    runs: list[Run]
    problems: list[str]
    recorded: list[str]
    # Programs not run: they need an outside service (requires=…) or are slow.
    skipped: list[Program] = field(default_factory=list)


def check(
    site: Site,
    only: list[str] | None = None,
    *,
    record: bool = False,
    jobs: int = 1,
    slow: bool | None = None,
) -> Report:
    """Excerpts, program runs and output panels for the selected pages.

    Slow programs (and their output panels) are skipped unless ``slow`` is true or,
    when it is None, ``DOCS_SLOW=1`` is set; excerpts are always checked.
    """
    include_slow = slow_enabled() if slow is None else slow
    paths = page_paths(site, only)
    all_programs(site)  # Names must be unique across the whole site.
    problems: list[str] = []
    for path in paths:
        problems += excerpt_problems(path, site)
    programs = [program for path in paths for program in programs_of(path, site)]
    skipped = [p for p in programs if p.requires or (p.slow and not include_slow)]
    runs = run_all([p for p in programs if p not in skipped], jobs)
    by_name = {r.program.name: r for r in runs}
    for item in runs:
        if item.returncode != 0:
            tail = "\n".join(item.stderr.strip().splitlines()[-12:])
            problems.append(
                f"{item.program.page.name}: {item.program.name} failed "
                f"(exit {item.returncode}):\n{tail}"
            )
    recorded: list[str] = []
    for path in paths:
        text = path.read_text()
        for fence in reversed(output_panels(text)):
            source = fence.attributes["from"]
            result = by_name.get(source)
            if result is None or result.returncode != 0:
                continue
            actual = select(result.stdout, fence.attributes.get("lines"))
            if record:
                if fence.code != actual:
                    text = replace_fence(text, fence, actual)
                    recorded.append(f"{path.name}: {source}")
                continue
            tol = float(fence.attributes.get("tol", 0))
            rtol = float(fence.attributes.get("rtol", 0))
            vary = fence.attributes.get("numbers") == "vary"
            if not same(fence.code, actual, tol, rtol, numbers_vary=vary):
                rule = "numbers vary" if vary else f"tol={tol}, rtol={rtol}"
                problems.append(
                    f"{path.name}: output of {source} changed "
                    f"({rule}); run `make docs-record` after review\n"
                    + diff(fence.code, actual, source)
                )
        if record and text != path.read_text():
            path.write_text(text)
    return Report(runs, problems, recorded, skipped)
