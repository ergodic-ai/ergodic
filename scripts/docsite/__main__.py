"""Command line: ``python -m scripts.docsite <command>`` from the repository root."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import ROOT, load_site
from .graphite import DocsError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.docsite", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="render the versioned site")
    build.add_argument("--output", type=Path, default=ROOT / "site")
    build.add_argument(
        "--allow-undocumented",
        action="store_true",
        help="report public names without a docstring instead of failing",
    )
    build.add_argument("--report", type=Path, help="write the docstring coverage report here")
    check = commands.add_parser("check", help="run programs, check excerpts and outputs")
    check.add_argument("pages", nargs="*", help="page names (default: every page)")
    record = commands.add_parser("record", help="rerun programs and record their output")
    record.add_argument("pages", nargs="*", help="page names (default: every page)")
    links = commands.add_parser("links", help="check internal links of a built site")
    links.add_argument("output", type=Path, nargs="?", default=ROOT / "site")
    notebooks = commands.add_parser("notebooks", help="execute every notebook (release CI)")
    notebooks.add_argument("--timeout", type=int, default=1800)
    notebooks.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    site = load_site()
    started = time.perf_counter()
    try:
        if args.command == "build":
            from .build import build as run_build

            summary = run_build(
                site, args.output, strict_reference=not args.allow_undocumented, report=args.report
            )
            print(
                f"Built {site.name} {summary['version']} docs: {summary['pages']} pages, "
                f"{summary['notebooks']} notebook previews, {summary['programs']} programs "
                f"in {summary['seconds']} s -> {summary['output']}"
            )
            print(
                f"Reference: {summary['undocumented']} of {summary['public_names']} public "
                "names have no docstring"
            )
            return 0
        if args.command in ("check", "record"):
            from .examples import check as run_check

            report = run_check(site, args.pages or None, record=args.command == "record")
            for item in report.runs:
                status = "ok" if item.returncode == 0 else "FAILED"
                where = f"{item.program.page.name}: {item.program.name}"
                print(f"{status:6} {item.seconds:6.1f} s  {where}")
            for line in report.recorded:
                print(f"recorded {line}")
            for problem in report.problems:
                print(problem, file=sys.stderr)
            elapsed = time.perf_counter() - started
            print(
                f"{len(report.runs)} program(s), {len(report.problems)} problem(s), {elapsed:.1f} s"
            )
            return 1 if report.problems else 0
        if args.command == "links":
            from .links import check_links

            problems = check_links(args.output)
            for problem in problems:
                print(problem, file=sys.stderr)
            print(f"Links: {len(problems)} problem(s) in {args.output}")
            return 1 if problems else 0
        if args.command == "notebooks":
            from .notebooks import execute

            runs = execute(site, args.timeout, args.workers)
            failed = [r for r in runs if r.returncode != 0]
            for item in runs:
                status = "ok" if item.returncode == 0 else "FAILED"
                print(f"{status:6} {item.seconds:7.1f} s  {item.program.name}")
            for item in failed:
                tail = "\n".join(item.stderr.strip().splitlines()[-12:])
                print(f"{item.program.name}:\n{tail}", file=sys.stderr)
            print(f"{len(runs)} notebook(s), {len(failed)} failed")
            return 1 if failed else 0
    except DocsError as error:
        print(f"docs: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
