#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import shutil


NUMBER_RE = r"-?(?:\d+(?:\.\d*)?|\.\d+)"
FILE_RE = re.compile(
    rf"L(?P<Lx>\d+)x(?P<Ly>\d+)Nup(?P<Nup>\d+)Ndn(?P<Ndn>\d+)"
    rf"U(?P<U>{NUMBER_RE})V(?P<V>{NUMBER_RE})"
    rf"tp(?P<tp>{NUMBER_RE})f(?P<f>[^Q]+)Q(?P<Q>[^_]+)_(?P<kind>G[hp])\.dat$"
)


@dataclass
class CheckResult:
    path: Path
    ok: bool
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether U and V encoded in .dat filenames exactly match "
            "the U and V values written inside the file header."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path(".")],
        help="Files or directories to scan. Directories are searched recursively.",
    )
    parser.add_argument(
        "--pattern",
        default="*.dat",
        help="Glob used when scanning directories recursively.",
    )
    parser.add_argument(
        "--move-bad-dir",
        type=Path,
        default=None,
        help=(
            "If set, create this directory and move mismatched files into it. "
            "Relative paths are resolved from the current working directory."
        ),
    )
    return parser.parse_args()


def parse_decimal(text: str, *, source: str) -> Decimal:
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Could not parse decimal from {source!r}: {text!r}") from exc


def iter_files(paths: list[Path], pattern: str) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(candidate for candidate in path.rglob(pattern) if candidate.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    return files


def parse_header_values(path: Path) -> tuple[Decimal, Decimal]:
    u_value: Decimal | None = None
    v_value: Decimal | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            fields = raw_line.split()
            if len(fields) < 2:
                continue

            key = fields[0]
            if key == "U":
                u_value = parse_decimal(fields[1], source=f"{path} header U")
            elif key == "V":
                v_value = parse_decimal(fields[1], source=f"{path} header V")

            if u_value is not None and v_value is not None:
                return u_value, v_value

    missing = []
    if u_value is None:
        missing.append("U")
    if v_value is None:
        missing.append("V")
    raise ValueError(f"Missing {' and '.join(missing)} in header: {path}")


def check_file(path: Path) -> CheckResult:
    match = FILE_RE.match(path.name)
    if match is None:
        return CheckResult(path=path, ok=False, message="filename does not match expected pattern")

    filename_u = parse_decimal(match.group("U"), source=f"{path.name} filename U")
    filename_v = parse_decimal(match.group("V"), source=f"{path.name} filename V")
    header_u, header_v = parse_header_values(path)

    mismatches: list[str] = []
    if filename_u != header_u:
        mismatches.append(f"U filename={filename_u} header={header_u}")
    if filename_v != header_v:
        mismatches.append(f"V filename={filename_v} header={header_v}")

    if mismatches:
        return CheckResult(path=path, ok=False, message="; ".join(mismatches))

    return CheckResult(path=path, ok=True, message="U/V match")


def move_bad_file(path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / path.name

    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    shutil.move(str(path), str(destination))
    return destination


def main() -> int:
    args = parse_args()
    try:
        files = iter_files(args.paths, args.pattern)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not files:
        print("No files found.")
        return 0

    results: list[CheckResult] = []
    for path in files:
        try:
            results.append(check_file(path))
        except ValueError as exc:
            results.append(CheckResult(path=path, ok=False, message=str(exc)))

    ok_count = sum(result.ok for result in results)
    bad_results = [result for result in results if not result.ok]

    moved_count = 0
    for result in bad_results:
        print(f"BAD  {result.path}: {result.message}")
        if args.move_bad_dir is not None and result.path.exists():
            try:
                moved_to = move_bad_file(result.path, args.move_bad_dir)
            except FileExistsError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            print(f"MOVED {result.path} -> {moved_to}")
            moved_count += 1

    if args.move_bad_dir is not None:
        print(
            f"Checked {len(results)} file(s): {ok_count} OK, {len(bad_results)} BAD, "
            f"{moved_count} moved to {args.move_bad_dir}"
        )
    else:
        print(f"Checked {len(results)} file(s): {ok_count} OK, {len(bad_results)} BAD")
    return 1 if bad_results else 0


if __name__ == "__main__":
    raise SystemExit(main())
