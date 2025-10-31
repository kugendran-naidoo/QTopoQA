#!/usr/bin/env python3
"""Scan repository files for sensitive strings (API keys, tokens, secrets).

Run from repo root:
  python scan_sensitive.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import os
from typing import Iterable, Iterator, List, Pattern, Tuple


DEFAULT_PATTERNS: List[Pattern[str]] = [
    re.compile(r"openai", re.IGNORECASE),
    re.compile(r"anthropic", re.IGNORECASE),
    re.compile(r"huggingface", re.IGNORECASE),
    re.compile(r"azure[_-]?(openai)?", re.IGNORECASE),
    re.compile(r"\baws\b", re.IGNORECASE),
    re.compile(r"\bgcp\b|\bgoogle\b", re.IGNORECASE),
    re.compile(r"cohere", re.IGNORECASE),
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"access[_-]?key", re.IGNORECASE),
    re.compile(r"client[_-]?secret", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"bearer", re.IGNORECASE),
]

SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".venv", "venv", "node_modules", "ml_runs"}
SKIP_EXTENSIONS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".png", ".jpg", ".jpeg", ".gif", ".bmp"}
MAX_BYTES = 2_000_000  # skip files larger than 2 MB


def iter_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        current = Path(dirpath)
        for filename in filenames:
            path = current / filename
            if path.suffix.lower() in SKIP_EXTENSIONS:
                continue
            try:
                if path.stat().st_size > MAX_BYTES:
                    continue
            except OSError:
                continue
            yield path


def scan_file(path: Path, patterns: List[Pattern[str]]) -> List[Tuple[int, str, Pattern[str]]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    findings: List[Tuple[int, str, Pattern[str]]] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines, 1):
        for pattern in patterns:
            if pattern.search(line):
                findings.append((idx, line.strip(), pattern))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan repository for sensitive strings.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root to scan (default: script directory).",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        help="Additional regex pattern to include (may be passed multiple times).",
    )
    args = parser.parse_args()

    patterns = list(DEFAULT_PATTERNS)
    if args.pattern:
        for raw in args.pattern:
            try:
                patterns.append(re.compile(raw, re.IGNORECASE))
            except re.error as exc:
                parser.error(f"Invalid regex '{raw}': {exc}")

    root = args.root.resolve()
    if not root.exists():
        parser.error(f"Root path does not exist: {root}")

    findings_total = 0
    for file_path in iter_files(root):
        matches = scan_file(file_path, patterns)
        if not matches:
            continue
        findings_total += len(matches)
        print(f"\n{file_path.relative_to(root)}")
        for idx, line, pattern in matches:
            sample = line if len(line) <= 160 else line[:157] + "..."
            print(f"  L{idx}: {pattern.pattern} -> {sample}")

    if findings_total == 0:
        print("No matches found.")
    else:
        print(f"\nTotal matches: {findings_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
