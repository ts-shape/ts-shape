#!/usr/bin/env python3
"""
Manage requirements with pip-tools.

Usage (from repo root):
  # Ensure pip-tools is installed in your active environment
  python -m pip install --upgrade pip-tools

  # Compile exact pins of the core dependencies (pyproject.toml) -> requirements.txt
  python scripts/requirements.py compile

  # Upgrade all pins to the newest compatible versions
  python scripts/requirements.py upgrade

  # Sync the current environment to requirements.txt (adds/removes packages)
  python scripts/requirements.py sync

Notes:
  - Direct dependencies live in pyproject.toml ([project.dependencies]);
    the same command runs in CI for Renovate updates
  - Writes fully pinned requirements.txt
  - pip-sync will uninstall anything not listed in requirements.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQ_TXT = REPO_ROOT / "requirements.txt"


def run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def check_piptools() -> None:
    try:
        import piptools  # noqa: F401
    except ImportError:
        print("[!] pip-tools is not installed. Install it first:")
        print("    python -m pip install --upgrade pip-tools")
        sys.exit(1)


def do_compile(upgrade: bool = False) -> None:
    check_piptools()
    args = [
        sys.executable,
        "-m",
        "piptools",
        "compile",
        "--strip-extras",
        str(PYPROJECT),
        "-o",
        str(REQ_TXT),
    ]
    if upgrade:
        args.insert(4, "--upgrade")
    code = run(args)
    sys.exit(code)


def do_sync() -> None:
    check_piptools()
    if not REQ_TXT.exists():
        print(f"[!] {REQ_TXT} not found. Compile it first:")
        print("    python scripts/requirements.py compile")
        sys.exit(1)
    code = run([sys.executable, "-m", "piptools", "sync", str(REQ_TXT)])
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage requirements with pip-tools")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compile", help="Compile pyproject.toml -> requirements.txt")
    sub.add_parser("upgrade", help="Compile with --upgrade to refresh pins")
    sub.add_parser("sync", help="Sync env to requirements.txt (adds/removes packages)")

    ns = parser.parse_args()
    if ns.cmd == "compile":
        do_compile(upgrade=False)
    elif ns.cmd == "upgrade":
        do_compile(upgrade=True)
    elif ns.cmd == "sync":
        do_sync()
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
