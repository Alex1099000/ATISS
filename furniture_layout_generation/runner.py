"""Command execution helpers for the project CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_python(script: str, args: list[str]) -> None:
    """Run a repository script with the current Python interpreter."""

    command = [sys.executable, str(PROJECT_ROOT / script), *map(str, args)]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_command(command: list[str]) -> None:
    """Run a subprocess command from the repository root."""

    subprocess.run([*map(str, command)], cwd=PROJECT_ROOT, check=True)


def git_commit_id() -> str:
    """Return the current git commit hash or 'unknown' outside git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"
