"""Data management helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .runner import PROJECT_ROOT


class DataDownloadError(RuntimeError):
    """Raised when DVC cannot fetch required artifacts."""


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def download_data(
    required_paths: list[str | Path] | None = None,
    strict: bool = True,
) -> list[str]:
    """Fetch DVC-tracked data into the local workspace.

    The project uses separate DVC remotes for data and model artifacts. Pulling
    both remotes keeps train, export, and inference commands self-contained on a
    clean clone.
    """

    required_paths = required_paths or []
    missing_before = [
        str(path) for path in required_paths if not _resolve_path(path).exists()
    ]
    dvc_binary = shutil.which("dvc")
    if dvc_binary is None:
        message = "DVC executable was not found in the current environment."
        if strict or missing_before:
            raise DataDownloadError(message)
        return [message]

    warnings = []
    for command in ([dvc_binary, "pull"], [dvc_binary, "pull", "--remote", "models"]):
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            continue
        output = "\n".join(
            part.strip()
            for part in (result.stdout, result.stderr)
            if part and part.strip()
        )
        warnings.append(
            f"{' '.join(command[1:])} failed with exit code {result.returncode}: {output}"
        )

    missing_after = [
        str(path) for path in required_paths if not _resolve_path(path).exists()
    ]
    if warnings and (strict or missing_after):
        missing_message = (
            f" Missing required paths after DVC pull: {missing_after}."
            if missing_after
            else ""
        )
        raise DataDownloadError("\n".join(warnings) + missing_message)
    return warnings
