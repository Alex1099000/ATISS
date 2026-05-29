"""Configuration helpers built on top of Hydra Compose API."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


@contextmanager
def hydra_config(
    config_name: str, overrides: list[str] | None = None
) -> Iterator[DictConfig]:
    """Load a Hydra config without making every command a Hydra entrypoint."""

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        yield compose(config_name=config_name, overrides=overrides or [])


def as_container(config: DictConfig) -> dict:
    """Return a plain Python dictionary with interpolations resolved."""

    return OmegaConf.to_container(config, resolve=True)


def write_yaml(config: dict, path: Path) -> None:
    """Write a dictionary as YAML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config), f=path)
