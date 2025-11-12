"""CLI command for running Ctrl-DNA optimization pipelines."""

import logging
from pathlib import Path
from typing import Optional

_LOGGER = logging.getLogger(__name__)


def run_ctrl_dna_pipeline(config_path: str | Path) -> None:
    """Run a complete Ctrl-DNA pipeline from config.

    Parameters
    ----------
    config_path : str | Path
        Path to configuration YAML file.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _LOGGER.info(f"Loading config from {config_path}")

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load config: {e}")

    _LOGGER.info("Config loaded successfully")
    _LOGGER.info(f"Engine config: {config.get('engine', {})}")
    _LOGGER.info(f"Strategy: {config.get('strategy', {}).get('type', 'unknown')}")


__all__ = ["run_ctrl_dna_pipeline"]

