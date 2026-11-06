from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load experiment configs. Install it with `pip install pyyaml`."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must define a mapping at the top level")
    return data
