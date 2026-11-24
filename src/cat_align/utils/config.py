import json
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path


def read_encoder_config(config: str | Path | Mapping | None) -> Mapping:
    if config is None:
        config = json.loads(files("cat_align.models").joinpath("conformer.json").read_text("utf-8"))
    elif isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("The encoder configuration must be a JSON object.")
    encoder = config.get("encoder", config)
    if not isinstance(encoder, Mapping) or not isinstance(encoder.get("kwargs"), Mapping):
        raise ValueError("The encoder configuration must contain a 'kwargs' object.")
    return encoder
