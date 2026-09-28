"""Load an NKE head checkpoint with its separately licensed backbone."""

from __future__ import annotations

import json
from pathlib import Path

from safetensors.torch import load_file

from .model import SnapshotDecisionModel


def load_checkpoint(directory: Path) -> SnapshotDecisionModel:
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = SnapshotDecisionModel(config["backbone"])
    weights = load_file(str(directory / "nke_heads.safetensors"))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise ValueError(f"checkpoint incompatible: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model
