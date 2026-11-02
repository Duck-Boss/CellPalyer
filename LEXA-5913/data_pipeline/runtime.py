from pathlib import Path
from typing import Any, Dict, Optional

import torch

from data_pipeline.clean_common import ensure_dir, load_config, now_run_id, progress, set_seed, write_json


def pick_device(name: str) -> torch.device:
    name = str(name or "auto")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cpu")


def make_output_dirs(config: Dict[str, Any], run_id: str) -> Dict[str, Path]:
    root = ensure_dir(Path(config["output"]["root"]) / run_id)
    figures = ensure_dir(root / "figures")
    checkpoints = ensure_dir(root / "checkpoints")
    return {"root": root, "figures": figures, "checkpoints": checkpoints}


def init_wandb(config: Dict[str, Any], run_id: str):
    import wandb

    logging_cfg = config["logging"]
    return wandb.init(
        project=logging_cfg["project"],
        entity=logging_cfg.get("entity"),
        mode=logging_cfg.get("mode", "online"),
        name=run_id,
        config=config,
        reinit=True,
    )


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["x"] = batch["x"].to(device, non_blocking=True)
    if "x_counts" in batch:
        out["x_counts"] = batch["x_counts"].to(device, non_blocking=True)
    if "x_mask" in batch:
        out["x_mask"] = batch["x_mask"].to(device, non_blocking=True)
    return out


def categorical_indices_for_model(batch: Dict[str, Any], model: torch.nn.Module, device: torch.device) -> Dict[str, torch.Tensor]:
    categories = getattr(model, "nuisance_categories", {})
    encoded: Dict[str, torch.Tensor] = {}
    for field, values in categories.items():
        if field not in batch:
            continue
        lookup = {str(value): idx for idx, value in enumerate(values)}
        unknown = lookup.get("<unknown>", 0)
        encoded[field] = torch.tensor(
            [lookup.get(str(value), unknown) for value in batch[field]],
            dtype=torch.long,
            device=device,
        )
    return encoded


def forward_batch(model: torch.nn.Module, batch: Dict[str, Any], device: torch.device, **kwargs):
    categorical = categorical_indices_for_model(batch, model, device)
    return model(
        batch["x"],
        mask=batch.get("x_mask"),
        count_target=batch.get("x_counts"),
        categorical_indices=categorical,
        **kwargs,
    )


__all__ = [
    "batch_to_device",
    "categorical_indices_for_model",
    "forward_batch",
    "init_wandb",
    "load_config",
    "make_output_dirs",
    "now_run_id",
    "pick_device",
    "progress",
    "set_seed",
    "write_json",
]
