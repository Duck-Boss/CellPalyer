from typing import Any, Dict, Optional

from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline.data import make_dataloader
from data_pipeline.runtime import forward_batch


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["x"] = batch["x"].to(device, non_blocking=True)
    if "x_counts" in batch:
        out["x_counts"] = batch["x_counts"].to(device, non_blocking=True)
    if "x_mask" in batch:
        out["x_mask"] = batch["x_mask"].to(device, non_blocking=True)
    return out


def collect_embeddings(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    include_expression: bool = True,
    include_indices: bool = True,
) -> Dict[str, Any]:
    embeddings = []
    encoder_embeddings = []
    recons = []
    targets = []
    indices = []
    labels = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            outputs = forward_batch(model, batch, device)
            z_q = outputs["z_q"].reshape(batch["x"].shape[0], -1).cpu().numpy()
            z_e = outputs.get("z_e", outputs["z_q"]).reshape(batch["x"].shape[0], -1).cpu().numpy()
            embeddings.append(z_q)
            encoder_embeddings.append(z_e)
            if include_expression:
                recons.append(outputs["x_recon"].cpu().numpy())
                targets.append(batch["x"].cpu().numpy())
            if include_indices:
                indices.append(outputs["indices"].cpu().numpy())
            for key in [
                "cell_type",
                "coarse_cell_type",
                "tissue",
                "disease",
                "assay",
                "group_id",
                "dataset_id",
                "donor_id",
                "sample_id",
                "batch_id",
                "technical_block",
                "author_cell_type",
                "development_stage",
                "age",
                "age_parse_source",
                "sex",
                "ethnicity",
                "suspension_type",
            ]:
                if key in batch:
                    labels[key].extend(batch[key])
    result = {
        "embeddings": np.concatenate(embeddings, axis=0),
        "embeddings_z_q": np.concatenate(embeddings, axis=0),
        "embeddings_z_e": np.concatenate(encoder_embeddings, axis=0),
        "labels": dict(labels),
    }
    if include_expression:
        result["recons"] = np.concatenate(recons, axis=0)
        result["targets"] = np.concatenate(targets, axis=0)
    if include_indices:
        result["indices"] = np.concatenate(indices, axis=0)
    return result


def collect_embeddings_sampled(
    model: torch.nn.Module,
    dataset: Dataset,
    config: Dict[str, Any],
    device: torch.device,
    max_samples: Optional[int],
    shuffle: bool,
    sample_seed: Optional[int] = None,
    include_expression: bool = True,
    include_indices: bool = True,
) -> Dict[str, Any]:
    if max_samples is not None and len(dataset) > int(max_samples):
        rng = np.random.default_rng(int(config["seed"]) if sample_seed is None else int(sample_seed))
        indices = np.sort(
            rng.choice(len(dataset), size=int(max_samples), replace=False)
        ).tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    loader = make_dataloader(dataset, int(config["training"]["batch_size"]), False, int(config["training"]["num_workers"]))
    return collect_embeddings(
        model,
        loader,
        device,
        include_expression=include_expression,
        include_indices=include_indices,
    )


__all__ = ["collect_embeddings_sampled"]
