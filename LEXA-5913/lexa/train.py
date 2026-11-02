import argparse
import fnmatch
import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW

from data_pipeline.bundle import load_data_bundle, warm_csr_caches
from data_pipeline.data import H5CSRDataset, make_dataloader
from data_pipeline.runtime import (
    batch_to_device,
    forward_batch,
    init_wandb,
    load_config,
    make_output_dirs,
    now_run_id,
    pick_device,
    progress,
    set_seed,
    write_json,
)
from data_pipeline.clean_common import apply_config_overrides
from lexa.lexa_model import build_lexa_model


def configure_torch_performance(config, device):
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(config["training"].get("allow_tf32", True))
    torch.backends.cudnn.benchmark = bool(config["training"].get("cudnn_benchmark", True))
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(config["training"].get("matmul_precision", "high")))


def autocast_dtype(config, device):
    if device.type != "cuda":
        return torch.float32
    requested = str(config["training"].get("amp_dtype", "float16")).lower()
    if requested in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_optimizer(model, config, device):
    kwargs = {
        "lr": float(config["training"]["lr"]),
        "weight_decay": float(config["training"]["weight_decay"]),
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable parameters remain after applying training.trainable_parameter_patterns")
    if device.type == "cuda" and bool(config["training"].get("fused_adamw", True)):
        try:
            return AdamW(parameters, fused=True, **kwargs)
        except TypeError:
            pass
    return AdamW(parameters, **kwargs)


def configure_trainable_parameters(model, config):
    patterns = config.get("training", {}).get("trainable_parameter_patterns")
    if not patterns:
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    patterns = [str(pattern) for pattern in patterns]
    selected = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(fnmatch.fnmatch(name, pattern) for pattern in patterns))
        if parameter.requires_grad:
            selected.append(name)
    if not selected:
        raise ValueError(f"No model parameters matched training.trainable_parameter_patterns={patterns}")
    return selected


def _warmup_scale(config, epoch_idx: int) -> float:
    warmup_epochs = int(config.get("training", {}).get("vq_warmup_epochs") or 0)
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, max(1, int(epoch_idx)) / float(warmup_epochs))


def apply_loss_weight_schedule(model, config, epoch_idx: int) -> float:
    scale = _warmup_scale(config, epoch_idx)
    model_cfg = config.get("model", {})
    scheduled = [
        "vq_aux_loss_weight",
        "anchor_loss_weight",
        "center_norm_weight",
        "intra_codebook_div_weight",
        "residual_center_weight",
        "codebook_sep_weight",
        "conditional_code_usage_weight",
    ]
    for name in scheduled:
        if hasattr(model, name) and name in model_cfg:
            setattr(model, name, float(model_cfg[name]) * scale)
    return scale


def metric_value(metrics, metric_name: str) -> float:
    if metric_name in metrics:
        return float(metrics[metric_name])
    aliases = {
        "loss": "loss",
        "val_loss": "loss",
        "recon_loss": "recon_loss",
        "val_recon_loss": "recon_loss",
    }
    key = aliases.get(metric_name)
    if key and key in metrics:
        return float(metrics[key])
    available = ", ".join(sorted(metrics))
    raise KeyError(f"checkpoint_metric={metric_name!r} is not in validation metrics. Available: {available}")


def torch_save_atomic(payload, path: Path) -> None:
    tmp_path = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def compact_summary(summary):
    keys = [
        "bundle_schema_version",
        "num_files",
        "num_cells_in_scanned_files",
        "num_genes",
        "num_records",
        "num_groups",
        "split_sizes",
        "split_group_counts",
        "split_manifest_paths",
        "matrix_key_counts",
        "matrix_format_counts",
        "source_semantics_counts",
        "expression_transform_counts",
        "expression_policy",
        "clean_pipeline",
        "qc_filter_stats",
        "qc_policy",
        "feature_namespace",
        "feature_source_counts",
        "files_with_duplicate_canonical_features",
        "data_limits",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lexa_r4.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--require-cuda", action="store_true", help="Fail fast instead of silently falling back to CPU.")
    parser.add_argument("--resume", action="store_true", help="Resume model and optimizer state from checkpoints/last.pt.")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Initialize model weights from another checkpoint but start a fresh optimizer/run. "
            "The parent checkpoint epoch is used as the epoch offset, so an epoch-1 parent with "
            "training.epochs=8 runs epochs 2..8."
        ),
    )
    parser.add_argument("--data-run-id", default=None, help="Read data_bundle/manifests from another run-id while writing outputs to --run-id.")
    parser.add_argument("--set", action="append", default=[], help="Override config key, e.g. --set model.num_axes=48")
    return parser.parse_args()


def train_epoch(model, loader, optimizer, scaler, device, config, epoch_idx, max_samples=None):
    model.train()
    totals = {}
    started = time.time()
    last_log = started
    samples_seen = 0
    for step, batch in enumerate(loader, start=1):
        batch_size = len(batch["group_id"])
        if max_samples is not None and samples_seen >= int(max_samples):
            break
        samples_seen += batch_size
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        amp_dtype = autocast_dtype(config, device)
        with autocast(
            device_type=device.type,
            enabled=bool(config["training"]["amp"]) and device.type == "cuda",
            dtype=amp_dtype if device.type == "cuda" else None,
        ):
            outputs = forward_batch(model, batch, device)
            loss = outputs["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
        scaler.step(optimizer)
        scaler.update()
        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim == 0:
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        if step % int(config["training"]["log_every"]) == 0:
            now = time.time()
            elapsed = now - started
            interval = now - last_log
            recent_samples = int(config["training"]["log_every"]) * batch_size
            parts = []
            for key in [
                "recon_loss",
                "count_nll",
                "library_size_mean",
                "inverse_dispersion_mean",
                "vq_aux_loss",
                "codebook_loss",
                "commitment_loss",
                "z_e_norm_mean",
                "z_q_norm_mean",
                "z_e_z_q_dist_mean",
                "z_e_z_q_cosine_mean",
                "code_entropy",
                "active_code_ratio",
                "module_effective_axes_mean",
                "module_effective_axes_median",
                "module_dominant_axis_weight_mean",
                "module_mask_prior_deviation_loss",
                "nuisance_effect_rms",
                "batch_adversary_loss",
                "batch_adversary_accuracy",
                "conditional_code_usage_loss",
                "conditional_code_usage_contexts",
                "conditional_code_usage_source_groups",
            ]:
                value = outputs.get(key)
                if torch.is_tensor(value) and value.ndim == 0:
                    parts.append(f"{key}={float(value.detach().cpu()):.4f}")
            if device.type == "cuda":
                parts.append(
                    f"gpu_peak_allocated_gib={torch.cuda.max_memory_allocated(device) / 1024**3:.2f}"
                )
                parts.append(
                    f"gpu_peak_reserved_gib={torch.cuda.max_memory_reserved(device) / 1024**3:.2f}"
                )
            progress(
                f"epoch={epoch_idx} step={step} loss={float(loss.detach().cpu()):.4f} "
                f"{' '.join(parts)} "
                f"samples={samples_seen} samples_per_s={samples_seen / max(elapsed, 1e-6):.1f} "
                f"recent_samples_per_s={recent_samples / max(interval, 1e-6):.1f}"
            )
            last_log = now
    steps_seen = max(1, step if samples_seen else 0)
    metrics = {key: value / steps_seen for key, value in totals.items()}
    return metrics, samples_seen


@torch.no_grad()
def eval_epoch(model, loader, device, config):
    model.eval()
    totals = {}
    steps_seen = 0
    max_steps = config.get("training", {}).get("max_validation_steps")
    max_steps = None if max_steps is None else int(max_steps)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("training.max_validation_steps must be positive when set")
    max_samples = config.get("training", {}).get("max_validation_cells")
    max_samples = None if max_samples is None else int(max_samples)
    if max_samples is not None and max_samples <= 0:
        raise ValueError("training.max_validation_cells must be positive when set")
    samples_seen = 0
    global_active = None
    amp_dtype = autocast_dtype(config, device)
    for step, batch in enumerate(loader, start=1):
        samples_seen += len(batch["group_id"])
        batch = batch_to_device(batch, device)
        with autocast(
            device_type=device.type,
            enabled=bool(config["training"]["amp"]) and device.type == "cuda",
            dtype=amp_dtype if device.type == "cuda" else None,
        ):
            outputs = forward_batch(model, batch, device)
        indices = outputs.get("indices")
        if torch.is_tensor(indices) and int(getattr(model, "codebook_size", 0)) > 1:
            if indices.ndim == 1:
                indices = indices[:, None]
            if global_active is None:
                global_active = torch.zeros(
                    indices.shape[1], int(model.codebook_size), dtype=torch.bool, device=indices.device
                )
            for axis in range(indices.shape[1]):
                global_active[axis, indices[:, axis].long().unique()] = True
        for key, value in outputs.items():
            if torch.is_tensor(value) and value.ndim == 0:
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu())
        steps_seen = step
        if max_steps is not None and step >= max_steps:
            break
        if max_samples is not None and samples_seen >= max_samples:
            break
    metrics = {key: value / max(1, steps_seen) for key, value in totals.items()}
    if global_active is not None:
        metrics["global_active_code_ratio"] = float(global_active.float().mean().cpu())
    return metrics


def main():
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    config = load_config(args.config)
    apply_config_overrides(config, args.set)
    set_seed(int(config["seed"]))
    run_id = now_run_id(config, args.run_id)

    out = make_output_dirs(config, run_id)
    device = pick_device(config["training"]["device"])
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required but is not available in this session.")
    configure_torch_performance(config, device)

    data_run_id = args.data_run_id or run_id
    config.setdefault("data", {})["_data_run_id"] = data_run_id
    data_root = Path(config["output"]["root"]) / data_run_id
    bundle_path = data_root / "data_bundle.json"
    slim_bundle_path = data_root / "data_bundle.slim.json"
    load_path = slim_bundle_path if slim_bundle_path.exists() else bundle_path
    if not load_path.exists():
        raise FileNotFoundError(
            f"Cannot find data bundle: {bundle_path}. "
            "Run src/preprocess_cellxgene_inclusive.py first, or pass --data-run-id for an existing bundle."
        )
    bundle = load_data_bundle(load_path, config)
    progress(f"loaded existing data bundle from {load_path}")
    if data_run_id != run_id:
        write_json(out["root"] / "data_source.json", {"data_run_id": data_run_id, "bundle_path": str(load_path)})
    progress(f"data summary: {compact_summary(bundle['summary'])}")

    warm_csr_caches(bundle, config, split_names=["train", "val"])

    train_ds = None
    val_ds = None
    wandb_run = None
    try:
        train_ds = H5CSRDataset(bundle["splits"]["train"], bundle["file_gene_maps"], bundle["gene_vocab"], config, bundle["file_metas"])
        val_ds = H5CSRDataset(bundle["splits"]["val"], bundle["file_gene_maps"], bundle["gene_vocab"], config, bundle["file_metas"])
        shuffle_train = bool(config["training"].get("shuffle_train", config["training"].get("shuffle", False)))
        label_balanced = bool(config["training"].get("label_balanced_batches", False)) or bool(config["training"].get("dataset_balanced_batches", False))
        balanced_label = str(config["training"].get("balanced_batch_label", "dataset_id"))
        progress(
            f"data loader: train shuffle={shuffle_train}, label_balanced_batches={label_balanced}, "
            f"balanced_batch_label={balanced_label}, balanced_epoch_fraction={config['training'].get('balanced_epoch_fraction')}, "
            f"batch_size={config['training']['batch_size']}, workers={config['training']['num_workers']}"
        )
        if label_balanced:
            progress(
                "data loader: training batches are label-balanced from manifest metadata; "
                "validation/test loaders keep the original split distribution"
            )
        if shuffle_train and not label_balanced:
            progress(
                "data loader: shuffle_train=true shuffles file-local batches by default; set training.group_batches_by_file=false for fully random batches"
            )
        train_loader = make_dataloader(train_ds, int(config["training"]["batch_size"]), shuffle_train, int(config["training"]["num_workers"]))
        val_loader = make_dataloader(val_ds, int(config["training"]["batch_size"]), False, int(config["training"]["num_workers"]))

        nuisance_cfg = config.setdefault("model", {}).get("nuisance_categories")
        nuisance_enabled = (
            float(config["model"].get("nuisance_decoder_weight", 0.0)) > 0
            or float(config["model"].get("batch_adversary_weight", 0.0)) > 0
            or float(config["model"].get("conditional_code_usage_weight", 0.0)) > 0
        )
        if nuisance_cfg is None and nuisance_enabled:
            nuisance_cfg = {}
            technical_fields = config["model"].get("technical_covariate_fields", ["dataset_id", "assay"])
            required_fields = list(technical_fields)
            if float(config["model"].get("batch_adversary_weight", 0.0)) > 0:
                required_fields.append(config["model"].get("batch_adversary_target_field", "dataset_id"))
                required_fields.append(config["model"].get("batch_adversary_context_field", "coarse_cell_type"))
            if float(config["model"].get("conditional_code_usage_weight", 0.0)) > 0:
                required_fields.append(config["model"].get("conditional_code_usage_target_field", "dataset_id"))
                required_fields.append(config["model"].get("conditional_code_usage_context_field", "coarse_cell_type"))
            for field in dict.fromkeys(str(value) for value in required_fields):
                groups = train_ds.records.label_indices(field)
                nuisance_cfg[field] = ["<unknown>"] + sorted(str(value) for value in groups)
            config["model"]["nuisance_categories"] = nuisance_cfg
            progress(f"inferred nuisance categories: { {key: len(values) for key, values in nuisance_cfg.items()} }")
        model = build_lexa_model(input_dim=len(bundle["gene_vocab"]), config=config, gene_vocab=bundle["gene_vocab"]).to(device)
        initialized_from = None
        initialized_start_epoch = 1
        if args.init_checkpoint:
            init_path = Path(args.init_checkpoint).expanduser().resolve()
            if not init_path.exists():
                raise FileNotFoundError(f"Cannot initialize from missing checkpoint: {init_path}")
            init_ckpt = torch.load(init_path, map_location="cpu")
            parent_vocab = init_ckpt.get("gene_vocab")
            if isinstance(parent_vocab, list) and parent_vocab != list(bundle["gene_vocab"]):
                raise ValueError(
                    "--init-checkpoint gene vocabulary does not exactly match the selected data bundle; "
                    "refusing a silent input-gene mismatch"
                )
            model.load_state_dict(init_ckpt["model_state"])
            digest = hashlib.sha1()
            with init_path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            parent_epoch = int(init_ckpt.get("epoch", 0))
            initialized_start_epoch = parent_epoch + 1
            initialized_from = {
                "checkpoint_path": str(init_path),
                "checkpoint_sha1": digest.hexdigest()[:16],
                "parent_run_id": init_ckpt.get("run_id"),
                "parent_epoch": parent_epoch,
            }
            progress(
                f"initialized model weights from {init_path} "
                f"(parent_run_id={init_ckpt.get('run_id')}, parent_epoch={parent_epoch}); "
                f"fresh optimizer starts at epoch {initialized_start_epoch}"
            )
        trainable_names = configure_trainable_parameters(model, config)
        progress(
            f"trainable parameters: {sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad):,}; "
            f"tensors={trainable_names}"
        )
        optimizer = make_optimizer(model, config, device)
        scaler_enabled = bool(config["training"]["amp"]) and device.type == "cuda" and autocast_dtype(config, device) == torch.float16
        scaler = GradScaler(enabled=scaler_enabled)
        progress(
            f"training runtime: device={device}, amp={bool(config['training']['amp'])}, "
            f"amp_dtype={autocast_dtype(config, device)}, grad_scaler={scaler_enabled}, "
            f"optimizer={optimizer.__class__.__name__}"
        )
        checkpoint_metric = str(config["training"].get("checkpoint_metric", "loss"))
        progress(f"checkpoint metric: val/{checkpoint_metric}")
        wandb_run = init_wandb(config, run_id)

        best_metric = float("inf")
        bad_validations = 0
        history = []
        start_epoch = initialized_start_epoch
        last_path = out["checkpoints"] / "last.pt"
        if args.resume:
            if not last_path.exists():
                raise FileNotFoundError(f"Cannot resume missing checkpoint: {last_path}")
            ckpt = torch.load(last_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scaler.load_state_dict(ckpt["scaler_state"])
            best_metric = float(ckpt["best_metric"])
            bad_validations = int(ckpt.get("bad_validations", 0))
            history = list(ckpt["history"])
            initialized_from = ckpt.get("initialized_from")
            start_epoch = int(ckpt["epoch"]) + 1
            progress(f"resuming from epoch {ckpt['epoch']} with best_metric={best_metric:.6f}")

        max_cell_updates = config["training"].get("max_cell_updates")
        max_cell_updates = None if max_cell_updates is None else int(max_cell_updates)
        if max_cell_updates is not None and max_cell_updates <= 0:
            raise ValueError("training.max_cell_updates must be positive when set")
        validation_every = max(1, int(config["training"].get("validation_every", 1)))
        save_last = bool(config["training"].get("save_last_checkpoint", True))
        early_stopping_patience = config["training"].get("early_stopping_patience")
        early_stopping_patience = None if early_stopping_patience is None else int(early_stopping_patience)
        early_stopping_min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
        if early_stopping_patience is not None and early_stopping_patience <= 0:
            raise ValueError("training.early_stopping_patience must be positive when set")
        cell_updates_seen = int(history[-1].get("cell_updates_seen", 0)) if history else 0
        final_epoch = int(config["training"]["epochs"])
        for epoch_idx in range(start_epoch, final_epoch + 1):
            weight_scale = apply_loss_weight_schedule(model, config, epoch_idx)
            if int(config["training"].get("vq_warmup_epochs") or 0) > 0:
                progress(f"epoch={epoch_idx} loss_weight_warmup_scale={weight_scale:.4f}")
            remaining_updates = None if max_cell_updates is None else max_cell_updates - cell_updates_seen
            if remaining_updates is not None and remaining_updates <= 0:
                progress(f"reached training.max_cell_updates={max_cell_updates}; stopping before epoch {epoch_idx}")
                break
            train_metrics, epoch_updates = train_epoch(
                model, train_loader, optimizer, scaler, device, config, epoch_idx, max_samples=remaining_updates
            )
            cell_updates_seen += int(epoch_updates)
            if (
                float(config.get("model", {}).get("conditional_code_usage_weight", 0.0)) > 0
                and bool(config.get("training", {}).get("require_conditional_code_usage_groups", True))
                and float(train_metrics.get("conditional_code_usage_contexts", 0.0)) < 1.0
            ):
                raise RuntimeError(
                    "conditional code-usage alignment was enabled, but no biological context "
                    "contained at least two sufficiently large source groups during this epoch; "
                    "adjust balanced_labels_per_batch, batch size, or the minimum group size"
                )
            should_validate = epoch_idx == final_epoch or epoch_idx % validation_every == 0
            if max_cell_updates is not None and cell_updates_seen >= max_cell_updates:
                should_validate = True
            val_metrics = eval_epoch(model, val_loader, device, config) if should_validate else {}
            row = {
                "epoch": epoch_idx,
                "cell_updates_seen": cell_updates_seen,
                "validated": should_validate,
                "loss_weight_warmup_scale": weight_scale,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            wandb_run.log(row)
            current_metric = metric_value(val_metrics, checkpoint_metric) if should_validate else None
            improved = should_validate and current_metric < best_metric - early_stopping_min_delta
            if improved:
                best_metric = current_metric
                bad_validations = 0
                payload = {
                    "epoch": epoch_idx,
                    "model_state": model.state_dict(),
                    "config": config,
                    "gene_vocab": bundle["gene_vocab"],
                    "run_id": run_id,
                    "checkpoint_metric": checkpoint_metric,
                    "best_metric": best_metric,
                    "bad_validations": bad_validations,
                    "history": history,
                    "initialized_from": initialized_from,
                }
                torch_save_atomic(payload, out["checkpoints"] / "best.pt")
            elif should_validate:
                bad_validations += 1
            last_payload = {
                "epoch": epoch_idx,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": config,
                "gene_vocab": bundle["gene_vocab"],
                "run_id": run_id,
                "checkpoint_metric": checkpoint_metric,
                "best_metric": best_metric,
                "bad_validations": bad_validations,
                "history": history,
                "initialized_from": initialized_from,
            }
            if save_last:
                torch_save_atomic(last_payload, last_path)
            if max_cell_updates is not None and cell_updates_seen >= max_cell_updates:
                progress(f"reached training.max_cell_updates={max_cell_updates}; stopping after epoch {epoch_idx}")
                break
            if early_stopping_patience is not None and bad_validations >= early_stopping_patience:
                progress(
                    f"early stopping after {bad_validations} non-improving validations "
                    f"(patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
                )
                break

        metrics = {
            "run_id": run_id,
            "data_summary": bundle["summary"],
            "checkpoint_metric": checkpoint_metric,
            "best_val_loss": best_metric,
            "history": history,
            "initialized_from": initialized_from,
        }
        write_json(out["root"] / "metrics.json", metrics)
        progress(f"saved checkpoint to {out['checkpoints'] / 'best.pt'}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if train_ds is not None:
            train_ds.close()
        if val_ds is not None:
            val_ds.close()


if __name__ == "__main__":
    main()
