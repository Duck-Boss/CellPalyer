from pathlib import Path
from typing import Any, Dict, Tuple
import hashlib
import json

import torch

from data_pipeline.clean_common import progress
from data_pipeline.runtime import load_config


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:16]


def resolve_checkpoint_path(run_root: Path, checkpoint_name: str = "last.pt") -> Path:
    name = str(checkpoint_name or "last.pt")
    if name in {"best", "best.pt"}:
        path = run_root / "checkpoints" / "best.pt"
        if not path.exists():
            fallback = run_root / "checkpoints" / "last.pt"
            if fallback.exists():
                progress(f"checkpoint {path} is missing; falling back to {fallback}")
                return fallback
        return path
    if name in {"last", "last.pt"}:
        return run_root / "checkpoints" / "last.pt"
    path = Path(name).expanduser()
    if path.exists():
        return path
    return run_root / "checkpoints" / name


def checkpoint_metadata(ckpt: Dict[str, Any], config: Dict[str, Any], checkpoint_path: Path) -> Dict[str, Any]:
    gene_vocab = ckpt.get("gene_vocab", [])
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha1": _file_sha1(checkpoint_path),
        "checkpoint_epoch": None if ckpt.get("epoch") is None else int(ckpt["epoch"]),
        "checkpoint_metric": ckpt.get("checkpoint_metric"),
        "best_metric": ckpt.get("best_metric"),
        "run_id": ckpt.get("run_id"),
        "config_sha1": _stable_hash(config),
        "gene_vocab_sha1": _stable_hash(gene_vocab),
        "num_genes": len(gene_vocab) if isinstance(gene_vocab, list) else None,
    }


def load_checkpoint_and_config(
    config_path: str,
    run_root: Path,
    checkpoint_name: str = "last.pt",
    return_path: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base_config = load_config(config_path)
    ckpt_path = resolve_checkpoint_path(run_root, checkpoint_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_config = ckpt.get("config")
    if isinstance(ckpt_config, dict):
        output_cfg = base_config.get("output", {})
        logging_cfg = base_config.get("logging", {})
        base_data = base_config.get("data", {})
        config = ckpt_config
        config["output"] = output_cfg
        config["logging"] = logging_cfg
        config.setdefault("data", {}).update({k: v for k, v in base_data.items() if k not in config.get("data", {})})
        progress(f"loaded model config from checkpoint {ckpt_path}")
    else:
        config = base_config
        progress(f"checkpoint {ckpt_path} has no config; using {config_path}")
    if return_path:
        return ckpt, config, ckpt_path
    return ckpt, config
