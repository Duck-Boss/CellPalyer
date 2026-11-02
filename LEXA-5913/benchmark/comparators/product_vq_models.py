import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lexa.lexa_model import LEXA


class ProductVQ(LEXA):
    """Per-axis VQ baseline with uniform gene responsibility."""

    def __init__(self, input_dim: int, **kwargs):
        kwargs.pop("module_mask_prior", None)
        kwargs.pop("module_mask_prior_path", None)
        kwargs["module_mask_parameterization"] = "uniform"
        kwargs["module_mask_prior_weight"] = 0.0
        super().__init__(input_dim=input_dim, **kwargs)


def masked_mse(pred, target, mask=None, eps=1e-8):
    diff = (pred - target).pow(2)
    if mask is None:
        return diff.mean()
    return (diff * mask).sum() / mask.sum().clamp_min(eps)


def code_usage_loss(indices: torch.Tensor, codebook_size: int, eps: float = 1e-8):
    """Compute hard code-usage diagnostics per position."""
    if indices.ndim == 1:
        indices = indices[:, None]
    entropies = []
    active = []
    for token_idx in range(indices.shape[1]):
        hist = torch.bincount(indices[:, token_idx], minlength=codebook_size).float()
        prob = hist / hist.sum().clamp_min(1.0)
        ent = -(prob * (prob + eps).log()).sum() / math.log(codebook_size)
        entropies.append(ent)
        active.append((hist > 0).float().mean())
    entropy = torch.stack(entropies).mean()
    active_ratio = torch.stack(active).mean()
    return 1.0 - entropy, entropy, active_ratio


def soft_code_usage_loss(assign_prob: torch.Tensor, eps: float = 1e-8):
    """Differentiable diversity loss, averaged over latent positions."""
    usage = assign_prob.mean(dim=0)
    entropy = -(usage * (usage + eps).log()).sum(dim=-1) / math.log(assign_prob.shape[-1])
    return 1.0 - entropy.mean(), entropy.mean()


def effective_shared_codebook(
    codebook: torch.Tensor,
    quantization_mode: str,
    center_norm_target: float,
) -> torch.Tensor:
    if quantization_mode in {"spherical", "cosine"}:
        return F.normalize(codebook, dim=-1) * float(center_norm_target)
    if quantization_mode == "euclidean":
        return codebook
    raise ValueError(f"Unknown quantization_mode: {quantization_mode}")


def quantize_shared_codebook(
    z_e: torch.Tensor,
    codebook: torch.Tensor,
    quantization_mode: str,
    center_norm_target: float,
    assignment_temperature: float,
    commitment_weight: float,
):
    codebook_for_vq = effective_shared_codebook(codebook, quantization_mode, center_norm_target)
    if quantization_mode in {"spherical", "cosine"}:
        scale = float(center_norm_target)
        z_e_for_vq = F.normalize(z_e, dim=-1) * scale
        dist = 2.0 * scale * scale - 2.0 * torch.einsum("bld,kd->blk", z_e_for_vq, codebook_for_vq)
    elif quantization_mode == "euclidean":
        z_e_for_vq = z_e
        dist = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            - 2.0 * torch.einsum("bld,kd->blk", z_e, codebook_for_vq)
            + codebook_for_vq.pow(2).sum(dim=-1).unsqueeze(0)
        )
    else:
        raise ValueError(f"Unknown quantization_mode: {quantization_mode}")

    indices = dist.argmin(dim=-1)
    selected = codebook_for_vq[indices]
    codebook_loss = F.mse_loss(selected, z_e_for_vq.detach())
    commitment_loss = F.mse_loss(z_e_for_vq, selected.detach())
    vq_aux_loss = codebook_loss + float(commitment_weight) * commitment_loss
    assign_prob = F.softmax(-dist / max(float(assignment_temperature), 1e-6), dim=-1)
    z_q = z_e + (selected - z_e).detach()
    diagnostics = {
        "codebook_loss": codebook_loss,
        "commitment_loss": commitment_loss,
        "z_e_norm_mean": z_e.norm(dim=-1).mean(),
        "z_e_norm_max": z_e.norm(dim=-1).amax(),
        "z_q_norm_mean": z_q.norm(dim=-1).mean(),
        "z_q_norm_max": z_q.norm(dim=-1).amax(),
        "z_e_z_q_dist_mean": (z_e_for_vq.detach() - z_q.detach()).norm(dim=-1).mean(),
    }
    return z_q, indices, vq_aux_loss, assign_prob, diagnostics


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...], output_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for idx in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[idx], dims[idx + 1]), nn.GELU(), nn.Dropout(dropout)])
        layers.extend([nn.Linear(dims[-1], output_dim), nn.LayerNorm(output_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleCodebookVQVAE(nn.Module):
    """VQ-VAE with one codebook shared across latent positions."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (2048,),
        latent_tokens: int = 32,
        code_dim: int = 64,
        codebook_size: int = 512,
        decoder_hidden_dim: int = 2048,
        decoder_hidden_dims: Optional[Tuple[int, ...]] = None,
        dropout: float = 0.1,
        commitment_weight: float = 1.0,
        vq_aux_loss_weight: float = 0.05,
        diversity_loss_weight: float = 0.0,
        quantization_mode: str = "euclidean",
        assignment_temperature: float = 1.0,
        center_norm_target: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_tokens = int(latent_tokens)
        self.num_axes = self.latent_tokens
        self.num_codebooks = 1
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_hidden_dims = list(decoder_hidden_dims) if decoder_hidden_dims is not None else [decoder_hidden_dim]
        self.commitment_weight = commitment_weight
        self.vq_aux_loss_weight = vq_aux_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.quantization_mode = str(quantization_mode).lower()
        self.assignment_temperature = float(assignment_temperature)
        self.center_norm_target = float(center_norm_target)

        latent_dim = self.latent_tokens * self.code_dim
        self.encoder = MLPEncoder(input_dim, tuple(hidden_dims), latent_dim, dropout)
        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim) * 0.02)
        decoder_layers = []
        decoder_dims = [latent_dim] + self.decoder_hidden_dims
        for idx in range(len(decoder_dims) - 1):
            decoder_layers.extend([nn.Linear(decoder_dims[idx], decoder_dims[idx + 1]), nn.GELU(), nn.Dropout(dropout)])
        decoder_layers.append(nn.Linear(decoder_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z.view(x.shape[0], self.latent_tokens, self.code_dim)

    def quantize(self, z_e: torch.Tensor):
        return quantize_shared_codebook(
            z_e,
            self.codebook,
            self.quantization_mode,
            self.center_norm_target,
            self.assignment_temperature,
            self.commitment_weight,
        )

    def quantization_codebook(self) -> torch.Tensor:
        return effective_shared_codebook(self.codebook, self.quantization_mode, self.center_norm_target)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        if z_q.ndim == 3:
            z_q = z_q.reshape(z_q.shape[0], self.latent_tokens * self.code_dim)
        return self.decoder(z_q)

    def module_mask(self) -> torch.Tensor:
        return torch.full(
            (self.latent_tokens, self.input_dim),
            1.0 / float(self.latent_tokens),
            device=self.codebook.device,
            dtype=self.codebook.dtype,
        )

    @torch.no_grad()
    def sweep_axis_states(self, baseline_z_q: torch.Tensor, axis_idx: int) -> torch.Tensor:
        if axis_idx < 0 or axis_idx >= self.latent_tokens:
            raise ValueError(f"axis_idx must be in [0, {self.latent_tokens})")
        if baseline_z_q.ndim == 2:
            baseline_z_q = baseline_z_q.view(baseline_z_q.shape[0], self.latent_tokens, self.code_dim)
        outs = []
        codebook = self.quantization_codebook()
        for code_idx in range(self.codebook_size):
            z = baseline_z_q.clone()
            z[:, axis_idx, :] = codebook[code_idx].unsqueeze(0).to(z.device)
            outs.append(self.decode(z))
        return torch.stack(outs, dim=0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_aux: bool = False, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        z_e = self.encode(x)
        z_q, indices, vq_aux_loss, assign_prob, vq_diagnostics = self.quantize(z_e)
        x_recon = self.decode(z_q)
        recon_loss = masked_mse(x_recon, x, mask)

        hard_diversity_loss, code_entropy, active_code_ratio = code_usage_loss(indices, self.codebook_size)
        diversity_loss, soft_code_entropy = soft_code_usage_loss(assign_prob)

        loss = recon_loss + self.vq_aux_loss_weight * vq_aux_loss + self.diversity_loss_weight * diversity_loss
        module_mask = self.module_mask()
        output = {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_aux_loss": vq_aux_loss,
            **vq_diagnostics,
            "diversity_loss": diversity_loss,
            "hard_diversity_loss": hard_diversity_loss,
            "code_entropy": code_entropy,
            "soft_code_entropy": soft_code_entropy,
            "active_code_ratio": active_code_ratio,
            "module_sparsity_loss": x.new_tensor(0.0),
            "module_balance_loss": x.new_tensor(0.0),
            "module_overlap_loss": x.new_tensor(0.0),
            "x_recon": x_recon,
            "z_e": z_e,
            "z_q": z_q,
            "indices": indices,
            "module_mask": module_mask,
        }
        if return_aux:
            module_recon = x_recon.unsqueeze(1)
            output["module_recon"] = module_recon
            output["raw_recon"] = module_recon
        return output


class AdditiveCodebookVQVAE(nn.Module):
    """Shared-codebook VQ-VAE with an additive decoder and FiLM."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (2048,),
        latent_tokens: int = 32,
        code_dim: int = 64,
        codebook_size: int = 16,
        decoder_hidden_dim: int = 2048,
        dropout: float = 0.1,
        commitment_weight: float = 1.0,
        vq_aux_loss_weight: float = 0.05,
        diversity_loss_weight: float = 0.0,
        quantization_mode: str = "euclidean",
        assignment_temperature: float = 1.0,
        center_norm_target: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.latent_tokens = int(latent_tokens)
        self.num_axes = self.latent_tokens
        self.num_codebooks = 1
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.commitment_weight = float(commitment_weight)
        self.vq_aux_loss_weight = float(vq_aux_loss_weight)
        self.diversity_loss_weight = float(diversity_loss_weight)
        self.quantization_mode = str(quantization_mode).lower()
        self.assignment_temperature = float(assignment_temperature)
        self.center_norm_target = float(center_norm_target)

        latent_dim = self.latent_tokens * self.code_dim
        self.encoder = MLPEncoder(input_dim, tuple(hidden_dims), latent_dim, dropout)
        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim) * 0.02)
        self.decoder_in = nn.Linear(self.code_dim, self.decoder_hidden_dim)
        self.decoder_out = nn.Linear(self.decoder_hidden_dim, self.input_dim)
        self.dropout = nn.Dropout(dropout)
        self.film_gamma = nn.Parameter(torch.ones(self.latent_tokens, self.decoder_hidden_dim))
        self.film_beta = nn.Parameter(torch.zeros(self.latent_tokens, self.decoder_hidden_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z.view(x.shape[0], self.latent_tokens, self.code_dim)

    def quantize(self, z_e: torch.Tensor):
        return quantize_shared_codebook(
            z_e,
            self.codebook,
            self.quantization_mode,
            self.center_norm_target,
            self.assignment_temperature,
            self.commitment_weight,
        )

    def quantization_codebook(self) -> torch.Tensor:
        return effective_shared_codebook(self.codebook, self.quantization_mode, self.center_norm_target)

    def decode_modules(self, z_q: torch.Tensor):
        batch_size = z_q.shape[0]
        h = self.decoder_in(z_q.reshape(batch_size * self.latent_tokens, self.code_dim))
        h = F.gelu(h).view(batch_size, self.latent_tokens, self.decoder_hidden_dim)
        h = h * self.film_gamma.unsqueeze(0) + self.film_beta.unsqueeze(0)
        h = self.dropout(F.gelu(h))
        raw = self.decoder_out(h.reshape(batch_size * self.latent_tokens, self.decoder_hidden_dim))
        raw = raw.view(batch_size, self.latent_tokens, self.input_dim)
        mask = self.module_mask().unsqueeze(0)
        modules = raw * mask
        return modules.sum(dim=1), modules, raw, mask

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        if z_q.ndim == 2:
            z_q = z_q.view(z_q.shape[0], self.latent_tokens, self.code_dim)
        return self.decode_modules(z_q)[0]

    def module_mask(self) -> torch.Tensor:
        return torch.full(
            (self.latent_tokens, self.input_dim),
            1.0 / float(self.latent_tokens),
            device=self.codebook.device,
            dtype=self.codebook.dtype,
        )

    @torch.no_grad()
    def sweep_axis_states(self, baseline_z_q: torch.Tensor, axis_idx: int) -> torch.Tensor:
        if axis_idx < 0 or axis_idx >= self.latent_tokens:
            raise ValueError(f"axis_idx must be in [0, {self.latent_tokens})")
        if baseline_z_q.ndim == 2:
            baseline_z_q = baseline_z_q.view(baseline_z_q.shape[0], self.latent_tokens, self.code_dim)
        outs = []
        codebook = self.quantization_codebook()
        for code_idx in range(self.codebook_size):
            z = baseline_z_q.clone()
            z[:, axis_idx, :] = codebook[code_idx].unsqueeze(0).to(z.device)
            outs.append(self.decode(z))
        return torch.stack(outs, dim=0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_aux: bool = False, **kwargs) -> Dict[str, torch.Tensor]:
        del kwargs
        z_e = self.encode(x)
        z_q, indices, vq_aux_loss, assign_prob, vq_diagnostics = self.quantize(z_e)
        x_recon, module_recon, raw_recon, module_mask = self.decode_modules(z_q)
        recon_loss = masked_mse(x_recon, x, mask)

        hard_diversity_loss, code_entropy, active_code_ratio = code_usage_loss(indices, self.codebook_size)
        diversity_loss, soft_code_entropy = soft_code_usage_loss(assign_prob)

        loss = recon_loss + self.vq_aux_loss_weight * vq_aux_loss + self.diversity_loss_weight * diversity_loss
        output = {
            "loss": loss,
            "recon_loss": recon_loss,
            "vq_aux_loss": vq_aux_loss,
            **vq_diagnostics,
            "diversity_loss": diversity_loss,
            "hard_diversity_loss": hard_diversity_loss,
            "code_entropy": code_entropy,
            "soft_code_entropy": soft_code_entropy,
            "active_code_ratio": active_code_ratio,
            "module_sparsity_loss": x.new_tensor(0.0),
            "module_balance_loss": x.new_tensor(0.0),
            "module_overlap_loss": x.new_tensor(0.0),
            "x_recon": x_recon,
            "z_e": z_e,
            "z_q": z_q,
            "indices": indices,
            "module_mask": module_mask.squeeze(0),
        }
        if return_aux:
            output["module_recon"] = module_recon
            output["raw_recon"] = raw_recon
        return output
