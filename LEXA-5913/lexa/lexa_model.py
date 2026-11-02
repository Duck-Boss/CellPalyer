"""LEXA model and quantizer."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    diff = (pred - target).pow(2)
    if mask is None:
        return diff.mean()
    return (diff * mask).sum() / mask.sum().clamp_min(eps)


def negative_binomial_nll(
    counts: torch.Tensor,
    mean: torch.Tensor,
    inverse_dispersion: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean negative-binomial loss parameterized by mean and inverse dispersion."""
    counts = counts.float().clamp_min(0.0)
    mean = mean.float().clamp_min(eps)
    inverse_dispersion = inverse_dispersion.float().clamp_min(eps)
    log_prob = (
        torch.lgamma(counts + inverse_dispersion)
        - torch.lgamma(inverse_dispersion)
        - torch.lgamma(counts + 1.0)
        + inverse_dispersion * (torch.log(inverse_dispersion) - torch.log(inverse_dispersion + mean))
        + counts * (torch.log(mean) - torch.log(inverse_dispersion + mean))
    )
    nll = -log_prob
    if mask is None:
        return nll.mean()
    return (nll * mask).sum() / mask.sum().clamp_min(eps)


def _offdiag(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


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


def code_usage_loss(indices: torch.Tensor, codebook_size: int, num_axes: int, eps: float = 1e-8):
    entropies = []
    active = []
    for axis_idx in range(num_axes):
        hist = torch.bincount(indices[:, axis_idx], minlength=codebook_size).float()
        prob = hist / hist.sum().clamp_min(1.0)
        ent = -(prob * (prob + eps).log()).sum() / math.log(codebook_size)
        entropies.append(ent)
        active.append((hist > 0).float().mean())
    entropy = torch.stack(entropies).mean()
    active_ratio = torch.stack(active).mean()
    return 1.0 - entropy, entropy, active_ratio


def soft_code_usage_loss(assign_prob: torch.Tensor, eps: float = 1e-8):
    usage = assign_prob.mean(dim=0)
    entropy = -(usage * (usage + eps).log()).sum(dim=-1) / math.log(assign_prob.shape[-1])
    return 1.0 - entropy.mean(), entropy.mean()


class LEXA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (2048,),
        num_axes: int = 32,
        code_dim: int = 64,
        codebook_size: int = 16,
        decoder_hidden_dim: int = 2048,
        dropout: float = 0.1,
        commitment_weight: float = 1.0,
        vq_aux_loss_weight: float = 0.05,
        diversity_loss_weight: float = 0.1,
        module_sparsity_weight: float = 0.02,
        module_balance_weight: float = 1.0,
        module_overlap_weight: float = 0.01,
        module_mask_parameterization: str = "learned_softmax",
        module_mask_prior: Optional[torch.Tensor] = None,
        module_effective_axes_target: float = 0.0,
        module_mask_prior_weight: float = 0.0,
        module_mask_prior_floor: float = 1e-3,
        module_mask_logit_init_std: float = 0.02,
        codebook_sep_weight: float = 0.0,
        sep_margin: float = 0.0,
        assignment_temperature: float = 0.1,
        module_mask_temperature: Optional[float] = None,
        quantization_mode: str = "euclidean",
        anchor_loss_weight: float = 0.02,
        center_norm_weight: float = 0.005,
        center_norm_target: float = 2.0,
        codebook_compact_weight: float = 0.001,
        intra_codebook_div_weight: float = 0.005,
        cluster_radius: float = 0.35,
        residual_center_weight: float = 0.02,
        codebook_parameterization: str = "anchored_cap",
        encoder_fixed_norm: bool = False,
        vq_loss_normalization: str = "per_coordinate",
        nuisance_categories: Optional[dict] = None,
        technical_covariate_fields: Optional[Tuple[str, ...]] = None,
        nuisance_rank: int = 32,
        nuisance_decoder_weight: float = 0.0,
        batch_adversary_weight: float = 0.0,
        batch_adversary_grl_scale: float = 1.0,
        batch_adversary_hidden_dim: int = 128,
        batch_adversary_target_field: str = "dataset_id",
        batch_adversary_context_field: str = "coarse_cell_type",
        conditional_code_usage_weight: float = 0.0,
        conditional_code_usage_temperature: float = 0.5,
        conditional_code_usage_min_cells: int = 8,
        conditional_code_usage_target_field: Optional[str] = None,
        conditional_code_usage_context_field: Optional[str] = None,
        exclude_unknown_from_nuisance_centering: bool = False,
        reconstruction_distribution: str = "mse_log1p",
        count_normalized_total: float = 10000.0,
        inverse_dispersion_init: float = 10.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims)
        self.num_axes = int(num_axes)
        self.num_codebooks = self.num_axes
        self.code_dim = int(code_dim)
        self.codebook_size = int(codebook_size)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.commitment_weight = float(commitment_weight)
        self.vq_aux_loss_weight = float(vq_aux_loss_weight)
        self.diversity_loss_weight = float(diversity_loss_weight)
        self.module_sparsity_weight = float(module_sparsity_weight)
        self.module_balance_weight = float(module_balance_weight)
        self.module_overlap_weight = float(module_overlap_weight)
        self.module_mask_parameterization = str(module_mask_parameterization).lower()
        self.module_effective_axes_target = float(module_effective_axes_target)
        self.module_mask_prior_weight = float(module_mask_prior_weight)
        self.module_mask_prior_floor = float(module_mask_prior_floor)
        self.module_mask_logit_init_std = float(module_mask_logit_init_std)
        self.codebook_sep_weight = float(codebook_sep_weight)
        self.sep_margin = float(sep_margin)
        self.assignment_temperature = float(assignment_temperature)
        self.module_mask_temperature = float(
            assignment_temperature if module_mask_temperature is None else module_mask_temperature
        )
        if self.assignment_temperature <= 0 or self.module_mask_temperature <= 0:
            raise ValueError("assignment_temperature and module_mask_temperature must both be positive")
        self.quantization_mode = str(quantization_mode).lower()
        self.anchor_loss_weight = float(anchor_loss_weight)
        self.center_norm_weight = float(center_norm_weight)
        self.center_norm_target = float(center_norm_target)
        self.codebook_compact_weight = float(codebook_compact_weight)
        self.intra_codebook_div_weight = float(intra_codebook_div_weight)
        self.cluster_radius = float(cluster_radius)
        self.residual_center_weight = float(residual_center_weight)
        self.codebook_parameterization = str(codebook_parameterization).lower()
        self.encoder_fixed_norm = bool(encoder_fixed_norm)
        self.vq_loss_normalization = str(vq_loss_normalization).lower()
        if self.vq_loss_normalization not in {"per_coordinate", "per_radius"}:
            raise ValueError(f"Unknown vq_loss_normalization: {self.vq_loss_normalization}")
        self.nuisance_rank = int(nuisance_rank)
        self.nuisance_decoder_weight = float(nuisance_decoder_weight)
        self.batch_adversary_weight = float(batch_adversary_weight)
        self.batch_adversary_grl_scale = float(batch_adversary_grl_scale)
        self.batch_adversary_target_field = str(batch_adversary_target_field)
        self.batch_adversary_context_field = str(batch_adversary_context_field)
        self.conditional_code_usage_weight = float(conditional_code_usage_weight)
        self.conditional_code_usage_temperature = float(conditional_code_usage_temperature)
        self.conditional_code_usage_min_cells = int(conditional_code_usage_min_cells)
        self.conditional_code_usage_target_field = str(
            conditional_code_usage_target_field or batch_adversary_target_field
        )
        self.conditional_code_usage_context_field = str(
            conditional_code_usage_context_field or batch_adversary_context_field
        )
        if self.conditional_code_usage_temperature <= 0:
            raise ValueError("conditional_code_usage_temperature must be positive")
        if self.conditional_code_usage_min_cells < 1:
            raise ValueError("conditional_code_usage_min_cells must be at least one")
        self.exclude_unknown_from_nuisance_centering = bool(exclude_unknown_from_nuisance_centering)
        self.reconstruction_distribution = str(reconstruction_distribution).lower()
        if self.reconstruction_distribution not in {"mse", "mse_log1p", "negative_binomial", "nb"}:
            raise ValueError(f"Unknown reconstruction_distribution: {self.reconstruction_distribution}")
        self.count_normalized_total = float(count_normalized_total)
        if self.count_normalized_total <= 0:
            raise ValueError("count_normalized_total must be positive")
        if self.reconstruction_distribution in {"negative_binomial", "nb"}:
            init = max(float(inverse_dispersion_init), 1e-4)
            raw_init = math.log(math.expm1(init)) if init < 20 else init
            self.inverse_dispersion_raw = nn.Parameter(torch.full((self.input_dim,), raw_init))
        else:
            self.register_parameter("inverse_dispersion_raw", None)

        self.encoder = MLPEncoder(
            input_dim=self.input_dim,
            hidden_dims=tuple(hidden_dims),
            output_dim=self.num_axes * self.code_dim,
            dropout=dropout,
        )
        if self.codebook_parameterization in {"free_spherical", "spherical_free", "free_euclidean", "euclidean_free"}:
            self.register_buffer("codebook_anchors", torch.empty(0), persistent=False)
            self.codebook_vectors = nn.Parameter(torch.randn(self.num_axes, self.codebook_size, self.code_dim))
            self.register_parameter("codebook_residual", None)
        elif self.codebook_parameterization in {"anchored_cap", "anchor_residual"}:
            anchors = self._make_anchors(self.num_axes, self.code_dim)
            self.register_buffer("codebook_anchors", anchors)
            self.codebook_residual = nn.Parameter(torch.randn(self.num_axes, self.codebook_size, self.code_dim) * 0.02)
            self.register_parameter("codebook_vectors", None)
        else:
            raise ValueError(f"Unknown codebook_parameterization: {self.codebook_parameterization}")

        self.decoder_in = nn.Linear(self.code_dim, self.decoder_hidden_dim)
        self.decoder_out = nn.Linear(self.decoder_hidden_dim, self.input_dim)
        self.dropout = nn.Dropout(dropout)

        fixed_mask_modes = {"fixed_prior", "fixed"}
        guided_mask_modes = {"prior_guided_softmax", "learned_prior_softmax"}
        if self.module_mask_parameterization == "uniform":
            self.register_buffer(
                "fixed_module_mask",
                torch.full((self.num_axes, self.input_dim), 1.0 / float(self.num_axes)),
            )
            self.register_parameter("assignment_logits", None)
        elif self.module_mask_parameterization in fixed_mask_modes | guided_mask_modes:
            if module_mask_prior is None:
                raise ValueError("module_mask_prior is required for a fixed or prior-guided module mask")
            prior = torch.as_tensor(module_mask_prior, dtype=torch.float32)
            expected_shape = (self.num_axes, self.input_dim)
            if tuple(prior.shape) != expected_shape:
                raise ValueError(f"module mask prior has shape {tuple(prior.shape)}, expected {expected_shape}")
            if not bool(torch.isfinite(prior).all()) or bool((prior < 0).any()):
                raise ValueError("module mask prior must be finite and nonnegative")
            prior = prior / prior.sum(dim=0, keepdim=True).clamp_min(1e-12)
            if self.module_mask_parameterization in guided_mask_modes:
                floor = max(self.module_mask_prior_floor, 0.0)
                prior = (prior + floor) / (prior + floor).sum(dim=0, keepdim=True)
            self.register_buffer("fixed_module_mask", prior)
            if self.module_mask_parameterization in guided_mask_modes:
                initial_logits = self.module_mask_temperature * prior.clamp_min(1e-12).log()
                if self.module_mask_logit_init_std > 0:
                    initial_logits = initial_logits + torch.randn_like(initial_logits) * self.module_mask_logit_init_std
                self.assignment_logits = nn.Parameter(initial_logits)
            else:
                self.register_parameter("assignment_logits", None)
        elif self.module_mask_parameterization in {"learned_softmax", "learned"}:
            self.register_buffer("fixed_module_mask", torch.empty(0), persistent=False)
            self.assignment_logits = nn.Parameter(torch.empty(self.num_axes, self.input_dim))
            nn.init.normal_(self.assignment_logits, std=self.module_mask_logit_init_std)
        else:
            raise ValueError(f"Unknown module_mask_parameterization: {self.module_mask_parameterization}")

        self.film_gamma = nn.Parameter(torch.ones(self.num_axes, self.decoder_hidden_dim))
        self.film_beta = nn.Parameter(torch.zeros(self.num_axes, self.decoder_hidden_dim))

        category_cfg = nuisance_categories or {}
        self.nuisance_categories = {str(key): [str(value) for value in values] for key, values in category_cfg.items()}
        requested_technical_fields = (
            tuple(str(field) for field in technical_covariate_fields)
            if technical_covariate_fields is not None
            else ("dataset_id", "assay")
        )
        self.nuisance_fields = tuple(field for field in requested_technical_fields if field in self.nuisance_categories)
        self.nuisance_embeddings = nn.ModuleDict()
        for field in self.nuisance_fields:
            self.nuisance_embeddings[field] = nn.Embedding(len(self.nuisance_categories[field]), self.nuisance_rank)
        self.nuisance_out = nn.Linear(self.nuisance_rank, self.input_dim, bias=False) if self.nuisance_fields else None

        target_categories = self.nuisance_categories.get(self.batch_adversary_target_field, [])
        self.batch_adversary_target_offset = int(
            self.exclude_unknown_from_nuisance_centering
            and bool(target_categories)
            and target_categories[0] == "<unknown>"
        )
        target_count = len(target_categories) - self.batch_adversary_target_offset
        context_count = len(self.nuisance_categories.get(self.batch_adversary_context_field, []))
        if self.batch_adversary_weight > 0 and target_count > 1:
            context_dim = min(32, max(4, self.nuisance_rank))
            self.batch_context_embedding = nn.Embedding(max(1, context_count), context_dim)
            self.batch_adversary = nn.Sequential(
                nn.Linear(self.num_axes * self.code_dim + context_dim, int(batch_adversary_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(batch_adversary_hidden_dim), target_count),
            )
        else:
            self.batch_context_embedding = None
            self.batch_adversary = None

        alignment_target_categories = self.nuisance_categories.get(
            self.conditional_code_usage_target_field, []
        )
        alignment_context_categories = self.nuisance_categories.get(
            self.conditional_code_usage_context_field, []
        )
        self.conditional_code_usage_target_offset = int(
            self.exclude_unknown_from_nuisance_centering
            and bool(alignment_target_categories)
            and alignment_target_categories[0] == "<unknown>"
        )
        self.conditional_code_usage_context_offset = int(
            bool(alignment_context_categories)
            and alignment_context_categories[0] == "<unknown>"
        )

    @staticmethod
    def _make_anchors(num_axes: int, code_dim: int) -> torch.Tensor:
        base = torch.randn(code_dim, num_axes)
        q, _ = torch.linalg.qr(base, mode="reduced")
        anchors = q.t()
        if anchors.shape[0] < num_axes:
            anchors = F.normalize(torch.randn(num_axes, code_dim), dim=-1)
        return F.normalize(anchors[:num_axes], dim=-1)

    def module_mask(self) -> torch.Tensor:
        if self.module_mask_parameterization in {"fixed_prior", "fixed", "uniform"}:
            return self.fixed_module_mask
        logits = self.assignment_logits
        if self.module_mask_parameterization in {"learned_softmax", "learned"}:
            logits = logits - logits.mean(dim=1, keepdim=True)
        return F.softmax(logits / self.module_mask_temperature, dim=0)

    @property
    def codebook(self) -> torch.Tensor:
        if self.codebook_parameterization in {"free_euclidean", "euclidean_free"}:
            return self.codebook_vectors
        if self.codebook_parameterization in {"free_spherical", "spherical_free"}:
            return F.normalize(self.codebook_vectors, dim=-1) * float(self.center_norm_target)
        anchor = F.normalize(self.codebook_anchors, dim=-1)
        residual = self.codebook_residual
        residual = residual - (residual * anchor[:, None, :]).sum(dim=-1, keepdim=True) * anchor[:, None, :]
        residual = F.normalize(residual, dim=-1)
        return self.center_norm_target * anchor[:, None, :] + self.cluster_radius * residual

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = z.view(x.shape[0], self.num_axes, self.code_dim)
        if self.encoder_fixed_norm:
            z = F.normalize(z, dim=-1) * float(self.center_norm_target)
        return z

    def quantization_codebook(self) -> torch.Tensor:
        codebook = self.codebook
        if self.quantization_mode in {"spherical", "cosine"}:
            return F.normalize(codebook, dim=-1) * float(self.center_norm_target)
        if self.quantization_mode == "euclidean":
            return codebook
        raise ValueError(f"Unknown quantization_mode: {self.quantization_mode}")

    def quantize(self, z_e: torch.Tensor):
        codebook = self.codebook
        if self.quantization_mode in {"spherical", "cosine"}:
            scale = float(self.center_norm_target)
            z_e_for_vq = F.normalize(z_e, dim=-1) * scale
            codebook_for_vq = self.quantization_codebook()
            dist = 2.0 * scale * scale - 2.0 * torch.einsum("bad,akd->bak", z_e_for_vq, codebook_for_vq)
        elif self.quantization_mode == "euclidean":
            z_e_for_vq = z_e
            codebook_for_vq = codebook
            dist = (
                z_e.pow(2).sum(dim=-1, keepdim=True)
                - 2.0 * torch.einsum("bad,akd->bak", z_e, codebook)
                + codebook.pow(2).sum(dim=-1).unsqueeze(0)
            )
        else:
            raise ValueError(f"Unknown quantization_mode: {self.quantization_mode}")
        indices = dist.argmin(dim=-1)
        assign_prob = F.softmax(-dist / self.assignment_temperature, dim=-1)
        z_q = codebook_for_vq.unsqueeze(0).expand(z_e.shape[0], -1, -1, -1).gather(
            2, indices[:, :, None, None].expand(-1, -1, 1, self.code_dim)
        ).squeeze(2)
        if self.quantization_mode in {"spherical", "cosine"} and self.vq_loss_normalization == "per_radius":
            radius_sq = max(float(self.center_norm_target) ** 2, 1e-12)
            codebook_loss = ((z_q - z_e_for_vq.detach()).pow(2).sum(dim=-1) / radius_sq).mean()
            commitment_loss = ((z_e_for_vq - z_q.detach()).pow(2).sum(dim=-1) / radius_sq).mean()
        else:
            codebook_loss = F.mse_loss(z_q, z_e_for_vq.detach())
            commitment_loss = F.mse_loss(z_e_for_vq, z_q.detach())
        vq_aux_loss = codebook_loss + self.commitment_weight * commitment_loss
        z_q = z_e_for_vq + (z_q - z_e_for_vq).detach()
        diagnostics = {
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "z_e_norm_mean": z_e.norm(dim=-1).mean(),
            "z_e_norm_max": z_e.norm(dim=-1).amax(),
            "z_q_norm_mean": z_q.norm(dim=-1).mean(),
            "z_q_norm_max": z_q.norm(dim=-1).amax(),
            "z_e_z_q_dist_mean": (z_e_for_vq.detach() - z_q.detach()).norm(dim=-1).mean(),
            "z_e_z_q_cosine_mean": F.cosine_similarity(z_e_for_vq.detach(), z_q.detach(), dim=-1).mean(),
        }
        return z_q, indices, vq_aux_loss, assign_prob, dist, diagnostics

    def decode_modules(self, z_q: torch.Tensor):
        batch_size = z_q.shape[0]
        h = self.decoder_in(z_q.reshape(batch_size * self.num_axes, self.code_dim))
        h = F.gelu(h)
        h = h.view(batch_size, self.num_axes, self.decoder_hidden_dim)
        h = h * self.film_gamma.unsqueeze(0) + self.film_beta.unsqueeze(0)
        h = self.dropout(F.gelu(h))
        raw = self.decoder_out(h.reshape(batch_size * self.num_axes, self.decoder_hidden_dim))
        raw = raw.view(batch_size, self.num_axes, self.input_dim)
        mask = self.module_mask().unsqueeze(0)
        modules = raw * mask
        return modules.sum(dim=1), modules, raw, mask

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        if z_q.ndim == 2:
            z_q = z_q.view(z_q.shape[0], self.num_axes, self.code_dim)
        biological = self.decode_modules(z_q)[0]
        if self.reconstruction_distribution in {"negative_binomial", "nb"}:
            fractions = F.softmax(biological.float(), dim=-1)
            return torch.log1p(fractions * self.count_normalized_total).to(biological.dtype)
        return biological

    def module_losses(self):
        eps = 1e-8
        mask = self.module_mask()
        entropy = -(mask * (mask + eps).log()).sum(dim=0) / math.log(self.num_axes)
        if self.module_effective_axes_target > 0:
            target_concentration = 1.0 / self.module_effective_axes_target
            concentration = mask.pow(2).sum(dim=0)
            sparsity = (concentration - target_concentration).pow(2).mean()
        else:
            sparsity = entropy.mean()
        mass = mask.mean(dim=1)
        balance = F.mse_loss(mass, torch.full_like(mass, 1.0 / self.num_axes))
        sim = F.normalize(mask, dim=-1) @ F.normalize(mask, dim=-1).t()
        overlap = _offdiag(sim).pow(2).mean() if sim.numel() > 1 else mask.new_tensor(0.0)
        if self.module_mask_parameterization in {"prior_guided_softmax", "learned_prior_softmax"}:
            prior = self.fixed_module_mask
            prior_deviation = (mask - prior).pow(2).sum(dim=0).mean()
        else:
            prior_deviation = mask.new_tensor(0.0)
        return sparsity, balance, overlap, prior_deviation

    def codebook_separation_loss(self):
        code = F.normalize(self.codebook, dim=-1)
        centers = F.normalize(code.mean(dim=1), dim=-1)
        sim = centers @ centers.t()
        return F.relu(_offdiag(sim) - self.sep_margin).pow(2).mean()

    def codebook_aux_losses(self):
        if self.codebook_parameterization in {"free_spherical", "spherical_free", "free_euclidean", "euclidean_free"}:
            zero = self.codebook_vectors.new_zeros(())
            code = F.normalize(self.codebook, dim=-1)
            sims = torch.einsum("akd,ald->akl", code, code)
            eye = torch.eye(self.codebook_size, device=sims.device, dtype=torch.bool).unsqueeze(0)
            intra_div_loss = sims.masked_select(~eye).pow(2).mean()
            return zero, zero, zero, intra_div_loss, zero
        codebook = self.codebook
        centers = codebook.mean(dim=1)
        center_dir = F.normalize(centers, dim=-1)
        anchor = F.normalize(self.codebook_anchors, dim=-1)
        anchor_loss = F.mse_loss(center_dir, anchor)
        center_norm = centers.norm(dim=-1)
        center_norm_loss = F.mse_loss(center_norm, torch.full_like(center_norm, self.center_norm_target))
        residual = codebook - centers[:, None, :]
        compact_loss = residual.pow(2).mean()
        raw_residual = self.codebook_residual
        raw_residual = raw_residual - (raw_residual * anchor[:, None, :]).sum(dim=-1, keepdim=True) * anchor[:, None, :]
        raw_residual = F.normalize(raw_residual, dim=-1)
        residual_center_loss = raw_residual.mean(dim=1).pow(2).sum(dim=-1).mean()
        code = F.normalize(residual, dim=-1)
        sims = torch.einsum("akd,ald->akl", code, code)
        eye = torch.eye(self.codebook_size, device=sims.device, dtype=torch.bool).unsqueeze(0)
        intra_div_loss = sims.masked_select(~eye).pow(2).mean()
        return anchor_loss, center_norm_loss, compact_loss, intra_div_loss, residual_center_loss

    @torch.no_grad()
    def sweep_axis_states(self, baseline_z_q: torch.Tensor, axis_idx: int) -> torch.Tensor:
        if baseline_z_q.ndim == 2:
            baseline_z_q = baseline_z_q.view(baseline_z_q.shape[0], self.num_axes, self.code_dim)
        outs = []
        codebook = self.quantization_codebook()
        for code_idx in range(self.codebook_size):
            z = baseline_z_q.clone()
            z[:, axis_idx, :] = codebook[axis_idx, code_idx].unsqueeze(0).to(z.device)
            outs.append(self.decode(z))
        return torch.stack(outs, dim=0)

    def nuisance_effect(self, categorical_indices: Optional[dict], batch_size: int, device: torch.device) -> torch.Tensor:
        if self.nuisance_out is None or not categorical_indices:
            return torch.zeros(batch_size, self.input_dim, device=device)
        embedding = torch.zeros(batch_size, self.nuisance_rank, device=device)
        for field in self.nuisance_fields:
            if field not in categorical_indices:
                continue
            weight = self.nuisance_embeddings[field].weight
            has_unknown = bool(self.nuisance_categories[field]) and self.nuisance_categories[field][0] == "<unknown>"
            if self.exclude_unknown_from_nuisance_centering and has_unknown and weight.shape[0] > 1:
                known_mean = weight[1:].mean(dim=0, keepdim=True)
                centered_weight = torch.cat([torch.zeros_like(weight[:1]), weight[1:] - known_mean], dim=0)
            else:
                centered_weight = weight - weight.mean(dim=0, keepdim=True)
            embedding = embedding + F.embedding(categorical_indices[field], centered_weight)
        return self.nuisance_out(embedding)

    def batch_adversary_objective(self, z_q: torch.Tensor, categorical_indices: Optional[dict]):
        zero = z_q.new_zeros(())
        target_field = self.batch_adversary_target_field
        if self.batch_adversary is None or not categorical_indices or target_field not in categorical_indices:
            return zero, zero
        context_idx = categorical_indices.get(self.batch_adversary_context_field)
        if context_idx is None:
            context_idx = torch.zeros(z_q.shape[0], dtype=torch.long, device=z_q.device)
        reversed_z = _GradientReverse.apply(z_q.reshape(z_q.shape[0], -1), self.batch_adversary_grl_scale)
        context = self.batch_context_embedding(context_idx)
        logits = self.batch_adversary(torch.cat([reversed_z, context], dim=-1))
        target = categorical_indices[target_field] - self.batch_adversary_target_offset
        if self.batch_adversary_target_offset:
            valid = target >= 0
            if not bool(valid.any()):
                return zero, zero
            logits = logits[valid]
            target = target[valid]
        loss = F.cross_entropy(logits, target)
        accuracy = (logits.argmax(dim=-1) == target).float().mean()
        return loss, accuracy

    def conditional_code_usage_objective(
        self,
        distance: torch.Tensor,
        categorical_indices: Optional[dict],
    ):
        """Match code usage across datasets within cell classes."""
        zero = distance.new_zeros(())
        if self.conditional_code_usage_weight <= 0 or not categorical_indices:
            return zero, zero, zero
        target_field = self.conditional_code_usage_target_field
        context_field = self.conditional_code_usage_context_field
        if target_field not in categorical_indices or context_field not in categorical_indices:
            return zero, zero, zero

        source = categorical_indices[target_field] - self.conditional_code_usage_target_offset
        context = categorical_indices[context_field] - self.conditional_code_usage_context_offset
        valid = (source >= 0) & (context >= 0)
        if not bool(valid.any()):
            return zero, zero, zero
        source = source[valid]
        context = context[valid]
        probabilities = F.softmax(
            -distance[valid] / self.conditional_code_usage_temperature,
            dim=-1,
        )

        context_losses = []
        used_source_groups = 0
        for context_value in torch.unique(context):
            in_context = context == context_value
            source_tables = []
            for source_value in torch.unique(source[in_context]):
                in_group = in_context & (source == source_value)
                if int(in_group.sum().item()) < self.conditional_code_usage_min_cells:
                    continue
                source_tables.append(probabilities[in_group].mean(dim=0))
            if len(source_tables) < 2:
                continue
            tables = torch.stack(source_tables, dim=0)
            equally_weighted_mean = tables.mean(dim=0, keepdim=True)
            moved_probability_mass = 0.5 * (
                tables - equally_weighted_mean
            ).abs().sum(dim=-1)
            context_losses.append(moved_probability_mass.mean())
            used_source_groups += len(source_tables)
        if not context_losses:
            return zero, zero, zero
        return (
            torch.stack(context_losses).mean(),
            distance.new_tensor(float(len(context_losses))),
            distance.new_tensor(float(used_source_groups)),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        categorical_indices: Optional[dict] = None,
        count_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_e = self.encode(x)
        z_q, indices, vq_aux_loss, assign_prob, distance, vq_diagnostics = self.quantize(z_e)
        biological_recon, module_recon, raw_recon, module_mask = self.decode_modules(z_q)
        nuisance_recon = self.nuisance_effect(categorical_indices, x.shape[0], x.device)
        count_nll = x.new_zeros(())
        library_size_mean = x.new_zeros(())
        inverse_dispersion_mean = x.new_zeros(())
        expected_counts = None
        gene_fraction = None
        if self.reconstruction_distribution in {"negative_binomial", "nb"}:
            if count_target is None:
                raise ValueError(
                    "negative_binomial reconstruction requires count_target; set data.return_raw_counts=true"
                )
            if count_target.shape != x.shape:
                raise ValueError(
                    f"count_target has shape {tuple(count_target.shape)}, expected {tuple(x.shape)}"
                )
            count_target = count_target.float()
            if bool((count_target < 0).any()):
                raise ValueError("negative_binomial reconstruction requires nonnegative raw counts")
            gene_logits = biological_recon + self.nuisance_decoder_weight * nuisance_recon
            observed_mask = None
            if mask is not None:
                if mask.shape != gene_logits.shape:
                    raise ValueError(f"mask has shape {tuple(mask.shape)}, expected {tuple(gene_logits.shape)}")
                observed_mask = mask > 0
                if bool((observed_mask.sum(dim=-1) == 0).any()):
                    raise ValueError("every cell must have at least one observed gene")
                gene_logits = gene_logits.masked_fill(~observed_mask, torch.finfo(torch.float32).min)
            gene_fraction = F.softmax(gene_logits.float(), dim=-1)
            library_size = count_target.sum(dim=-1, keepdim=True).clamp_min(1.0)
            expected_counts = library_size * gene_fraction
            inverse_dispersion = F.softplus(self.inverse_dispersion_raw.float()).clamp_min(1e-4)
            recon_loss = negative_binomial_nll(
                count_target,
                expected_counts,
                inverse_dispersion.unsqueeze(0),
                mask,
            )
            count_nll = recon_loss
            library_size_mean = library_size.mean()
            inverse_dispersion_mean = inverse_dispersion.mean()
            x_recon = torch.log1p(gene_fraction * self.count_normalized_total).to(x.dtype)
            biological_logits_for_output = biological_recon
            if observed_mask is not None:
                biological_logits_for_output = biological_logits_for_output.masked_fill(
                    ~observed_mask, torch.finfo(torch.float32).min
                )
            biological_output = torch.log1p(
                F.softmax(biological_logits_for_output.float(), dim=-1) * self.count_normalized_total
            ).to(x.dtype)
        else:
            x_recon = biological_recon + self.nuisance_decoder_weight * nuisance_recon
            biological_output = biological_recon
            recon_loss = masked_mse(x_recon, x, mask)
        hard_diversity_loss, entropy, active = code_usage_loss(indices, self.codebook_size, self.num_axes)
        diversity_loss, soft_entropy = soft_code_usage_loss(assign_prob)
        sparsity, balance, overlap, module_prior_deviation = self.module_losses()
        anchor_loss, center_norm_loss, compact_loss, intra_div_loss, residual_center_loss = self.codebook_aux_losses()
        codebook_sep_loss = self.codebook_separation_loss()
        batch_adversary_loss, batch_adversary_accuracy = self.batch_adversary_objective(z_q, categorical_indices)
        conditional_code_usage_loss, aligned_contexts, aligned_source_groups = (
            self.conditional_code_usage_objective(distance, categorical_indices)
        )
        # (many loss terms are actually weighted by zero)
        loss = (
            recon_loss
            + self.vq_aux_loss_weight * vq_aux_loss
            + self.diversity_loss_weight * diversity_loss
            + self.module_sparsity_weight * sparsity
            + self.module_balance_weight * balance
            + self.module_overlap_weight * overlap
            + self.module_mask_prior_weight * module_prior_deviation
            + self.anchor_loss_weight * anchor_loss
            + self.center_norm_weight * center_norm_loss
            + self.codebook_compact_weight * compact_loss
            + self.intra_codebook_div_weight * intra_div_loss
            + self.residual_center_weight * residual_center_loss
            + self.batch_adversary_weight * batch_adversary_loss
            + self.conditional_code_usage_weight * conditional_code_usage_loss
        )
        if self.codebook_sep_weight > 0:
            loss = loss + self.codebook_sep_weight * codebook_sep_loss

        output = {
            "loss": loss,
            "recon_loss": recon_loss,
            "count_nll": count_nll,
            "library_size_mean": library_size_mean,
            "inverse_dispersion_mean": inverse_dispersion_mean,
            "vq_aux_loss": vq_aux_loss,
            **vq_diagnostics,
            "diversity_loss": diversity_loss,
            "hard_diversity_loss": hard_diversity_loss,
            "code_entropy": entropy,
            "soft_code_entropy": soft_entropy,
            "active_code_ratio": active,
            "module_sparsity_loss": sparsity,
            "module_balance_loss": balance,
            "module_overlap_loss": overlap,
            "module_mask_prior_deviation_loss": module_prior_deviation,
            "module_effective_axes_mean": module_mask.squeeze(0).pow(2).sum(dim=0).clamp_min(1e-12).reciprocal().mean(),
            "module_effective_axes_median": module_mask.squeeze(0).pow(2).sum(dim=0).clamp_min(1e-12).reciprocal().median(),
            "module_dominant_axis_weight_mean": module_mask.squeeze(0).max(dim=0).values.mean(),
            "codebook_sep_loss": codebook_sep_loss,
            "anchor_loss": anchor_loss,
            "center_norm_loss": center_norm_loss,
            "codebook_compact_loss": compact_loss,
            "intra_codebook_div_loss": intra_div_loss,
            "residual_center_loss": residual_center_loss,
            "batch_adversary_loss": batch_adversary_loss,
            "batch_adversary_accuracy": batch_adversary_accuracy,
            "conditional_code_usage_loss": conditional_code_usage_loss,
            "conditional_code_usage_contexts": aligned_contexts,
            "conditional_code_usage_source_groups": aligned_source_groups,
            "nuisance_effect_rms": nuisance_recon.pow(2).mean().sqrt(),
            "x_recon": x_recon,
            "biological_recon": biological_output,
            "nuisance_recon": nuisance_recon,
            "biological_logits": biological_recon,
            "z_e": z_e,
            "z_q": z_q,
            "indices": indices,
            "module_mask": module_mask.squeeze(0),
        }
        if return_aux:
            output["module_recon"] = module_recon
            output["raw_recon"] = raw_recon
            if expected_counts is not None:
                output["expected_counts"] = expected_counts
                output["gene_fraction"] = gene_fraction
        return output


def build_lexa_model(
    input_dim: int,
    config: Dict[str, Any],
    gene_vocab: Optional[Sequence[str]] = None,
) -> LEXA:
    """Build LEXA with its configured scaffold."""
    model_cfg = dict(config["model"])
    architecture = model_cfg.pop("architecture", "fixed_axes_vqvae")
    if architecture not in {"fixed_axes_vqvae", "modular_vqvae"}:
        raise ValueError(f"Expected a LEXA architecture, received {architecture!r}")
    prior_path = model_cfg.pop("module_mask_prior_path", None)
    mask_mode = str(model_cfg.get("module_mask_parameterization", "learned_softmax")).lower()
    if mask_mode in {"fixed_prior", "fixed", "prior_guided_softmax", "learned_prior_softmax"}:
        if not prior_path:
            raise ValueError("model.module_mask_prior_path is required for the selected scaffold mode")
        payload = np.load(prior_path, allow_pickle=False)
        try:
            if "module_mask" not in payload.files:
                raise ValueError(f"{prior_path} does not contain module_mask")
            prior = payload["module_mask"].astype(np.float32)
            prior_genes = payload["genes"].astype(str).tolist() if "genes" in payload.files else None
        finally:
            payload.close()
        expected = (int(model_cfg.get("num_axes", 32)), int(input_dim))
        if tuple(prior.shape) != expected:
            raise ValueError(f"Scaffold shape {prior.shape} does not match {expected}")
        if prior_genes is not None:
            if gene_vocab is None or prior_genes != [str(gene) for gene in gene_vocab]:
                raise ValueError("Scaffold gene order does not match the expression vocabulary")
        model_cfg["module_mask_prior"] = prior
    return LEXA(input_dim=input_dim, **model_cfg)
