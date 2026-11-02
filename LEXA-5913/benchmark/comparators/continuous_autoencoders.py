"""Continuous autoencoder baselines."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    error = (pred - target).pow(2)
    if mask is None:
        return error.mean()
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


class _TechnicalBranch(nn.Module):
    def _init_technical(self, input_dim, nuisance_categories, technical_covariate_fields, nuisance_rank):
        self.nuisance_categories = {
            str(key): [str(value) for value in values]
            for key, values in (nuisance_categories or {}).items()
        }
        requested = tuple(technical_covariate_fields or ("dataset_id", "assay"))
        self.nuisance_fields = tuple(field for field in requested if field in self.nuisance_categories)
        self.nuisance_embeddings = nn.ModuleDict({
            field: nn.Embedding(len(self.nuisance_categories[field]), int(nuisance_rank))
            for field in self.nuisance_fields
        })
        self.nuisance_out = (
            nn.Linear(int(nuisance_rank), int(input_dim), bias=False) if self.nuisance_fields else None
        )

    def nuisance_effect(self, categorical_indices, batch_size, device):
        if self.nuisance_out is None or not categorical_indices:
            return torch.zeros(batch_size, self.input_dim, device=device)
        embedding = torch.zeros(batch_size, self.nuisance_out.in_features, device=device)
        for field in self.nuisance_fields:
            if field not in categorical_indices:
                continue
            weight = self.nuisance_embeddings[field].weight
            embedding = embedding + F.embedding(categorical_indices[field], weight - weight.mean(dim=0, keepdim=True))
        return self.nuisance_out(embedding)


class ContinuousAE(_TechnicalBranch):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (2048,),
        latent_dim: int = 128,
        decoder_hidden_dim: int = 2048,
        dropout: float = 0.1,
        nuisance_categories: Optional[dict] = None,
        technical_covariate_fields: Optional[Tuple[str, ...]] = None,
        nuisance_rank: int = 16,
        nuisance_decoder_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_axes = 1
        self.code_dim = self.latent_dim
        self.codebook_size = 0
        self.reconstruction_distribution = "mse_log1p"
        dims = [self.input_dim, *hidden_dims, self.latent_dim]
        enc = []
        for left, right in zip(dims[:-1], dims[1:]):
            enc.extend([nn.Linear(left, right), nn.GELU(), nn.Dropout(dropout)])
        enc[-2:] = [nn.LayerNorm(self.latent_dim)]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, int(decoder_hidden_dim)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(decoder_hidden_dim), self.input_dim),
        )
        self.nuisance_decoder_weight = float(nuisance_decoder_weight)
        self._init_technical(input_dim, nuisance_categories, technical_covariate_fields, nuisance_rank)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z.reshape(z.shape[0], -1))

    def module_mask(self):
        return torch.ones(1, self.input_dim, device=next(self.parameters()).device)

    def forward(self, x, mask=None, count_target=None, categorical_indices=None, return_aux=False):
        del count_target, return_aux
        z = self.encode(x)
        biological = self.decode(z)
        nuisance = self.nuisance_effect(categorical_indices, len(x), x.device)
        recon = biological + self.nuisance_decoder_weight * nuisance
        loss = masked_mse(recon, x, mask)
        zero = loss.new_zeros(())
        return {
            "loss": loss, "recon_loss": loss, "vq_aux_loss": zero,
            "x_recon": recon, "z_e": z, "z_q": z,
            "module_mask": self.module_mask(),
            "nuisance_effect_rms": nuisance.pow(2).mean().sqrt(),
        }


class ModularContinuousAE(_TechnicalBranch):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (2048,),
        num_axes: int = 32,
        code_dim: int = 64,
        decoder_hidden_dim: int = 2048,
        dropout: float = 0.1,
        module_mask_parameterization: str = "uniform",
        module_mask_prior: Optional[torch.Tensor] = None,
        module_mask_temperature: float = 0.5,
        module_mask_logit_init_std: float = 0.02,
        nuisance_categories: Optional[dict] = None,
        technical_covariate_fields: Optional[Tuple[str, ...]] = None,
        nuisance_rank: int = 16,
        nuisance_decoder_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.input_dim = int(input_dim)
        self.num_axes = int(num_axes)
        self.code_dim = int(code_dim)
        self.codebook_size = 0
        self.reconstruction_distribution = "mse_log1p"
        self.module_mask_parameterization = str(module_mask_parameterization).lower()
        self.module_mask_temperature = float(module_mask_temperature)
        dims = [self.input_dim, *hidden_dims, self.num_axes * self.code_dim]
        enc = []
        for left, right in zip(dims[:-1], dims[1:]):
            enc.extend([nn.Linear(left, right), nn.GELU(), nn.Dropout(dropout)])
        enc[-2:] = [nn.LayerNorm(self.num_axes * self.code_dim)]
        self.encoder = nn.Sequential(*enc)
        self.decoder_in = nn.Linear(self.code_dim, int(decoder_hidden_dim))
        self.decoder_out = nn.Linear(int(decoder_hidden_dim), self.input_dim)
        self.film_gamma = nn.Parameter(torch.ones(self.num_axes, int(decoder_hidden_dim)))
        self.film_beta = nn.Parameter(torch.zeros(self.num_axes, int(decoder_hidden_dim)))
        self.dropout = nn.Dropout(dropout)
        if self.module_mask_parameterization == "uniform":
            self.register_buffer("fixed_module_mask", torch.full((self.num_axes, self.input_dim), 1.0 / self.num_axes))
            self.register_parameter("assignment_logits", None)
        elif module_mask_prior is not None:
            prior = torch.as_tensor(module_mask_prior, dtype=torch.float32)
            prior = prior / prior.sum(dim=0, keepdim=True).clamp_min(1e-12)
            self.register_buffer("fixed_module_mask", prior)
            self.assignment_logits = nn.Parameter(self.module_mask_temperature * prior.clamp_min(1e-12).log())
        else:
            self.register_buffer("fixed_module_mask", torch.empty(0), persistent=False)
            self.assignment_logits = nn.Parameter(torch.randn(self.num_axes, self.input_dim) * float(module_mask_logit_init_std))
        self.nuisance_decoder_weight = float(nuisance_decoder_weight)
        self._init_technical(input_dim, nuisance_categories, technical_covariate_fields, nuisance_rank)

    def module_mask(self):
        if self.module_mask_parameterization == "uniform":
            return self.fixed_module_mask
        return F.softmax(self.assignment_logits / self.module_mask_temperature, dim=0)

    def encode(self, x):
        return self.encoder(x).view(len(x), self.num_axes, self.code_dim)

    def decode_modules(self, z):
        h = F.gelu(self.decoder_in(z.reshape(-1, self.code_dim)))
        h = h.view(len(z), self.num_axes, -1)
        h = self.dropout(F.gelu(h * self.film_gamma.unsqueeze(0) + self.film_beta.unsqueeze(0)))
        raw = self.decoder_out(h.reshape(-1, h.shape[-1])).view(len(z), self.num_axes, self.input_dim)
        modules = raw * self.module_mask().unsqueeze(0)
        return modules.sum(dim=1), modules, raw

    def decode(self, z):
        return self.decode_modules(z.view(len(z), self.num_axes, self.code_dim))[0]

    def forward(self, x, mask=None, count_target=None, categorical_indices=None, return_aux=False):
        del count_target
        z = self.encode(x)
        biological, modules, raw = self.decode_modules(z)
        nuisance = self.nuisance_effect(categorical_indices, len(x), x.device)
        recon = biological + self.nuisance_decoder_weight * nuisance
        loss = masked_mse(recon, x, mask)
        zero = loss.new_zeros(())
        out = {
            "loss": loss, "recon_loss": loss, "vq_aux_loss": zero,
            "x_recon": recon, "z_e": z, "z_q": z,
            "module_mask": self.module_mask(),
            "nuisance_effect_rms": nuisance.pow(2).mean().sqrt(),
        }
        if return_aux:
            out.update(module_recon=modules, raw_recon=raw)
        return out
