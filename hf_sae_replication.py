from __future__ import annotations

import heapq
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lazy_import_datasets():
    from datasets import load_dataset

    return load_dataset


def _lazy_import_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return AutoModelForCausalLM, AutoTokenizer


def _lazy_import_pandas():
    import pandas as pd

    return pd


def _lazy_import_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str | torch.dtype, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if not hasattr(torch, dtype):
        raise ValueError(f"Unsupported dtype: {dtype}")
    return getattr(torch, dtype)


def inverse_softplus(x: float) -> float:
    if x <= 0:
        return -20.0
    return float(torch.log(torch.expm1(torch.tensor(x))).item())


def paper_like_layer_index(num_hidden_layers: int) -> int:
    return min(num_hidden_layers - 1, max(0, int(num_hidden_layers * 5 / 6)))


def normalize_residual(
    x: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mean = x.mean(dim=-1, keepdim=True)
    centered = x - mean
    norm = centered.norm(dim=-1, keepdim=True).clamp_min(eps)
    return centered / norm, {"mean": mean, "norm": norm}


def denormalize_residual(x: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return x * stats["norm"] + stats["mean"]


def center_and_unit_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    normalized, _ = normalize_residual(x, eps=eps)
    return normalized


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return ((x - y) ** 2).mean()


def normalized_mse(reconstruction: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return (((reconstruction - original) ** 2).mean(dim=1) / original.pow(2).mean(dim=1)).mean()


def normalized_l1(latents: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return (latents.abs().sum(dim=1) / original.norm(dim=1).clamp_min(1e-8)).mean()


def geometric_median(points: torch.Tensor, max_iter: int = 50, tol: float = 1e-5) -> torch.Tensor:
    current = points.mean(dim=0)
    for _ in range(max_iter):
        distances = (points - current).norm(dim=1).clamp_min(1e-6)
        weights = 1.0 / distances
        updated = (weights[:, None] * points).sum(dim=0) / weights.sum()
        if torch.norm(updated - current) <= tol:
            return updated
        current = updated
    return current


class JumpReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        threshold: torch.Tensor,
        ste_gradient: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, threshold)
        ctx.ste_gradient = ste_gradient
        return torch.where(x > threshold, x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        active = (x > threshold).to(grad_output.dtype)
        if ctx.ste_gradient:
            grad_x = grad_output
        else:
            grad_x = grad_output * active
        grad_threshold = -(grad_output * active)
        grad_threshold = grad_threshold.sum(dim=0)
        return grad_x, grad_threshold, None


def jump_relu(
    x: torch.Tensor,
    threshold: torch.Tensor,
    ste_gradient: bool,
) -> torch.Tensor:
    return JumpReLUFunction.apply(x, threshold, ste_gradient)


@dataclass
class SubjectModelConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B"
    dtype: str = "auto"
    device: str | None = None
    layer_index: int | None = None
    sequence_length: int = 64
    hf_token: str | None = None


@dataclass
class DatasetConfig:
    dataset_name: str = "Skylion007/openwebtext"
    dataset_config_name: str | None = None
    split: str = "train"
    text_field: str = "text"
    streaming: bool = True
    shuffle_buffer: int = 1_000
    seed: int = 0


@dataclass
class SAEConfig:
    variant: str = "topk"
    n_latents: int = 16_384
    k: int = 32
    auxk: int = 256
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_steps: int = 500
    stats_batches: int = 8
    eval_batches: int = 8
    downstream_batches: int = 8
    dead_steps_threshold: int = 250
    auxk_coef: float = 1 / 32
    l1_coef: float = 0.0
    jump_relu_init: float = 0.1
    grad_clip_norm: float | None = 1.0
    sae_dtype: str = "float32"
    log_every: int = 25
    label: str | None = None


@dataclass
class Figure5SweepConfig:
    n_latents: int = 131_072
    batch_size: int = 16
    num_steps: int = 300
    stats_batches: int = 8
    eval_batches: int = 8
    downstream_batches: int = 8
    learning_rate: float = 3e-4
    topk_values: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 256)
    relu_l1_coefs: tuple[float, ...] = (1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1)
    prolu_l1_coefs: tuple[float, ...] = (1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1)
    gated_l1_coefs: tuple[float, ...] = (2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1)
    jump_relu_init: float = 0.1
    sae_dtype: str = "float32"


@dataclass
class FixedL0SweepConfig:
    target_l0: int = 128
    n_latents_values: tuple[int, ...] = (4_096, 8_192, 16_384, 32_768)
    batch_size: int = 16
    num_steps: int = 300
    stats_batches: int = 8
    eval_batches: int = 8
    downstream_batches: int = 8
    learning_rate: float = 3e-4
    relu_l1_coefs: tuple[float, ...] = (1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1)
    prolu_l1_coefs: tuple[float, ...] = (1e-3, 2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1)
    gated_l1_coefs: tuple[float, ...] = (2.5e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1)
    jump_relu_init: float = 0.1
    sae_dtype: str = "float32"


@dataclass
class SubjectModelBundle:
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    layer_index: int
    d_model: int
    sequence_length: int
    model_name: str


def sae_variant_label(variant: str) -> str:
    labels = {
        "relu": "ReLU",
        "prolu_relu": "ProLU ReLU",
        "prolu_ste": "ProLU STE",
        "gated": "Gated",
        "topk": "TopK",
    }
    return labels.get(variant, variant)


def unit_norm_decoder_(sae: "SparseAutoencoder") -> None:
    sae.decoder.weight.data /= sae.decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)


def unit_norm_decoder_grad_adjustment_(sae: "SparseAutoencoder") -> None:
    if sae.decoder.weight.grad is None:
        return
    parallel = (sae.decoder.weight.grad * sae.decoder.weight.data).sum(dim=0, keepdim=True)
    sae.decoder.weight.grad -= sae.decoder.weight.data * parallel


def sample_uniform_directions(n_latents: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    directions = torch.randn(n_latents, d_model, device=device, dtype=dtype)
    return directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)


def align_encoder_to_decoder_directions(
    encoder_weight: torch.Tensor,
    directions: torch.Tensor,
) -> None:
    row_norms = encoder_weight.data.norm(dim=1, keepdim=True).clamp_min(1e-8)
    encoder_weight.data = directions * row_norms


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        cfg: SAEConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.variant = cfg.variant
        self.d_model = d_model
        self.n_latents = cfg.n_latents
        self.k = cfg.k
        self.auxk = cfg.auxk
        self.dead_steps_threshold = cfg.dead_steps_threshold
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
        self.register_buffer("stats_last_nonzero", torch.zeros(cfg.n_latents, dtype=torch.long))

        if self.variant == "gated":
            self.gate_encoder = nn.Linear(d_model, cfg.n_latents, bias=False)
            self.mag_encoder = nn.Linear(d_model, cfg.n_latents, bias=False)
            self.gate_bias = nn.Parameter(torch.zeros(cfg.n_latents))
            self.mag_bias = nn.Parameter(torch.zeros(cfg.n_latents))
        else:
            self.encoder = nn.Linear(d_model, cfg.n_latents, bias=False)
            self.latent_bias = nn.Parameter(torch.zeros(cfg.n_latents))

        if self.variant in {"prolu_relu", "prolu_ste"}:
            raw_init = inverse_softplus(cfg.jump_relu_init)
            self.raw_threshold = nn.Parameter(torch.full((cfg.n_latents,), raw_init))

        self.decoder = nn.Linear(cfg.n_latents, d_model, bias=False)
        directions = sample_uniform_directions(
            cfg.n_latents,
            d_model,
            self.decoder.weight.device,
            self.decoder.weight.dtype,
        )
        self.decoder.weight.data = directions.T.contiguous()

        if self.variant == "gated":
            align_encoder_to_decoder_directions(self.gate_encoder.weight, directions)
            align_encoder_to_decoder_directions(self.mag_encoder.weight, directions)
        else:
            align_encoder_to_decoder_directions(self.encoder.weight, directions)

        unit_norm_decoder_(self)

    @property
    def threshold(self) -> torch.Tensor:
        if not hasattr(self, "raw_threshold"):
            raise AttributeError("Threshold is only defined for ProLU variants.")
        return F.softplus(self.raw_threshold)

    def encode_pre_act(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - self.pre_bias
        if self.variant == "gated":
            return F.linear(centered, self.gate_encoder.weight, self.gate_bias)
        return F.linear(centered, self.encoder.weight, self.latent_bias)

    def _topk_relu(self, latents_pre_act: torch.Tensor, k: int) -> torch.Tensor:
        k = min(k, latents_pre_act.shape[-1])
        topk = torch.topk(latents_pre_act, k=k, dim=-1)
        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, topk.indices, F.relu(topk.values))
        return latents

    def _auxiliary_latents(self, latents_pre_act: torch.Tensor) -> torch.Tensor | None:
        if self.variant != "topk" or self.auxk <= 0:
            return None
        dead_mask = self.stats_last_nonzero > self.dead_steps_threshold
        if not bool(dead_mask.any()):
            return None
        k = min(self.auxk, int(dead_mask.sum().item()))
        masked = latents_pre_act.masked_fill(~dead_mask.unsqueeze(0), float("-inf"))
        topk = torch.topk(masked, k=k, dim=-1)
        aux_latents = torch.zeros_like(latents_pre_act)
        aux_latents.scatter_(-1, topk.indices, F.relu(topk.values))
        return aux_latents

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        centered = x - self.pre_bias
        if self.variant == "gated":
            gate_pre = F.linear(centered, self.gate_encoder.weight, self.gate_bias)
            mag_pre = F.linear(centered, self.mag_encoder.weight, self.mag_bias)
            gate_mask = (gate_pre > 0).to(centered.dtype)
            magnitudes = F.relu(mag_pre)
            latents = magnitudes * gate_mask
            return {
                "latents_pre_act": gate_pre,
                "latents": latents,
                "aux_latents": None,
                "sparsity_proxy": F.relu(gate_pre),
            }

        latents_pre_act = F.linear(centered, self.encoder.weight, self.latent_bias)
        if self.variant == "relu":
            latents = F.relu(latents_pre_act)
        elif self.variant == "prolu_relu":
            latents = jump_relu(latents_pre_act, self.threshold, ste_gradient=False)
        elif self.variant == "prolu_ste":
            latents = jump_relu(latents_pre_act, self.threshold, ste_gradient=True)
        elif self.variant == "topk":
            latents = self._topk_relu(latents_pre_act, self.k)
        else:
            raise ValueError(f"Unsupported SAE variant: {self.variant}")

        return {
            "latents_pre_act": latents_pre_act,
            "latents": latents,
            "aux_latents": self._auxiliary_latents(latents_pre_act),
            "sparsity_proxy": latents,
        }

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        info = self.encode(x)
        latents = info["latents"]
        assert latents is not None
        if self.training:
            self.stats_last_nonzero += 1
            self.stats_last_nonzero[(latents > 1e-3).any(dim=0)] = 0
        return self.decode(latents), info


def load_subject_model(cfg: SubjectModelConfig) -> SubjectModelBundle:
    AutoModelForCausalLM, AutoTokenizer = _lazy_import_transformers()
    device = torch.device(cfg.device) if cfg.device else default_device()
    dtype = resolve_dtype(cfg.dtype, device)
    token = cfg.hf_token or os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        token=token,
    )
    model.to(device)
    model.eval()

    num_hidden_layers = int(model.config.num_hidden_layers)
    layer_index = cfg.layer_index if cfg.layer_index is not None else paper_like_layer_index(num_hidden_layers)
    d_model = int(model.config.hidden_size)

    return SubjectModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        layer_index=layer_index,
        d_model=d_model,
        sequence_length=cfg.sequence_length,
        model_name=cfg.model_name,
    )


def iter_token_windows(
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    max_windows: int | None = None,
) -> Iterator[list[int]]:
    load_dataset = _lazy_import_datasets()
    dataset = load_dataset(
        data_cfg.dataset_name,
        data_cfg.dataset_config_name,
        split=data_cfg.split,
        streaming=data_cfg.streaming,
    )
    if data_cfg.streaming:
        dataset = dataset.shuffle(seed=data_cfg.seed, buffer_size=data_cfg.shuffle_buffer)

    produced = 0
    eos_token_id = bundle.tokenizer.eos_token_id
    for row in dataset:
        text = row.get(data_cfg.text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        token_ids = bundle.tokenizer(text, add_special_tokens=False)["input_ids"]
        if eos_token_id is not None:
            token_ids = token_ids + [eos_token_id]
        for start in range(0, len(token_ids) - bundle.sequence_length + 1, bundle.sequence_length):
            window = token_ids[start : start + bundle.sequence_length]
            if len(window) != bundle.sequence_length:
                continue
            yield window
            produced += 1
            if max_windows is not None and produced >= max_windows:
                return


def batched_token_windows(
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    batch_size: int,
    max_windows: int | None = None,
) -> Iterator[torch.Tensor]:
    batch: list[list[int]] = []
    for window in iter_token_windows(bundle, data_cfg, max_windows=max_windows):
        batch.append(window)
        if len(batch) == batch_size:
            yield torch.tensor(batch, dtype=torch.long, device=bundle.device)
            batch = []
    if batch:
        yield torch.tensor(batch, dtype=torch.long, device=bundle.device)


def collect_token_batches(
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    batch_size: int,
    num_batches: int,
) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    iterator = batched_token_windows(
        bundle,
        data_cfg,
        batch_size=batch_size,
        max_windows=batch_size * num_batches,
    )
    for _, token_batch in zip(range(num_batches), iterator):
        batches.append(token_batch)
    if not batches:
        raise RuntimeError("No token batches were collected from the dataset.")
    return batches


@torch.no_grad()
def subject_hidden_states(
    bundle: SubjectModelBundle,
    input_ids: torch.Tensor,
    return_stats: bool = False,
) -> Any:
    outputs = bundle.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_state = outputs.hidden_states[bundle.layer_index + 1].float()
    normalized, stats = normalize_residual(hidden_state)
    if return_stats:
        return normalized, stats, hidden_state
    return normalized


def iter_activation_batches(
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    batch_size: int,
    max_windows: int | None = None,
) -> Iterator[torch.Tensor]:
    for token_batch in batched_token_windows(bundle, data_cfg, batch_size=batch_size, max_windows=max_windows):
        acts = subject_hidden_states(bundle, token_batch)
        yield acts.reshape(-1, acts.shape[-1])


def collect_stats_sample(
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    batch_size: int,
    num_batches: int,
) -> torch.Tensor:
    sample: list[torch.Tensor] = []
    iterator = iter_activation_batches(
        bundle,
        data_cfg,
        batch_size=batch_size,
        max_windows=batch_size * num_batches,
    )
    for _, batch in zip(range(num_batches), iterator):
        sample.append(batch)
    if not sample:
        raise RuntimeError("No activation batches were collected from the dataset.")
    return torch.cat(sample, dim=0)


def initialize_sae(sae: SparseAutoencoder, stats_sample: torch.Tensor) -> float:
    with torch.no_grad():
        sae.pre_bias.copy_(geometric_median(stats_sample))
        if sae.variant == "topk":
            # Match the OpenAI reference: rescale the TopK encoder so initial
            # reconstructed vectors have roughly the same norm as the inputs.
            x = torch.randn(256, sae.d_model, device=stats_sample.device, dtype=stats_sample.dtype)
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            x = x + sae.pre_bias.data
            recons, _ = sae(x)
            recons_norm = (recons - sae.pre_bias.data).norm(dim=-1).mean().clamp_min(1e-8)
            sae.encoder.weight.data /= recons_norm.item()
    mse_scale = 1.0 / ((stats_sample.float().mean(dim=0) - stats_sample.float()) ** 2).mean().item()
    return float(mse_scale)


def build_sae(bundle: SubjectModelBundle, cfg: SAEConfig) -> SparseAutoencoder:
    sae = SparseAutoencoder(bundle.d_model, cfg)
    sae_dtype = resolve_dtype(cfg.sae_dtype, bundle.device)
    return sae.to(device=bundle.device, dtype=sae_dtype)


def compute_training_loss(
    sae: SparseAutoencoder,
    batch: torch.Tensor,
    recons: torch.Tensor,
    info: dict[str, torch.Tensor | None],
    cfg: SAEConfig,
    mse_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    recons_loss = mse_scale * mse(recons.float(), batch.float())
    loss = recons_loss
    logs = {"recons_loss": float(recons_loss.item())}

    if sae.variant == "topk":
        aux_latents = info["aux_latents"]
        if aux_latents is not None:
            aux_recons = sae.decode(aux_latents)
            residual_target = batch - recons.detach() + sae.pre_bias.detach()
            aux_loss = cfg.auxk_coef * normalized_mse(aux_recons.float(), residual_target.float()).nan_to_num(0.0)
            loss = loss + aux_loss
            logs["aux_loss"] = float(aux_loss.item())
        else:
            logs["aux_loss"] = 0.0
    else:
        sparsity_proxy = info["sparsity_proxy"]
        assert sparsity_proxy is not None
        l1_loss = cfg.l1_coef * normalized_l1(sparsity_proxy.float(), batch.float())
        loss = loss + l1_loss
        logs["l1_loss"] = float(l1_loss.item())

    return loss, logs


def train_sparse_autoencoder(
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    cfg: SAEConfig,
) -> tuple[SparseAutoencoder, list[dict[str, float]]]:
    stats_sample = collect_stats_sample(
        bundle,
        data_cfg,
        batch_size=cfg.batch_size,
        num_batches=cfg.stats_batches,
    )
    sae = build_sae(bundle, cfg)
    mse_scale = initialize_sae(sae, stats_sample.to(bundle.device, dtype=sae.pre_bias.dtype))

    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.learning_rate)
    logs: list[dict[str, float]] = []
    activation_iterator = iter_activation_batches(bundle, data_cfg, batch_size=cfg.batch_size)

    sae.train()
    for step, batch in enumerate(activation_iterator, start=1):
        if step > cfg.num_steps:
            break

        batch = batch.to(bundle.device, dtype=sae.pre_bias.dtype)
        optimizer.zero_grad(set_to_none=True)
        recons, info = sae(batch)
        loss, train_logs = compute_training_loss(sae, batch, recons, info, cfg, mse_scale)
        loss.backward()

        if cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip_norm)
        unit_norm_decoder_grad_adjustment_(sae)
        optimizer.step()
        unit_norm_decoder_(sae)

        latents = info["latents"]
        assert latents is not None
        log = {
            "step": float(step),
            "loss": float(loss.item()),
            "avg_l0": float((latents > 0).sum(dim=1).float().mean().item()),
            "dead_fraction": float((sae.stats_last_nonzero > sae.dead_steps_threshold).float().mean().item()),
            **train_logs,
        }
        logs.append(log)

        if cfg.log_every and step % cfg.log_every == 0:
            print(cfg.label or sae_variant_label(cfg.variant), log)

    return sae.eval(), logs


@torch.no_grad()
def evaluate_sae(
    sae: SparseAutoencoder,
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    batch_size: int,
    num_batches: int,
) -> dict[str, float]:
    sae.eval()
    nmse_values: list[float] = []
    l0_values: list[float] = []
    for _, batch in zip(
        range(num_batches),
        iter_activation_batches(
            bundle,
            data_cfg,
            batch_size=batch_size,
            max_windows=batch_size * num_batches,
        ),
    ):
        batch = batch.to(bundle.device, dtype=sae.pre_bias.dtype)
        recons, info = sae(batch)
        latents = info["latents"]
        assert latents is not None
        nmse_values.append(float(normalized_mse(recons.float(), batch.float()).item()))
        l0_values.append(float((latents > 0).sum(dim=1).float().mean().item()))
    return {
        "normalized_mse": sum(nmse_values) / max(len(nmse_values), 1),
        "avg_l0": sum(l0_values) / max(len(l0_values), 1),
        "dead_fraction": float((sae.stats_last_nonzero > sae.dead_steps_threshold).float().mean().item()),
    }


def get_model_layers(bundle: SubjectModelBundle):
    model = bundle.model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture for residual patching.")


@torch.no_grad()
def language_model_cross_entropy(bundle: SubjectModelBundle, input_ids: torch.Tensor) -> float:
    outputs = bundle.model(input_ids=input_ids, labels=input_ids, use_cache=False)
    return float(outputs.loss.item())


@torch.no_grad()
def evaluate_downstream_delta_cross_entropy(
    sae: SparseAutoencoder,
    bundle: SubjectModelBundle,
    token_batches: list[torch.Tensor],
) -> dict[str, float]:
    sae.eval()
    layers = get_model_layers(bundle)
    layer_module = layers[bundle.layer_index]
    base_losses: list[float] = []
    patched_losses: list[float] = []
    l0_values: list[float] = []

    for token_batch in token_batches:
        token_batch = token_batch.to(bundle.device)
        base_losses.append(language_model_cross_entropy(bundle, token_batch))
        batch_l0 = {"value": 0.0}

        def patch_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            normalized, stats = normalize_residual(hidden.float())
            flat = normalized.reshape(-1, normalized.shape[-1]).to(bundle.device, dtype=sae.pre_bias.dtype)
            recons_norm, info = sae(flat)
            latents = info["latents"]
            assert latents is not None
            batch_l0["value"] = float((latents > 0).sum(dim=1).float().mean().item())
            recons_norm = recons_norm.reshape_as(normalized).float()
            recons = denormalize_residual(recons_norm, stats).to(hidden.dtype)
            if isinstance(output, tuple):
                return (recons,) + output[1:]
            return recons

        handle = layer_module.register_forward_hook(patch_hook)
        try:
            patched_losses.append(language_model_cross_entropy(bundle, token_batch))
        finally:
            handle.remove()
        l0_values.append(batch_l0["value"])

    base_loss = sum(base_losses) / max(len(base_losses), 1)
    patched_loss = sum(patched_losses) / max(len(patched_losses), 1)
    return {
        "cross_entropy": patched_loss,
        "base_cross_entropy": base_loss,
        "delta_cross_entropy": patched_loss - base_loss,
        "avg_l0": sum(l0_values) / max(len(l0_values), 1),
    }


@torch.no_grad()
def collect_feature_examples(
    sae: SparseAutoencoder,
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    feature_idx: int,
    num_examples: int = 8,
    search_windows: int = 256,
) -> list[dict[str, Any]]:
    heap: list[tuple[float, int, dict[str, Any]]] = []
    counter = 0
    token_iterator = batched_token_windows(
        bundle,
        data_cfg,
        batch_size=1,
        max_windows=search_windows,
    )
    for token_batch in token_iterator:
        acts = subject_hidden_states(bundle, token_batch)
        flat = acts.reshape(-1, acts.shape[-1]).to(bundle.device, dtype=sae.pre_bias.dtype)
        encoded = sae.encode(flat)
        latents = encoded["latents"]
        assert latents is not None
        top_value, top_position = latents[:, feature_idx].max(dim=0)
        example = {
            "activation": float(top_value.item()),
            "text": bundle.tokenizer.decode(token_batch[0], skip_special_tokens=True),
            "token_position": int(top_position.item()),
        }
        heap_item = (example["activation"], counter, example)
        counter += 1
        if len(heap) < num_examples:
            heapq.heappush(heap, heap_item)
        else:
            heapq.heappushpop(heap, heap_item)
    return [item[2] for item in sorted(heap, key=lambda x: x[0], reverse=True)]


def explain_feature_with_open_model(
    examples: list[dict[str, Any]],
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    hf_token: str | None = None,
    max_new_tokens: int = 192,
) -> str:
    AutoModelForCausalLM, AutoTokenizer = _lazy_import_transformers()
    device = default_device()
    dtype = resolve_dtype("auto", device)
    token = hf_token or os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, token=token).to(device)
    model.eval()

    formatted_examples = "\n\n".join(
        f"Example {i + 1} (activation={ex['activation']:.3f}, token_position={ex['token_position']}):\n{ex['text']}"
        for i, ex in enumerate(examples)
    )
    prompt = (
        "You are analyzing a sparse autoencoder feature from a language model.\n"
        "Given the top activating text windows below, write a short explanation of what concept this feature detects.\n"
        "Then list two likely false positives.\n\n"
        f"{formatted_examples}"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
        except (AttributeError, ImportError):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = output_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_figure5_sae_configs(sweep_cfg: Figure5SweepConfig) -> list[SAEConfig]:
    base = dict(
        n_latents=sweep_cfg.n_latents,
        batch_size=sweep_cfg.batch_size,
        num_steps=sweep_cfg.num_steps,
        stats_batches=sweep_cfg.stats_batches,
        eval_batches=sweep_cfg.eval_batches,
        downstream_batches=sweep_cfg.downstream_batches,
        learning_rate=sweep_cfg.learning_rate,
        sae_dtype=sweep_cfg.sae_dtype,
    )
    configs: list[SAEConfig] = []

    for l1 in sweep_cfg.relu_l1_coefs:
        configs.append(
            SAEConfig(
                variant="relu",
                l1_coef=l1,
                label=f"ReLU λ={l1:g}",
                **base,
            )
        )

    for l1 in sweep_cfg.prolu_l1_coefs:
        configs.append(
            SAEConfig(
                variant="prolu_relu",
                l1_coef=l1,
                jump_relu_init=sweep_cfg.jump_relu_init,
                label=f"ProLU ReLU λ={l1:g}",
                **base,
            )
        )
        configs.append(
            SAEConfig(
                variant="prolu_ste",
                l1_coef=l1,
                jump_relu_init=sweep_cfg.jump_relu_init,
                label=f"ProLU STE λ={l1:g}",
                **base,
            )
        )

    for l1 in sweep_cfg.gated_l1_coefs:
        configs.append(
            SAEConfig(
                variant="gated",
                l1_coef=l1,
                label=f"Gated λ={l1:g}",
                **base,
            )
        )

    for k in sweep_cfg.topk_values:
        configs.append(
            SAEConfig(
                variant="topk",
                k=k,
                auxk=max(256, 2 * k),
                label=f"TopK k={k}",
                **base,
            )
        )

    return configs


def build_fixed_l0_sae_configs(sweep_cfg: FixedL0SweepConfig) -> list[SAEConfig]:
    configs: list[SAEConfig] = []
    for n_latents in sweep_cfg.n_latents_values:
        base = dict(
            n_latents=n_latents,
            batch_size=sweep_cfg.batch_size,
            num_steps=sweep_cfg.num_steps,
            stats_batches=sweep_cfg.stats_batches,
            eval_batches=sweep_cfg.eval_batches,
            downstream_batches=sweep_cfg.downstream_batches,
            learning_rate=sweep_cfg.learning_rate,
            sae_dtype=sweep_cfg.sae_dtype,
        )

        configs.append(
            SAEConfig(
                variant="topk",
                k=min(sweep_cfg.target_l0, n_latents),
                auxk=max(256, 2 * min(sweep_cfg.target_l0, n_latents)),
                label=f"TopK n={n_latents} k={min(sweep_cfg.target_l0, n_latents)}",
                **base,
            )
        )

        for l1 in sweep_cfg.relu_l1_coefs:
            configs.append(
                SAEConfig(
                    variant="relu",
                    l1_coef=l1,
                    label=f"ReLU n={n_latents} λ={l1:g}",
                    **base,
                )
            )

        for l1 in sweep_cfg.prolu_l1_coefs:
            configs.append(
                SAEConfig(
                    variant="prolu_relu",
                    l1_coef=l1,
                    jump_relu_init=sweep_cfg.jump_relu_init,
                    label=f"ProLU ReLU n={n_latents} λ={l1:g}",
                    **base,
                )
            )
            configs.append(
                SAEConfig(
                    variant="prolu_ste",
                    l1_coef=l1,
                    jump_relu_init=sweep_cfg.jump_relu_init,
                    label=f"ProLU STE n={n_latents} λ={l1:g}",
                    **base,
                )
            )

        for l1 in sweep_cfg.gated_l1_coefs:
            configs.append(
                SAEConfig(
                    variant="gated",
                    l1_coef=l1,
                    label=f"Gated n={n_latents} λ={l1:g}",
                    **base,
                )
            )
    return configs


def run_figure5_activation_sweep(
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    sweep_cfg: Figure5SweepConfig,
) -> Any:
    pd = _lazy_import_pandas()
    token_batches = collect_token_batches(
        bundle,
        data_cfg,
        batch_size=sweep_cfg.batch_size,
        num_batches=sweep_cfg.downstream_batches,
    )
    rows: list[dict[str, Any]] = []

    for cfg in build_figure5_sae_configs(sweep_cfg):
        print(f"Training {cfg.label or cfg.variant}")
        sae, train_logs = train_sparse_autoencoder(bundle, data_cfg, cfg)
        recons_metrics = evaluate_sae(
            sae,
            bundle,
            data_cfg,
            batch_size=cfg.batch_size,
            num_batches=cfg.eval_batches,
        )
        downstream_metrics = evaluate_downstream_delta_cross_entropy(sae, bundle, token_batches)
        rows.append(
            {
                "label": cfg.label or sae_variant_label(cfg.variant),
                "variant": cfg.variant,
                "variant_label": sae_variant_label(cfg.variant),
                "n_latents": cfg.n_latents,
                "k": cfg.k,
                "l1_coef": cfg.l1_coef,
                "jump_relu_init": cfg.jump_relu_init,
                "num_steps": cfg.num_steps,
                "train_logs": train_logs,
                **recons_metrics,
                **downstream_metrics,
            }
        )

    df = pd.DataFrame(rows)
    return df


def run_fixed_l0_width_sweep(
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    sweep_cfg: FixedL0SweepConfig,
) -> tuple[Any, Any]:
    pd = _lazy_import_pandas()
    token_batches = collect_token_batches(
        bundle,
        data_cfg,
        batch_size=sweep_cfg.batch_size,
        num_batches=sweep_cfg.downstream_batches,
    )
    rows: list[dict[str, Any]] = []

    for cfg in build_fixed_l0_sae_configs(sweep_cfg):
        print(f"Training {cfg.label or cfg.variant}")
        sae, train_logs = train_sparse_autoencoder(bundle, data_cfg, cfg)
        recons_metrics = evaluate_sae(
            sae,
            bundle,
            data_cfg,
            batch_size=cfg.batch_size,
            num_batches=cfg.eval_batches,
        )
        downstream_metrics = evaluate_downstream_delta_cross_entropy(sae, bundle, token_batches)
        rows.append(
            {
                "label": cfg.label or sae_variant_label(cfg.variant),
                "variant": cfg.variant,
                "variant_label": sae_variant_label(cfg.variant),
                "target_l0": sweep_cfg.target_l0,
                "n_latents": cfg.n_latents,
                "k": cfg.k,
                "l1_coef": cfg.l1_coef,
                "jump_relu_init": cfg.jump_relu_init,
                "num_steps": cfg.num_steps,
                "train_logs": train_logs,
                **recons_metrics,
                **downstream_metrics,
            }
        )

    all_df = pd.DataFrame(rows)
    selected_parts = []
    for (variant_label, n_latents), group in all_df.groupby(["variant_label", "n_latents"], sort=True):
        if variant_label == "TopK":
            chosen = group.iloc[[0]]
        else:
            deltas = (group["avg_l0"] - sweep_cfg.target_l0).abs()
            chosen = group.loc[[deltas.idxmin()]]
        selected_parts.append(chosen)
    selected_df = pd.concat(selected_parts, ignore_index=True)
    selected_df["l0_distance"] = (selected_df["avg_l0"] - sweep_cfg.target_l0).abs()
    return all_df, selected_df


def plot_figure5_tradeoff(df: Any, title: str | None = None):
    plt = _lazy_import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    styles = {
        "ReLU": {"color": "#5b9bd5", "marker": "o"},
        "ProLU ReLU": {"color": "#e06666", "marker": "<"},
        "ProLU STE": {"color": "#f6b26b", "marker": ">"},
        "Gated": {"color": "#6aa84f", "marker": "s"},
        "TopK": {"color": "#b394d6", "marker": "*"},
    }

    for label, group in df.groupby("variant_label"):
        if label == "ProLU ReLU":
            continue
        group = group.sort_values("avg_l0")
        style = styles.get(label, {"color": None, "marker": "o"})
        ax.plot(
            group["avg_l0"],
            group["delta_cross_entropy"],
            label=label,
            linewidth=2.0,
            markersize=9 if style["marker"] != "*" else 12,
            marker=style["marker"],
            color=style["color"],
            alpha=0.95,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sparsity (L0)")
    ax.set_ylabel("Delta cross-entropy")
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", alpha=0.5)
    ax.legend(framealpha=0.9)
    ax.text(
        0.94,
        0.94,
        "better",
        transform=ax.transAxes,
        fontsize=12,
        ha="right",
    )
    ax.annotate(
        "",
        xy=(0.82, 0.90),
        xytext=(0.92, 0.90),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0),
    )
    ax.annotate(
        "",
        xy=(0.92, 0.78),
        xytext=(0.92, 0.90),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0),
    )
    fig.tight_layout()
    return fig, ax


def plot_fixed_l0_mse_tradeoff(
    df: Any,
    title: str | None = None,
):
    plt = _lazy_import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    styles = {
        "ReLU": {"color": "#5b9bd5", "marker": "o"},
        "ProLU ReLU": {"color": "#e06666", "marker": "<"},
        "ProLU STE": {"color": "#f6b26b", "marker": ">"},
        "Gated": {"color": "#6aa84f", "marker": "s"},
        "TopK": {"color": "#b394d6", "marker": "*"},
    }

    for label, group in df.groupby("variant_label"):
        group = group.sort_values("normalized_mse")
        style = styles.get(label, {"color": None, "marker": "o"})
        ax.plot(
            group["normalized_mse"],
            group["delta_cross_entropy"],
            label=label,
            linewidth=2.0,
            markersize=9 if style["marker"] != "*" else 12,
            marker=style["marker"],
            color=style["color"],
            alpha=0.95,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Normalized MSE")
    ax.set_ylabel("Delta cross-entropy")
    if title:
        ax.set_title(title)
    ax.grid(True, which="major", alpha=0.5)
    ax.legend(framealpha=0.9)
    ax.text(
        0.94,
        0.07,
        "better",
        transform=ax.transAxes,
        fontsize=12,
        ha="right",
    )
    ax.annotate(
        "",
        xy=(0.82, 0.12),
        xytext=(0.92, 0.12),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0),
    )
    ax.annotate(
        "",
        xy=(0.92, 0.12),
        xytext=(0.92, 0.22),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1.0),
    )
    fig.tight_layout()
    return fig, ax


def run_qwen_figure5_replication(
    hf_token: str | None = None,
    subject_cfg: SubjectModelConfig | None = None,
    data_cfg: DatasetConfig | None = None,
    sweep_cfg: Figure5SweepConfig | None = None,
) -> tuple[Any, Any]:
    subject_cfg = subject_cfg or SubjectModelConfig(hf_token=hf_token)
    data_cfg = data_cfg or DatasetConfig()
    sweep_cfg = sweep_cfg or Figure5SweepConfig()
    bundle = load_subject_model(subject_cfg)
    df = run_figure5_activation_sweep(bundle, data_cfg, sweep_cfg)
    fig, ax = plot_figure5_tradeoff(
        df,
        title=f"Figure 5-style sweep on {bundle.model_name}",
    )
    return df, (fig, ax)


def train_sae(
    bundle: SubjectModelBundle,
    data_cfg: DatasetConfig,
    sae_cfg: SAEConfig,
) -> tuple[SparseAutoencoder, list[dict[str, float]]]:
    return train_sparse_autoencoder(bundle, data_cfg, sae_cfg)


def run_qwen_open_replication(
    hf_token: str | None = None,
    subject_cfg: SubjectModelConfig | None = None,
    data_cfg: DatasetConfig | None = None,
    sae_cfg: SAEConfig | None = None,
) -> dict[str, Any]:
    subject_cfg = subject_cfg or SubjectModelConfig(hf_token=hf_token)
    data_cfg = data_cfg or DatasetConfig()
    sae_cfg = sae_cfg or SAEConfig()

    bundle = load_subject_model(subject_cfg)
    sae, train_logs = train_sparse_autoencoder(bundle, data_cfg, sae_cfg)
    eval_metrics = evaluate_sae(
        sae,
        bundle,
        data_cfg,
        batch_size=sae_cfg.batch_size,
        num_batches=sae_cfg.eval_batches,
    )
    downstream_batches = collect_token_batches(
        bundle,
        data_cfg,
        batch_size=sae_cfg.batch_size,
        num_batches=sae_cfg.downstream_batches,
    )
    downstream_metrics = evaluate_downstream_delta_cross_entropy(sae, bundle, downstream_batches)
    return {
        "subject_cfg": asdict(subject_cfg),
        "data_cfg": asdict(data_cfg),
        "sae_cfg": asdict(sae_cfg),
        "bundle": bundle,
        "sae": sae,
        "train_logs": train_logs,
        "eval_metrics": eval_metrics,
        "downstream_metrics": downstream_metrics,
    }
