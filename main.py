"""EmbeddingGemma 2 — a single-file PyTorch implementation.

EmbeddingGemma 2 (Google DeepMind, Oct 2026) is a 740M-parameter multimodal embedding model that maps text,
code, images, video and audio (and interleavings of them) into one 768-d vector space. It is an
*encoder*: a bidirectional Gemma-4-style transformer whose token outputs are projected 512 -> 768,
mean-pooled and L2-normalised.

This file re-implements the architecture from the official sources (model card, launch blog, the
Hugging Face ``config.json`` and the reference ``transformers`` code for ``embedding_gemma2`` /
``gemma4``) without depending on ``transformers``. Module and parameter names mirror the official
checkpoint, so ``EmbeddingGemma2.from_pretrained`` loads the released safetensors directly.

Architecture (default config)
-----------------------------
Text model (270M = 130M transformer + 140M embedder)
    * vocab 262,144, d_model 512, 24 layers, gated-GELU(tanh) FFN with d_ff 2048
    * local:global = 5:1. Local layers: bidirectional sliding window ``|i-j| <= 512``, 4 query heads x 256,
      2 KV heads (GQA), RoPE theta 1e4. Global layers (5, 11, 17, 23): full attention, 4 query heads x 512,
      1 KV head (MQA), RoPE theta 1e6.
    * QK-RMSNorm (learned scale), V-RMSNorm (no scale), attention scale 1.0, sandwich norms
      (pre + post RMSNorm around attention and FFN).
    * "Projection-only" per-layer embeddings (PLE): ``inputs_embeds`` is projected to 24 x 512 signals;
      every layer gates its residual stream with its own slice through a 512 -> 512 -> 512 bottleneck.
    * final RMSNorm -> linear 512 -> 768 per token -> mean pooling -> L2 normalise.
Vision encoder (170M, shared by images and video frames)
    * 16-pixel patches -> linear(768 -> 768) + learned 2-D (x, y) position tables (10,240 each)
    * 16 bidirectional layers, d 768, 12 heads x 64, gated GELU FFN 3072, axial 2-D RoPE (theta 100)
    * 3x3 average pooling over the patch grid -> soft tokens (280 per image / 140 per video frame by default)
    * scaled by sqrt(768), then RMSNorm (no scale) + linear 768 -> 512 into the text embedding stream.
Audio encoder (300M, the Gemma 4 / USM-style conformer)
    * 128-bin log-mel @ 16 kHz (20 ms window, 10 ms hop) -> two stride-2 Conv2d blocks (4x subsampling,
      40 ms per token = 25 tokens/s) -> linear 1024 -> 1024
    * 12 conformer layers (d 1024, 8 heads): macaron FFN (x0.5 residual) -> chunked local self-attention
      (chunk 12, 12 frames of left context, Transformer-XL relative positions, logit soft-cap 50) ->
      causal depthwise light-conv (k=5) -> FFN -> RMSNorm. Projections use clipped linears.
    * linear 1024 -> 1536, then RMSNorm (no scale) + linear 1536 -> 512 into the text embedding stream.

Multimodal inputs are spliced into the token sequence. Each image becomes
``<boi> <|image|> x N <eoi>``, each video frame ``<boi> <|video|> x N <eoi>`` and each audio clip
``<boa> <|audio|> x N <eoa>``; the placeholder embeddings are replaced by the encoder soft tokens and
the whole sequence (8,192-token context) runs through the text model.

Quick start
-----------
>>> import torch
>>> from transformers import AutoTokenizer          # tokenizer only; the model is pure PyTorch
>>> from embedding_gemma2 import EmbeddingGemma2
>>> tok = AutoTokenizer.from_pretrained("google/embeddinggemma-2")
>>> model = EmbeddingGemma2.from_pretrained("google/embeddinggemma-2", modalities=("text",)).eval()
>>> q = model.encode_text(["What causes the northern lights?"], tok, prompt_name="SearchQuery")
>>> d = model.encode_text(["The northern lights are caused by charged particles from the sun."], tok,
...                       prompt_name="Document")
>>> (q @ d.T).item()                                 # cosine similarity (embeddings are unit length)

Interleaved text + image (see ``build_multimodal_input_ids``):
>>> patches, positions, n_tok = preprocess_image(image_uint8_chw)                # 280 soft tokens
>>> ids = tok("Waterproof running shoes. <|image|>")["input_ids"]               # one placeholder id
>>> ids = expand_multimodal_placeholders(ids, image_soft_tokens=[n_tok])
>>> out = model(torch.tensor([ids]), pixel_values=patches[None], image_position_ids=positions[None])
>>> out.embeddings.shape                                                        # (1, 768)

Numerical precision: run in bfloat16 or float32, never float16 (activations overflow its range).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "AudioConfig",
    "EmbeddingGemma2",
    "EmbeddingGemma2Config",
    "EmbeddingGemma2Output",
    "PROMPTS",
    "TextConfig",
    "VisionConfig",
    "build_multimodal_input_ids",
    "expand_multimodal_placeholders",
    "log_mel_spectrogram",
    "num_audio_soft_tokens",
    "preprocess_audio",
    "preprocess_image",
    "preprocess_video",
    "sample_video_frames",
    "truncate_embeddings",
]

# =====================================================================================================
# Special tokens and task prompts
# =====================================================================================================

PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
BOS_TOKEN_ID = 2
BOI_TOKEN_ID = 255_999  #: begin-of-image (also wraps every video frame)
BOA_TOKEN_ID = 256_000  #: begin-of-audio
IMAGE_TOKEN_ID = 258_880  #: ``<|image|>`` placeholder
AUDIO_TOKEN_ID = 258_881  #: ``<|audio|>`` placeholder
EOI_TOKEN_ID = 258_882  #: end-of-image
EOA_TOKEN_ID = 258_883  #: end-of-audio
VIDEO_TOKEN_ID = 258_884  #: ``<|video|>`` placeholder

#: Task-instruction prefixes (from the official ``config_sentence_transformers.json``). Prefixes apply to
#: text only. Asymmetric tasks use a query prompt for queries and ``Document`` for the corpus; for a
#: titled document use ``f"title: {title} | text: {content}"`` instead of the ``Document`` prompt.
PROMPTS: dict[str, str] = {
    "SearchQuery": "task: search result | query: ",
    "QuestionAnswering": "task: question answering | query: ",
    "FactChecking": "task: fact checking | query: ",
    "CodeRetrieval": "task: code retrieval | query: ",
    "Classification": "task: classification | query: ",
    "Clustering": "task: clustering | query: ",
    "SentenceSimilarity": "task: sentence similarity | query: ",
    "Document": "title: none | text: ",
    # MTEB-style aliases
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "Retrieval": "task: search result | query: ",
    "Retrieval-query": "task: search result | query: ",
    "Retrieval-document": "title: none | text: ",
    "Reranking": "task: search result | query: ",
    "BitextMining": "task: search result | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
    "MultilabelClassification": "task: classification | query: ",
    "PairClassification": "task: sentence similarity | query: ",
    "STS": "task: sentence similarity | query: ",
    "Summarization": "task: sentence similarity | query: ",
}

#: Supported Matryoshka (MRL) output sizes.
MRL_DIMS: tuple[int, ...] = (768, 512, 256, 128)

# =====================================================================================================
# Configuration
# =====================================================================================================


@dataclass
class TextConfig:
    """Text backbone hyper-parameters (defaults = released checkpoint)."""

    vocab_size: int = 262_144
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 4
    num_key_value_heads: int = 2  #: KV heads on local (sliding) layers -> GQA
    head_dim: int = 256  #: head size on local layers
    global_num_key_value_heads: int = 1  #: KV heads on global layers -> MQA
    global_head_dim: int = 512  #: head size on global layers
    hidden_size_per_layer_input: int = 512  #: per-layer-embedding (PLE) width
    sliding_window: int = 512  #: inclusive radius: local layers attend where |i - j| <= sliding_window
    sliding_window_pattern: int = 6  #: every 6th layer is global (5 local : 1 global)
    rope_theta_local: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    embedding_dim: int = 768  #: output size of the final per-token projection
    pad_token_id: int = PAD_TOKEN_ID
    layer_types: list[str] | None = None  #: "sliding_attention" / "full_attention" per layer

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.sliding_window_pattern == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must have one entry per layer")


@dataclass
class VisionConfig:
    """Vision encoder hyper-parameters (Gemma 4 vision tower)."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    head_dim: int = 64
    patch_size: int = 16
    pooling_kernel_size: int = 3  #: k x k patches are averaged into one soft token
    position_embedding_size: int = 10_240  #: rows in each of the x / y position tables
    rope_theta: float = 100.0
    rms_norm_eps: float = 1e-6
    standardize: bool = False
    use_clipped_linears: bool = False
    default_output_length: int = 280


@dataclass
class AudioConfig:
    """Audio encoder hyper-parameters (Gemma 4 conformer audio tower)."""

    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    input_feat_size: int = 128  #: mel bins
    subsampling_conv_channels: tuple[int, int] = (128, 32)
    conv_kernel_size: int = 5  #: depthwise light-conv kernel
    attention_chunk_size: int = 12
    attention_context_left: int = 13
    attention_context_right: int = 0
    attention_logit_cap: float = 50.0
    attention_invalid_logits_value: float = -1e9
    residual_weight: float = 0.5  #: macaron FFN half-step
    gradient_clipping: float = 1e10
    output_proj_dims: int = 1536
    rms_norm_eps: float = 1e-6
    use_clipped_linears: bool = True


@dataclass
class EmbeddingGemma2Config:
    """Full model configuration. ``vision`` / ``audio`` set to ``None`` drops that encoder."""

    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig | None = field(default_factory=VisionConfig)
    audio: AudioConfig | None = field(default_factory=AudioConfig)

    @classmethod
    def from_hf_dict(cls, d: Mapping[str, Any]) -> EmbeddingGemma2Config:
        """Build from the Hugging Face ``config.json`` dictionary."""
        t = dict(d["text_config"])
        per_layer = t.pop("per_layer_config", None) or {}
        if per_layer:
            overrides = {(v.get("head_dim"), v.get("num_key_value_heads")) for v in per_layer.values()}
            if len(overrides) != 1:
                raise NotImplementedError("Only a single global-layer override is supported.")
            head_dim, kv_heads = overrides.pop()
            t["global_head_dim"], t["global_num_key_value_heads"] = head_dim, kv_heads
        rope = t.pop("rope_parameters", None) or {}
        if "sliding_attention" in rope:
            t["rope_theta_local"] = rope["sliding_attention"]["rope_theta"]
        if "full_attention" in rope:
            t["rope_theta_global"] = rope["full_attention"]["rope_theta"]

        vision = None
        if d.get("vision_config"):
            v = dict(d["vision_config"])
            v["rope_theta"] = (v.get("rope_parameters") or {}).get("rope_theta", 100.0)
            vision = _dataclass_from_dict(VisionConfig, v)

        audio = None
        if d.get("audio_config"):
            a = dict(d["audio_config"])
            if "subsampling_conv_channels" in a:
                a["subsampling_conv_channels"] = tuple(a["subsampling_conv_channels"])
            audio = _dataclass_from_dict(AudioConfig, a)

        return cls(text=_dataclass_from_dict(TextConfig, t), vision=vision, audio=audio)

    @classmethod
    def from_json(cls, path: str | Path) -> EmbeddingGemma2Config:
        with open(path, encoding="utf-8") as f:
            return cls.from_hf_dict(json.load(f))


def _dataclass_from_dict(cls: type, d: Mapping[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


# =====================================================================================================
# Shared building blocks
# =====================================================================================================


class RMSNorm(nn.Module):
    """RMSNorm computed in float32: ``x / sqrt(mean(x^2) + eps) * weight``.

    Gemma 4 uses a plain multiplicative ``weight`` (initialised to 1), not Gemma 1-3's ``1 + weight``.
    ``with_scale=False`` gives a parameter-free normalisation (used for V-norm and the multimodal
    embedders).
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, x: Tensor) -> Tensor:
        x32 = x.float()
        out = x32 * torch.pow(x32.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            out = out * self.weight.float()
        return out.type_as(x)


class ClippableLinear(nn.Module):
    """Bias-free linear layer with optional input/output clamping.

    The clamp bounds are checkpoint buffers (``input_min`` ... ``output_max``) used by the audio tower.
    The weight lives in ``.linear`` so names match the official checkpoint.
    """

    def __init__(self, in_features: int, out_features: int, use_clipping: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.use_clipping = use_clipping
        if use_clipping:
            for name, value in (("input_min", -math.inf), ("input_max", math.inf),
                                ("output_min", -math.inf), ("output_max", math.inf)):
                self.register_buffer(name, torch.tensor(value))

    def forward(self, x: Tensor) -> Tensor:
        if self.use_clipping:
            x = torch.clamp(x, self.input_min.to(x.dtype), self.input_max.to(x.dtype))
        x = self.linear(x)
        if self.use_clipping:
            x = torch.clamp(x, self.output_min.to(x.dtype), self.output_max.to(x.dtype))
        return x


def gelu_tanh(x: Tensor) -> Tensor:
    """``gelu_pytorch_tanh`` activation used by the text and vision FFNs."""
    return F.gelu(x, approximate="tanh")


class GatedMLP(nn.Module):
    """``down(gelu(gate(x)) * up(x))``. ``clippable=True`` wraps projections in :class:`ClippableLinear`."""

    def __init__(self, hidden: int, intermediate: int, clippable: bool = False, clip: bool = False) -> None:
        super().__init__()
        if clippable:
            self.gate_proj: nn.Module = ClippableLinear(hidden, intermediate, clip)
            self.up_proj: nn.Module = ClippableLinear(hidden, intermediate, clip)
            self.down_proj: nn.Module = ClippableLinear(intermediate, hidden, clip)
        else:
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotary embedding (half-split convention); ``cos``/``sin`` must broadcast against ``x``."""
    return x * cos + rotate_half(x) * sin


def rope_cos_sin(positions: Tensor, dim: int, theta: float, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """1-D RoPE tables. ``positions``: (..., T) -> cos, sin: (..., T, dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    freqs = positions[..., None].float() * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def additive_mask(allowed: Tensor, dtype: torch.dtype) -> Tensor:
    """Boolean "may attend" mask -> additive float mask (0 / dtype-min). Never yields NaN rows."""
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )


def attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
    """Scaled-dot-product attention with scale 1.0 (QK-norm replaces 1/sqrt(d)) and GQA expansion.

    q: (B, Hq, T, D); k, v: (B, Hkv, S, D); mask: additive, broadcastable to (B, Hq, T, S).
    """
    if k.shape[1] != q.shape[1]:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)


# =====================================================================================================
# Text backbone
# =====================================================================================================


class TextAttention(nn.Module):
    """Bidirectional attention with QK-norm, V-norm and RoPE.

    Local layers: GQA (4 q-heads / 2 kv-heads, head 256), sliding window, RoPE theta 1e4.
    Global layers: MQA (4 q-heads / 1 kv-head, head 512), full attention, RoPE theta 1e6.
    """

    def __init__(self, cfg: TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_global = cfg.layer_types[layer_idx] == "full_attention"
        self.head_dim = cfg.global_head_dim if self.is_global else cfg.head_dim
        n_kv = cfg.global_num_key_value_heads if self.is_global else cfg.num_key_value_heads
        h, n_q = cfg.hidden_size, cfg.num_attention_heads
        self.q_proj = nn.Linear(h, n_q * self.head_dim, bias=False)
        self.k_proj = nn.Linear(h, n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(h, n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_q * self.head_dim, h, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps, with_scale=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None) -> Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, -1, self.head_dim)).transpose(1, 2)
        v = self.v_norm(self.v_proj(x).view(b, t, -1, self.head_dim)).transpose(1, 2)
        cos, sin = cos[:, None], sin[:, None]  # (1, 1, T, D)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = attention(q, k, v, mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class PerLayerEmbedder(nn.Module):
    """Projection-only PLE: derives one ``hidden_size_per_layer_input`` signal per layer from the input
    embeddings (no token-id lookup). Output: (B, T, num_layers, P)."""

    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.num_layers, self.dim = cfg.num_hidden_layers, cfg.hidden_size_per_layer_input
        self.per_layer_model_projection = nn.Linear(cfg.hidden_size, self.num_layers * self.dim, bias=False)
        self.scale = cfg.hidden_size**-0.5
        self.per_layer_projection_norm = RMSNorm(self.dim, cfg.rms_norm_eps)

    def forward(self, inputs_embeds: Tensor) -> Tensor:
        x = self.per_layer_model_projection(inputs_embeds) * self.scale
        x = x.reshape(*inputs_embeds.shape[:-1], self.num_layers, self.dim)
        return self.per_layer_projection_norm(x)


class PerLayerBlock(nn.Module):
    """Third residual sub-block of every layer: ``x + norm(proj(gelu(gate(x)) * ple_i))``."""

    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.per_layer_input_gate = nn.Linear(cfg.hidden_size, cfg.hidden_size_per_layer_input, bias=False)
        self.per_layer_projection = nn.Linear(cfg.hidden_size_per_layer_input, cfg.hidden_size, bias=False)
        self.post_per_layer_input_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x: Tensor, per_layer_input: Tensor) -> Tensor:
        h = gelu_tanh(self.per_layer_input_gate(x)) * per_layer_input
        return x + self.post_per_layer_input_norm(self.per_layer_projection(h))


class TextLayer(nn.Module):
    """Sandwich-norm transformer block: attention, FFN, PLE gate, then a learned ``layer_scalar``."""

    def __init__(self, cfg: TextConfig, layer_idx: int) -> None:
        super().__init__()
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.self_attn = TextAttention(cfg, layer_idx)
        self.mlp = GatedMLP(h, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(h, eps)
        self.post_attention_layernorm = RMSNorm(h, eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps)
        self.ple_block = PerLayerBlock(cfg)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x: Tensor, per_layer_input: Tensor, cos: Tensor, sin: Tensor, mask: Tensor | None) -> Tensor:
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos, sin, mask))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        x = self.ple_block(x, per_layer_input)
        return x * self.layer_scalar.to(x.dtype)


class TextModel(nn.Module):
    """Bidirectional Gemma-4-style text encoder that outputs per-token ``embedding_dim`` vectors."""

    def __init__(self, cfg: TextConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        # sqrt(d) input scaling, applied in the weight dtype (bf16 rounds sqrt(512) to 22.625, as in Gemma)
        self.register_buffer("embed_scale", torch.tensor(cfg.hidden_size**0.5), persistent=False)
        self.ple = PerLayerEmbedder(cfg)
        self.layers = nn.ModuleList(TextLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # Linear, so projecting every token then mean-pooling == pooling then projecting.
        self.embedding_projection = nn.Linear(cfg.hidden_size, cfg.embedding_dim, bias=False)

    def embed(self, input_ids: Tensor) -> Tensor:
        """Scaled token embeddings, (B, T) -> (B, T, hidden)."""
        w = self.embed_tokens.weight
        return self.embed_tokens(input_ids) * self.embed_scale.to(w.dtype)

    def forward(self, inputs_embeds: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """inputs_embeds: (B, T, hidden); attention_mask: (B, T) with 1 = real token (right padding).

        Returns per-token embeddings (B, T, embedding_dim) before pooling.
        """
        cfg = self.config
        _, t, _ = inputs_embeds.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        per_layer_inputs = self.ple(inputs_embeds)

        # Masks: padding keys are never attended; local layers additionally restrict |i - j| <= window.
        if attention_mask is None:
            key_ok = torch.ones(1, 1, 1, t, dtype=torch.bool, device=device)
        else:
            key_ok = attention_mask.bool()[:, None, None, :]
        idx = torch.arange(t, device=device)
        near = (idx[:, None] - idx[None, :]).abs() <= cfg.sliding_window
        masks = {
            "full_attention": additive_mask(key_ok, dtype),
            "sliding_attention": additive_mask(key_ok & near, dtype),
        }
        positions = idx[None]  # (1, T); right-padded inputs share positions 0..T-1
        rope = {
            "full_attention": rope_cos_sin(positions, cfg.global_head_dim, cfg.rope_theta_global, dtype),
            "sliding_attention": rope_cos_sin(positions, cfg.head_dim, cfg.rope_theta_local, dtype),
        }

        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            kind = cfg.layer_types[i]
            cos, sin = rope[kind]
            x = layer(x, per_layer_inputs[:, :, i, :], cos, sin, masks[kind])
        return self.embedding_projection(self.norm(x))


# =====================================================================================================
# Vision encoder
# =====================================================================================================


def vision_rope_cos_sin(positions: Tensor, head_dim: int, theta: float, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Axial 2-D RoPE. positions: (B, N, 2) as (x, y) -> cos, sin: (B, N, 1, head_dim).

    The first half of each head is rotated by the x coordinate, the second half by y.
    """
    spatial = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial, 2, device=positions.device, dtype=torch.float32) / spatial))
    freqs = positions[..., None].float() * inv_freq  # (B, N, 2, spatial/2)
    fx, fy = freqs[..., 0, :], freqs[..., 1, :]
    angles = torch.cat([fx, fx, fy, fy], dim=-1)
    return angles.cos().to(dtype)[:, :, None], angles.sin().to(dtype)[:, :, None]


def apply_axial_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Applies :func:`vision_rope_cos_sin` tables to x: (B, N, H, D)."""
    h = x.shape[-1] // 2
    return torch.cat(
        [apply_rope(x[..., :h], cos[..., :h], sin[..., :h]), apply_rope(x[..., h:], cos[..., h:], sin[..., h:])],
        dim=-1,
    )


class VisionPatchEmbedder(nn.Module):
    """Linear patch projection + summed learned x / y position embeddings."""

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(3 * cfg.patch_size**2, cfg.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(torch.ones(2, cfg.position_embedding_size, cfg.hidden_size))

    def forward(self, pixel_values: Tensor, positions: Tensor, padding: Tensor) -> Tensor:
        # Pixels arrive in [0, 1]; Gemma 4 rescales to [-1, 1] in-model instead of mean/std normalisation.
        x = self.input_proj((2.0 * (pixel_values - 0.5)).to(self.input_proj.weight.dtype))
        p = positions.clamp(min=0)
        pos = F.embedding(p[..., 0], self.position_embedding_table[0]) + F.embedding(
            p[..., 1], self.position_embedding_table[1]
        )
        return x + pos.masked_fill(padding[..., None], 0.0)


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        h, c = cfg.hidden_size, cfg.use_clipped_linears
        self.q_proj = ClippableLinear(h, cfg.num_attention_heads * cfg.head_dim, c)
        self.k_proj = ClippableLinear(h, cfg.num_key_value_heads * cfg.head_dim, c)
        self.v_proj = ClippableLinear(h, cfg.num_key_value_heads * cfg.head_dim, c)
        self.o_proj = ClippableLinear(cfg.num_attention_heads * cfg.head_dim, h, c)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
        self.v_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps, with_scale=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        b, n, _ = x.shape
        shape = (b, n, -1, self.head_dim)
        q = apply_axial_rope(self.q_norm(self.q_proj(x).view(shape)), cos, sin).transpose(1, 2)
        k = apply_axial_rope(self.k_norm(self.k_proj(x).view(shape)), cos, sin).transpose(1, 2)
        v = self.v_norm(self.v_proj(x).view(shape)).transpose(1, 2)
        out = attention(q, k, v, mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class VisionLayer(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.self_attn = VisionAttention(cfg)
        self.mlp = GatedMLP(h, cfg.intermediate_size, clippable=True, clip=cfg.use_clipped_linears)
        self.input_layernorm = RMSNorm(h, eps)
        self.post_attention_layernorm = RMSNorm(h, eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Tensor) -> Tensor:
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos, sin, mask))
        return x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))


class VisionEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.config = cfg
        self.layers = nn.ModuleList(VisionLayer(cfg) for _ in range(cfg.num_hidden_layers))

    def forward(self, x: Tensor, positions: Tensor, valid: Tensor) -> Tensor:
        cos, sin = vision_rope_cos_sin(positions, self.config.head_dim, self.config.rope_theta, x.dtype)
        mask = additive_mask(valid[:, None, None, :], x.dtype)  # bidirectional, padding keys masked
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return x


class VisionModel(nn.Module):
    """Gemma 4 vision tower: patches -> encoder -> k x k spatial average pooling -> soft tokens."""

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.config = cfg
        self.patch_embedder = VisionPatchEmbedder(cfg)
        self.encoder = VisionEncoder(cfg)
        if cfg.standardize:
            self.register_buffer("std_bias", torch.zeros(cfg.hidden_size))
            self.register_buffer("std_scale", torch.ones(cfg.hidden_size))

    @staticmethod
    def _avg_pool(x: Tensor, positions: Tensor, length: int) -> tuple[Tensor, Tensor]:
        """Average k x k patch neighbourhoods by grid position. Returns pooled (B, L, D) and valid (B, L)."""
        k = int(math.isqrt(x.shape[1] // length))
        if k * k * length != x.shape[1]:
            raise ValueError(f"Cannot pool {x.shape[1]} patches to {length} soft tokens.")
        p = positions.clamp(min=0)
        grid_w = p[..., 0].max(dim=-1, keepdim=True).values + 1  # patches per row, per image
        cell = torch.div(p, k, rounding_mode="floor")
        idx = cell[..., 0] + (grid_w // k) * cell[..., 1]  # (B, N) pooled-cell index
        weights = F.one_hot(idx.long(), length).float() / (k * k)  # (B, N, L)
        pooled = weights.transpose(1, 2) @ x.float()
        valid = (weights != 0).any(dim=1)
        return pooled.to(x.dtype), valid

    def forward(self, pixel_values: Tensor, positions: Tensor) -> Tensor:
        """pixel_values: (B, N, 3*p*p) in [0, 1] (see :func:`preprocess_image`); positions: (B, N, 2) with
        (-1, -1) for padding. N must be ``max_soft_tokens * k^2``.

        Returns the real (non-padding) soft tokens of all images concatenated: (num_tokens, hidden).
        """
        cfg = self.config
        length = pixel_values.shape[-2] // cfg.pooling_kernel_size**2
        padding = (positions == -1).all(dim=-1)
        x = self.patch_embedder(pixel_values, positions, padding)
        x = self.encoder(x, positions, ~padding)
        dtype = x.dtype
        x = x.masked_fill(padding[..., None], 0.0)
        if x.shape[1] != length:
            x, valid = self._avg_pool(x, positions, length)
        else:
            valid = ~padding
        tokens = x.float()[valid] * cfg.hidden_size**0.5  # scaled in fp32 (can exceed fp16 range)
        if cfg.standardize:
            tokens = (tokens - self.std_bias.float()) * self.std_scale.float()
        return tokens.to(dtype)


# =====================================================================================================
# Audio encoder (conformer)
# =====================================================================================================


class AudioSubsampleLayer(nn.Module):
    """Conv2d(3x3, stride 2) -> LayerNorm over channels -> ReLU, with padding frames zeroed first."""

    def __init__(self, in_channels: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = nn.LayerNorm(out_channels, eps=eps, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        x = x * mask[:, None, :, None].to(x.dtype)  # x: (B, C, T, F)
        x = self.conv(x.to(self.conv.weight.dtype))
        x = F.relu(self.norm(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x, mask[:, ::2]


class AudioSubsample(nn.Module):
    """Two stride-2 conv blocks (4x time and frequency reduction) -> linear to ``hidden_size``."""

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        c0, c1 = cfg.subsampling_conv_channels
        self.layer0 = AudioSubsampleLayer(1, c0, cfg.rms_norm_eps)
        self.layer1 = AudioSubsampleLayer(c0, c1, cfg.rms_norm_eps)
        freq_out = math.ceil(math.ceil(cfg.input_feat_size / 2) / 2)  # mel bins after two stride-2 convs
        self.input_proj_linear = nn.Linear(freq_out * c1, cfg.hidden_size, bias=False)

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        x, mask = self.layer0(features[:, None], mask)
        x, mask = self.layer1(x, mask)
        b, _, t, _ = x.shape
        return self.input_proj_linear(x.permute(0, 2, 3, 1).reshape(b, t, -1)), mask


class AudioAttention(nn.Module):
    """Chunked local self-attention with Transformer-XL relative positions and logit soft-capping.

    The sequence is cut into blocks of ``chunk`` queries; each block attends to a ``chunk + past + future``
    key window. With the default config every frame sees itself and the previous 11 frames.
    """

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        h, c = cfg.hidden_size, cfg.use_clipped_linears
        self.num_heads = cfg.num_attention_heads
        self.head_dim = h // self.num_heads
        self.chunk = cfg.attention_chunk_size
        self.past = cfg.attention_context_left - 1
        self.future = cfg.attention_context_right
        self.context = self.chunk + self.past + self.future
        self.softcap = cfg.attention_logit_cap
        self.invalid_logit = cfg.attention_invalid_logits_value
        self.q_scale = self.head_dim**-0.5 / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        self.q_proj = ClippableLinear(h, h, c)
        self.k_proj = ClippableLinear(h, h, c)
        self.v_proj = ClippableLinear(h, h, c)
        self.post = ClippableLinear(h, h, c)
        self.relative_k_proj = nn.Linear(h, h, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

    def _blocks(self, x: Tensor) -> Tensor:
        """(B, T, H, D) -> (B, num_blocks, chunk, H, D)."""
        b, t, h, d = x.shape
        nb = -(-t // self.chunk)
        return F.pad(x, (0, 0, 0, 0, 0, nb * self.chunk - t)).reshape(b, nb, self.chunk, h, d)

    def _context_windows(self, x: Tensor) -> Tensor:
        """(B, T, H, D) -> overlapping key windows (B, num_blocks, context, H, D)."""
        x = F.pad(x, (0, 0, 0, 0, self.past, self.future + self.chunk - 1))
        return x.unfold(1, self.context, self.chunk).movedim(-1, 2)

    def _rel_shift(self, x: Tensor) -> Tensor:
        """Re-index (.., chunk, R) relative logits into (.., chunk, context) key-window logits."""
        b, h, nb, c, r = x.shape
        x = F.pad(x, (0, self.context + 1 - r)).reshape(b, h, nb, c * (self.context + 1))
        return x[..., : c * self.context].reshape(b, h, nb, c, self.context)

    def forward(self, x: Tensor, rel_pos: Tensor, mask: Tensor) -> Tensor:
        """x: (B, T, hidden); rel_pos: (R, hidden) sinusoids; mask: (B, 1, nb, chunk, context) bool."""
        b, t, _ = x.shape
        shape = (b, t, self.num_heads, self.head_dim)
        q = self.q_proj(x).float().view(shape) * (self.q_scale * F.softplus(self.per_dim_scale).float())
        k = self.k_proj(x).float().view(shape) * self.k_scale
        v = self.v_proj(x).float().view(shape)

        q = self._blocks(q).permute(0, 3, 1, 2, 4)  # (B, H, nb, chunk, D)
        k = self._context_windows(k).permute(0, 3, 1, 4, 2)  # (B, H, nb, D, context)
        v = self._context_windows(v).permute(0, 3, 1, 2, 4)  # (B, H, nb, context, D)
        nb = q.shape[2]

        rel_k = self.relative_k_proj(rel_pos).view(-1, self.num_heads, self.head_dim).float()  # (R, H, D)
        content = q @ k  # (B, H, nb, chunk, context)
        position = q.reshape(b, self.num_heads, nb * self.chunk, self.head_dim) @ rel_k.permute(1, 2, 0)
        position = self._rel_shift(position.reshape(b, self.num_heads, nb, self.chunk, -1))

        logits = torch.tanh((content + position) / self.softcap) * self.softcap
        logits = logits.masked_fill(~mask, self.invalid_logit)
        out = logits.softmax(dim=-1) @ v  # (B, H, nb, chunk, D)
        out = out.permute(0, 2, 3, 1, 4).reshape(b, nb * self.chunk, -1)[:, :t]
        return self.post(out.to(x.dtype))


class AudioFeedForward(nn.Module):
    """Macaron half-step FFN: ``x + 0.5 * norm(W2 silu(W1 norm(x)))``."""

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        h, c = cfg.hidden_size, cfg.use_clipped_linears
        self.ffw_layer_1 = ClippableLinear(h, 4 * h, c)
        self.ffw_layer_2 = ClippableLinear(4 * h, h, c)
        self.pre_layer_norm = RMSNorm(h, cfg.rms_norm_eps)
        self.post_layer_norm = RMSNorm(h, cfg.rms_norm_eps)
        self.clip, self.residual_weight = cfg.gradient_clipping, cfg.residual_weight

    def forward(self, x: Tensor) -> Tensor:
        clip = min(self.clip, torch.finfo(x.dtype).max)
        h = self.pre_layer_norm(torch.clamp(x, -clip, clip))
        h = self.ffw_layer_2(F.silu(self.ffw_layer_1(h)))
        h = self.post_layer_norm(torch.clamp(h, -clip, clip))
        return x + h * self.residual_weight


class AudioLightConv(nn.Module):
    """GLU -> causal depthwise Conv1d -> RMSNorm -> SiLU -> linear, with a residual connection."""

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        h, c = cfg.hidden_size, cfg.use_clipped_linears
        self.linear_start = ClippableLinear(h, 2 * h, c)
        self.linear_end = ClippableLinear(h, h, c)
        self.depthwise_conv1d = nn.Conv1d(h, h, cfg.conv_kernel_size, groups=h, bias=False)
        self.pre_layer_norm = RMSNorm(h, cfg.rms_norm_eps)
        self.conv_norm = RMSNorm(h, cfg.rms_norm_eps)
        self.clip = cfg.gradient_clipping

    def forward(self, x: Tensor) -> Tensor:
        h = F.glu(self.linear_start(self.pre_layer_norm(x)), dim=-1)
        h = F.pad(h.transpose(1, 2), (self.depthwise_conv1d.kernel_size[0] - 1, 0))  # causal: left pad only
        h = self.depthwise_conv1d(h).transpose(1, 2)
        clip = min(self.clip, torch.finfo(h.dtype).max)
        h = F.silu(self.conv_norm(torch.clamp(h, -clip, clip)))
        return x + self.linear_end(h)


class AudioLayer(nn.Module):
    """Conformer block: FFN/2 -> attention -> light-conv -> FFN/2 -> RMSNorm."""

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        self.feed_forward1 = AudioFeedForward(cfg)
        self.feed_forward2 = AudioFeedForward(cfg)
        self.self_attn = AudioAttention(cfg)
        self.lconv1d = AudioLightConv(cfg)
        self.norm_pre_attn = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm_post_attn = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm_out = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.clip = cfg.gradient_clipping

    def forward(self, x: Tensor, rel_pos: Tensor, mask: Tensor) -> Tensor:
        clip = min(self.clip, torch.finfo(x.dtype).max)
        x = self.feed_forward1(x)
        h = self.self_attn(self.norm_pre_attn(torch.clamp(x, -clip, clip)), rel_pos, mask)
        x = x + self.norm_post_attn(torch.clamp(h, -clip, clip))
        x = self.feed_forward2(self.lconv1d(x))
        return self.norm_out(torch.clamp(x, -clip, clip))


class AudioModel(nn.Module):
    """Gemma 4 audio tower: log-mel (B, T, 128) -> soft tokens (B, T/4, output_proj_dims)."""

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        self.config = cfg
        self.subsample_conv_projection = AudioSubsample(cfg)
        self.layers = nn.ModuleList(AudioLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.output_proj = nn.Linear(cfg.hidden_size, cfg.output_proj_dims, bias=True)

    def _relative_positions(self, like: Tensor) -> Tensor:
        """Sinusoids for relative distances ``context//2 ... 0`` -> (R, hidden), [sin | cos] layout."""
        cfg = self.config
        ctx = cfg.attention_chunk_size + cfg.attention_context_left - 1 + cfg.attention_context_right
        n = cfg.hidden_size // 2
        inv_timescales = torch.exp(
            torch.arange(n, device=like.device, dtype=torch.float32) * -(math.log(10_000.0) / max(n - 1, 1))
        )
        pos = torch.arange(ctx // 2, -1, -1, device=like.device, dtype=torch.float32)[:, None]
        scaled = pos * inv_timescales[None]
        return torch.cat([scaled.sin(), scaled.cos()], dim=-1).to(like.dtype)

    def _block_mask(self, valid: Tensor) -> Tensor:
        """Chunked local mask (B, 1, nb, chunk, context): key must be real and 0 <= q - k < past
        (or 0 < k - q < future)."""
        cfg = self.config
        chunk, past, future = cfg.attention_chunk_size, cfg.attention_context_left - 1, cfg.attention_context_right
        ctx = chunk + past + future
        _, t = valid.shape
        nb = -(-t // chunk)
        dev = valid.device
        blocks = torch.arange(nb, device=dev)[:, None] * chunk
        q_pos = blocks + torch.arange(chunk, device=dev)[None]  # (nb, chunk)
        k_pos = blocks - past + torch.arange(ctx, device=dev)[None]  # (nb, ctx)
        dist = q_pos[:, :, None] - k_pos[:, None, :]
        window = ((dist >= 0) & (dist < past)) | ((dist < 0) & (-dist < future))
        k_real = valid[:, k_pos.clamp(0, t - 1)] & ((k_pos >= 0) & (k_pos < t))  # (B, nb, ctx)
        return window[None, None] & k_real[:, None, :, None, :]

    def forward(self, features: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """features: (B, T, 128) log-mel; mask: (B, T) bool, True = real frame.

        Returns soft tokens (B, ceil(T/4), output_proj_dims) and their validity mask (B, ceil(T/4)).
        """
        if mask is None:
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        x, valid = self.subsample_conv_projection(features, mask.bool())
        rel_pos = self._relative_positions(x)
        block_mask = self._block_mask(valid)
        for layer in self.layers:
            x = layer(x, rel_pos, block_mask)
        return self.output_proj(x), valid


# =====================================================================================================
# Full model
# =====================================================================================================


class MultimodalEmbedder(nn.Module):
    """Maps encoder soft tokens into the text embedding space: parameter-free RMSNorm + linear."""

    def __init__(self, in_dim: int, out_dim: int, eps: float) -> None:
        super().__init__()
        self.embedding_pre_projection_norm = RMSNorm(in_dim, eps, with_scale=False)
        self.embedding_projection = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding_projection(self.embedding_pre_projection_norm(x))


@dataclass
class EmbeddingGemma2Output:
    """``embeddings``: (B, 768) mean-pooled, L2-normalised (float32).
    ``token_embeddings``: (B, T, 768) per-token outputs before pooling (model dtype)."""

    embeddings: Tensor
    token_embeddings: Tensor
    attention_mask: Tensor


class EmbeddingGemma2(nn.Module):
    """EmbeddingGemma 2: text model + optional vision and audio towers, mean pooling and normalisation.

    Attribute names (``language_model``, ``vision_tower``, ``embed_vision``, ``audio_tower``,
    ``embed_audio``) match the official checkpoint keys.
    """

    def __init__(self, config: EmbeddingGemma2Config | None = None) -> None:
        super().__init__()
        self.config = config = config or EmbeddingGemma2Config()
        d_text = config.text.hidden_size
        self.language_model = TextModel(config.text)
        self.vision_tower = VisionModel(config.vision) if config.vision else None
        self.embed_vision = (
            MultimodalEmbedder(config.vision.hidden_size, d_text, config.vision.rms_norm_eps)
            if config.vision else None
        )
        self.audio_tower = AudioModel(config.audio) if config.audio else None
        self.embed_audio = (
            MultimodalEmbedder(config.audio.output_proj_dims, d_text, config.audio.rms_norm_eps)
            if config.audio else None
        )

    # ------------------------------------------------------------------ modality encoders

    def get_image_features(self, pixel_values: Tensor, positions: Tensor) -> Tensor:
        """Images (or video frames) -> soft tokens in text space, (num_tokens, 512)."""
        if self.vision_tower is None or self.embed_vision is None:
            raise ValueError("Model was built without the vision encoder.")
        return self.embed_vision(self.vision_tower(pixel_values, positions))

    def get_audio_features(self, features: Tensor, mask: Tensor) -> Tensor:
        """Log-mel batch -> real (non-padding) soft tokens in text space, (num_tokens, 512)."""
        if self.audio_tower is None or self.embed_audio is None:
            raise ValueError("Model was built without the audio encoder.")
        tokens, valid = self.audio_tower(features, mask)
        return self.embed_audio(tokens)[valid]

    @staticmethod
    def _splice(embeds: Tensor, slots: Tensor, features: Tensor, name: str) -> Tensor:
        n_slots, n_feats = int(slots.sum()), features.shape[0]
        if n_slots != n_feats:
            raise ValueError(f"{name}: {n_slots} placeholder tokens but {n_feats} soft tokens were produced.")
        return embeds.masked_scatter(slots[..., None], features.to(embeds.dtype))

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        pixel_values: Tensor | None = None,
        image_position_ids: Tensor | None = None,
        pixel_values_videos: Tensor | None = None,
        video_position_ids: Tensor | None = None,
        input_features: Tensor | None = None,
        input_features_mask: Tensor | None = None,
    ) -> EmbeddingGemma2Output:
        """Embeds a batch of (possibly interleaved multimodal) sequences.

        Args:
            input_ids: (B, T) token ids, right-padded, with placeholders already expanded
                (see :func:`expand_multimodal_placeholders`).
            attention_mask: (B, T), 1 for real tokens. Defaults to all ones.
            pixel_values, image_position_ids: (num_images, N, 768) / (num_images, N, 2) from
                :func:`preprocess_image`, in the order the images appear in the batch.
            pixel_values_videos, video_position_ids: (total_frames, N, 768) / (total_frames, N, 2) from
                :func:`preprocess_video`, all frames of all videos concatenated in order.
            input_features, input_features_mask: (num_clips, F, 128) / (num_clips, F) from
                :func:`preprocess_audio`.

        Soft tokens are written into the ``<|image|>`` / ``<|video|>`` / ``<|audio|>`` slots in row-major
        order, so the number of slots must equal the number of real soft tokens.
        """
        lm = self.language_model
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        image_slots = input_ids == IMAGE_TOKEN_ID
        video_slots = input_ids == VIDEO_TOKEN_ID
        audio_slots = input_ids == AUDIO_TOKEN_ID
        text_ids = input_ids.masked_fill(image_slots | video_slots | audio_slots, lm.config.pad_token_id)
        embeds = lm.embed(text_ids)

        if pixel_values is not None:
            feats = self.get_image_features(pixel_values, image_position_ids)
            embeds = self._splice(embeds, image_slots, feats, "image")
        if pixel_values_videos is not None:
            feats = self.get_image_features(pixel_values_videos, video_position_ids)
            embeds = self._splice(embeds, video_slots, feats, "video")
        if input_features is not None:
            if input_features_mask is None:
                input_features_mask = torch.ones(input_features.shape[:2], dtype=torch.bool,
                                                 device=input_features.device)
            feats = self.get_audio_features(input_features, input_features_mask)
            embeds = self._splice(embeds, audio_slots, feats, "audio")

        tokens = lm(embeds, attention_mask)
        m = attention_mask.to(torch.float32)[..., None]
        pooled = (tokens.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # mean pooling
        return EmbeddingGemma2Output(F.normalize(pooled, dim=-1), tokens, attention_mask)

    # ------------------------------------------------------------------ convenience

    @torch.no_grad()
    def encode_text(
        self,
        texts: Sequence[str],
        tokenizer: Callable[..., Any],
        *,
        prompt_name: str | None = None,
        prompt: str | None = None,
        truncate_dim: int | None = None,
        batch_size: int = 32,
        max_length: int = 8192,
    ) -> Tensor:
        """Embeds plain texts with an optional task prefix; returns (N, dim) unit vectors (float32).

        ``tokenizer`` is the Gemma tokenizer (e.g. ``AutoTokenizer.from_pretrained("google/embeddinggemma-2")``),
        which adds ``<bos>`` and ``<eos>``.
        """
        if prompt is None and prompt_name is not None:
            prompt = PROMPTS[prompt_name]
        prefix = prompt or ""
        device = next(self.parameters()).device
        old_side = getattr(tokenizer, "padding_side", None)
        if old_side is not None:
            tokenizer.padding_side = "right"  # positions assume right padding
        try:
            chunks = []
            for i in range(0, len(texts), batch_size):
                batch = tokenizer([prefix + t for t in texts[i : i + batch_size]], padding=True,
                                  truncation=True, max_length=max_length, return_tensors="pt")
                out = self(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                chunks.append(out.embeddings)
        finally:
            if old_side is not None:
                tokenizer.padding_side = old_side
        emb = torch.cat(chunks)
        return truncate_embeddings(emb, truncate_dim) if truncate_dim else emb

    def num_parameters(self) -> dict[str, int]:
        """Parameter counts per component (cf. model card: 130M + 140M text, 170M vision, 300M audio)."""
        def count(*mods: nn.Module | None) -> int:
            return sum(p.numel() for m in mods if m is not None for p in m.parameters())

        lm = self.language_model
        counts = {
            "text_embedder": count(lm.embed_tokens, lm.ple),
            "text_transformer": count(lm.layers, lm.norm, lm.embedding_projection),
            "vision": count(self.vision_tower, self.embed_vision),
            "audio": count(self.audio_tower, self.embed_audio),
        }
        counts["total"] = sum(counts.values())
        return counts

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str | Path = "google/embeddinggemma-2",
        *,
        modalities: Sequence[str] = ("text", "image", "audio"),
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> EmbeddingGemma2:
        """Loads the official weights from a local directory or a Hugging Face repo id.

        ``modalities`` selects which encoders to build: text is always present, "image"/"video" load the
        vision tower, "audio" the audio tower (card sizes: text 270M, +image 440M, +audio 570M, all 740M).
        Requires ``safetensors`` (and ``huggingface_hub`` for repo ids).
        """
        if dtype == torch.float16:
            raise ValueError("float16 overflows EmbeddingGemma 2 activations; use bfloat16 or float32.")
        path = Path(path_or_repo)
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(str(path_or_repo), allow_patterns=["config.json", "*.safetensors"]))
        from safetensors.torch import load_file

        config = EmbeddingGemma2Config.from_json(path / "config.json")
        mods = set(modalities)
        if not mods & {"image", "video", "vision"}:
            config.vision = None
        if "audio" not in mods:
            config.audio = None
        model = cls(config)

        dropped = tuple(
            p for p, keep in (("vision_tower.", config.vision), ("embed_vision.", config.vision),
                              ("audio_tower.", config.audio), ("embed_audio.", config.audio)) if not keep
        )
        state: dict[str, Tensor] = {}
        for shard in sorted(path.glob("*.safetensors")):
            for k, v in load_file(str(shard)).items():
                k = k.removeprefix("model.")
                if not k.startswith(dropped or ("\0",)):
                    state[k] = v
        if not state:
            raise FileNotFoundError(f"No *.safetensors weights found in {path}")

        missing, unexpected = model.load_state_dict(state, strict=False)
        buffer_suffixes = ("layer_scalar", "input_min", "input_max", "output_min", "output_max")
        hard_missing = [k for k in missing if not k.endswith(buffer_suffixes)]
        if hard_missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch. Missing: {hard_missing[:10]} Unexpected: {unexpected[:10]}")
        if missing:
            warnings.warn(f"{len(missing)} buffers kept at their defaults (e.g. {missing[0]}).", stacklevel=2)
        return model.to(device=device, dtype=dtype)


def truncate_embeddings(embeddings: Tensor, dim: int) -> Tensor:
    """Matryoshka truncation: keep the leading ``dim`` values, then re-normalise to unit length.

    Supported sizes are 768, 512, 256 and 128 (near-lossless to 256; 128 is best kept to text-only use).
    Queries and documents must use the same size.
    """
    if dim not in MRL_DIMS:
        warnings.warn(f"{dim} is not a trained MRL size {MRL_DIMS}.", stacklevel=2)
    return F.normalize(embeddings[..., :dim].float(), dim=-1)


# =====================================================================================================
# Pre-processing (mirrors the official processors; pure torch)
# =====================================================================================================

SUPPORTED_SOFT_TOKENS: tuple[int, ...] = (70, 140, 280, 560, 1120)


def _target_size(height: int, width: int, patch: int, max_patches: int, k: int) -> tuple[int, int]:
    """Largest aspect-preserving size within ``max_patches`` with sides divisible by ``k * patch``."""
    target_px = max_patches * patch**2
    factor = math.sqrt(target_px / (height * width))
    side = k * patch
    th = int(math.floor(factor * height / side)) * side
    tw = int(math.floor(factor * width / side)) * side
    max_side = (max_patches // k**2) * side
    if th == 0 and tw == 0:
        raise ValueError("Image is too small to resize.")
    if th == 0:
        th, tw = side, min(int(math.floor(width / height)) * side, max_side)
    elif tw == 0:
        tw, th = side, min(int(math.floor(height / width)) * side, max_side)
    return th, tw


def preprocess_image(
    image: Tensor,
    max_soft_tokens: int = 280,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
) -> tuple[Tensor, Tensor, int]:
    """Resizes and patchifies one RGB image for the vision tower.

    Args:
        image: (3, H, W), uint8 in [0, 255] or float in [0, 1].
        max_soft_tokens: soft-token budget, one of 70 / 140 / 280 (default) / 560 / 1120. Higher budgets
            give finer visual detail at the cost of context.

    Returns:
        patches: (max_soft_tokens * k^2, 3 * p * p) in [0, 1], zero-padded.
        positions: (max_soft_tokens * k^2, 2) patch (x, y) grid coordinates, (-1, -1) for padding.
        num_soft_tokens: number of real soft tokens this image produces (<= max_soft_tokens).
    """
    if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
        raise ValueError(f"max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}")
    is_uint8 = image.dtype == torch.uint8
    img = image.float() if is_uint8 else image.float() * 255.0
    max_patches = max_soft_tokens * pooling_kernel_size**2
    th, tw = _target_size(img.shape[-2], img.shape[-1], patch_size, max_patches, pooling_kernel_size)
    if (th, tw) != tuple(img.shape[-2:]):
        img = F.interpolate(img[None], size=(th, tw), mode="bicubic", antialias=True, align_corners=False)[0]
        img = img.round() if is_uint8 else img  # torchvision resizes uint8 images in uint8
    img = img.clamp(0, 255) / 255.0

    c, nh, nw = img.shape[0], th // patch_size, tw // patch_size
    patches = img.reshape(c, nh, patch_size, nw, patch_size).permute(1, 3, 2, 4, 0).reshape(nh * nw, -1)
    xs, ys = torch.meshgrid(torch.arange(nw), torch.arange(nh), indexing="xy")
    positions = torch.stack([xs, ys], dim=-1).reshape(nh * nw, 2)

    pad = max_patches - patches.shape[0]
    patches = F.pad(patches, (0, 0, 0, pad), value=0.0)
    positions = F.pad(positions, (0, 0, 0, pad), value=-1)
    return patches, positions, (nh * nw) // pooling_kernel_size**2


def sample_video_frames(frames: Tensor, video_fps: float, fps: float = 1.0, max_frames: int = 32) -> Tensor:
    """Uniformly samples ``fps`` frames per second (default 1), capped at ``max_frames`` (default 32).

    frames: (num_frames, 3, H, W). Returns the selected frames.
    """
    total = frames.shape[0]
    n = max(1, min(max_frames, int(total / video_fps * fps)))
    idx = torch.linspace(0, total - 1, n).round().long()
    return frames[idx]


def preprocess_video(frames: Tensor, max_soft_tokens: int = 140, **kwargs: Any) -> tuple[Tensor, Tensor, int]:
    """Patchifies already-sampled frames (num_frames, 3, H, W) with the image pipeline.

    Returns patches (F, N, 3*p*p), positions (F, N, 2) and soft tokens per frame (default budget 140).
    """
    out = [preprocess_image(f, max_soft_tokens=max_soft_tokens, **kwargs) for f in frames]
    return torch.stack([o[0] for o in out]), torch.stack([o[1] for o in out]), out[0][2]


def _mel_filter_bank(n_freqs: int, n_mels: int, f_min: float, f_max: float, sr: int) -> Tensor:
    """HTK-scale triangular mel filters without area normalisation, (n_freqs, n_mels), float64."""
    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    mels = torch.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2, dtype=torch.float64)
    edges = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    fft_freqs = torch.linspace(0, sr // 2, n_freqs, dtype=torch.float64)
    diff = edges[1:] - edges[:-1]
    slopes = edges[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    return torch.clamp(torch.minimum(down, up), min=0.0)


def log_mel_spectrogram(
    waveform: Tensor,
    sample_mask: Tensor | None = None,
    *,
    sampling_rate: int = 16_000,
    n_mels: int = 128,
    frame_length: int = 320,
    hop_length: int = 160,
    fft_length: int = 512,
    mel_floor: float = 1e-3,
) -> tuple[Tensor, Tensor]:
    """Gemma 4 audio features: semicausal STFT (20 ms Hann window, 10 ms hop) -> |.| -> 128 HTK mels ->
    ``log(x + 1e-3)``.

    Args:
        waveform: (S,) mono float waveform at 16 kHz.
        sample_mask: (S,) 1 for real samples (for padded inputs). Defaults to all ones.

    Returns:
        features (num_frames, 128) float32 and mask (num_frames,) bool; a frame is valid only if its whole
        analysis window lies on real samples.
    """
    wav = waveform.detach().cpu().to(torch.float64)
    if sample_mask is None:
        sample_mask = torch.ones_like(wav, dtype=torch.bool)
    wav = F.pad(wav, (frame_length // 2, 0))  # first frame centred at t = 0
    sample_mask = torch.cat([torch.zeros(frame_length // 2, dtype=torch.bool), sample_mask.bool().cpu()])
    frames = wav.unfold(-1, frame_length + 1, hop_length)[..., :-1]  # (num_frames, frame_length)
    window = torch.hann_window(frame_length, periodic=True, dtype=torch.float64)
    spec = torch.fft.rfft(frames * window, n=fft_length).abs()
    mel = spec @ _mel_filter_bank(fft_length // 2 + 1, n_mels, 0.0, sampling_rate / 2, sampling_rate)
    features = torch.log(mel + mel_floor).to(torch.float32)
    ends = torch.arange(frames.shape[0]) * hop_length + frame_length
    return features, sample_mask[ends]


def preprocess_audio(
    waveforms: Sequence[Tensor],
    max_samples: int = 480_000,
    pad_to_multiple_of: int = 128,
) -> tuple[Tensor, Tensor]:
    """Batches mono 16 kHz waveforms into log-mel features.

    Each clip is truncated to ``max_samples`` (30 s), right-padded to the longest clip (rounded up to a
    multiple of 128 samples). Returns features (B, F, 128) and mask (B, F).
    """
    clips = [w.flatten()[:max_samples].float() for w in waveforms]
    longest = max(c.numel() for c in clips)
    longest = -(-longest // pad_to_multiple_of) * pad_to_multiple_of
    feats, masks = [], []
    for c in clips:
        sample_mask = torch.zeros(longest, dtype=torch.bool)
        sample_mask[: c.numel()] = True
        f, m = log_mel_spectrogram(F.pad(c, (0, longest - c.numel())), sample_mask)
        feats.append(f)
        masks.append(m)
    return torch.stack(feats), torch.stack(masks)


def num_audio_soft_tokens(frame_mask: Tensor) -> int:
    """Soft tokens produced from one clip's mel-frame mask (two stride-2 subsamplings, 25 tokens/s)."""
    return int(frame_mask.bool()[::2][::2].sum())


def expand_multimodal_placeholders(
    input_ids: Sequence[int],
    *,
    image_soft_tokens: Sequence[int] = (),
    video_soft_tokens: Sequence[tuple[int, int]] = (),
    audio_soft_tokens: Sequence[int] = (),
) -> list[int]:
    """Expands single placeholder ids into the full soft-token spans the model expects.

    Each ``<|image|>`` becomes ``<boi> <|image|> x n <eoi>``; each ``<|video|>`` becomes
    ``(<boi> <|video|> x n <eoi>) x num_frames``; each ``<|audio|>`` becomes ``<boa> <|audio|> x n <eoa>``.
    Counts are consumed in order: ``image_soft_tokens[i]`` for the i-th image, ``video_soft_tokens[i] =
    (tokens_per_frame, num_frames)``, ``audio_soft_tokens[i]`` for the i-th clip.
    """
    images, videos, audios = iter(image_soft_tokens), iter(video_soft_tokens), iter(audio_soft_tokens)
    out: list[int] = []
    try:
        for tok in input_ids:
            if tok == IMAGE_TOKEN_ID:
                out += [BOI_TOKEN_ID] + [IMAGE_TOKEN_ID] * next(images) + [EOI_TOKEN_ID]
            elif tok == VIDEO_TOKEN_ID:
                per_frame, num_frames = next(videos)
                out += ([BOI_TOKEN_ID] + [VIDEO_TOKEN_ID] * per_frame + [EOI_TOKEN_ID]) * num_frames
            elif tok == AUDIO_TOKEN_ID:
                out += [BOA_TOKEN_ID] + [AUDIO_TOKEN_ID] * next(audios) + [EOA_TOKEN_ID]
            else:
                out.append(int(tok))
    except StopIteration:
        raise ValueError("More placeholders in input_ids than soft-token counts were given.") from None
    return out


def build_multimodal_input_ids(
    parts: Sequence[str | tuple[str, int] | tuple[str, int, int]],
    tokenize: Callable[[str], list[int]],
) -> list[int]:
    """Builds ``<bos> ... <eos>`` ids from text pieces and media spans without a placeholder-aware tokenizer.

    ``parts`` mixes strings with ``("image", n)``, ``("audio", n)`` or ``("video", n_per_frame, n_frames)``.
    ``tokenize`` must return ids *without* special tokens, e.g.
    ``lambda s: tok(s, add_special_tokens=False)["input_ids"]``. Put any task prompt in the first string.
    """
    ids = [BOS_TOKEN_ID]
    for part in parts:
        if isinstance(part, str):
            ids += tokenize(part)
        elif part[0] == "image":
            ids += expand_multimodal_placeholders([IMAGE_TOKEN_ID], image_soft_tokens=[part[1]])
        elif part[0] == "audio":
            ids += expand_multimodal_placeholders([AUDIO_TOKEN_ID], audio_soft_tokens=[part[1]])
        elif part[0] == "video":
            ids += expand_multimodal_placeholders([VIDEO_TOKEN_ID], video_soft_tokens=[(part[1], part[2])])
        else:
            raise ValueError(f"Unknown part {part!r}")
    return ids + [EOS_TOKEN_ID]


# =====================================================================================================
# Smoke test: python embedding_gemma2.py
# =====================================================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # 1) Full-size parameter budget, built on the meta device (no memory used).
    with torch.device("meta"):
        full = EmbeddingGemma2()
    for name, n in full.num_parameters().items():
        print(f"{name:>17}: {n / 1e6:7.1f}M")

    # 2) Tiny random model through every code path.
    cfg = EmbeddingGemma2Config(
        text=TextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=16, global_head_dim=32, hidden_size_per_layer_input=16,
                        sliding_window=4, embedding_dim=48),
        vision=VisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=2, head_dim=16, position_embedding_size=64),
        audio=AudioConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                          subsampling_conv_channels=(16, 8), output_proj_dims=48),
    )
    model = EmbeddingGemma2(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():  # break the all-ones init so tests are meaningful
            p.add_(torch.randn_like(p) * 0.02)

        # text + padding invariance
        a = torch.randint(10, 1000, (1, 9))
        b = torch.randint(10, 1000, (1, 14))
        batch = torch.zeros(2, 14, dtype=torch.long)
        batch[0, :9], batch[1] = a[0], b[0]
        mask = (torch.arange(14) < torch.tensor([[9], [14]])).long()
        e_batch = model(batch, mask).embeddings
        e_alone = model(a).embeddings
        print("text padding invariance err:", (e_batch[0] - e_alone[0]).abs().max().item())

        # image + video + audio interleaved
        img = torch.randint(0, 256, (3, 120, 200), dtype=torch.uint8)
        pv, pos, n_img = preprocess_image(img, max_soft_tokens=70)
        vid = torch.rand(3, 3, 64, 96)
        vpv, vpos, n_vid = preprocess_video(vid, max_soft_tokens=70)
        feats, fmask = preprocess_audio([torch.randn(16_000), torch.randn(9_000)])
        n_aud = [num_audio_soft_tokens(m) for m in fmask]
        ids = build_multimodal_input_ids(
            ["task: search result | query: shoes ", ("image", n_img), " clip ", ("video", n_vid, 3),
             " sound ", ("audio", n_aud[0]), " and ", ("audio", n_aud[1])],
            tokenize=lambda s: [10 + (ord(ch) % 200) for ch in s],
        )
        out = model(torch.tensor([ids]), pixel_values=pv[None], image_position_ids=pos[None],
                    pixel_values_videos=vpv, video_position_ids=vpos,
                    input_features=feats, input_features_mask=fmask)
        print("multimodal:", tuple(out.embeddings.shape), "norm", out.embeddings.norm().item(),
              f"(seq {len(ids)}, image {n_img}, video {n_vid}x3, audio {n_aud})")

        # vision padding invariance: same image, smaller vs larger padded budget
        pv2, pos2, n2 = preprocess_image(img, max_soft_tokens=70)
        big_pv = F.pad(pv2, (0, 0, 0, 9 * 70))
        big_pos = F.pad(pos2, (0, 0, 0, 9 * 70), value=-1)
        t_small = model.vision_tower(pv2[None], pos2[None])
        t_big = model.vision_tower(big_pv[None], big_pos[None])
        print("vision padding invariance err:", (t_small - t_big).abs().max().item())

        # audio padding invariance: valid tokens unchanged by extra right padding
        f1, m1 = preprocess_audio([torch.randn(12_345)])
        f2 = F.pad(f1, (0, 0, 0, 40))
        m2 = torch.cat([m1, torch.zeros(1, 40, dtype=torch.bool)], dim=1)
        a1 = model.get_audio_features(f1, m1)
        a2 = model.get_audio_features(f2, m2)
        print("audio padding invariance err:", (a1 - a2).abs().max().item())

        print("MRL 128-d norm:", truncate_embeddings(out.embeddings, 128).norm().item())
