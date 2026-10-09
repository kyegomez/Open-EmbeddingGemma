import math
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from open_eg.main import (
    AUDIO_TOKEN_ID,
    BOA_TOKEN_ID,
    BOI_TOKEN_ID,
    BOS_TOKEN_ID,
    EOA_TOKEN_ID,
    EOI_TOKEN_ID,
    EOS_TOKEN_ID,
    IMAGE_TOKEN_ID,
    VIDEO_TOKEN_ID,
    AudioConfig,
    EmbeddingGemma2,
    EmbeddingGemma2Config,
    TextConfig,
    VisionConfig,
    truncate_embeddings,
)

# =====================================================================================================
# Pre-processing (mirrors the official processors; pure torch)
# =====================================================================================================

SUPPORTED_SOFT_TOKENS: tuple[int, ...] = (70, 140, 280, 560, 1120)


def _target_size(
    height: int, width: int, patch: int, max_patches: int, k: int
) -> tuple[int, int]:
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
    th, tw = _target_size(
        img.shape[-2], img.shape[-1], patch_size, max_patches, pooling_kernel_size
    )
    if (th, tw) != tuple(img.shape[-2:]):
        img = F.interpolate(
            img[None],
            size=(th, tw),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        )[0]
        img = (
            img.round() if is_uint8 else img
        )  # torchvision resizes uint8 images in uint8
    img = img.clamp(0, 255) / 255.0

    c, nh, nw = img.shape[0], th // patch_size, tw // patch_size
    patches = (
        img.reshape(c, nh, patch_size, nw, patch_size)
        .permute(1, 3, 2, 4, 0)
        .reshape(nh * nw, -1)
    )
    xs, ys = torch.meshgrid(torch.arange(nw), torch.arange(nh), indexing="xy")
    positions = torch.stack([xs, ys], dim=-1).reshape(nh * nw, 2)

    pad = max_patches - patches.shape[0]
    patches = F.pad(patches, (0, 0, 0, pad), value=0.0)
    positions = F.pad(positions, (0, 0, 0, pad), value=-1)
    return patches, positions, (nh * nw) // pooling_kernel_size**2


def sample_video_frames(
    frames: Tensor, video_fps: float, fps: float = 1.0, max_frames: int = 32
) -> Tensor:
    """Uniformly samples ``fps`` frames per second (default 1), capped at ``max_frames`` (default 32).

    frames: (num_frames, 3, H, W). Returns the selected frames.
    """
    total = frames.shape[0]
    n = max(1, min(max_frames, int(total / video_fps * fps)))
    idx = torch.linspace(0, total - 1, n).round().long()
    return frames[idx]


def preprocess_video(
    frames: Tensor, max_soft_tokens: int = 140, **kwargs: Any
) -> tuple[Tensor, Tensor, int]:
    """Patchifies already-sampled frames (num_frames, 3, H, W) with the image pipeline.

    Returns patches (F, N, 3*p*p), positions (F, N, 2) and soft tokens per frame (default budget 140).
    """
    out = [
        preprocess_image(f, max_soft_tokens=max_soft_tokens, **kwargs) for f in frames
    ]
    return torch.stack([o[0] for o in out]), torch.stack([o[1] for o in out]), out[0][2]


def _mel_filter_bank(
    n_freqs: int, n_mels: int, f_min: float, f_max: float, sr: int
) -> Tensor:
    """HTK-scale triangular mel filters without area normalisation, (n_freqs, n_mels), float64."""

    def hz_to_mel(f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    mels = torch.linspace(
        hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2, dtype=torch.float64
    )
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
    sample_mask = torch.cat(
        [torch.zeros(frame_length // 2, dtype=torch.bool), sample_mask.bool().cpu()]
    )
    frames = wav.unfold(-1, frame_length + 1, hop_length)[
        ..., :-1
    ]  # (num_frames, frame_length)
    window = torch.hann_window(frame_length, periodic=True, dtype=torch.float64)
    spec = torch.fft.rfft(frames * window, n=fft_length).abs()
    mel = spec @ _mel_filter_bank(
        fft_length // 2 + 1, n_mels, 0.0, sampling_rate / 2, sampling_rate
    )
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
    images, videos, audios = (
        iter(image_soft_tokens),
        iter(video_soft_tokens),
        iter(audio_soft_tokens),
    )
    out: list[int] = []
    try:
        for tok in input_ids:
            if tok == IMAGE_TOKEN_ID:
                out += [BOI_TOKEN_ID] + [IMAGE_TOKEN_ID] * next(images) + [EOI_TOKEN_ID]
            elif tok == VIDEO_TOKEN_ID:
                per_frame, num_frames = next(videos)
                out += (
                    [BOI_TOKEN_ID] + [VIDEO_TOKEN_ID] * per_frame + [EOI_TOKEN_ID]
                ) * num_frames
            elif tok == AUDIO_TOKEN_ID:
                out += [BOA_TOKEN_ID] + [AUDIO_TOKEN_ID] * next(audios) + [EOA_TOKEN_ID]
            else:
                out.append(int(tok))
    except StopIteration:
        raise ValueError(
            "More placeholders in input_ids than soft-token counts were given."
        ) from None
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
            ids += expand_multimodal_placeholders(
                [IMAGE_TOKEN_ID], image_soft_tokens=[part[1]]
            )
        elif part[0] == "audio":
            ids += expand_multimodal_placeholders(
                [AUDIO_TOKEN_ID], audio_soft_tokens=[part[1]]
            )
        elif part[0] == "video":
            ids += expand_multimodal_placeholders(
                [VIDEO_TOKEN_ID], video_soft_tokens=[(part[1], part[2])]
            )
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
        text=TextConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            hidden_size_per_layer_input=16,
            sliding_window=4,
            embedding_dim=48,
        ),
        vision=VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
            position_embedding_size=64,
        ),
        audio=AudioConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            subsampling_conv_channels=(16, 8),
            output_proj_dims=48,
        ),
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
        print(
            "text padding invariance err:", (e_batch[0] - e_alone[0]).abs().max().item()
        )

        # image + video + audio interleaved
        img = torch.randint(0, 256, (3, 120, 200), dtype=torch.uint8)
        pv, pos, n_img = preprocess_image(img, max_soft_tokens=70)
        vid = torch.rand(3, 3, 64, 96)
        vpv, vpos, n_vid = preprocess_video(vid, max_soft_tokens=70)
        feats, fmask = preprocess_audio([torch.randn(16_000), torch.randn(9_000)])
        n_aud = [num_audio_soft_tokens(m) for m in fmask]
        ids = build_multimodal_input_ids(
            [
                "task: search result | query: shoes ",
                ("image", n_img),
                " clip ",
                ("video", n_vid, 3),
                " sound ",
                ("audio", n_aud[0]),
                " and ",
                ("audio", n_aud[1]),
            ],
            tokenize=lambda s: [10 + (ord(ch) % 200) for ch in s],
        )
        out = model(
            torch.tensor([ids]),
            pixel_values=pv[None],
            image_position_ids=pos[None],
            pixel_values_videos=vpv,
            video_position_ids=vpos,
            input_features=feats,
            input_features_mask=fmask,
        )
        print(
            "multimodal:",
            tuple(out.embeddings.shape),
            "norm",
            out.embeddings.norm().item(),
            f"(seq {len(ids)}, image {n_img}, video {n_vid}x3, audio {n_aud})",
        )

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
