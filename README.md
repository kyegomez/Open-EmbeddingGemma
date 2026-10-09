<img src="./img.png" width="800px"></img>

## Open EmbeddingGemma

Implementation of <a href="https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/">EmbeddingGemma 2</a>, Google DeepMind's 740M parameter multimodal embedding model, in Pytorch

Text, code, images, video and audio (and any interleaving of them) all go through one bidirectional Gemma 4 style encoder and come out as a single 768 dimensional unit vector. The whole thing is one file, and the model itself does not depend on `transformers`. Module and parameter names mirror the official checkpoint, so the released safetensors load directly.

## Install

```bash
$ git clone https://github.com/kyegomez/Open-EmbeddingGemma
$ cd Open-EmbeddingGemma
$ pip install torch safetensors huggingface_hub transformers
```

`safetensors` and `huggingface_hub` are only needed for `from_pretrained`, and `transformers` only for the tokenizer

## Usage

```python
import torch
from open_eg.main import EmbeddingGemma2, EmbeddingGemma2Config

model = EmbeddingGemma2(EmbeddingGemma2Config(vision = None, audio = None)) # text only, 270M

ids = torch.randint(0, 262_144, (2, 1024))
mask = torch.ones_like(ids)

out = model(ids, attention_mask = mask)

out.embeddings       # (2, 768) - mean pooled, l2 normalized
out.token_embeddings # (2, 1024, 768)
```

With the official weights

```python
from transformers import AutoTokenizer
from open_eg.main import EmbeddingGemma2

tokenizer = AutoTokenizer.from_pretrained('google/embeddinggemma-2')
model = EmbeddingGemma2.from_pretrained('google/embeddinggemma-2', modalities = ('text',)).eval()

queries = model.encode_text(['What causes the northern lights?'], tokenizer, prompt_name = 'SearchQuery')

docs = model.encode_text([
    'The northern lights are caused by charged particles from the sun hitting the atmosphere.',
    'Pytorch is a deep learning framework.'
], tokenizer, prompt_name = 'Document')

scores = queries @ docs.T # (1, 2) cosine similarities
```

Task prompts live in `PROMPTS`. Use `SearchQuery`, `QuestionAnswering`, `FactChecking`, `CodeRetrieval`, `Classification`, `Clustering` or `SentenceSimilarity` for queries and `Document` for the corpus (MTEB style aliases are included too)

Matryoshka truncation to 512, 256 or 128 dimensions

```python
from open_eg.main import truncate_embeddings

docs_256 = model.encode_text(texts, tokenizer, prompt_name = 'Document', truncate_dim = 256)

# or after the fact - keeps the leading dimensions and renormalizes

docs_128 = truncate_embeddings(docs, 128)
```

Image and text, embedded into the same space

```python
import torch
from open_eg.main import EmbeddingGemma2, PROMPTS, preprocess_image, build_multimodal_input_ids

model = EmbeddingGemma2.from_pretrained('google/embeddinggemma-2', modalities = ('text', 'image')).eval()

tokenize = lambda text: tokenizer(text, add_special_tokens = False)['input_ids']

image = torch.randint(0, 256, (3, 480, 640), dtype = torch.uint8) # (c, h, w)

pixels, positions, num_image_tokens = preprocess_image(image, max_soft_tokens = 280) # 70 | 140 | 280 | 560 | 1120

ids = build_multimodal_input_ids([
    PROMPTS['Document'] + 'waterproof trail runners ',
    ('image', num_image_tokens)
], tokenize = tokenize)

out = model(
    torch.tensor([ids]),
    pixel_values = pixels[None],
    image_position_ids = positions[None]
)

out.embeddings # (1, 768)
```

Video and audio

```python
from open_eg.main import sample_video_frames, preprocess_video, preprocess_audio, num_audio_soft_tokens

model = EmbeddingGemma2.from_pretrained('google/embeddinggemma-2').eval() # all modalities, 740M

video = torch.rand(90, 3, 360, 640) # (frames, c, h, w) in [0, 1]
waveform = torch.randn(16_000 * 3)  # mono, 16kHz

frames = sample_video_frames(video, video_fps = 30.)                       # 1 fps, at most 32 frames
video_pixels, video_positions, tokens_per_frame = preprocess_video(frames) # up to 140 soft tokens per frame

audio_features, audio_mask = preprocess_audio([waveform])  # log-mel, clips up to 30s
num_audio_tokens = num_audio_soft_tokens(audio_mask[0])    # 25 per second

ids = build_multimodal_input_ids([
    PROMPTS['Document'] + 'a dog barking at the mailman ',
    ('video', tokens_per_frame, len(frames)),
    ('audio', num_audio_tokens)
], tokenize = tokenize)

out = model(
    torch.tensor([ids]),
    pixel_values_videos = video_pixels,
    video_position_ids = video_positions,
    input_features = audio_features,
    input_features_mask = audio_mask
)

out.embeddings # (1, 768)
```

Run it in `bfloat16` or `float32`. `float16` overflows the activations, so `from_pretrained` refuses it

## Architecture

- **Text** (270M = 130M transformer + 140M embedder). 24 bidirectional layers with d 512 and a 5:1 local to global ratio. Local layers attend within `|i - j| <= 512` using GQA (4 query / 2 kv heads, head dim 256, RoPE θ 1e4). Every 6th layer is global, with full attention and MQA (4 query / 1 kv head, head dim 512, RoPE θ 1e6). Also QK-norm, V-norm, attention scale of 1, sandwich norms, gated GELU feedforward, and projection-only per-layer embeddings that gate every layer
- **Vision** (170M). 16 layer ViT on 16px patches, with learned x / y position tables plus axial 2D RoPE. Patches are 3x3 average pooled down to at most 280 soft tokens per image (140 per video frame)
- **Audio** (300M). 12 layer USM-style conformer on 128 bin log-mel. 4x conv subsampling gives 25 tokens per second, followed by chunked local attention with Transformer-XL relative positions and logit soft-capping
- **Head**. Final RMSNorm, then linear 512 → 768, mean pool and L2 normalize. Matryoshka-trained at 768 / 512 / 256 / 128

Each encoder's soft tokens are projected to 512 and spliced into the token sequence in place of the `<|image|>`, `<|video|>` and `<|audio|>` placeholders. The interleaved sequence (up to 8192 tokens) then runs through the text encoder as one

## Test

```bash
$ python -m open_eg.main
```

This prints the full size parameter budget, then runs a tiny random model through every code path and checks text, vision and audio padding invariance

```
    text_embedder:   140.5M
 text_transformer:   130.5M
           vision:   167.8M
            audio:   305.6M
            total:   744.4M
```

## Todo

- [x] bidirectional text encoder with 5:1 local / global attention and per-layer embeddings
- [x] vision tower, shared by images and video frames
- [x] conformer audio tower
- [x] interleaved multimodal sequences
- [x] matryoshka truncation
- [x] `from_pretrained` for the official safetensors

- [ ] numerical parity test against the reference `transformers` implementation
- [ ] training, with contrastive loss, geometric embedding distillation and spread-out regularizer
- [ ] package for `pip install`

## Citations

```bibtex
@misc{embeddinggemma2,
    title   = {EmbeddingGemma 2 is a best-in-class open model for natively multimodal embeddings},
    author  = {Google DeepMind},
    year    = {2026},
    url     = {https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/}
}
```

```bibtex
@article{Vera2025EmbeddingGemma,
    title   = {EmbeddingGemma: Powerful and Lightweight Text Representations},
    author  = {Henrique Schechter Vera and Sahil Dua and Biao Zhang and Daniel Salz and Ryan Mullins and Sindhu Raghuram Panyam and Sara Smoot and Iftekhar Naim and Joe Zou and Feiyang Chen and others},
    journal = {arXiv preprint arXiv:2509.20354},
    year    = {2025}
}
```

```bibtex
@article{GemmaTeam2025Gemma3,
    title   = {Gemma 3 Technical Report},
    author  = {Gemma Team},
    journal = {arXiv preprint arXiv:2503.19786},
    year    = {2025}
}
```

```bibtex
@inproceedings{Kusupati2022Matryoshka,
    title     = {Matryoshka Representation Learning},
    author    = {Aditya Kusupati and Gantavya Bhatt and Aniket Rege and Matthew Wallingford and Aditya Sinha and Vivek Ramanujan and William Howard-Snyder and Kaifeng Chen and Sham Kakade and Prateek Jain and Ali Farhadi},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2022}
}
```

```bibtex
@inproceedings{Gulati2020Conformer,
    title     = {Conformer: Convolution-augmented Transformer for Speech Recognition},
    author    = {Anmol Gulati and James Qin and Chung-Cheng Chiu and Niki Parmar and Yu Zhang and Jiahui Yu and Wei Han and Shibo Wang and Zhengdong Zhang and Yonghui Wu and Ruoming Pang},
    booktitle = {Interspeech},
    year      = {2020}
}
```

```bibtex
@article{Zhang2023USM,
    title   = {Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages},
    author  = {Yu Zhang and Wei Han and James Qin and Yongqiang Wang and Ankur Bapna and Zhehuai Chen and others},
    journal = {arXiv preprint arXiv:2303.01037},
    year    = {2023}
}
```

```bibtex
@inproceedings{Dai2019TransformerXL,
    title     = {Transformer-XL: Attentive Language Models beyond a Fixed-Length Context},
    author    = {Zihang Dai and Zhilin Yang and Yiming Yang and Jaime Carbonell and Quoc V. Le and Ruslan Salakhutdinov},
    booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year      = {2019}
}
```

```bibtex
@article{Su2021RoFormer,
    title   = {RoFormer: Enhanced Transformer with Rotary Position Embedding},
    author  = {Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
    journal = {arXiv preprint arXiv:2104.09864},
    year    = {2021}
}
```

```bibtex
@inproceedings{Ainslie2023GQA,
    title     = {GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
    author    = {Joshua Ainslie and James Lee-Thorp and Michiel de Jong and Yury Zemlyanskiy and Federico Lebr{\'o}n and Sumit Sanghai},
    booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
    year      = {2023}
}
```

```bibtex
@article{Shazeer2019MQA,
    title   = {Fast Transformer Decoding: One Write-Head is All You Need},
    author  = {Noam Shazeer},
    journal = {arXiv preprint arXiv:1911.02150},
    year    = {2019}
}
```

```bibtex
@inproceedings{Henry2020QKNorm,
    title     = {Query-Key Normalization for Transformers},
    author    = {Alex Henry and Prudhvi Raj Dachapally and Shubham Shantaram Pawar and Yuxuan Chen},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
    year      = {2020}
}
```

```bibtex
@article{Ding2021CogView,
    title   = {CogView: Mastering Text-to-Image Generation via Transformers},
    author  = {Ming Ding and Zhuoyi Yang and Wenyi Hong and Wendi Zheng and Chang Zhou and Da Yin and Junyang Lin and Xu Zou and Zhou Shao and Hongxia Yang and Jie Tang},
    journal = {arXiv preprint arXiv:2105.13290},
    year    = {2021}
}
```
