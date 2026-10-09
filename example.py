import torch
from open_eg.main import EmbeddingGemma2, EmbeddingGemma2Config, truncate_embeddings

# text only EmbeddingGemma 2 (270M), randomly initialized

model = EmbeddingGemma2(EmbeddingGemma2Config(vision = None, audio = None)).eval()

print(f"params: {model.num_parameters()['total'] / 1e6:.1f}M")

# batch of 2 sequences, the second one right padded
# token ids stay below 255_999 to avoid the special / multimodal placeholder ids

ids = torch.randint(3, 255_999, (2, 128))
mask = torch.ones_like(ids)

ids[1, 100:] = 0
mask[1, 100:] = 0

with torch.no_grad():
    out = model(ids, attention_mask = mask)

print(out.embeddings.shape)           # (2, 768) - mean pooled, l2 normalized
print(out.token_embeddings.shape)     # (2, 128, 768)
print(out.embeddings.norm(dim = -1))  # (1., 1.)

# matryoshka truncation to 512 | 256 | 128

small = truncate_embeddings(out.embeddings, 256)

print(small.shape)                    # (2, 256)
