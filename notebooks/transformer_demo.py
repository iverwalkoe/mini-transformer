# %%
# Mini Transformer Language Model in PyTorch
#
# Notebook-style script for VS Code / Jupyter that reads training text
# from the repository file: data/input.txt

from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# Device setup and reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
random.seed(42)


# %%
# Locate and load the dataset from the repository
#
# Assumes this file lives at:
#   notebooks/transformer_demo.py
# and the dataset lives at:
#   data/input.txt

repo_root = Path.cwd()
if repo_root.name == "notebooks":
    repo_root = repo_root.parent

data_path = repo_root / "data" / "input.txt"

if not data_path.exists():
    raise FileNotFoundError(
        f"Could not find dataset at {data_path}. "
        "Make sure your repo contains data/input.txt."
    )

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read().strip()

print("Loaded dataset from:", data_path)
print("Corpus length:", len(text))
print()
print(text[:500])


# %%
# Vocabulary and tokenization
#
# This is a character-level model: each unique character is a token.

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(tokens: list[int]) -> str:
    return "".join(itos[i] for i in tokens)


data = torch.tensor(encode(text), dtype=torch.long)

print("Vocabulary size:", vocab_size)
print("First 40 characters:", chars[:40])
print("Encoded sample:", data[:30].tolist())


# %%
# Train / validation split and batching

batch_size = 32
block_size = 64
train_ratio = 0.9

n = int(train_ratio * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str = "train"):
    source = train_data if split == "train" else val_data

    if len(source) <= block_size + 1:
        raise ValueError(
            f"Dataset split '{split}' is too small for block_size={block_size}. "
            "Use a larger input file or reduce block_size."
        )

    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


xb, yb = get_batch("train")
print("Batch shapes:", xb.shape, yb.shape)


# %%
# Transformer components

class Head(nn.Module):
    def __init__(self, embed_dim: int, head_size: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape

        k = self.key(x)
        q = self.query(x)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)

        # Causal masking: prevent attention to future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        head_size = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(embed_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sa = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        block_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(
            *[Block(embed_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


# %%
# Model setup

embed_dim = 128
num_heads = 4
num_layers = 4
dropout = 0.1
learning_rate = 3e-4

model = MiniTransformerLM(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    block_size=block_size,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")


# %%
# Loss estimation helper

@torch.no_grad()
def estimate_loss(eval_iters: int = 50):
    model.eval()
    out = {}

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


# %%
# Training

max_iters = 2000
eval_interval = 200

train_history = []
val_history = []
steps = []

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        train_history.append(losses["train"])
        val_history.append(losses["val"])
        steps.append(step)
        print(
            f"step {step:4d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# %%
# Plot loss curves

plt.figure(figsize=(8, 5))
plt.plot(steps, train_history, label="train")
plt.plot(steps, val_history, label="val")
plt.xlabel("Training step")
plt.ylabel("Cross-entropy loss")
plt.title("Training and validation loss")
plt.legend()
plt.grid(True)
plt.show()


# %%
# Generate sample text

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=400, temperature=0.9)[0].tolist()
print(decode(generated))


# %%
# Save checkpoint in the repo root

checkpoint_path = repo_root / "checkpoint.pt"

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "config": {
        "batch_size": batch_size,
        "block_size": block_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "data_path": str(data_path),
    },
}

torch.save(checkpoint, checkpoint_path)
print(f"Saved checkpoint to {checkpoint_path}")


# %%
# Optional: if running in Colab, uncomment to download the checkpoint
#
# from google.colab import files
# files.download(str(checkpoint_path))