from dataclasses import dataclass
import torch


@dataclass
class Vocabulary:
    stoi: dict
    itos: dict
    vocab_size: int


def build_vocab(text: str) -> Vocabulary:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return Vocabulary(stoi=stoi, itos=itos, vocab_size=len(chars))


def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in text]


def decode(tokens: list[int], itos: dict) -> str:
    return "".join(itos[i] for i in tokens)


def build_dataset(text: str, stoi: dict) -> torch.Tensor:
    return torch.tensor(encode(text, stoi), dtype=torch.long)


def split_dataset(data: torch.Tensor, train_ratio: float = 0.9):
    n = int(train_ratio * len(data))
    return data[:n], data[n:]


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)