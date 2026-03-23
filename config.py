from dataclasses import dataclass


@dataclass
class Config:
    batch_size: int = 32
    block_size: int = 64
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 50
    seed: int = 42

    data_path: str = "data/input.txt"
    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"