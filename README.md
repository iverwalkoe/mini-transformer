# mini-transformer

# Mini Transformer Language Model in PyTorch

A compact character-level transformer language model implemented from scratch in PyTorch.

This project was built to understand the core mechanics of transformer-based language models, including:

- token embeddings
- positional embeddings
- masked self-attention
- multi-head attention
- feed-forward blocks
- residual connections
- layer normalization
- autoregressive text generation

## Why this project

This implementation is intentionally small and readable. The goal is not to train a large model, but to understand how transformer language models work end-to-end.

## Project structure

```text
mini-transformer-pytorch/
├── README.md
├── requirements.txt
├── config.py
├── data.py
├── model.py
├── train.py
└── generate.py
