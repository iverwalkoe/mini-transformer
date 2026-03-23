# Mini Transformer Language Model in PyTorch

A compact character-level transformer language model implemented from scratch in PyTorch.

This project demonstrates the core components of transformer-based language models, including:

- token embeddings
- positional embeddings
- masked self-attention
- multi-head attention
- feed-forward layers
- residual connections
- layer normalization
- autoregressive text generation

---

## 🚀 Try it in Google Colab

Run the project instantly in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iverwalkoe/mini-transformer/blob/main/notebooks/transformer_demo.ipynb)

---

## 📂 Project Structure

```text
mini-transformer-pytorch/
├── README.md
├── requirements.txt
├── config.py
├── data.py
├── model.py
├── train.py
├── generate.py
├── data/
│   └── input.txt
├── notebooks/
│   └── transformer_demo.ipynb
└── outputs/