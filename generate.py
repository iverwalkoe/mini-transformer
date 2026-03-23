import torch

from model import MiniTransformerLM


def decode(tokens: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in tokens)


def main():
    checkpoint = torch.load("checkpoint.pt", map_location="cpu")

    cfg = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    vocab_size = len(stoi)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MiniTransformerLM(
        vocab_size=vocab_size,
        embed_dim=cfg["embed_dim"],
        block_size=cfg["block_size"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    output = model.generate(context, max_new_tokens=400, temperature=0.9)[0].tolist()
    print(decode(output, itos))


if __name__ == "__main__":
    main()