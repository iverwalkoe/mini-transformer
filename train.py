import torch

from config import Config
from data import build_dataset, build_vocab, get_batch, split_dataset
from model import MiniTransformerLM


def estimate_loss(model, train_data, val_data, cfg, device):
    model.eval()
    out = {}

    with torch.no_grad():
        for split in ["train", "val"]:
            losses = torch.zeros(cfg.eval_iters)
            for k in range(cfg.eval_iters):
                x, y = get_batch(
                    split=split,
                    train_data=train_data,
                    val_data=val_data,
                    batch_size=cfg.batch_size,
                    block_size=cfg.block_size,
                    device=device,
                )
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()

    model.train()
    return out


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    text = cfg.text.strip()
    vocab = build_vocab(text)
    data = build_dataset(text, vocab.stoi)
    train_data, val_data = split_dataset(data)

    model = MiniTransformerLM(
        vocab_size=vocab.vocab_size,
        embed_dim=cfg.embed_dim,
        block_size=cfg.block_size,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    for step in range(cfg.max_iters):
        if step % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(
            split="train",
            train_data=train_data,
            val_data=val_data,
            batch_size=cfg.batch_size,
            block_size=cfg.block_size,
            device=device,
        )

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "stoi": vocab.stoi,
            "itos": vocab.itos,
            "config": cfg.__dict__,
        },
        "checkpoint.pt",
    )
    print("Saved checkpoint to checkpoint.pt")


if __name__ == "__main__":
    main()