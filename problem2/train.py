import os
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_dataloader, NameDataset
from models  import VanillaRNN, BidirectionalLSTM, AttentionRNN

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NAMES_FILE = os.path.join(SCRIPT_DIR, "..", "TrainingNames.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_HPARAMS = {
    "embed_size"   : 32,
    "hidden_size"  : 128,
    "attn_size"    : 64,
    "learning_rate": 0.001,
    "batch_size"   : 32,
    "num_epochs"   : 200,
    "grad_clip"    : 5.0,
}

SEARCH_GRID = [
    {"embed_size": 32,  "hidden_size": 128, "attn_size": 64,  "learning_rate": 0.001, "batch_size": 32, "grad_clip": 5.0, "dropout": 0.2},
    {"embed_size": 64,  "hidden_size": 128, "attn_size": 64,  "learning_rate": 0.001, "batch_size": 32, "grad_clip": 5.0, "dropout": 0.2},
    {"embed_size": 64,  "hidden_size": 256, "attn_size": 128, "learning_rate": 0.001, "batch_size": 32, "grad_clip": 5.0, "dropout": 0.3},
    {"embed_size": 64,  "hidden_size": 256, "attn_size": 128, "learning_rate": 0.003, "batch_size": 32, "grad_clip": 5.0, "dropout": 0.3},
    {"embed_size": 64,  "hidden_size": 256, "attn_size": 128, "learning_rate": 0.001, "batch_size": 64, "grad_clip": 5.0, "dropout": 0.3},
    {"embed_size": 128, "hidden_size": 256, "attn_size": 128, "learning_rate": 0.001, "batch_size": 64, "grad_clip": 5.0, "dropout": 0.3},
]

SEARCH_EPOCHS = 60
BEST_EPOCHS   = 400


def train_model(model, loader, dataset, hparams, model_name, device):
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    num_epochs   = hparams["num_epochs"]
    loss_history = []

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Epochs: {num_epochs}  |  LR: {hparams['learning_rate']}  |  "
          f"Batch: {hparams['batch_size']}  |  Clip: {hparams['grad_clip']}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        epoch_loss, num_batches = 0.0, 0

        for inputs, targets, lengths in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(inputs, lengths)

            batch_size, seq_len, vocab_size = logits.shape
            loss = criterion(logits.view(batch_size * seq_len, vocab_size),
                             targets.view(batch_size * seq_len))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hparams["grad_clip"])
            optimizer.step()

            epoch_loss  += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{num_epochs}  |  loss = {avg_loss:.4f}  |  "
                  f"lr = {optimizer.param_groups[0]['lr']:.6f}")

    print(f"\n  Final loss: {loss_history[-1]:.4f}")
    return loss_history


def save_checkpoint(model, model_name, hparams):
    path = os.path.join(OUTPUT_DIR, f"{model_name}_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "hparams": hparams}, path)
    print(f"  Checkpoint saved: {path}")


def save_loss_curve(loss_history, model_name):
    path = os.path.join(OUTPUT_DIR, f"{model_name}_loss_curve.png")
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"{model_name} -- Training Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Loss curve saved: {path}")


def save_training_summary(model_name, model, loss_history, hparams, dataset):
    summary = {
        "model_name"  : model_name,
        "num_params"  : model.count_parameters(),
        "vocab_size"  : dataset.vocab_size,
        "dataset_size": len(dataset),
        "hparams"     : hparams,
        "final_loss"  : round(loss_history[-1], 5),
        "min_loss"    : round(min(loss_history), 5),
    }
    path = os.path.join(OUTPUT_DIR, f"{model_name}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved: {path}")


def build_model(name, vocab_size, hparams):
    dropout = hparams.get("dropout", 0.0)
    if name == "rnn":
        return VanillaRNN(vocab_size, hparams["embed_size"], hparams["hidden_size"], dropout=dropout)
    elif name == "blstm":
        return BidirectionalLSTM(vocab_size, hparams["embed_size"], hparams["hidden_size"], dropout=dropout)
    elif name == "attention":
        return AttentionRNN(vocab_size, hparams["embed_size"], hparams["hidden_size"], hparams["attn_size"], dropout=dropout)
    else:
        raise ValueError(f"Unknown model name '{name}'. Choose: rnn / blstm / attention")


def run_hyperparameter_search(model_names, device, dataset):
    print(f"\n{'='*65}")
    print(f"  Hyperparameter Search: {len(SEARCH_GRID)} configs x {len(model_names)} models")
    print(f"  Search epochs: {SEARCH_EPOCHS}  |  Best retrain epochs: {BEST_EPOCHS}")
    print(f"{'='*65}")

    best_configs = {}

    for model_name in model_names:
        print(f"\n  -- Searching {model_name.upper()} --")
        print(f"  {'Cfg':>4} | {'embed':>5} | {'hidden':>6} | {'lr':>7} | {'batch':>5} | {'loss':>8}")
        print("  " + "-" * 46)

        best_loss, best_cfg = float("inf"), None

        for ci, cfg in enumerate(SEARCH_GRID):
            search_loader, _ = get_dataloader(NAMES_FILE, batch_size=cfg["batch_size"], shuffle=True)
            search_hparams   = {**cfg, "num_epochs": SEARCH_EPOCHS}
            model      = build_model(model_name, dataset.vocab_size, search_hparams)
            loss_hist  = train_model(model, search_loader, dataset, search_hparams,
                                     f"{model_name}_search_cfg{ci}", device)
            final_loss = loss_hist[-1]
            marker = " <-- best" if final_loss < best_loss else ""
            print(f"  {ci:>4} | {cfg['embed_size']:>5} | {cfg['hidden_size']:>6} | "
                  f"{cfg['learning_rate']:>7.4f} | {cfg['batch_size']:>5} | {final_loss:>8.4f}{marker}")
            if final_loss < best_loss:
                best_loss, best_cfg = final_loss, cfg

        best_configs[model_name] = best_cfg
        print(f"\n  Best config for {model_name}: {best_cfg}  (loss={best_loss:.4f})")

    print(f"\n{'='*65}")
    print(f"  Full retraining -- best configs ({BEST_EPOCHS} epochs each)")
    print(f"{'='*65}")

    search_summary = {}

    for model_name in model_names:
        best_cfg    = best_configs[model_name]
        full_loader, _ = get_dataloader(NAMES_FILE, batch_size=best_cfg["batch_size"], shuffle=True)
        hparams     = {**best_cfg, "num_epochs": BEST_EPOCHS}
        model       = build_model(model_name, dataset.vocab_size, hparams)
        loss_history = train_model(model, full_loader, dataset, hparams, model_name, device)
        save_checkpoint(model, model_name, hparams)
        save_loss_curve(loss_history, model_name)
        save_training_summary(model_name, model, loss_history, hparams, dataset)
        search_summary[model_name] = {"best_config": best_cfg, "final_loss": round(loss_history[-1], 5)}

    summary_path = os.path.join(OUTPUT_DIR, "hyperparam_search_results.json")
    with open(summary_path, "w") as f:
        json.dump(search_summary, f, indent=2)
    print(f"\n  Search summary saved: {summary_path}")

    return best_configs


def main(model_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, dataset = get_dataloader(NAMES_FILE, batch_size=DEFAULT_HPARAMS["batch_size"], shuffle=True)
    print(f"\nDataset: {len(dataset)} names | Vocab size: {dataset.vocab_size}")

    run_hyperparameter_search(model_names, device, dataset)
    print("\nAll models trained and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train character-level name generation models.")
    parser.add_argument("--model", type=str, default="all",
                        help='Model to train: "rnn", "blstm", "attention", or "all"')
    parser.add_argument("--no-search", action="store_true",
                        help="Skip hyperparameter search and train once with DEFAULT_HPARAMS.")
    args = parser.parse_args()

    names_to_train = ["rnn", "blstm", "attention"] if args.model == "all" else [args.model]

    if args.no_search:
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader, dataset = get_dataloader(NAMES_FILE, batch_size=DEFAULT_HPARAMS["batch_size"], shuffle=True)
        for name in names_to_train:
            model = build_model(name, dataset.vocab_size, DEFAULT_HPARAMS)
            loss_history = train_model(model, loader, dataset, DEFAULT_HPARAMS, name, device)
            save_checkpoint(model, name, DEFAULT_HPARAMS)
            save_loss_curve(loss_history, name)
            save_training_summary(name, model, loss_history, DEFAULT_HPARAMS, dataset)
        print("\nAll models trained and saved.")
    else:
        main(names_to_train)
