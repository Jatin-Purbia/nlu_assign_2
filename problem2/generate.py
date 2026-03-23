import os
import argparse
import json
from collections import Counter

import torch
import torch.nn.functional as F

from dataset import NameDataset, load_names, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from models  import VanillaRNN, BidirectionalLSTM, AttentionRNN
from train   import DEFAULT_HPARAMS, OUTPUT_DIR

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
NAMES_FILE   = os.path.join(SCRIPT_DIR, "..", "TrainingNames.txt")
TEMPERATURES = [0.5, 0.8, 1.0, 1.2, 1.5]
MODEL_NAMES  = ["rnn", "blstm", "attention"]
MAX_LEN      = 20


def load_trained_model(model_name, dataset, device):
    checkpoint_path = os.path.join(OUTPUT_DIR, f"{model_name}_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\nRun train.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams    = checkpoint["hparams"]
    vocab_size = dataset.vocab_size

    dropout = hparams.get("dropout", 0.0)
    if model_name == "rnn":
        model = VanillaRNN(vocab_size, hparams["embed_size"], hparams["hidden_size"], dropout=dropout)
    elif model_name == "blstm":
        model = BidirectionalLSTM(vocab_size, hparams["embed_size"], hparams["hidden_size"], dropout=dropout)
    elif model_name == "attention":
        model = AttentionRNN(vocab_size, hparams["embed_size"], hparams["hidden_size"], hparams["attn_size"], dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def sample_next_char(logits, temperature):
    probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def generate_rnn_name(model, dataset, temperature, device):
    sos_idx = dataset.char2idx[SOS_TOKEN]
    eos_idx = dataset.char2idx[EOS_TOKEN]
    h = model.rnn_cell.init_hidden(batch_size=1, device=device)
    x = torch.tensor([[sos_idx]], device=device)
    generated_chars = []
    with torch.no_grad():
        for _ in range(MAX_LEN):
            logits, h = model.generate_step(x, h)
            next_idx  = sample_next_char(logits.squeeze(0), temperature)
            if next_idx == eos_idx:
                break
            generated_chars.append(next_idx)
            x = torch.tensor([[next_idx]], device=device)
    return dataset.decode_tensor(torch.tensor(generated_chars))


def generate_blstm_name(model, dataset, temperature, device):
    sos_idx = dataset.char2idx[SOS_TOKEN]
    eos_idx = dataset.char2idx[EOS_TOKEN]
    seed    = torch.tensor([[sos_idx]], device=device)
    h, c    = model.get_initial_decoder_state(seed, torch.tensor([1], device=device), device)
    x       = torch.tensor([[sos_idx]], device=device)
    generated_chars = []
    with torch.no_grad():
        for _ in range(MAX_LEN):
            logits, h, c = model.generate_step(x, h, c)
            next_idx     = sample_next_char(logits.squeeze(0), temperature)
            if next_idx == eos_idx:
                break
            generated_chars.append(next_idx)
            x = torch.tensor([[next_idx]], device=device)
    return dataset.decode_tensor(torch.tensor(generated_chars))


def generate_attention_name(model, dataset, temperature, device):
    sos_idx = dataset.char2idx[SOS_TOKEN]
    eos_idx = dataset.char2idx[EOS_TOKEN]
    prefix  = torch.tensor([[sos_idx]], device=device)
    enc_outputs, h_dec = model.encode_prefix(prefix)
    x = torch.tensor([[sos_idx]], device=device)
    generated_chars = []
    with torch.no_grad():
        for _ in range(MAX_LEN):
            logits, h_dec = model.generate_step(x, h_dec, enc_outputs)
            next_idx      = sample_next_char(logits.squeeze(0), temperature)
            if next_idx == eos_idx:
                break
            generated_chars.append(next_idx)
            x = torch.tensor([[next_idx]], device=device)
            new_h = model.encoder_cell(model.embedding(x).squeeze(1), h_dec)
            enc_outputs = torch.cat([enc_outputs, new_h.unsqueeze(1)], dim=1)
    return dataset.decode_tensor(torch.tensor(generated_chars))


def generate_names(model, model_name, dataset, n, temperature, device):
    generate_fn = {
        "rnn":       generate_rnn_name,
        "blstm":     generate_blstm_name,
        "attention": generate_attention_name,
    }[model_name]
    names = []
    for _ in range(n):
        name = generate_fn(model, dataset, temperature, device)
        if len(name) >= 2:
            names.append(name.title())
    return names


def novelty_rate(generated, training_set):
    if not generated:
        return 0.0
    return sum(1 for name in generated if name.lower() not in training_set) / len(generated)


def diversity(generated):
    if not generated:
        return 0.0
    return len(set(generated)) / len(generated)


def print_metrics_table(all_results):
    print("\n== Quantitative Evaluation =============================================")
    print(f"  {'Model':<12} {'Temp':>6} {'N generated':>12} {'Novelty':>10} {'Diversity':>11}")
    print(f"  {'-'*56}")
    for model_name, temp_results in all_results.items():
        for temp, metrics in temp_results.items():
            print(f"  {model_name:<12} {temp:>6.1f} "
                  f"{metrics['n_generated']:>12,} "
                  f"{metrics['novelty_rate']:>10.3f} "
                  f"{metrics['diversity']:>11.3f}")


def save_generated_names(names, model_name, temperature):
    fname = f"generated_{model_name}_t{str(temperature).replace('.','')}.txt"
    path  = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")
    return path


def load_generated(model_name, temperature):
    fname = f"generated_{model_name}_t{str(temperature).replace('.','')}.txt"
    path  = os.path.join(OUTPUT_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(n_generate=500):
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset     = NameDataset(NAMES_FILE)
    training_set = {name.lower() for name in dataset.names}
    all_results  = {}

    for model_name in MODEL_NAMES:
        print(f"\n-- Generating names with {model_name.upper()} --")
        model = load_trained_model(model_name, dataset, device)
        all_results[model_name] = {}

        for temp in TEMPERATURES:
            names = generate_names(model, model_name, dataset, n_generate, temp, device)
            nov   = novelty_rate(names, training_set)
            div   = diversity(names)
            all_results[model_name][temp] = {
                "n_generated" : len(names),
                "novelty_rate": nov,
                "diversity"   : div,
            }
            path = save_generated_names(names, model_name, temp)
            print(f"\n  Temperature={temp}  |  Novelty={nov:.3f}  |  Diversity={div:.3f}")
            print(f"  Sample: {', '.join(names[:10])}")
            print(f"  Saved: {path}")

    print_metrics_table(all_results)

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    main(n_generate=args.n)
