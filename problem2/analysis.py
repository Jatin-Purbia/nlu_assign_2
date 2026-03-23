import os
import json
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_names

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")
NAMES_FILE  = os.path.join(SCRIPT_DIR, "..", "TrainingNames.txt")
REPORT_FILE = os.path.join(OUTPUT_DIR, "qualitative_report.txt")
PLOTS_DIR   = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

TEMPERATURES = [0.5, 0.8, 1.0, 1.2, 1.5]
MODEL_NAMES  = ["rnn", "blstm", "attention"]

INDIAN_PATTERNS = [
    "ar", "ra", "an", "av", "sh", "kr", "vi", "pr", "ka", "na", "pa",
    "ma", "sa", "la", "ha", "da", "ta", "ga", "ba", "va", "ya", "ja",
    "ni", "ri", "ki", "si", "ti", "di", "li", "hi", "mi", "pi",
    "deep", "dev", "raj", "ram", "lax", "man", "sun", "nee", "san",
    "pra", "sri", "esh", "ish", "iya", "aya", "eka", "uja", "aka",
]


def looks_indian(name: str) -> bool:
    name_lower = name.lower()
    if not (3 <= len(name) <= 16):
        return False
    return any(pat in name_lower for pat in INDIAN_PATTERNS)


def load_generated(model_name, temperature):
    fname = f"generated_{model_name}_t{str(temperature).replace('.','')}.txt"
    path  = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def analyse_realism(names):
    if not names:
        return {"realistic": 0, "total": 0, "rate": 0.0}
    realistic = sum(1 for n in names if looks_indian(n))
    return {"realistic": realistic, "total": len(names), "rate": realistic / len(names)}


def analyse_failure_modes(names, training_set):
    failures = {"too_short": [], "too_long": [], "memorised": [],
                "repeated_char": [], "no_vowel": []}
    vowels = set("aeiou")
    for name in names:
        n = name.lower()
        if len(n) < 3:           failures["too_short"].append(name)
        if len(n) > 16:          failures["too_long"].append(name)
        if n in training_set:    failures["memorised"].append(name)
        if len(set(n)) == 1:     failures["repeated_char"].append(name)
        if not any(ch in vowels for ch in n): failures["no_vowel"].append(name)
    return {k: len(v) for k, v in failures.items()}


def length_distribution(names):
    return dict(Counter(len(n) for n in names))


def char_frequency(names):
    return Counter(ch.lower() for name in names for ch in name if ch.isalpha())


def plot_length_distributions(all_data, training_names):
    train_lengths = Counter(len(n) for n in training_names)
    max_len = max(max(train_lengths.keys()), 16)
    x = list(range(1, max_len + 1))

    fig, axes = plt.subplots(1, len(MODEL_NAMES) + 1, figsize=(20, 4), sharey=True)

    def bar_plot(ax, length_counter, title, color):
        heights = [length_counter.get(l, 0) for l in x]
        total   = sum(heights)
        ax.bar(x, [h / max(total, 1) for h in heights], color=color, alpha=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Name length")
        ax.set_ylabel("Fraction")
        ax.set_xticks(x)

    colors = ["#adb5bd", "#e63946", "#2a9d8f", "#e9c46a"]
    bar_plot(axes[0], train_lengths, "Training Set", colors[0])
    for i, model_name in enumerate(MODEL_NAMES):
        names  = all_data.get(model_name, {}).get(1.0, [])
        counts = Counter(len(n) for n in names)
        bar_plot(axes[i + 1], counts, f"{model_name.upper()} (T=1.0)", colors[i + 1])

    fig.suptitle("Name Length Distribution: Training vs. Generated", fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "length_distributions.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Length distribution plot saved: {save_path}")


def plot_char_frequencies(all_data, training_names):
    train_freq = char_frequency(training_names)
    top10 = [ch for ch, _ in train_freq.most_common(10)]

    fig, axes = plt.subplots(1, len(MODEL_NAMES) + 1, figsize=(22, 4), sharey=False)

    def bar_plot_chars(ax, freq_counter, title, color):
        total  = sum(freq_counter.values())
        values = [freq_counter.get(ch, 0) / max(total, 1) for ch in top10]
        ax.bar(top10, values, color=color, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Character")
        ax.set_ylabel("Relative frequency")

    colors = ["#adb5bd", "#e63946", "#2a9d8f", "#e9c46a"]
    bar_plot_chars(axes[0], train_freq, "Training Set", colors[0])
    for i, model_name in enumerate(MODEL_NAMES):
        names = all_data.get(model_name, {}).get(1.0, [])
        freq  = char_frequency(names) if names else Counter()
        bar_plot_chars(axes[i + 1], freq, f"{model_name.upper()} (T=1.0)", colors[i + 1])

    fig.suptitle("Top-10 Character Frequencies: Training vs. Generated", fontsize=12)
    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "char_frequencies.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Char frequency plot saved: {save_path}")


def build_report(all_data, training_names):
    training_set = {n.lower() for n in training_names}
    lines = []
    lines.append("== QUALITATIVE ANALYSIS -- Character-level Name Generation ==")

    for model_name in MODEL_NAMES:
        lines.append(f"\n{'='*60}")
        lines.append(f"  Model: {model_name.upper()}")
        lines.append(f"{'='*60}")

        for temp in [0.5, 1.0, 1.5]:
            names = all_data.get(model_name, {}).get(temp, [])
            if not names:
                lines.append(f"  Temperature {temp}: no data found.")
                continue
            real   = analyse_realism(names)
            fail   = analyse_failure_modes(names, training_set)
            sample = names[:20]
            lines.append(f"\n  -- Temperature = {temp} --")
            lines.append(f"  Total generated : {len(names)}")
            lines.append(f"  Realism rate    : {real['rate']:.3f}  ({real['realistic']}/{real['total']})")
            lines.append(f"  Failure modes:")
            for mode, count in fail.items():
                lines.append(f"    {mode:<20} : {count}")
            lines.append(f"  Sample (first 20):")
            for i in range(0, len(sample), 5):
                chunk = sample[i:i+5]
                lines.append("    " + "  |  ".join(f"{n:<15}" for n in chunk))

    lines.append(f"\n{'='*60}")
    lines.append("  Architecture Comparison")
    lines.append(f"{'='*60}")
    lines.append("""
  VanillaRNN:
    Simple model; suffers from vanishing gradients on long names.
    Common failure: names that start well but deteriorate at the end.

  BidirectionalLSTM:
    LSTM gating alleviates vanishing gradients. Bidirectional encoder
    provides richer gradient signal during training. Generated names
    tend to be more coherent. Decoder is unidirectional at inference.

  AttentionRNN:
    Attention allows the decoder to selectively attend back to any
    position in the generated prefix. Particularly helpful for Indian
    names with repeated syllabic patterns. At low temperature (T=0.5),
    may collapse to a small set of common prefixes.
    """)
    return "\n".join(lines)


if __name__ == "__main__":
    training_names = load_names(NAMES_FILE)
    print(f"Training set: {len(training_names)} names")

    all_data = {}
    for model_name in MODEL_NAMES:
        all_data[model_name] = {}
        for temp in TEMPERATURES:
            names = load_generated(model_name, temp)
            all_data[model_name][temp] = names
            status = f"{len(names)} names" if names else "[missing]"
            print(f"  {model_name}, T={temp}: {status}")

    plot_length_distributions(all_data, training_names)
    plot_char_frequencies(all_data, training_names)

    report = build_report(all_data, training_names)
    print(report)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {REPORT_FILE}")
