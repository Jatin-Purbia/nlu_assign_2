import os
import sys
import argparse
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NAMES_FILE = os.path.join(SCRIPT_DIR, "..", "TrainingNames.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
sys.path.insert(0, SCRIPT_DIR)


def step(title):
    print(f"\n{'#'*64}\n  {title}\n{'#'*64}")


def run_step0_verify():
    step("STEP 0 -- Verify TrainingNames.txt")
    if not os.path.exists(NAMES_FILE):
        print(f"  TrainingNames.txt not found at {NAMES_FILE}")
        raise SystemExit(1)
    with open(NAMES_FILE, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    print(f"  Found {len(names)} names")
    if len(names) < 900:
        print(f"  Warning: expected ~1000 names, found {len(names)}")


def run_step1_train(skip):
    step("STEP 1 -- Model Training (VanillaRNN / BLSTM / AttentionRNN)")
    if skip:
        missing = [
            m for m in ["rnn", "blstm", "attention"]
            if not os.path.exists(os.path.join(OUTPUT_DIR, f"{m}_model.pt"))
        ]
        if missing:
            print(f"  Checkpoints not found for: {missing}. Running training.")
        else:
            print("  Skipping -- checkpoints found.")
            return
    from train import main as train_main
    train_main(["rnn", "blstm", "attention"])


def run_step2_generate(n_generate):
    step("STEP 2 -- Name Generation & Quantitative Evaluation")
    from generate import main as gen_main
    gen_main(n_generate=n_generate)


def run_step3_analyse():
    step("STEP 3 -- Qualitative Analysis")
    import analysis
    from dataset import load_names
    from generate import TEMPERATURES, MODEL_NAMES, load_generated

    training_names = load_names(NAMES_FILE)
    all_data = {}
    for model_name in MODEL_NAMES:
        all_data[model_name] = {}
        for temp in TEMPERATURES:
            all_data[model_name][temp] = load_generated(model_name, temp)

    analysis.plot_length_distributions(all_data, training_names)
    analysis.plot_char_frequencies(all_data, training_names)

    report = analysis.build_report(all_data, training_names)
    print(report)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(analysis.REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved: {analysis.REPORT_FILE}")


def print_summary():
    step("SUMMARY -- Output Files")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel   = os.path.relpath(fpath, OUTPUT_DIR)
            print(f"  {rel:<55}  {os.path.getsize(fpath):>8,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Run Problem 2 end-to-end pipeline.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    steps = [
        ("Verify",   run_step0_verify),
        ("Train",    lambda: run_step1_train(args.skip_train)),
        ("Generate", lambda: run_step2_generate(args.n)),
        ("Analyse",  run_step3_analyse),
    ]

    failed = []
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:
            print(f"\n  Step '{name}' failed: {exc}")
            traceback.print_exc()
            failed.append(name)

    print_summary()

    print(f"\n{'#'*64}")
    if failed:
        print(f"  Problem 2 complete with errors in: {failed}")
    else:
        print("  Problem 2 pipeline completed successfully!")
    print(f"{'#'*64}\n")


if __name__ == "__main__":
    main()
