# NLU Assignment 2

Two independent NLP/DL problems — each has its own folder with a single `run_all.py` entry-point.

---

## Repository Layout

```
assign-2/
├── TrainingNames.txt          # ~1000 Indian names (required by Problem 2)
├── problem1/                  # Word2Vec on IIT Jodhpur corpus
│   ├── scraper.py
│   ├── preprocessor.py
│   ├── train_word2vec.py
│   ├── semantic_analysis.py
│   ├── visualize.py
│   └── run_all.py             ← single entry-point
└── problem2/                  # Character-level Indian name generation
    ├── dataset.py
    ├── models.py
    ├── train.py
    ├── generate.py
    ├── analysis.py
    └── run_all.py             ← single entry-point
```

---

## Problem 1 — Word2Vec on IIT Jodhpur Corpus

### What it does

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `scraper.py` | BFS-crawls `iitj.ac.in` (up to 200 pages, depth 2) and saves raw text to `data/raw_corpus.txt` |
| 2 | `preprocessor.py` | Cleans text, tokenises with NLTK, removes stopwords/boilerplate, saves `data/processed_corpus.txt` + word-cloud PNG |
| 3 | `train_word2vec.py` | Grid-searches CBOW & Skip-gram (dim ∈ {50,100,200}, window ∈ {3,5,7}, neg ∈ {5,10}, epochs=20) using gensim; saves the best models to `models/` |
| 4 | `semantic_analysis.py` | Reports top-5 nearest neighbours for *research / student / phd / exam* and runs 5 word-analogy experiments (e.g. `UG:BTech::PG:?`) |
| 5 | `visualize.py` | PCA and t-SNE plots for colour-coded word groups; side-by-side CBOW vs. Skip-gram comparison |

### Output files produced

```
problem1/
├── data/
│   ├── raw_corpus.txt
│   └── processed_corpus.txt
├── models/
│   ├── best_cbow.model
│   └── best_skipgram.model
└── outputs/
    ├── wordcloud.png
    ├── semantic_results.txt
    └── plots/
        ├── pca_cbow.png
        ├── tsne_cbow.png
        ├── pca_skip-gram.png
        ├── tsne_skip-gram.png
        ├── comparison_pca.png
        └── comparison_tsne.png
```

### Dependencies

```
pip install requests beautifulsoup4 lxml nltk gensim wordcloud matplotlib scikit-learn
```

NLTK data (downloaded automatically on first run): `punkt`, `punkt_tab`, `stopwords`.

### How to run

```bash
cd problem1

# Full pipeline (scrape + preprocess + train + analyse + visualise)
python run_all.py

# Skip scraping if raw_corpus.txt already exists
python run_all.py --skip-scrape
```

Or run individual steps:

```bash
python scraper.py              # Step 1 only
python preprocessor.py         # Step 2 only (needs raw_corpus.txt)
python train_word2vec.py       # Step 3 only (needs processed_corpus.txt)
python semantic_analysis.py    # Step 4 only (needs trained models)
python visualize.py            # Step 5 only (needs trained models)
```

> **Note:** Scraping ~200 pages takes a few minutes (1 s polite delay per request). If you already have `data/raw_corpus.txt` from a previous run, use `--skip-scrape`.

---

## Problem 2 — Character-Level Indian Name Generation

### What it does

Three sequence models are implemented **from scratch** using only PyTorch primitives (no `nn.RNN`, `nn.LSTM`, etc.).

| Step | Script | Purpose |
|------|--------|---------|
| 0 | — | Verifies `TrainingNames.txt` exists and has ~1000 names |
| 1 | `train.py` | Trains **VanillaRNN**, **BidirectionalLSTM**, **AttentionRNN** for 200 epochs each; saves checkpoints + loss curves |
| 2 | `generate.py` | Generates 500 names per model × 5 temperatures (0.5, 0.8, 1.0, 1.2, 1.5); computes **Novelty Rate** and **Diversity** |
| 3 | `analysis.py` | Qualitative analysis — realism rate, failure modes, length distributions, character frequency plots, architecture comparison report |

### Models

| Model | Architecture |
|-------|-------------|
| `VanillaRNN` | Elman RNN cell → embedding → linear output |
| `BidirectionalLSTM` | Bi-LSTM encoder (fwd + bwd) → bridge → uni-directional LSTM decoder |
| `AttentionRNN` | RNN encoder → Bahdanau additive attention → RNN decoder |

### Output files produced

```
problem2/outputs/
├── rnn_model.pt
├── blstm_model.pt
├── attention_model.pt
├── rnn_loss_curve.png
├── blstm_loss_curve.png
├── attention_loss_curve.png
├── rnn_summary.json
├── blstm_summary.json
├── attention_summary.json
├── generated_rnn_t05.txt          # one file per model × temperature
├── generated_rnn_t08.txt
├── ...  (15 files total)
├── metrics.json
├── qualitative_report.txt
└── plots/
    ├── length_distributions.png
    └── char_frequencies.png
```

### Dependencies

```
pip install torch matplotlib numpy
```

GPU is used automatically if CUDA is available; falls back to CPU.

### How to run

```bash
cd problem2

# Full pipeline (train + generate + analyse)
python run_all.py

# Skip training if checkpoints already exist
python run_all.py --skip-train

# Generate fewer names (faster for testing)
python run_all.py --n 100
```

Or run individual steps:

```bash
python train.py                    # train all three models
python train.py --model rnn        # train one model only
python generate.py --n 200         # generate names (needs checkpoints)
python analysis.py                 # qualitative report (needs generated files)
```

> **Note:** Training 3 models × 200 epochs on CPU takes ~5–15 minutes depending on hardware. Use `--skip-train` on subsequent runs.

---

## Quick-start (both problems)

```bash
# Install all dependencies
pip install requests beautifulsoup4 lxml nltk gensim wordcloud matplotlib scikit-learn torch numpy

# Problem 1
cd problem1 && python run_all.py --skip-scrape   # omit --skip-scrape on first run

# Problem 2
cd ../problem2 && python run_all.py
```

---

## Hyperparameter Summary

### Problem 1 — Word2Vec grid search

| Parameter | Values |
|-----------|--------|
| Architecture | CBOW, Skip-gram |
| `vector_size` | 50, 100, 200 |
| `window` | 3, 5, 7 |
| `negative` | 5, 10 |
| `epochs` | 20 |

Best model selected by average top-5 neighbour hit count for probe words.

### Problem 2 — Name generation

| Parameter | Value |
|-----------|-------|
| `embed_size` | 32 |
| `hidden_size` | 128 |
| `attn_size` | 64 (AttentionRNN only) |
| `learning_rate` | 0.001 (Adam) |
| `batch_size` | 32 |
| `num_epochs` | 200 |
| `grad_clip` | 5.0 |
| Temperatures | 0.5, 0.8, 1.0, 1.2, 1.5 |
