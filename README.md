# NLU Assignment 2

Two problems, each in its own folder with a `run_all.py` to run everything end-to-end.

---

## Folder structure

```
assign-2/
├── TrainingNames.txt          # ~1000 Indian names (needed by Problem 2)
├── problem1/                  # Word2Vec on IIT Jodhpur corpus
└── problem2/                  # Character-level Indian name generation
```

---

## Problem 1 — Word2Vec on IIT Jodhpur Corpus

Scrapes the IIT Jodhpur website, preprocesses the text, trains CBOW and Skip-gram Word2Vec models from scratch (PyTorch), then runs semantic analysis and visualisation.

### Setup

```bash
pip install requests beautifulsoup4 lxml nltk gensim wordcloud matplotlib scikit-learn torch
```

NLTK data is downloaded automatically on first run.

### Running

```bash
cd problem1

# Full pipeline (scrape → preprocess → train → analyse → visualise)
python run_all.py

# If you already have data/raw_corpus.txt from a previous run
python run_all.py --skip-scrape
```

Individual steps can also be run separately:

```bash
python scraper.py              # crawl iitj.ac.in, saves data/raw_corpus.txt
python preprocessor.py         # clean + tokenise, saves data/processed_corpus.txt
python train_word2vec.py       # grid search + train best CBOW & Skip-gram
python semantic_analysis.py    # nearest neighbours + analogy experiments
python visualize.py            # PCA and t-SNE plots
```

> Scraping ~500 pages takes a few minutes. Use `--skip-scrape` on subsequent runs.

### Output

```
problem1/
├── data/
│   ├── raw_corpus.txt
│   └── processed_corpus.txt
├── models/
│   ├── best_cbow.pkl
│   └── best_skipgram.pkl
└── outputs/
    ├── wordcloud.png
    ├── semantic_results.txt
    └── plots/
        ├── pca_cbow.png  /  tsne_cbow.png
        ├── pca_skip-gram.png  /  tsne_skip-gram.png
        └── comparison_pca.png  /  comparison_tsne.png
```

---

## Problem 2 — Character-Level Indian Name Generation

Trains three sequence models (VanillaRNN, BidirectionalLSTM, AttentionRNN) from scratch on a dataset of Indian names, then generates new names and analyses the results.

### Setup

```bash
pip install torch matplotlib numpy
```

GPU is used automatically if available, otherwise falls back to CPU.

### Running

Make sure `TrainingNames.txt` is in the `assign-2/` root before running.

```bash
cd problem2

# Full pipeline (train → generate → analyse)
python run_all.py

# Skip training if checkpoints already exist
python run_all.py --skip-train

# Generate fewer names (useful for quick testing)
python run_all.py --n 100
```

Individual steps:

```bash
python train.py                    # train all three models (runs hyperparameter search)
python train.py --model rnn        # train one model only
python train.py --no-search        # skip grid search, use default hyperparameters
python generate.py --n 200         # generate names (needs trained checkpoints)
python analysis.py                 # qualitative report + plots
```

> Training all three models with the full hyperparameter search takes a while on CPU (~30–60 min depending on hardware). Use `--skip-train` on subsequent runs, or `--no-search` to skip the grid search.

### Output

```
problem2/outputs/
├── rnn_model.pt / blstm_model.pt / attention_model.pt
├── rnn_loss_curve.png / blstm_loss_curve.png / attention_loss_curve.png
├── rnn_summary.json / blstm_summary.json / attention_summary.json
├── generated_rnn_t05.txt ... generated_attention_t15.txt  (15 files)
├── metrics.json
├── qualitative_report.txt
└── plots/
    ├── length_distributions.png
    └── char_frequencies.png
```

---

## Quick start (both problems)

```bash
pip install requests beautifulsoup4 lxml nltk gensim wordcloud matplotlib scikit-learn torch numpy

# Problem 1 (skip scrape if raw_corpus.txt already exists)
cd problem1 && python run_all.py --skip-scrape

# Problem 2
cd ../problem2 && python run_all.py
```
