import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_word2vec import (
    load_corpus, Vocabulary, train_model, WordVectors, Word2VecResult
)

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROCESSED_FILE = os.path.join(SCRIPT_DIR, "data", "processed_corpus.txt")
MODELS_DIR     = os.path.join(SCRIPT_DIR, "models")
OUT_PATH       = os.path.join(MODELS_DIR, "word2vec_300d.pkl")

TARGET_WORD    = "btech"

def get_300d_model() -> Word2VecResult:
    if os.path.exists(OUT_PATH):
        print(f"[1] Loading cached 300-dim model from {OUT_PATH}")
        return Word2VecResult.load(OUT_PATH)

    print("[1] Training 300-dim Word2Vec on your corpus ...")
    sentences = load_corpus(PROCESSED_FILE)
    vocab     = Vocabulary(sentences, min_count=3)
    print(f"    Vocab size: {vocab.vocab_size:,}")

    result = train_model(
        sentences, vocab,
        sg=0,
        embed_size=300,
        window=5,
        negative=10,
        epochs=20,
        lr=0.025,
    )
    result.save(OUT_PATH)
    print(f"    Saved -> {OUT_PATH}")
    return result


def show_embedding(label: str, word: str, vec: np.ndarray):
    print(f"\n{'='*60}")
    print(f"  Model  : {label}")
    print(f"  Word   : '{word}'")
    print(f"  Shape  : {vec.shape}")
    print(f"  Norm   : {np.linalg.norm(vec):.4f}")
    print(f"  Min    : {vec.min():.4f}   Max: {vec.max():.4f}")
    print(f"  First 20 dims:")
    for i in range(0, 20, 5):
        chunk = vec[i:i+5]
        print(f"    [{i:>3}-{i+4:>3}]  " + "  ".join(f"{v:+.4f}" for v in chunk))
    print(f"{'='*60}")


if __name__ == "__main__":
    result = get_300d_model()
    wv     = result.wv

    print(f"\n  Model dim   : {result.vector_size}")
    print(f"  Vocab size  : {len(wv):,}")

    test_words = ["research", "professor", "student", "thesis", "engineering"]
    for w in test_words:
        if w in wv:
            show_embedding("Custom Word2Vec 300d", w, wv[w])
            top5 = wv.most_similar(w, topn=5)
            print(f"  Top-5 similar: {[(w, round(s,3)) for w,s in top5]}")
            break

    if TARGET_WORD in wv:
        show_embedding("Custom CBOW 300d", TARGET_WORD, wv[TARGET_WORD])
        vec = wv[TARGET_WORD]
        save_path = os.path.join(MODELS_DIR, f"{TARGET_WORD}_cbow_300d.json")
        with open(save_path, "w") as f:
            json.dump({"word": TARGET_WORD, "dim": len(vec), "vector": vec.tolist()}, f)
        print(f"\n  Embedding saved -> {save_path}")
    else:
        print(f"\n  '{TARGET_WORD}' not in vocabulary.")
