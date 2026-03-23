import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from sklearn.manifold       import TSNE
from train_word2vec         import Word2VecResult, WordVectors

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR     = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_CBOW_PATH = os.path.join(MODELS_DIR, "best_cbow.pkl")
BEST_SG_PATH   = os.path.join(MODELS_DIR, "best_skipgram.pkl")

WORD_GROUPS = {
    "Academic Roles"   : ["professor", "student", "faculty", "researcher", "phd",
                          "doctor", "lecturer", "dean", "director", "advisor"],
    "Programmes"       : ["btech", "mtech", "msc", "bsc", "ug", "pg",
                          "undergraduate", "postgraduate", "degree", "diploma"],
    "Academic Activity": ["research", "exam", "thesis", "project", "lecture",
                          "course", "seminar", "publication", "paper", "study"],
    "Spaces"           : ["lab", "library", "classroom", "hostel", "campus",
                          "department", "building", "hall", "centre", "office"],
    "Administration"   : ["institute", "committee", "board", "regulation",
                         "admission", "academic", "programme", "scholarship"],
}

GROUP_COLORS = {
    "Academic Roles"   : "#e63946",
    "Programmes"       : "#2a9d8f",
    "Academic Activity": "#e9c46a",
    "Spaces"           : "#457b9d",
    "Administration"   : "#a8dadc",
}


def collect_embeddings(result):
    vectors, words, colors = [], [], []
    for group_name, word_list in WORD_GROUPS.items():
        color = GROUP_COLORS[group_name]
        for word in word_list:
            if word in result.wv:
                vectors.append(result.wv[word])
                words.append(word)
                colors.append(color)
    if not vectors:
        raise ValueError("None of the probe words found in model vocabulary.")
    return np.array(vectors), words, colors


def reduce_pca(vectors):
    return PCA(n_components=2, random_state=42).fit_transform(vectors)


def reduce_tsne(vectors, perplexity=None):
    n = len(vectors)
    p = perplexity if perplexity is not None else min(30, max(5, n // 3))
    return TSNE(n_components=2, perplexity=p, random_state=42, max_iter=1000).fit_transform(vectors)


def plot_embeddings(coords, words, colors, title, save_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=60, alpha=0.8,
               edgecolors="white", linewidths=0.5)
    for i, word in enumerate(words):
        ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=8,
                    ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
              title="Word Groups", title_fontsize=10, framealpha=0.8)
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Component 1", fontsize=10)
    ax.set_ylabel("Component 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_model(result, model_label):
    print(f"\n-- Visualising: {model_label} --")
    vectors, words, colors = collect_embeddings(result)
    total = sum(len(v) for v in WORD_GROUPS.values())
    print(f"  Words found: {len(words)} / {total}")

    pca_coords = reduce_pca(vectors)
    plot_embeddings(pca_coords, words, colors,
                    f"{model_label} -- PCA (2D)",
                    os.path.join(OUTPUT_DIR, f"pca_{model_label.lower().replace(' ', '_')}.png"))

    tsne_coords = reduce_tsne(vectors)
    plot_embeddings(tsne_coords, words, colors,
                    f"{model_label} -- t-SNE (2D)",
                    os.path.join(OUTPUT_DIR, f"tsne_{model_label.lower().replace(' ', '_')}.png"))


def plot_side_by_side_comparison(cbow_result, sg_result, method="pca"):
    vecs_c, words_c, cols_c = collect_embeddings(cbow_result)
    vecs_s, words_s, cols_s = collect_embeddings(sg_result)

    reduce_fn = reduce_pca if method == "pca" else reduce_tsne
    coords_c  = reduce_fn(vecs_c)
    coords_s  = reduce_fn(vecs_s)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for ax, coords, words, colors, label in [
        (axes[0], coords_c, words_c, cols_c, "CBOW"),
        (axes[1], coords_s, words_s, cols_s, "Skip-gram"),
    ]:
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=50, alpha=0.8,
                   edgecolors="white", linewidths=0.4)
        for i, w in enumerate(words):
            ax.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=7,
                        xytext=(2, 2), textcoords="offset points")
        ax.set_title(f"{label} -- {method.upper()}", fontsize=12)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, linestyle="--", alpha=0.3)

    legend_patches = [mpatches.Patch(color=c, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"CBOW vs. Skip-gram -- {method.upper()}", fontsize=14, y=1.01)
    fig.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"comparison_{method}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved: {save_path}")


if __name__ == "__main__":
    if not os.path.exists(BEST_CBOW_PATH) or not os.path.exists(BEST_SG_PATH):
        print("Model files not found. Run train_word2vec.py first.")
        raise SystemExit(1)

    cbow_result = Word2VecResult.load(BEST_CBOW_PATH)
    sg_result   = Word2VecResult.load(BEST_SG_PATH)

    print(f"Loaded CBOW      (vocab: {len(cbow_result.wv)}, dim: {cbow_result.vector_size})")
    print(f"Loaded Skip-gram (vocab: {len(sg_result.wv)}, dim: {sg_result.vector_size})")

    visualize_model(cbow_result, "CBOW")
    visualize_model(sg_result,   "Skip-gram")
    plot_side_by_side_comparison(cbow_result, sg_result, method="pca")
    plot_side_by_side_comparison(cbow_result, sg_result, method="tsne")

    print("\nAll visualisations saved to:", OUTPUT_DIR)
