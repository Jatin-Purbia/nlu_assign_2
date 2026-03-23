import os
from train_word2vec import Word2VecResult, WordVectors

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR     = os.path.join(SCRIPT_DIR, "models")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "outputs")
BEST_CBOW_PATH = os.path.join(MODELS_DIR, "best_cbow.pkl")
BEST_SG_PATH   = os.path.join(MODELS_DIR, "best_skipgram.pkl")
RESULTS_FILE   = os.path.join(OUTPUT_DIR, "semantic_results.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROBE_WORDS = ["research", "student", "phd", "exam"]

ANALOGIES = [
    ("ug",        "btech",         "pg",     "UG : BTech :: PG : ?"),
    ("professor", "research",      "student","professor : research :: student : ?"),
    ("lab",       "experiment",    "class",  "lab : experiment :: class : ?"),
    ("iit",       "jodhpur",       "nit",    "IIT : Jodhpur :: NIT : ?"),
    ("btech",     "undergraduate", "mtech",  "BTech : undergraduate :: MTech : ?"),
]


def safe_most_similar(wv, word, topn=5):
    if word not in wv:
        return []
    return wv.most_similar(word, topn=topn)


def safe_analogy(wv, word_a, word_b, word_c, topn=5):
    if any(w not in wv for w in (word_a, word_b, word_c)):
        return []
    return wv.most_similar(positive=[word_b, word_c], negative=[word_a], topn=topn)


def format_nn_table(result, model_name):
    lines = [
        f"\n{'='*58}",
        f"  Nearest Neighbours -- {model_name}",
        f"  (vector_size={result.vector_size}, window={result.window})",
        f"{'-'*58}",
    ]
    for word in PROBE_WORDS:
        similar = safe_most_similar(result.wv, word, topn=5)
        if not similar:
            lines.append(f"  {word!r:<12} -> [OUT OF VOCABULARY]")
        else:
            lines.append(f"  {word!r:<12} -> top-5 similar:")
            for rank, (nb, cos) in enumerate(similar, 1):
                lines.append(f"    {rank}. {nb:<18} (cosine = {cos:.4f})")
    lines.append(f"{'='*58}")
    return "\n".join(lines)


def format_analogy_table(result, model_name):
    lines = [
        f"\n{'='*68}",
        f"  Analogy Experiments -- {model_name}",
        f"{'-'*68}",
    ]
    for word_a, word_b, word_c, description in ANALOGIES:
        lines.append(f"\n  Query : {description}")
        lines.append(f"  Formula: most_similar(+[{word_b!r},{word_c!r}], -[{word_a!r}])")
        oov = [w for w in (word_a, word_b, word_c) if w not in result.wv]
        if oov:
            lines.append(f"  OOV: {oov}")
            continue
        res = safe_analogy(result.wv, word_a, word_b, word_c, topn=5)
        if not res:
            lines.append("  No results.")
            continue
        lines.append("  Top answers:")
        for rank, (ans, sc) in enumerate(res, 1):
            lines.append(f"    {rank}. {ans:<20} (cosine = {sc:.4f})")
    lines.append(f"\n{'='*68}")
    return "\n".join(lines)


def semantic_discussion(results_summary):
    return "\n".join([
        "",
        "== Semantic Meaningfulness Discussion ===================================",
        "",
        "  1. UG : BTech :: PG : ?",
        "     Expected answer: 'mtech'.",
        "",
        "  2. professor : research :: student : ?",
        "     Expected: 'project' or 'thesis'.",
        "",
        "  3. lab : experiment :: class : ?",
        "     Expected: 'lecture' or 'course'.",
        "",
        "  4. IIT : Jodhpur :: NIT : ?",
        "     Expected: a city name.",
        "",
        "  General observation:",
        "     Skip-gram models tend to produce better analogy results on small",
        "     corpora because Skip-gram predicts context from centre words,",
        "     producing richer individual word representations.",
        "=========================================================================",
    ])


def run_analysis(result, model_name):
    return format_nn_table(result, model_name) + format_analogy_table(result, model_name)


if __name__ == "__main__":
    if not os.path.exists(BEST_CBOW_PATH) or not os.path.exists(BEST_SG_PATH):
        print("Model files not found. Run train_word2vec.py first.")
        raise SystemExit(1)

    cbow_result = Word2VecResult.load(BEST_CBOW_PATH)
    sg_result   = Word2VecResult.load(BEST_SG_PATH)

    print(f"Loaded CBOW      (vocab: {len(cbow_result.wv)}, dim: {cbow_result.vector_size})")
    print(f"Loaded Skip-gram (vocab: {len(sg_result.wv)}, dim: {sg_result.vector_size})")

    full_report  = "\nSEMANTIC ANALYSIS REPORT -- IIT Jodhpur Word2Vec\n"
    full_report += run_analysis(cbow_result, "CBOW (from scratch)")
    full_report += run_analysis(sg_result,   "Skip-gram with Negative Sampling (from scratch)")
    full_report += semantic_discussion({})

    print(full_report)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"\nResults saved to: {RESULTS_FILE}")
