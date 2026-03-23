import os
import re
import string
from collections import Counter

import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus   import stopwords

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
RAW_CORPUS     = os.path.join(SCRIPT_DIR, "data", "raw_corpus.txt")
PROCESSED_FILE = os.path.join(SCRIPT_DIR, "data", "processed_corpus.txt")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "outputs")
WORDCLOUD_FILE = os.path.join(OUTPUT_DIR, "wordcloud.png")

STOP_WORDS = set(stopwords.words("english"))
DOMAIN_KEEP = {
    "research", "student", "students", "phd", "faculty", "department",
    "lab", "project", "course", "exam", "thesis", "degree", "institute",
    "technology", "professor", "lecture", "seminar", "publication",
    "btech", "mtech", "msc", "bsc", "ug", "pg",
}
STOP_WORDS -= DOMAIN_KEEP

ABBREV_MAP = {
    r"\bph\.d\.?\b":   "phd",
    r"\bm\.tech\.?\b": "mtech",
    r"\bb\.tech\.?\b": "btech",
    r"\bm\.sc\.?\b":   "msc",
    r"\bb\.sc\.?\b":   "bsc",
    r"\bm\.e\.?\b":    "me",
    r"\bb\.e\.?\b":    "be",
    r"\bi\.i\.t\.?\b": "iit",
    r"\bu\.g\.?\b":    "ug",
    r"\bp\.g\.?\b":    "pg",
    r"\bprof\.?\b":    "professor",
    r"\bdr\.?\b":      "doctor",
    r"\bst\.?\b":      "student",
    r"\bdept\.?\b":    "department",
    r"\bno\.?\b":      "number",
    r"\bvs\.?\b":      "versus",
}


def _looks_english(text: str, threshold: float = 0.20) -> bool:
    if not text:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return (non_ascii / len(text)) < threshold


def _normalise_abbreviations(text: str) -> str:
    text = text.lower()
    for pattern, replacement in ABBREV_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _remove_boilerplate(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"\b\d{10,}\b", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"[^\w\s]{4,}", " ", text)
    return text


def preprocess_sentence(raw: str) -> list[str]:
    if not _looks_english(raw):
        return []
    text = _remove_boilerplate(raw)
    text = _normalise_abbreviations(text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = word_tokenize(text)
    clean_tokens = []
    for tok in tokens:
        tok = tok.strip()
        if len(tok) < 2:
            continue
        if not re.match(r"^[a-z][a-z0-9\-]*$", tok):
            continue
        if tok in STOP_WORDS:
            continue
        clean_tokens.append(tok)
    return clean_tokens


def build_processed_corpus(raw_path: str) -> list[list[str]]:
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw corpus not found at {raw_path}.\nRun scraper.py first."
        )
    processed_sentences = []
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    print(f"Raw lines read: {len(raw_lines)}")
    for raw_line in raw_lines:
        for sent in sent_tokenize(raw_line):
            tokens = preprocess_sentence(sent)
            if len(tokens) >= 4:
                processed_sentences.append(tokens)
    return processed_sentences


def save_processed_corpus(sentences: list[list[str]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for tokens in sentences:
            f.write(" ".join(tokens) + "\n")
    print(f"Processed corpus saved: {path}")


def report_statistics(sentences: list[list[str]]):
    total_tokens = sum(len(s) for s in sentences)
    vocab_size   = len(set(tok for s in sentences for tok in s))
    print("\n-- Dataset Statistics --")
    print(f"  Total sentences : {len(sentences):>8,}")
    print(f"  Total tokens    : {total_tokens:>8,}")
    print(f"  Vocabulary size : {vocab_size:>8,}")
    return total_tokens, vocab_size


def generate_wordcloud(sentences: list[list[str]], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    freq = Counter(tok for s in sentences for tok in s)
    wc = WordCloud(
        width=1200, height=600, background_color="white",
        max_words=200, colormap="viridis", prefer_horizontal=0.9,
    ).generate_from_frequencies(freq)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Words in IIT Jodhpur Corpus", fontsize=16, pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Word cloud saved: {save_path}")


if __name__ == "__main__":
    sentences = build_processed_corpus(RAW_CORPUS)
    save_processed_corpus(sentences, PROCESSED_FILE)
    report_statistics(sentences)
    generate_wordcloud(sentences, WORDCLOUD_FILE)
