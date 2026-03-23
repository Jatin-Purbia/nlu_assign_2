import os
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from collections import Counter
from itertools import product

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROCESSED_FILE = os.path.join(SCRIPT_DIR, "data", "processed_corpus.txt")
MODELS_DIR     = os.path.join(SCRIPT_DIR, "models")
BEST_CBOW_PATH = os.path.join(MODELS_DIR, "best_cbow.pkl")
BEST_SG_PATH   = os.path.join(MODELS_DIR, "best_skipgram.pkl")
GLOVE_PATH     = os.path.join(SCRIPT_DIR, "data", "glove.6B.50d.txt")

PROBE_WORDS = [
    "research", "student", "phd", "exam", "department", "faculty",
    "professor", "thesis", "lecture", "course", "graduate", "engineering",
]

SEMANTIC_PAIRS = [
    ("research",    "phd"),
    ("student",     "exam"),
    ("faculty",     "department"),
    ("professor",   "lecture"),
    ("thesis",      "research"),
    ("course",      "lecture"),
    ("graduate",    "phd"),
    ("engineering", "technology"),
    ("student",     "graduate"),
    ("faculty",     "professor"),
]

# (a, b, c) -> expected: word ~ b - a + c
ANALOGY_TRIPLES = [
    ("student",     "exam",       "faculty"),
    ("phd",         "research",   "mtech"),
    ("professor",   "department", "student"),
    ("lecture",     "course",     "research"),
    ("engineering", "department", "science"),
]


class Vocabulary:
    def __init__(self, sentences: list, min_count: int = 3):
        counts = Counter(w for sent in sentences for w in sent)
        counts = {w: c for w, c in counts.items() if c >= min_count}

        vocab = sorted(counts.keys())
        self.word2idx: dict = {w: i for i, w in enumerate(vocab)}
        self.idx2word: dict = {i: w for i, w in enumerate(vocab)}
        self.vocab_size: int = len(vocab)

        total = sum(counts.values())
        t = 1e-5
        self.keep_prob: dict = {}
        for w in vocab:
            f = counts[w] / total
            ratio = t / f
            self.keep_prob[w] = min(1.0, (ratio ** 0.5) + ratio)

        freqs = np.array([counts[self.idx2word[i]] for i in range(self.vocab_size)],
                         dtype=np.float32)
        noise = freqs ** 0.75
        self.noise_weights = torch.tensor(noise / noise.sum(), dtype=torch.float32)

    def subsample(self, sentence: list) -> list:
        out = []
        for w in sentence:
            if w not in self.word2idx:
                continue
            if np.random.random() < self.keep_prob[w]:
                out.append(self.word2idx[w])
        return out

    def encode(self, sentence: list) -> list:
        return [self.word2idx[w] for w in sentence if w in self.word2idx]


class _W2VModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.in_embed  = nn.Embedding(vocab_size, embed_size, sparse=True)
        self.out_embed = nn.Embedding(vocab_size, embed_size, sparse=True)
        nn.init.uniform_(self.in_embed.weight, -0.5 / embed_size, 0.5 / embed_size)
        nn.init.zeros_(self.out_embed.weight)

    def forward_skipgram(self, center, context, negatives):
        v_c   = self.in_embed(center)
        v_ctx = self.out_embed(context)
        v_neg = self.out_embed(negatives)
        pos_loss = nn.functional.logsigmoid((v_c * v_ctx).sum(dim=1))
        neg_loss = nn.functional.logsigmoid(
            -torch.bmm(v_neg, v_c.unsqueeze(2)).squeeze(2)
        ).sum(dim=1)
        return -(pos_loss + neg_loss).mean()

    def forward_cbow(self, context, target, negatives, mask):
        v_ctx = self.in_embed(context)
        mf    = mask.unsqueeze(2).float()
        v_avg = (v_ctx * mf).sum(1) / mf.sum(1).clamp(min=1)
        v_tgt = self.out_embed(target)
        v_neg = self.out_embed(negatives)
        pos_loss = nn.functional.logsigmoid((v_avg * v_tgt).sum(1))
        neg_loss = nn.functional.logsigmoid(
            -torch.bmm(v_neg, v_avg.unsqueeze(2)).squeeze(2)
        ).sum(1)
        return -(pos_loss + neg_loss).mean()


class _SkipgramDataset(Dataset):
    def __init__(self, encoded: list, window: int, neg_k: int,
                 noise_weights: torch.Tensor):
        total_tokens = sum(len(s) for s in encoded)
        max_pairs    = total_tokens * 2 * window
        centers  = np.empty(max_pairs, dtype=np.int32)
        contexts = np.empty(max_pairs, dtype=np.int32)
        idx = 0
        for sent in encoded:
            arr = np.array(sent, dtype=np.int32)
            n   = len(arr)
            for i in range(n):
                lo  = max(0, i - window)
                hi  = min(n, i + window + 1)
                ctx = np.concatenate([arr[lo:i], arr[i + 1:hi]])
                k   = len(ctx)
                centers[idx:idx + k]  = arr[i]
                contexts[idx:idx + k] = ctx
                idx += k
        self._centers  = centers[:idx]
        self._contexts = contexts[:idx]
        self._neg_k    = neg_k
        self._noise    = noise_weights

    def __len__(self):
        return len(self._centers)

    def __getitem__(self, idx):
        return (torch.tensor(int(self._centers[idx]),  dtype=torch.long),
                torch.tensor(int(self._contexts[idx]), dtype=torch.long))


class _CBOWDataset(Dataset):
    def __init__(self, encoded: list, window: int, neg_k: int,
                 noise_weights: torch.Tensor):
        max_ctx  = window * 2
        total_tokens = sum(len(s) for s in encoded)
        ctx_arr  = np.zeros((total_tokens, max_ctx), dtype=np.int32)
        mask_arr = np.zeros((total_tokens, max_ctx), dtype=bool)
        tgt_arr  = np.empty(total_tokens, dtype=np.int32)
        idx = 0
        for sent in encoded:
            arr = np.array(sent, dtype=np.int32)
            n   = len(arr)
            for i in range(n):
                lo  = max(0, i - window)
                hi  = min(n, i + window + 1)
                ctx = np.concatenate([arr[lo:i], arr[i + 1:hi]])
                k   = min(len(ctx), max_ctx)
                ctx_arr[idx, :k]  = ctx[:k]
                mask_arr[idx, :k] = True
                tgt_arr[idx]      = arr[i]
                idx += 1
        self._ctx  = ctx_arr[:idx]
        self._mask = mask_arr[:idx]
        self._tgt  = tgt_arr[:idx]
        self._neg_k = neg_k
        self._noise = noise_weights

    def __len__(self):
        return len(self._tgt)

    def __getitem__(self, idx):
        return (torch.from_numpy(self._ctx[idx].copy()).long(),
                torch.tensor(int(self._tgt[idx]), dtype=torch.long),
                torch.from_numpy(self._mask[idx].copy()))


class WordVectors:
    def __init__(self, word2idx: dict, idx2word: dict, vectors: np.ndarray):
        self.word2idx  = word2idx
        self.idx2word  = idx2word
        self._vectors  = vectors.astype(np.float32)
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        self._norm_vecs = self._vectors / norms

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __getitem__(self, word: str) -> np.ndarray:
        return self._vectors[self.word2idx[word]]

    def __len__(self) -> int:
        return len(self.word2idx)

    def most_similar(self, word=None, positive=None, negative=None, topn: int = 5) -> list:
        if word is not None:
            positive = [word]
        positive = list(positive or [])
        negative = list(negative or [])

        query   = np.zeros(self._vectors.shape[1], dtype=np.float32)
        exclude = set()

        for w in positive:
            if w not in self.word2idx:
                raise KeyError(f"'{w}' not in vocabulary")
            query += self._norm_vecs[self.word2idx[w]]
            exclude.add(self.word2idx[w])

        for w in negative:
            if w not in self.word2idx:
                raise KeyError(f"'{w}' not in vocabulary")
            query -= self._norm_vecs[self.word2idx[w]]
            exclude.add(self.word2idx[w])

        norm = np.linalg.norm(query)
        if norm > 1e-12:
            query /= norm

        sims = self._norm_vecs @ query
        for idx in exclude:
            sims[idx] = -np.inf

        top_idx = np.argpartition(sims, -topn)[-topn:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
        return [(self.idx2word[i], float(sims[i])) for i in top_idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"word2idx": self.word2idx, "idx2word": self.idx2word,
                 "vectors":  self._vectors},
                f, protocol=4
            )

    @staticmethod
    def load(path: str) -> "WordVectors":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return WordVectors(d["word2idx"], d["idx2word"], d["vectors"])


class Word2VecResult:
    def __init__(self, wv: WordVectors, vector_size: int, window: int, sg: int):
        self.wv          = wv
        self.vector_size = vector_size
        self.window      = window
        self.sg          = sg

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)

    @staticmethod
    def load(path: str) -> "Word2VecResult":
        with open(path, "rb") as f:
            return pickle.load(f)


def train_model(
    sentences:  list,
    vocab:      Vocabulary,
    sg:         int,
    embed_size: int,
    window:     int,
    negative:   int,
    epochs:     int   = 5,
    batch_size: int   = 2048,
    lr:         float = 0.025,
) -> Word2VecResult:
    arch_name = "Skip-gram" if sg else "CBOW"
    print(f"\n  Training {arch_name:>10} | dim={embed_size:>3} | "
          f"window={window} | neg={negative} | epochs={epochs}")
    t0 = time.time()

    encoded = [vocab.subsample(s) for s in sentences]
    encoded = [s for s in encoded if len(s) >= 2]

    print(f"    Building dataset ...", end=" ", flush=True)
    if sg:
        dataset    = _SkipgramDataset(encoded, window, negative, vocab.noise_weights)
        n_pairs    = len(dataset)
        centers_t  = torch.from_numpy(dataset._centers).long()
        contexts_t = torch.from_numpy(dataset._contexts).long()
    else:
        dataset = _CBOWDataset(encoded, window, negative, vocab.noise_weights)
        n_pairs = len(dataset)
        ctx_t   = torch.from_numpy(dataset._ctx).long()
        tgt_t   = torch.from_numpy(dataset._tgt).long()
        mask_t  = torch.from_numpy(dataset._mask)
    print(f"{n_pairs:,} samples")

    model     = _W2VModel(vocab.vocab_size, embed_size)
    opt       = optim.SparseAdam(list(model.parameters()), lr=lr)
    noise     = vocab.noise_weights
    n_batches = (n_pairs + batch_size - 1) // batch_size

    for epoch in range(1, epochs + 1):
        frac   = (epoch - 1) / max(epochs, 1)
        cur_lr = lr * (1.0 - frac * (1.0 - 1e-4))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        perm       = torch.randperm(n_pairs)
        total_loss = 0.0

        for b in range(n_batches):
            idx  = perm[b * batch_size: (b + 1) * batch_size]
            B    = len(idx)
            negs = torch.multinomial(noise, B * negative, replacement=True).view(B, negative)

            opt.zero_grad()
            if sg:
                loss = model.forward_skipgram(centers_t[idx], contexts_t[idx], negs)
            else:
                loss = model.forward_cbow(ctx_t[idx], tgt_t[idx], negs, mask_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg = total_loss / n_batches
        pct = epoch / epochs * 100
        print(f"    Epoch {epoch:>2}/{epochs}  loss={avg:.4f}  {pct:.0f}%")

    print(f"  Done in {time.time() - t0:.1f}s")

    vectors = model.in_embed.weight.detach().cpu().numpy()
    wv = WordVectors(vocab.word2idx, vocab.idx2word, vectors)
    return Word2VecResult(wv, embed_size, window, sg)


def evaluate_model(result: Word2VecResult, probe_words: list) -> float:
    """
    Three-component analogy-quality score:
      0.3 * neighbour coherence  +  0.5 * semantic pair similarity  +  0.2 * 3CosAdd hit-rate
    """
    wv = result.wv

    # 1. Neighbour coherence
    nbr_total, nbr_count = 0.0, 0
    for word in probe_words:
        if word in wv:
            for _, sim in wv.most_similar(word, topn=5):
                nbr_total += max(sim, 0.0)
                nbr_count += 1
    nbr_score = nbr_total / max(nbr_count, 1)

    # 2. Semantic pair similarity
    pair_total, pair_count = 0.0, 0
    for w1, w2 in SEMANTIC_PAIRS:
        if w1 in wv and w2 in wv:
            v1 = wv._norm_vecs[wv.word2idx[w1]]
            v2 = wv._norm_vecs[wv.word2idx[w2]]
            pair_total += float(np.dot(v1, v2))
            pair_count += 1
    pair_score = pair_total / max(pair_count, 1)

    # 3. 3CosAdd analogy hit-rate
    hits, triples = 0, 0
    for a, b, c in ANALOGY_TRIPLES:
        if a in wv and b in wv and c in wv:
            triples += 1
            try:
                top = wv.most_similar(positive=[b, c], negative=[a], topn=10)
                if any(len(w) > 2 for w, _ in top):
                    hits += 1
            except KeyError:
                pass
    analogy_score = hits / max(triples, 1)

    return 0.3 * nbr_score + 0.5 * pair_score + 0.2 * analogy_score


def run_experiments(sentences: list) -> dict:
    print("\n  Building vocabulary ...")
    vocab = Vocabulary(sentences, min_count=3)
    print(f"  Vocabulary: {vocab.vocab_size:,} words (min_count=3)")

    vector_sizes  = [200, 300]
    windows       = [5, 7, 10]
    negatives     = [10, 15]
    lrs           = [0.025, 0.01]
    search_epochs = 10
    best_epochs   = 30

    results = []
    best_cbow_score, best_cbow_cfg = -1, None
    best_sg_score,   best_sg_cfg   = -1, None

    total = len(vector_sizes) * len(windows) * len(negatives) * len(lrs) * 2
    done  = 0

    print(f"\n  {'Model':>10} | {'dim':>4} | {'win':>4} | {'neg':>4} | {'lr':>6} | {'score':>7} | progress")
    print("  " + "-" * 68)

    for vsize, win, neg, lr in product(vector_sizes, windows, negatives, lrs):
        cfg = {"embed_size": vsize, "window": win, "negative": neg, "lr": lr}

        cbow = train_model(sentences, vocab, sg=0, epochs=search_epochs, **cfg)
        cs   = evaluate_model(cbow, PROBE_WORDS)
        done += 1
        print(f"  {'CBOW':>10} | {vsize:>4} | {win:>4} | {neg:>4} | {lr:>6.3f} | {cs:>7.4f} | {done/total*100:.0f}%")
        if cs > best_cbow_score:
            best_cbow_score, best_cbow_cfg = cs, cfg

        sg_m = train_model(sentences, vocab, sg=1, epochs=search_epochs, **cfg)
        ss   = evaluate_model(sg_m, PROBE_WORDS)
        done += 1
        print(f"  {'Skip-gram':>10} | {vsize:>4} | {win:>4} | {neg:>4} | {lr:>6.3f} | {ss:>7.4f} | {done/total*100:.0f}%")
        if ss > best_sg_score:
            best_sg_score, best_sg_cfg = ss, cfg

        results.append((cfg, cs, ss))

    print("\n  -- Re-training best models for", best_epochs, "epochs --")
    print(f"  Best CBOW config     : {best_cbow_cfg}  (score={best_cbow_score:.4f})")
    print(f"  Best Skip-gram config: {best_sg_cfg}  (score={best_sg_score:.4f})")
    best_cbow = train_model(sentences, vocab, sg=0, epochs=best_epochs, **best_cbow_cfg)
    best_sg   = train_model(sentences, vocab, sg=1, epochs=best_epochs, **best_sg_cfg)

    return {
        "vocab":         vocab,
        "best_cbow":     best_cbow,
        "best_skipgram": best_sg,
        "results_table": results,
    }


def load_glove(path: str) -> WordVectors:
    word2idx, idx2word, vectors = {}, {}, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            idx   = len(word2idx)
            word2idx[word] = idx
            idx2word[idx]  = word
            vectors.append(vec)
    return WordVectors(word2idx, idx2word, np.stack(vectors))


def compare_with_pretrained(cbow_result, sg_result, probe_words):
    print("\n" + "=" * 72)
    print("  Comparison: scratch CBOW  vs.  scratch Skip-gram  vs.  GloVe-50d")
    print("=" * 72)

    glove_wv = None
    if os.path.exists(GLOVE_PATH):
        print("  Loading GloVe-50d ...")
        glove_wv = load_glove(GLOVE_PATH)
        print(f"  GloVe vocab: {len(glove_wv):,} words")
    else:
        print(f"  GloVe-50d not found at {GLOVE_PATH}")

    for word in probe_words:
        print(f"\n  Query word: '{word}'")
        sources = [("CBOW (scratch)", cbow_result.wv),
                   ("Skip-gram (scratch)", sg_result.wv)]
        if glove_wv:
            sources.append(("GloVe-50d", glove_wv))
        for label, wv in sources:
            if word not in wv:
                print(f"    {label:<22}: [OOV]")
                continue
            neighbours = wv.most_similar(word, topn=5)
            top5 = ", ".join(f"{w}({s:.2f})" for w, s in neighbours)
            print(f"    {label:<22}: {top5}")

    print("=" * 72)


def load_corpus(path: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed corpus not found: {path}\nRun preprocessor.py first.")
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) >= 2:
                sentences.append(toks)
    print(f"  Loaded {len(sentences):,} sentences.")
    return sentences


def save_models(best_cbow: Word2VecResult, best_sg: Word2VecResult):
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_cbow.save(BEST_CBOW_PATH)
    best_sg.save(BEST_SG_PATH)
    print(f"\n  CBOW saved      -> {BEST_CBOW_PATH}")
    print(f"  Skip-gram saved -> {BEST_SG_PATH}")


def load_best_models():
    return Word2VecResult.load(BEST_CBOW_PATH), Word2VecResult.load(BEST_SG_PATH)


def print_results_table(results: list):
    print("\n  == Hyperparameter Search Results =======================================")
    print(f"  {'dim':>5} | {'win':>5} | {'neg':>5} | {'lr':>7} | {'CBOW':>8} | {'Skip-gram':>10}")
    print("  " + "-" * 58)
    for cfg, cs, ss in results:
        print(
            f"  {cfg['embed_size']:>5} | {cfg['window']:>5} | {cfg['negative']:>5} | "
            f"{cfg.get('lr', 0.025):>7.3f} | {cs:>8.4f} | {ss:>10.4f}"
        )
    print("  " + "=" * 58)


if __name__ == "__main__":
    t_start = time.time()

    sentences    = load_corpus(PROCESSED_FILE)
    results_dict = run_experiments(sentences)

    print_results_table(results_dict["results_table"])
    save_models(results_dict["best_cbow"], results_dict["best_skipgram"])
    compare_with_pretrained(
        results_dict["best_cbow"],
        results_dict["best_skipgram"],
        PROBE_WORDS,
    )

    m, s = divmod(int(time.time() - t_start), 60)
    print(f"\n  Total time: {m}m {s}s")
    print("  Models ready for: semantic_analysis.py  visualize.py")
