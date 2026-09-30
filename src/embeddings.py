import io
import urllib.request
import zipfile

import numpy as np

from src import config
from src import preprocessing as pp
from src.dataset import PAD, UNK, tweet_tokens

# glove twitter: 2B tweets, same <user>/<url>/<hashtag> placeholders as our data
# fasttext crawl: 600B tokens of web text, bigger vocab but not twitter specific
SOURCES = {
    "glove": {"url": "https://nlp.stanford.edu/data/glove.twitter.27B.zip",
              "zip": "glove.twitter.27B.zip", "member": "glove.twitter.27B.200d.txt", "dim": 200},
    "fasttext": {"url": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip",
                 "zip": "crawl-300d-2M.vec.zip", "member": "crawl-300d-2M.vec", "dim": 300},
}


def candidates(token):
    # "#vaccineswork" has no vector of its own most of the time, so try the word
    out = [token]
    if token.startswith("#") and len(token) > 1:
        out.append(token[1:])
    return out


def dataset_words():
    # every word the vocab could ever ask for, so one pass over the zip is enough.
    # only the text is used here, the labels are never touched
    texts = list(pp.load_train()[config.TEXT_COL]) + list(pp.load_test()[config.TEXT_COL])
    words = set()
    for text in texts:
        for t in set(tweet_tokens(text)) | set(tweet_tokens(text, split_hashtags=True)):
            words.update(candidates(t))
    return words


def load_vectors(name):
    src = SOURCES[name]
    cache = config.EMBEDDINGS_DIR / f"{name}_filtered.txt"
    if not cache.exists():
        zip_path = config.EMBEDDINGS_DIR / src["zip"]
        if not zip_path.exists():
            config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"downloading {src['url']} (this takes a few minutes)")
            urllib.request.urlretrieve(src["url"], zip_path)
        words = dataset_words()
        with zipfile.ZipFile(zip_path) as z, z.open(src["member"]) as raw, \
                open(cache, "w", encoding="utf-8") as out:
            for line in io.TextIOWrapper(raw, encoding="utf-8", errors="ignore"):
                word = line.split(" ", 1)[0]
                if word in words:
                    out.write(line)

    vectors = {}
    with open(cache, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != src["dim"] + 1:
                continue  # fasttext header line
            # fasttext has some words in several cases, keep the first (most frequent)
            vectors.setdefault(parts[0], np.asarray(parts[1:], dtype=np.float32))
    return vectors


def lookup(token, vectors):
    for c in candidates(token):
        if c in vectors:
            return vectors[c]
    return None


def build_matrix(vocab, vectors, dim, seed=config.SEED):
    # words without a pretrained vector start random, on the same scale as the real ones
    found = np.array([lookup(w, vectors) is not None for w in vocab.itos])
    known = np.stack([lookup(w, vectors) for w, ok in zip(vocab.itos, found) if ok])
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, known.std(), size=(len(vocab), dim)).astype(np.float32)
    for i, w in enumerate(vocab.itos):
        if found[i]:
            matrix[i] = lookup(w, vectors)
    matrix[vocab.stoi[PAD]] = 0
    return matrix, found


def embedding_coverage(vocab, vectors, df, split_hashtags=False):
    # share of the vocab and of the running text in df that gets a pretrained vector
    in_vocab = [w for w in vocab.itos if w not in (PAD, UNK)]
    tokens = [t for text in df[config.TEXT_COL] for t in tweet_tokens(text, split_hashtags)]
    return {
        "vocab_with_vector": round(float(np.mean([lookup(w, vectors) is not None for w in in_vocab])), 4),
        "tokens_with_vector": round(float(np.mean([lookup(t, vectors) is not None for t in tokens])), 4),
    }
