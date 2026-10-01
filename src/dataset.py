from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import config
from src import preprocessing as pp

PAD, UNK, HASHTAG = "<pad>", "<unk>", "<hashtag>"


def tweet_tokens(text, split_hashtags=False):
    tokens = pp.tokenize(text)
    if not split_hashtags:
        return tokens
    # glove twitter was trained with "#word" written as "<hashtag> word"
    out = []
    for t in tokens:
        if t.startswith("#") and len(t) > 1:
            out += [HASHTAG, t[1:]]
        else:
            out.append(t)
    return out


class Vocab:
    def __init__(self, counts, min_freq=config.MIN_FREQ, keep=None):
        # keep: rare train words worth keeping anyway (they have a pretrained vector,
        # so the model doesn't have to learn them from a single example)
        keep = keep or (lambda w: False)
        words = sorted(w for w, c in counts.items() if c >= min_freq or keep(w))
        # pad is 0 so padded positions can be masked / ignored by the embedding
        self.itos = [PAD, UNK] + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = counts

    @classmethod
    def build(cls, texts, min_freq=config.MIN_FREQ, split_hashtags=False, keep=None):
        counts = Counter(t for text in texts for t in tweet_tokens(text, split_hashtags))
        return cls(counts, min_freq, keep)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]


class TweetDataset(Dataset):
    def __init__(self, df, vocab, max_len=config.MAX_LEN, split_hashtags=False, weights=None):
        self.ids = np.zeros((len(df), max_len), dtype=np.int64)
        self.lengths = np.zeros(len(df), dtype=np.int64)
        for i, text in enumerate(df[config.TEXT_COL]):
            ids = vocab.encode(tweet_tokens(text, split_hashtags))[:max_len]
            # an empty tweet would break the packed lstm, give it one <unk>
            ids = ids or [vocab.stoi[UNK]]
            self.ids[i, :len(ids)] = ids
            self.lengths[i] = len(ids)
        self.labels = np.array([config.LABEL2IDX[int(l)] for l in df[config.LABEL_COL]])
        self.weights = np.ones(len(df), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (torch.from_numpy(self.ids[i]), torch.tensor(self.lengths[i]),
                torch.tensor(self.labels[i]), torch.tensor(self.weights[i]))


def make_loader(dataset, batch_size=64, shuffle=False, seed=config.SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def coverage_report(vocab, df, split_hashtags=False):
    # how much of the val/test text the train vocab can actually represent
    tokens = [t for text in df[config.TEXT_COL] for t in tweet_tokens(text, split_hashtags)]
    known = sum(t in vocab.stoi for t in tokens)
    lengths = [len(tweet_tokens(text, split_hashtags)) for text in df[config.TEXT_COL]]
    return {
        "vocab_size": len(vocab),
        "token_coverage": round(known / len(tokens), 4),
        "type_coverage": round(float(np.mean([t in vocab.stoi for t in set(tokens)])), 4),
        "truncated_tweets": int(sum(n > config.MAX_LEN for n in lengths)),
    }
