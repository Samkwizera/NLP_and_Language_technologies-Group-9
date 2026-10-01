import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src import config
from src.dataset import TweetDataset, Vocab, make_loader
from src.embeddings import SOURCES, build_matrix, load_vectors, lookup
from src.evaluate import compute_metrics
from src.models.baselines import agreement_subset
from src.utils import log_experiment, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS_DIR = config.RESULTS_DIR / "runs"


def make_embedding(vocab_size, dim, matrix=None, freeze=False):
    emb = nn.Embedding(vocab_size, dim, padding_idx=0)
    if matrix is not None:
        emb.weight.data.copy_(torch.from_numpy(matrix))
    emb.weight.requires_grad = not freeze
    return emb


def masked_pool(out, lengths, pooling):
    # out: (batch, time, features). padded steps must not win the max or drag the mean
    mask = (torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]).unsqueeze(-1)
    if pooling == "max":
        return out.masked_fill(~mask, -1e4).max(dim=1).values
    return (out * mask).sum(dim=1) / lengths[:, None].float()


class BiRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, matrix=None, freeze=False, cell="lstm",
                 hidden=128, layers=1, dropout=0.5, pooling="max"):
        super().__init__()
        self.emb = make_embedding(vocab_size, emb_dim, matrix, freeze)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                       dropout=dropout if layers > 1 else 0)
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, len(config.LABELS))

    def forward(self, ids, lengths):
        x = self.drop(self.emb(ids))
        # packing makes the backward direction start at the last real token, not at the padding
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        if self.pooling == "last":
            h = h[0] if isinstance(h, tuple) else h
            feats = torch.cat([h[-2], h[-1]], dim=1)  # final forward + final backward state
        else:
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
            feats = masked_pool(out, lengths, self.pooling)
        return self.fc(self.drop(feats))


class TextCNN(nn.Module):
    # kim (2014): parallel convolutions over 3/4/5-word windows, max over time
    def __init__(self, vocab_size, emb_dim, matrix=None, freeze=False, filters=100,
                 kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = make_embedding(vocab_size, emb_dim, matrix, freeze)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, filters, k) for k in kernel_sizes)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernel_sizes), len(config.LABELS))

    def forward(self, ids, lengths):
        x = self.drop(self.emb(ids)).transpose(1, 2)  # conv1d wants (batch, channels, time)
        feats = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))


class CNNBiLSTM(nn.Module):
    # conv picks up local phrases, the bilstm then reads those phrase features in order
    def __init__(self, vocab_size, emb_dim, matrix=None, freeze=False, filters=128,
                 kernel_size=3, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = make_embedding(vocab_size, emb_dim, matrix, freeze)
        self.conv = nn.Conv1d(emb_dim, filters, kernel_size, padding=kernel_size // 2)
        self.rnn = nn.LSTM(filters, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, len(config.LABELS))

    def forward(self, ids, lengths):
        x = self.drop(self.emb(ids)).transpose(1, 2)
        x = torch.relu(self.conv(x)).transpose(1, 2)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
        return self.fc(self.drop(masked_pool(out, lengths, "max")))


def build_model(name, vocab_size, emb_dim, matrix=None, freeze=False, **kwargs):
    if name in ("bilstm", "bigru"):
        return BiRNN(vocab_size, emb_dim, matrix, freeze, cell=name[2:], **kwargs)
    if name == "cnn":
        return TextCNN(vocab_size, emb_dim, matrix, freeze, **kwargs)
    if name == "cnn-bilstm":
        return CNNBiLSTM(vocab_size, emb_dim, matrix, freeze, **kwargs)
    raise ValueError(name)


def class_weights(labels):
    # "balanced" as in sklearn: n / (k * count), so the 10% negative class counts ~3x more
    counts = np.bincount(labels, minlength=len(config.LABELS))
    return torch.tensor(len(labels) / (len(counts) * counts), dtype=torch.float32)


def predict_probs(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for ids, lengths, _, _ in loader:
            probs.append(torch.softmax(model(ids.to(DEVICE), lengths.to(DEVICE)), dim=1).cpu())
    return torch.cat(probs).numpy()


def to_labels(probs):
    return np.array([config.IDX2LABEL[i] for i in probs.argmax(axis=1)])


def fit(model, train_ds, val_ds, val_labels, epochs=30, lr=1e-3, batch_size=64, patience=5,
        weights=None, seed=config.SEED):
    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(DEVICE), reduction="none")
    train_loader = make_loader(train_ds, batch_size, shuffle=True, seed=seed)
    val_loader = make_loader(val_ds, 256)

    history, best, best_state, bad = [], -1, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for ids, lengths, y, w in train_loader:
            ids, lengths, y, w = ids.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE), w.to(DEVICE)
            opt.zero_grad()
            # w is the agreement weight (all ones unless agreement="weighted")
            loss = (loss_fn(model(ids, lengths), y) * w).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item() * len(y)
            n += len(y)

        probs = predict_probs(model, val_loader)
        val_loss = nn.functional.nll_loss(torch.log(torch.tensor(probs) + 1e-9),
                                          torch.tensor(val_ds.labels)).item()
        val_f1 = compute_metrics(val_labels, to_labels(probs))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, "val_macro_f1": val_f1})

        # early stopping on the metric we actually report, not on the loss
        if val_f1 > best:
            best, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return history


def prepare(train_df, embeddings="random", split_hashtags=False, keep_rare=True):
    # returns vocab + embedding matrix (None for random) built from the training rows only
    if embeddings == "random":
        return Vocab.build(train_df[config.TEXT_COL], split_hashtags=split_hashtags), None, 200
    vectors = load_vectors(embeddings)
    keep = (lambda w: lookup(w, vectors) is not None) if keep_rare else None
    vocab = Vocab.build(train_df[config.TEXT_COL], split_hashtags=split_hashtags, keep=keep)
    matrix, _ = build_matrix(vocab, vectors, SOURCES[embeddings]["dim"])
    return vocab, matrix, SOURCES[embeddings]["dim"]


def model_name(model, embeddings, freeze):
    if embeddings == "random":
        return f"{model}+random-emb"
    return f"{model}+{embeddings}-{'frozen' if freeze else 'tuned'}"


def run_experiment(exp_id, train_df, val_df, eval_df=None, model="bilstm", embeddings="random",
                   freeze=False, class_weight=None, agreement="all", split_hashtags=False,
                   keep_rare=True, lr=1e-3, epochs=30, batch_size=64, patience=5, seed=config.SEED,
                   change="", reason="", takeaway="", member="Divine", log=True, **model_kwargs):
    # early stopping always watches val. eval_df is where the reported scores come from
    # (val while developing, test only for the final runs)
    eval_df = val_df if eval_df is None else eval_df
    set_seed(seed)
    data, agree_w = agreement_subset(train_df, agreement)
    vocab, matrix, dim = prepare(data, embeddings, split_hashtags, keep_rare)

    train_ds = TweetDataset(data, vocab, split_hashtags=split_hashtags, weights=agree_w)
    val_ds = TweetDataset(val_df, vocab, split_hashtags=split_hashtags)
    eval_ds = TweetDataset(eval_df, vocab, split_hashtags=split_hashtags)
    net = build_model(model, len(vocab), dim, matrix, freeze, **model_kwargs)
    weights = class_weights(train_ds.labels) if class_weight == "balanced" else None

    start = time.time()
    history = fit(net, train_ds, val_ds, val_df[config.LABEL_COL].values, epochs, lr, batch_size,
                  patience, weights, seed)
    probs = predict_probs(net, make_loader(eval_ds, 256))
    preds = to_labels(probs)
    metrics = compute_metrics(eval_df[config.LABEL_COL], preds)
    metrics["epochs"] = len(history)
    metrics["seconds"] = round(time.time() - start)
    metrics["params"] = sum(p.numel() for p in net.parameters() if p.requires_grad)

    if log:
        log_experiment(exp_id=exp_id, member=member, model=model_name(model, embeddings, freeze),
                       change=change, reason=reason, macro_f1=metrics["macro_f1"],
                       accuracy=metrics["accuracy"], rmse=metrics["rmse"], takeaway=takeaway)
    return {"model": net, "vocab": vocab, "preds": preds, "confidence": probs.max(axis=1),
            "probs": probs, "metrics": metrics, "history": history}


def run_or_load(exp_id, train_df, val_df, eval_df=None, retrain=False, **kwargs):
    # a bilstm run takes 10-20 min on a laptop cpu, so every finished run is saved and
    # reloaded next time. retrain=True trains it again (much faster on a colab gpu)
    path = RUNS_DIR / f"{exp_id}.json"
    if path.exists() and not retrain:
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        run["preds"] = np.array(run["preds"])
        run["probs"] = np.array(run["probs"])
        run["confidence"] = run["probs"].max(axis=1)
        return run

    run = run_experiment(exp_id, train_df, val_df, eval_df, **kwargs)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in kwargs.items() if k not in ("change", "reason", "takeaway", "member")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"exp_id": exp_id, "change": kwargs.get("change", ""), "reason": kwargs.get("reason", ""),
                   "settings": settings, "metrics": run["metrics"],
                   "history": run["history"], "preds": run["preds"].tolist(),
                   "probs": np.round(run["probs"], 4).tolist()}, f)
    run["settings"] = settings
    run["change"] = kwargs.get("change", "")
    return run


def learning_curve(train_df, val_df, fractions=(0.1, 0.25, 0.5, 0.75, 1.0), **kwargs):
    # same sampling as the baseline learning curve so the two can go on one plot
    sizes, val_scores = [], []
    for frac in fractions:
        sub = train_df.groupby(config.LABEL_COL, group_keys=False).sample(frac=frac, random_state=config.SEED)
        run = run_experiment("lc", sub, val_df, log=False, **kwargs)
        sizes.append(len(sub))
        val_scores.append(run["metrics"]["macro_f1"])
    return np.array(sizes), val_scores
