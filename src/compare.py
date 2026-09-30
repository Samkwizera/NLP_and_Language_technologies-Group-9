import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from src import config
from src.evaluate import compute_metrics


def preds_from_errors(df, name):
    # the error files only list the wrong predictions, so every other tweet was predicted correctly
    errors = pd.read_csv(config.ERRORS_DIR / f"{name}_val_errors.csv")
    pred = df.set_index(config.ID_COL)[config.LABEL_COL].copy()
    pred.loc[errors[config.ID_COL]] = errors["predicted"].values
    return pred.loc[df[config.ID_COL]].values


def preds_from_run(exp_id):
    with open(config.RESULTS_DIR / "runs" / f"{exp_id}.json", encoding="utf-8") as f:
        return np.array(json.load(f)["preds"])


def macro_f1(y, p):
    return f1_score(y, p, labels=config.LABELS, average="macro", zero_division=0)


def bootstrap_ci(y, p, n=2000, seed=config.SEED):
    # resample tweets with replacement, 95% interval of macro-F1
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    scores = [macro_f1(y[idx], p[idx]) for idx in rng.integers(0, len(y), size=(n, len(y)))]
    return np.percentile(scores, [2.5, 97.5])


def paired_bootstrap(y, a, b, n=2000, seed=config.SEED):
    # same resampled tweets for both models, so the difference is paired.
    # returns the 95% interval of f1(a) - f1(b) and how often b came out ahead
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    diffs = np.array([macro_f1(y[idx], a[idx]) - macro_f1(y[idx], b[idx])
                      for idx in rng.integers(0, len(y), size=(n, len(y)))])
    return np.percentile(diffs, [2.5, 97.5]), float((diffs <= 0).mean())


def mcnemar(y, a, b):
    # only tweets where exactly one of the two models is right carry information
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    only_a = int(((a == y) & (b != y)).sum())
    only_b = int(((a != y) & (b == y)).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, p


def comparison_table(y, preds):
    rows = []
    for name, p in preds.items():
        m = compute_metrics(y, p)
        lo, hi = bootstrap_ci(y, p)
        rows.append({"model": name, "macro_f1": m["macro_f1"], "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                     "accuracy": m["accuracy"], "rmse": m["rmse"], "f1_negative": m["f1_negative"],
                     "f1_neutral": m["f1_neutral"], "f1_positive": m["f1_positive"]})
    return pd.DataFrame(rows).set_index("model")


def plot_per_class(table, title, name=None):
    cols = ["f1_negative", "f1_neutral", "f1_positive", "macro_f1"]
    ax = table[cols].rename(columns=lambda c: c.replace("f1_", "").replace("_", "-")).T.plot.bar(
        figsize=(10, 4.5), rot=0, width=0.8)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    if name:
        plt.savefig(config.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
