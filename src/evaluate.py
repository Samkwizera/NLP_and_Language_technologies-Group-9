import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay

from src import config


def compute_metrics(y_true, y_pred):
    # labels stay on the original -1/0/1 scale so rmse matches the zindi leaderboard
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    per_class = f1_score(y_true, y_pred, labels=config.LABELS, average=None, zero_division=0)
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, labels=config.LABELS, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }
    for lab, f in zip(config.LABELS, per_class):
        metrics[f"f1_{config.LABEL_NAMES[lab]}"] = f
    return {k: round(float(v), 4) for k, v in metrics.items()}


def plot_confusion_matrix(y_true, y_pred, title, name=None):
    names = [config.LABEL_NAMES[l] for l in config.LABELS]
    cm = confusion_matrix(y_true, y_pred, labels=config.LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"{title} (counts)")
    # row-normalised so the small negative class is readable
    ConfusionMatrixDisplay(cm / cm.sum(axis=1, keepdims=True), display_labels=names).plot(
        ax=axes[1], colorbar=False, cmap="Blues", values_format=".2f")
    axes[1].set_title(f"{title} (recall per class)")
    plt.tight_layout()
    if name:
        plt.savefig(config.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_learning_curve(sizes, train_scores, val_scores, title, name=None):
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, train_scores, "o-", label="train")
    plt.plot(sizes, val_scores, "o-", label="val")
    plt.xlabel("training tweets")
    plt.ylabel("macro-F1")
    plt.title(title)
    plt.legend()
    if name:
        plt.savefig(config.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history, title, name=None):
    # per-epoch curves for the neural models: where it starts to overfit and where
    # early stopping picked the weights
    h = pd.DataFrame(history)
    best = h.loc[h["val_macro_f1"].idxmax(), "epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["epoch"], h["train_loss"], "o-", label="train")
    axes[0].plot(h["epoch"], h["val_loss"], "o-", label="val")
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].plot(h["epoch"], h["val_macro_f1"], "o-", color="C2")
    axes[1].set_ylabel("val macro-F1")
    for ax in axes:
        ax.axvline(best, color="grey", ls="--", lw=1)
        ax.set_xlabel("epoch")
    axes[0].legend()
    fig.suptitle(f"{title} (dashed line = kept epoch)")
    plt.tight_layout()
    if name:
        plt.savefig(config.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def export_errors(df, y_pred, confidence, exp_id):
    out = pd.DataFrame({
        config.ID_COL: df[config.ID_COL].values,
        "text": df[config.TEXT_COL].values,
        "true": df[config.LABEL_COL].values,
        "predicted": np.asarray(y_pred),
        "agreement": df[config.AGREEMENT_COL].values,
        "confidence": np.round(confidence, 4),
    })
    out = out[out["true"] != out["predicted"]].sort_values("confidence", ascending=False)
    path = config.ERRORS_DIR / f"{exp_id}_val_errors.csv"
    out.to_csv(path, index=False)
    return out
