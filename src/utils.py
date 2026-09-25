import csv
import os
import random

import numpy as np

from src import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs():
    for d in [config.FIGURES_DIR, config.ERRORS_DIR, config.SPLITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def log_experiment(**row):
    # fail on typos so we don't end up with half-empty rows in the log
    unknown = set(row) - set(config.EXPERIMENT_COLUMNS)
    if unknown:
        raise KeyError(f"unknown columns: {unknown}")
    path = config.EXPERIMENTS_CSV
    rows = []
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    # rerunning a notebook replaces the old row with the same exp_id
    old = next((r for r in rows if r["exp_id"] == row["exp_id"]), None)
    if old is not None:
        if not row.get("takeaway"):
            row["takeaway"] = old.get("takeaway", "")
        rows[rows.index(old)] = row
    else:
        rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.EXPERIMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def add_takeaway(exp_id, takeaway):
    path = config.EXPERIMENTS_CSV
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["exp_id"] == exp_id:
            r["takeaway"] = takeaway
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.EXPERIMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
