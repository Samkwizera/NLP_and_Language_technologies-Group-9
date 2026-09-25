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
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.EXPERIMENT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
