import csv
import random
from collections import defaultdict

from src import config
from src import preprocessing as pp


def assign_splits(rows, seed=config.SEED):
    # copies of the same text all go to the same split, otherwise the model
    # gets tested on tweets it saw in training
    groups = defaultdict(list)
    for r in rows:
        groups[pp.normalize(r[config.TEXT_COL]).lower()].append(r)

    # stratify on label + agreement so every split gets the same share of the
    # noisy 0.333 negatives. a group uses the label/agreement of its first tweet
    strata = defaultdict(list)
    for key in sorted(groups):
        first = min(groups[key], key=lambda r: r[config.ID_COL])
        strata[(int(float(first[config.LABEL_COL])), round(float(first[config.AGREEMENT_COL]), 3))].append(key)

    rng = random.Random(seed)
    split_of = {}
    for stratum in sorted(strata):
        keys = strata[stratum]
        rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        seen = 0
        for k in keys:
            if seen < total * config.TEST_SIZE:
                name = "test"
            elif seen < total * (config.TEST_SIZE + config.VAL_SIZE):
                name = "val"
            else:
                name = "train"
            for r in groups[k]:
                split_of[r[config.ID_COL]] = name
            seen += len(groups[k])
    return split_of


def make_split(path=config.SPLIT_FILE):
    rows, _ = pp.read_raw_rows(config.TRAIN_CSV)
    split_of = assign_splits(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([config.ID_COL, "split"])
        for tweet_id in sorted(split_of):
            writer.writerow([tweet_id, split_of[tweet_id]])
    return split_of


def load_split(df=None):
    import pandas as pd

    if df is None:
        df = pp.load_train()
    split = pd.read_csv(config.SPLIT_FILE)
    df = df.merge(split, on=config.ID_COL, how="left")
    if df["split"].isna().any():
        raise ValueError("some tweets are missing from the split file")
    return tuple(df[df["split"] == s].drop(columns="split").reset_index(drop=True)
                 for s in ["train", "val", "test"])


if __name__ == "__main__":
    make_split()
    print("saved", config.SPLIT_FILE)
