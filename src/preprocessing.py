import csv
import html
import re

import pandas as pd

from src import config

ID_RE = re.compile(config.TWEET_ID_PATTERN)


def read_raw_rows(path, has_labels=True):
    # A tweet with a raw newline gets split into two rows and the second one has
    # its fields shifted left (text in tweet_id, label in safe_text, ...).
    # We glue it back onto the previous row instead of dropping it.
    with open(path, encoding="utf-8", newline="") as f:
        raw = list(csv.DictReader(f))

    rows, report = [], []
    for r in raw:
        rid = (r[config.ID_COL] or "").strip()
        if ID_RE.match(rid):
            rows.append(r)
            continue

        prev = rows[-1] if rows else None
        if has_labels and prev is not None and not prev[config.LABEL_COL].strip():
            prev[config.TEXT_COL] = prev[config.TEXT_COL].rstrip() + " " + rid
            prev[config.LABEL_COL] = r[config.TEXT_COL]
            prev[config.AGREEMENT_COL] = r[config.LABEL_COL]
            report.append({"action": "merged", "tweet_id": prev[config.ID_COL], "text": rid})
        else:
            # test file: the ID line is lost, keep the text anyway
            rows.append({config.ID_COL: None, config.TEXT_COL: rid})
            report.append({"action": "orphan", "tweet_id": None, "text": rid})
    return rows, report


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_train(path=config.TRAIN_CSV, drop_invalid=True, return_report=False):
    rows, report = read_raw_rows(path)
    df = pd.DataFrame(rows)
    df[config.TEXT_COL] = df[config.TEXT_COL].fillna("")
    df[config.LABEL_COL] = df[config.LABEL_COL].map(to_float)
    df[config.AGREEMENT_COL] = df[config.AGREEMENT_COL].map(to_float)

    df["valid"] = df[config.LABEL_COL].isin([-1.0, 0.0, 1.0]) & (df[config.TEXT_COL].str.strip() != "")
    for _, r in df[~df["valid"]].iterrows():
        report.append({"action": "invalid", "tweet_id": r[config.ID_COL], "text": r[config.TEXT_COL][:40]})

    if drop_invalid:
        df = df[df["valid"]].copy()
        df[config.LABEL_COL] = df[config.LABEL_COL].astype(int)
    df = df.reset_index(drop=True)
    return (df, report) if return_report else df


def load_test(path=config.TEST_CSV, return_report=False):
    rows, report = read_raw_rows(path, has_labels=False)
    df = pd.DataFrame(rows)
    df[config.TEXT_COL] = df[config.TEXT_COL].fillna("")
    return (df, report) if return_report else df


def normalize(text):
    text = html.unescape(text or "")  # &amp; -> &
    return re.sub(r"\s+", " ", text).strip()


TOKEN_RE = re.compile(r"<user>|<url>|https?://\S+|www\.\S+|#\w+|@\w+|\w+(?:['’]\w+)?|[^\w\s]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"#\w+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]")


def tokenize(text, lower=True):
    text = normalize(text)
    if lower:
        text = text.lower()
    return TOKEN_RE.findall(text)


def word_tokens(text):
    return [t for t in tokenize(text) if t.isalpha()]


def hashtags(text):
    return [h.lower() for h in HASHTAG_RE.findall(normalize(text))]


def leftover_urls(text):
    return URL_RE.findall(normalize(text))


def leftover_mentions(text):
    return MENTION_RE.findall(normalize(text))


def emojis(text):
    return EMOJI_RE.findall(normalize(text))


# Rough check only, not real language ID. The short words clash with English
# ("ya", "si", "mama") so they only count if two different ones show up.
SWAHILI_WORDS = {
    "hii", "hiyo", "hizi", "huyo", "kwa", "kwenye", "katika", "lakini", "sana",
    "hapa", "leo", "kesho", "jana", "kama", "hakuna", "nini", "wewe", "mimi",
    "sisi", "wao", "yeye", "tena", "bado", "pia", "sasa", "watu", "mtu",
    "mtoto", "watoto", "wazazi", "kila", "sababu", "habari", "asante", "karibu",
    "ndio", "hapana", "sijui", "nataka", "kweli", "serikali", "daktari",
    "hospitali", "ugonjwa", "magonjwa", "chanjo", "sindano", "dawa", "afya",
    "homa", "surua",
    # sheng
    "manze", "msee", "wasee", "niaje", "poa", "fiti", "mbogi", "buda", "maze",
    "vitu", "sare", "mathe", "fathe", "mzae", "kanjo", "noma", "sanse",
}
SWAHILI_SHORT = {"na", "ya", "wa", "za", "la", "ni", "si", "tu", "mama", "baba", "hata", "juu", "moja"}


def swahili_tokens(text):
    toks = word_tokens(text)
    return [t for t in toks if t in SWAHILI_WORDS], [t for t in toks if t in SWAHILI_SHORT]


def is_swahili(text):
    strong, short = swahili_tokens(text)
    return len(strong) >= 1 or len(set(short)) >= 2
