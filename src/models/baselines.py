import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src import config
from src import preprocessing as pp
from src.evaluate import compute_metrics
from src.utils import log_experiment

# keeps placeholders, hashtags and punctuation/emojis like ! ? as tokens
WORD_PATTERN = r"<user>|<url>|#?\w\w+|[^\w\s]"


def build_pipeline(model="logreg", features="word+char", C=1.0, class_weight=None):
    word = TfidfVectorizer(preprocessor=pp.clean_for_tfidf, token_pattern=WORD_PATTERN,
                           ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(preprocessor=pp.clean_for_tfidf, analyzer="char_wb",
                           ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    if features == "word":
        vec = word
    else:
        vec = FeatureUnion([("word", word), ("char", char)])

    if model == "logreg":
        clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000)
    elif model == "svm":
        clf = LinearSVC(C=C, class_weight=class_weight)
    elif model == "nb":
        clf = ComplementNB(alpha=C)
    else:
        raise ValueError(model)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def agreement_subset(train_df, mode):
    # returns the training rows and sample weights for each agreement setting
    a = train_df[config.AGREEMENT_COL]
    if mode == "all":
        return train_df, None
    if mode == "high":
        return train_df[a == 1.0], None
    if mode == "drop_333":
        return train_df[a > 0.34], None
    if mode == "weighted":
        return train_df, a.values
    raise ValueError(mode)


def confidence(pipe, X):
    # svm has no probabilities, so we use the top decision score instead
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X).max(axis=1)
    return pipe.decision_function(X).max(axis=1)


def run_experiment(exp_id, train_df, eval_df, model="logreg", features="word+char", C=1.0,
                   class_weight=None, agreement="all", change="", reason="", takeaway="",
                   member="Samuel", log=True):
    data, weights = agreement_subset(train_df, agreement)
    pipe = build_pipeline(model, features, C, class_weight)
    pipe.fit(data[config.TEXT_COL], data[config.LABEL_COL], clf__sample_weight=weights)

    preds = pipe.predict(eval_df[config.TEXT_COL])
    metrics = compute_metrics(eval_df[config.LABEL_COL], preds)
    if log:
        name = f"tfidf-{features}+{model}"
        log_experiment(exp_id=exp_id, member=member, model=name, change=change, reason=reason,
                       macro_f1=metrics["macro_f1"], accuracy=metrics["accuracy"],
                       rmse=metrics["rmse"], takeaway=takeaway)
    return pipe, preds, metrics


def learning_curve(train_df, val_df, fractions=(0.1, 0.25, 0.5, 0.75, 1.0), **kwargs):
    sizes, train_scores, val_scores = [], [], []
    for frac in fractions:
        # sample within each label so small subsets keep the class balance
        sub = train_df.groupby(config.LABEL_COL, group_keys=False).sample(frac=frac, random_state=config.SEED)
        pipe = build_pipeline(**kwargs)
        pipe.fit(sub[config.TEXT_COL], sub[config.LABEL_COL])
        sizes.append(len(sub))
        train_scores.append(compute_metrics(sub[config.LABEL_COL], pipe.predict(sub[config.TEXT_COL]))["macro_f1"])
        val_scores.append(compute_metrics(val_df[config.LABEL_COL], pipe.predict(val_df[config.TEXT_COL]))["macro_f1"])
    return np.array(sizes), train_scores, val_scores
