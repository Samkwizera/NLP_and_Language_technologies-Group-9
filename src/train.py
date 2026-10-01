# python -m src.train --model logreg --class-weight balanced --agreement all --exp-id B3
# python -m src.train --model bilstm --embeddings glove --class-weight balanced --exp-id R8
import argparse

from src.models.baselines import run_experiment
from src.split import load_split
from src.utils import ensure_dirs, set_seed

NEURAL = ["bilstm", "bigru", "cnn", "cnn-bilstm"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logreg", choices=["logreg", "svm", "nb"] + NEURAL)
    parser.add_argument("--features", default="word+char", choices=["word", "word+char"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    parser.add_argument("--agreement", default="all", choices=["all", "high", "drop_333", "weighted"])
    # rnn / cnn only
    parser.add_argument("--embeddings", default="random", choices=["random", "glove", "fasttext"])
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--split-hashtags", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="score on the test split (final runs only)")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--member", default="Samuel")
    parser.add_argument("--change", default="")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    set_seed()
    ensure_dirs()
    train, val, test = load_split()
    if args.model in NEURAL:
        from src.models.rnn_cnn import run_or_load
        run = run_or_load(args.exp_id, train, val, test if args.test else None, retrain=True,
                          model=args.model, embeddings=args.embeddings, freeze=args.freeze,
                          class_weight=args.class_weight, agreement=args.agreement,
                          split_hashtags=args.split_hashtags, seed=args.seed,
                          change=args.change, reason=args.reason, member=args.member)
        print(run["metrics"])
        return

    _, _, metrics = run_experiment(args.exp_id, train, val, model=args.model, features=args.features,
                                   C=args.C, class_weight=args.class_weight, agreement=args.agreement,
                                   change=args.change, reason=args.reason, member=args.member)
    print(metrics)


if __name__ == "__main__":
    main()
