# python -m src.train --model logreg --class-weight balanced --agreement all --exp-id B3
import argparse

from src.models.baselines import run_experiment
from src.split import load_split
from src.utils import ensure_dirs, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logreg", choices=["logreg", "svm", "nb"])
    parser.add_argument("--features", default="word+char", choices=["word", "word+char"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--class-weight", default=None, choices=[None, "balanced"])
    parser.add_argument("--agreement", default="all", choices=["all", "high", "drop_333", "weighted"])
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--member", default="Samuel")
    parser.add_argument("--change", default="")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    set_seed()
    ensure_dirs()
    train, val, _ = load_split()
    _, _, metrics = run_experiment(args.exp_id, train, val, model=args.model, features=args.features,
                                   C=args.C, class_weight=args.class_weight, agreement=args.agreement,
                                   change=args.change, reason=args.reason, member=args.member)
    print(metrics)


if __name__ == "__main__":
    main()
