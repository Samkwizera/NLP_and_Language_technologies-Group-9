from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "Train.csv"
TEST_CSV = DATA_DIR / "Test.csv"

ID_COL = "tweet_id"
TEXT_COL = "safe_text"
LABEL_COL = "label"
AGREEMENT_COL = "agreement"

LABELS = [-1, 0, 1]
LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}
LABEL2IDX = {lab: i for i, lab in enumerate(LABELS)}
IDX2LABEL = {i: lab for lab, i in LABEL2IDX.items()}

# real ids are 8 uppercase letters/digits, anything else is a broken row
TWEET_ID_PATTERN = r"^[A-Z0-9]{8}$"

SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SPLITS_DIR = ROOT / "splits"
SPLIT_FILE = SPLITS_DIR / "split_seed42.csv"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
ERRORS_DIR = RESULTS_DIR / "errors"
EXPERIMENTS_CSV = RESULTS_DIR / "experiments.csv"
EXPERIMENT_COLUMNS = ["exp_id", "member", "model", "change", "reason",
                      "macro_f1", "accuracy", "rmse", "takeaway"]
