import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                           DataCollatorWithPadding, Trainer, TrainingArguments)

from src import config
from src.evaluate import compute_metrics
from src.utils import log_experiment, set_seed

# add xlm-r / afroxlmr checkpoints here as you get to them
CHECKPOINTS = {
    "twitter-roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "bertweet": "vinai/bertweet-base",
    "xlm-r": "xlm-roberta-base",
    "afroxlmr": "Davlan/afro-xlmr-base",
}


def build_dataset(df, tokenizer, max_length=128, has_labels=True):
    enc = tokenizer(df[config.TEXT_COL].tolist(), truncation=True, max_length=max_length)
    data = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if has_labels:
        data["labels"] = [config.LABEL2IDX[int(l)] for l in df[config.LABEL_COL]]
    return Dataset.from_dict(data)


def hf_metrics(eval_pred):
    # wraps the team's shared compute_metrics so numbers line up with the
    # baseline/rnn results in experiments.csv (same macro_f1/accuracy/rmse)
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    y_true = [config.IDX2LABEL[i] for i in labels]
    y_pred = [config.IDX2LABEL[i] for i in preds]
    return compute_metrics(y_true, y_pred)


def run_experiment(exp_id, train_df, eval_df, model_key="twitter-roberta", epochs=3,
                   batch_size=16, lr=2e-5, max_length=128, change="", reason="",
                   takeaway="", member="Sheilla Keza", log=True):
    set_seed()
    checkpoint = CHECKPOINTS.get(model_key, model_key)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(config.LABELS))

    train_ds = build_dataset(train_df, tokenizer, max_length)
    eval_ds = build_dataset(eval_df, tokenizer, max_length)

    args = TrainingArguments(
        output_dir=str(config.RESULTS_DIR / "checkpoints" / exp_id),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=config.SEED,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=hf_metrics,
    )
    trainer.train()
    eval_result = trainer.evaluate()
    metrics = {
        "macro_f1": round(float(eval_result["eval_macro_f1"]), 4),
        "accuracy": round(float(eval_result["eval_accuracy"]), 4),
        "rmse": round(float(eval_result["eval_rmse"]), 4),
    }

    if log:
        name = f"transformer-{model_key}"
        log_experiment(exp_id=exp_id, member=member, model=name, change=change, reason=reason,
                       macro_f1=metrics["macro_f1"], accuracy=metrics["accuracy"],
                       rmse=metrics["rmse"], takeaway=takeaway)
    return trainer, tokenizer, metrics


def predict(trainer, tokenizer, df, max_length=128):
    # for confusion matrices / error export, same shape as the baseline predictions
    ds = build_dataset(df, tokenizer, max_length, has_labels=False)
    logits = trainer.predict(ds).predictions
    idx_preds = np.argmax(logits, axis=1)
    y_pred = np.array([config.IDX2LABEL[i] for i in idx_preds])
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    confidence = probs.max(axis=1)
    return y_pred, confidence