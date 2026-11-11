"""LAB 08 - Pretrained Transformer classification on IMDb.

One call of this script trains and evaluates one configuration, then appends a
row to results.csv. Running each configuration in its own process keeps the
peak memory figure of Experiment 4 clean.

Modes:
    finetune  pretrained encoder + linear head, everything trainable
    frozen    pretrained encoder frozen, only the linear head is trained
    scratch   small Transformer with random weights, same tokenizer and data

Example:
    python pretrained_classification.py --experiment exp1 --mode finetune --seed 42
"""

import argparse
import csv
import json
import os
import random
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results.csv"
RUNS_DIR = Path(os.environ.get("LAB08_RUNS", Path(tempfile.gettempdir()) / "lab08_runs"))

# Fixed splits. The data seed never changes, so every run sees the same
# validation and test sets. Only the training subset depends on the run seed.
TRAIN_SIZE = 2000
VAL_SIZE = 500
TEST_SIZE = 1000
DOMAIN_B_SIZE = 1000
DATA_SEED = 2026

FIELDS = [
    "run", "experiment", "mode", "seed", "train_fraction", "train_size", "max_length",
    "epochs_max", "best_epoch", "batch_size", "lr", "test_domain", "accuracy", "precision",
    "recall", "f1", "train_time_s", "peak_memory_mb", "n_params", "n_trainable_params",
]


# 8 threads gave the best throughput on the i9-13900H; more threads did not help
torch.set_num_threads(int(os.environ.get("LAB08_THREADS", 8)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean(text):
    return re.sub(r"<br\s*/?>", " ", text).strip()


def balanced_sample(texts, labels, size, rng):
    """Draw size // 2 examples from each class."""
    labels = np.asarray(labels)
    chosen = []
    for c in (0, 1):
        idx = np.flatnonzero(labels == c)
        chosen.extend(rng.choice(idx, size // 2, replace=False).tolist())
    rng.shuffle(chosen)
    return [texts[i] for i in chosen], labels[chosen]


def load_imdb():
    """Return fixed, class-balanced train pool, validation and test sets."""
    raw = load_dataset("stanfordnlp/imdb")
    rng = np.random.default_rng(DATA_SEED)

    seen, train_texts, train_labels = set(), [], []
    for row in raw["train"]:
        text = clean(row["text"])
        if text not in seen:
            seen.add(text)
            train_texts.append(text)
            train_labels.append(row["label"])
    pool_x, pool_y = balanced_sample(train_texts, train_labels, TRAIN_SIZE + VAL_SIZE, rng)
    # pool is already shuffled; split per class to keep both parts balanced
    val_idx = []
    for c in (0, 1):
        val_idx.extend(np.flatnonzero(pool_y == c)[: VAL_SIZE // 2].tolist())
    val_set = set(val_idx)
    train_idx = [i for i in range(len(pool_x)) if i not in val_set]
    train = ([pool_x[i] for i in train_idx], pool_y[train_idx])
    val = ([pool_x[i] for i in val_idx], pool_y[val_idx])

    test_texts, test_labels = [], []
    for row in raw["test"]:
        text = clean(row["text"])
        if text not in seen:  # no review shared with the training side
            seen.add(text)
            test_texts.append(text)
            test_labels.append(row["label"])
    test = balanced_sample(test_texts, test_labels, TEST_SIZE, rng)
    return train, val, test


def load_domain_b():
    """Amazon product reviews, used only for testing in Experiment 5."""
    stream = load_dataset("fancyzhx/amazon_polarity", split="test", streaming=True)
    stream = stream.shuffle(seed=DATA_SEED, buffer_size=10_000)
    texts, labels, per_class = [], [], {0: 0, 1: 0}
    for row in stream:
        c = row["label"]
        if per_class[c] < DOMAIN_B_SIZE // 2:
            per_class[c] += 1
            texts.append(clean(f"{row['title']}. {row['content']}"))
            labels.append(c)
        if len(texts) == DOMAIN_B_SIZE:
            break
    return texts, np.asarray(labels)


def subset_train(train, fraction, seed):
    """Class-balanced training subset. Smaller fractions are nested in larger ones."""
    texts, labels = train
    rng = np.random.default_rng(seed)
    keep = []
    for c in (0, 1):
        idx = rng.permutation(np.flatnonzero(labels == c))
        keep.extend(idx[: round(len(idx) * fraction)].tolist())
    keep = rng.permutation(keep)
    return [texts[i] for i in keep], labels[keep]


def encode(tokenizer, texts, labels, max_length):
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.as_tensor(labels))


class PretrainedClassifier(nn.Module):
    """Input IDs -> pretrained Transformer -> [CLS] -> linear head -> logits."""

    def __init__(self, model_name=MODEL_NAME, num_labels=2, dropout=0.1):
        super().__init__()
        # eager attention measured faster than sdpa on this CPU
        self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def cls_representation(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask):
        return self.head(self.dropout(self.cls_representation(input_ids, attention_mask)))


class ScratchTransformer(nn.Module):
    """Small Transformer Encoder with random initialization, as in LAB 07."""

    def __init__(self, vocab_size, pad_id, max_length, d_model=128, n_heads=4, n_layers=4,
                 d_ff=256, num_labels=2, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask):
        pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.dropout(self.tok(input_ids) + self.pos(pos))
        h = self.encoder(x, src_key_padding_mask=attention_mask == 0)
        return self.head(self.dropout(h[:, 0]))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def predict_probs(model, dataset, batch_size=64):
    model.eval()
    out = []
    for ids, mask, _ in DataLoader(dataset, batch_size=batch_size):
        out.append(torch.softmax(model(ids, mask), dim=-1))
    return torch.cat(out).numpy()


def metrics(y_true, probs):
    pred = probs.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, pred), "precision": p, "recall": r, "f1": f1}


def val_report(y_true, probs):
    """Validation accuracy and loss for one epoch."""
    picked = probs[np.arange(len(y_true)), y_true]
    return accuracy_score(y_true, probs.argmax(1)), float(-np.log(np.clip(picked, 1e-12, 1)).mean())


def train_loop(model, train_ds, val_ds, y_val, epochs, lr, batch_size, log):
    """AdamW training. Keeps the weights of the epoch with the best validation accuracy.

    Validation is class-balanced. F1 of the positive class is not used for selection
    because a model that always predicts positive already reaches F1 = 0.667.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    steps = epochs * max(1, len(train_ds) // batch_size + (len(train_ds) % batch_size > 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: max(0.0, 1 - s / steps))
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for ids, mask, y in DataLoader(train_ds, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model(ids, mask), y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(y)
            n += len(y)
        val_acc, val_loss = val_report(y_val, predict_probs(model, val_ds))
        log(f"  epoch {epoch}/{epochs} train_loss={total / n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return best_epoch


@torch.no_grad()
def extract_cls(model, dataset, batch_size=64):
    model.eval()
    feats, labels = [], []
    for ids, mask, y in DataLoader(dataset, batch_size=batch_size):
        feats.append(model.cls_representation(ids, mask))
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_frozen_head(model, train_ds, val_ds, y_val, epochs, lr, batch_size, log):
    """Frozen encoder never changes, so [CLS] vectors are computed once and reused."""
    x_train, y_train = extract_cls(model, train_ds)
    x_val, _ = extract_cls(model, val_ds)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_epoch, best_state = -1.0, 0, None

    for epoch in range(1, epochs + 1):
        model.head.train()
        total = 0.0
        for xb, yb in DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = loss_fn(model.head(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(yb)
        with torch.no_grad():
            model.head.eval()
            val_acc, val_loss = val_report(y_val, torch.softmax(model.head(x_val), -1).numpy())
        if epoch % 5 == 0 or epoch == 1:
            log(f"  epoch {epoch}/{epochs} train_loss={total / len(y_train):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.head.state_dict().items()}

    model.head.load_state_dict(best_state)
    return best_epoch


def done_runs():
    if not RESULTS_CSV.exists():
        return set()
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return {row["run"] for row in csv.DictReader(f)}


def append_rows(rows):
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


DEFAULTS = {
    # mode: (epochs, learning rate)
    "finetune": (2, 2e-5),
    "frozen": (30, 1e-3),
    "scratch": (10, 5e-4),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--mode", choices=DEFAULTS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--domain-b", action="store_true", help="also test on Amazon reviews")
    args = ap.parse_args()

    run = f"{args.mode}_f{args.fraction:g}_T{args.max_length}_s{args.seed}"
    if run in done_runs():
        print(f"{run}: already in results.csv, skipped")
        return

    log = lambda msg: print(msg, flush=True)
    epochs, lr = DEFAULTS[args.mode]
    set_seed(args.seed)
    train, val, test = load_imdb()
    train =subset_train(train, args.fraction, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = encode(tokenizer, *train, args.max_length)
    val_ds = encode(tokenizer, *val, args.max_length)
    test_ds = encode(tokenizer, *test, args.max_length)

    if args.mode == "scratch":
        model = ScratchTransformer(tokenizer.vocab_size, tokenizer.pad_token_id, args.max_length)
    else:
        model = PretrainedClassifier()
        if args.mode == "frozen":
            model.freeze_encoder()
    total, trainable = count_params(model)
    log(f"{run}: train={len(train[0])} params={total:,} trainable={trainable:,}")

    start = time.perf_counter()
    if args.mode == "frozen":
        best_epoch = train_frozen_head(model, train_ds, val_ds, val[1], epochs, lr, args.batch_size, log)
    else:
        best_epoch = train_loop(model, train_ds, val_ds, val[1], epochs, lr, args.batch_size, log)
    train_time = time.perf_counter() - start

    test_probs = predict_probs(model, test_ds)
    # peak_wset exists on Windows only; other systems fall back to current RSS
    mem = psutil.Process().memory_info()
    peak_mb = getattr(mem, "peak_wset", mem.rss) / 1e6

    base = {
        "run": run, "experiment": args.experiment, "mode": args.mode, "seed": args.seed,
        "train_fraction": args.fraction, "train_size": len(train[0]),
        "max_length": args.max_length, "epochs_max": epochs, "best_epoch": best_epoch,
        "batch_size": args.batch_size, "lr": lr, "train_time_s": round(train_time, 1),
        "peak_memory_mb": round(peak_mb), "n_params": total, "n_trainable_params": trainable,
    }
    rows = [{**base, "test_domain": "imdb", **{k: round(v, 4) for k, v in metrics(test[1], test_probs).items()}}]
    log(f"  test imdb {rows[0]['accuracy']=} {rows[0]['f1']=}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    saved = [("imdb", test[0], test[1], test_probs)]
    if args.domain_b:
        b_texts, b_labels = load_domain_b()
        b_probs = predict_probs(model, encode(tokenizer, b_texts, b_labels, args.max_length))
        row = {**base, "test_domain": "amazon", **{k: round(v, 4) for k, v in metrics(b_labels, b_probs).items()}}
        rows.append(row)
        log(f"  test amazon {row['accuracy']=} {row['f1']=}")
        saved.append(("amazon", b_texts, b_labels, b_probs))

    # Per-example predictions for the confusion matrix, confidence and error analysis
    for domain, texts, labels, probs in saved:
        with open(RUNS_DIR / f"{run}_{domain}.jsonl", "w", encoding="utf-8") as f:
            for text, y, p in zip(texts, labels, probs):
                f.write(json.dumps({"text": text, "label": int(y), "p_pos": float(p[1])}) + "\n")

    append_rows(rows)


if __name__ == "__main__":
    main()
