"""Topic classification on Yahoo! Answers: preprocessing, baselines, fine-tuning, benchmark.

One file, five stages. Run them in order, or run `all` to do everything:

    python yahoo_topics_finetune.py prepare      # download, clean, split -> data/
    python yahoo_topics_finetune.py baselines    # TF-IDF baselines      -> results/
    python yahoo_topics_finetune.py train        # fine-tune one model   -> results/
    python yahoo_topics_finetune.py benchmark    # sweep + build the table
    python yahoo_topics_finetune.py all          # all of the above

    python yahoo_topics_finetune.py train --model deberta-v3-base --max_len 256 --seed 1
    python yahoo_topics_finetune.py benchmark --quick        # one seed each, faster
    python yahoo_topics_finetune.py selftest                 # no GPU or data needed

Every Config field below is also a flag of the same name: --lr 3e-5, --epochs 3,
--no-clean_html, and so on.

Outputs:
    data/raw/*.jsonl.gz      exactly what was downloaded, uncleaned
    data/clean/*.jsonl.gz    what the models train on, plus stats.json
    results/*.json           one file per model and seed
    results/benchmark.md     the comparison table, ready to paste into a README
    runs/<model>_seed<n>/    checkpoints (best.pt and a resumable state.pt)
"""

# %% imports
import argparse
import dataclasses
import gzip
import html
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# The fast tokenizer runs its own threads and the DataLoader workers fork, which
# produces a warning. We tokenize inside the workers anyway, so turn its threading off.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

HF_DATASET = "community-datasets/yahoo_answers_topics"

# __file__ does not exist when these cells are pasted into a notebook, so look it up
# defensively. If the file really is on disk the sweep can run each stage as its own
# subprocess; in a notebook it falls back to running them in-process.
SCRIPT_PATH = Path(globals().get("__file__", "yahoo_topics_finetune.py")).resolve()
SCRIPT_NAME = SCRIPT_PATH.name

# The 10 Yahoo! Answers top-level categories in the dataset's own label order. Hardcoded
# so a results file is readable on its own; `prepare` checks the order still matches.
CLASS_NAMES = [
    "Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
    "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
    "Family & Relationships", "Politics & Government",
]

# The three text fields, concatenated in this order (as Zhang et al. 2015 did).
# Order matters: truncation drops the tail, so the answer is what gets cut first.
TEXT_FIELDS = ("question_title", "question_content", "best_answer")

# Short name -> (architecture, specific model). The two are kept apart deliberately:
# BERT is an architecture, bert-base-uncased is one model of it.
MODELS = {
    "bert-base": ("BERT", "bert-base-uncased"),
    "bert-large": ("BERT", "bert-large-uncased"),
    "roberta-base": ("RoBERTa", "roberta-base"),
    "roberta-large": ("RoBERTa", "roberta-large"),
    "deberta-v3-xsmall": ("DeBERTa-v3", "microsoft/deberta-v3-xsmall"),
    "deberta-v3-small": ("DeBERTa-v3", "microsoft/deberta-v3-small"),
    "deberta-v3-base": ("DeBERTa-v3", "microsoft/deberta-v3-base"),
    "deberta-v3-large": ("DeBERTa-v3", "microsoft/deberta-v3-large"),
}

# Published test accuracy (%) on this exact benchmark: 10 classes, 1.4M train / 60k test.
# Most of these papers report error rate, so accuracy here is 100 - error.
LITERATURE = [
    {"method": "Bag-of-words + logistic regression", "architecture": "linear",
     "model": "BoW", "accuracy": 68.89,
     "source": "Zhang et al. 2015, arXiv:1509.01626 (err 31.11)"},
    {"method": "n-grams TF-IDF + logistic regression", "architecture": "linear",
     "model": "ngrams-TFIDF", "accuracy": 68.51,
     "source": "Zhang et al. 2015, arXiv:1509.01626 (err 31.49)"},
    {"method": "Bag-of-words TF-IDF + logistic regression", "architecture": "linear",
     "model": "BoW-TFIDF", "accuracy": 71.04,
     "source": "Zhang et al. 2015, arXiv:1509.01626 (err 28.96)"},
    {"method": "LSTM (word-level, word2vec)", "architecture": "LSTM",
     "model": "LSTM", "accuracy": 70.84,
     "source": "Zhang et al. 2015, arXiv:1509.01626 (err 29.16)"},
    {"method": "Character-level CNN (large, thesaurus aug.)", "architecture": "char-CNN",
     "model": "char-CNN-large", "accuracy": 71.20,
     "source": "Zhang et al. 2015, arXiv:1509.01626 (err 28.80)"},
    {"method": "fastText (bigrams)", "architecture": "fastText",
     "model": "fastText h=10 bigram", "accuracy": 72.30,
     "source": "Joulin et al. 2016, arXiv:1607.01759"},
    {"method": "Very Deep CNN (29 layers)", "architecture": "VDCNN",
     "model": "VDCNN-29", "accuracy": 73.43,
     "source": "Conneau et al. 2016, arXiv:1606.01781 (err 26.57)"},
    {"method": "Hierarchical Attention Network", "architecture": "HAN",
     "model": "HAN", "accuracy": 75.80,
     "source": "Yang et al. 2016, via arXiv:1905.05583 (err 24.20)"},
    {"method": "Deep Pyramid CNN", "architecture": "DPCNN",
     "model": "DPCNN", "accuracy": 76.10,
     "source": "Johnson & Zhang 2017, via arXiv:1905.05583 (err 23.90)"},
    {"method": "BERT fine-tuned", "architecture": "BERT",
     "model": "bert-base-uncased", "accuracy": 77.58,
     "source": "Sun et al. 2019, arXiv:1905.05583 (err 22.42, 512 tokens)"},
    {"method": "BERT + in-domain further pre-training", "architecture": "BERT",
     "model": "bert-base-uncased + IDPT", "accuracy": 78.14,
     "source": "Sun et al. 2019, arXiv:1905.05583 (err 21.86)"},
]


# %% config
@dataclass
class Config:
    # --- model and training
    model: str = "roberta-base"
    max_len: int = 128                 # 256 truncates fewer answers and costs ~2x the time
    epochs: int = 2
    batch_size: int = 32
    grad_accum: int = 1                # >1 imitates a bigger batch on a smaller card
    lr: float = 2e-5
    layer_decay: float = 1.0           # <1 gives lower layers a smaller lr (Sun et al.: 0.95)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    scheduler: str = "linear"          # linear | cosine
    label_smoothing: float = 0.0
    max_grad_norm: float = 1.0
    precision: str = "auto"            # auto | bf16 | fp16 | fp32
    # --- data (used by `prepare`)
    clean_html: bool = True            # the raw text is full of <br /> and &#xd;
    train_size: int = 100_000          # 0 = everything left after the val split (~1.39M)
    val_size: int = 10_000             # taken out of train; the dataset has no val split
    test_size: int = 0                 # 0 = the full official 60k test set
    # --- plumbing
    resume: bool = True                # carry on from a saved state.pt if there is one
    log_every: int = 100               # optimizer steps between printed loss lines
    num_workers: int = 2
    seed: int = 42
    data_dir: str = "data"
    results_dir: str = "results"
    runs_dir: str = "runs"
    limit_train: int = 0               # cap the prepared train split; 0 = use all of it
    smoke: bool = False                # tiny everything, just to check the plumbing


def add_config_flags(parser):
    """Turn every Config field into a command-line flag of the same name."""
    for field in dataclasses.fields(Config):
        if isinstance(field.default, bool):
            # gives both --clean_html and --no-clean_html
            parser.add_argument(f"--{field.name}", default=field.default,
                                action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(f"--{field.name}", default=field.default,
                                type=type(field.default))
    return parser


def parse_cli(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage")
    for name in ("prepare", "baselines", "train", "all", "selftest"):
        add_config_flags(stages.add_parser(name))
    bench = add_config_flags(stages.add_parser("benchmark"))
    bench.add_argument("--models", nargs="+", default=["bert-base", "roberta-base",
                                                       "deberta-v3-base"])
    bench.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    bench.add_argument("--quick", action="store_true", help="one seed per model")
    bench.add_argument("--report_only", action="store_true", help="rebuild the table only")
    # parse_known_args so a notebook kernel's own arguments do not break it
    args, _ = parser.parse_known_args(argv)
    extras = {k: v for k, v in vars(args).items()
              if k in ("stage", "models", "seeds", "quick", "report_only")}
    fields = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in vars(args).items() if k in fields})
    if cfg.smoke:   # one tiny pass over tiny data, for checking the plumbing works
        cfg = dataclasses.replace(cfg, train_size=2000, val_size=500, test_size=1000,
                                  epochs=1, log_every=10)
    return cfg, extras


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# %% dataset files
# One JSON object per line, gzipped. Chosen over CSV because the text contains commas,
# quotes and newlines, and over Parquet so the files stay readable with the standard
# library alone.
def write_jsonl(path, rows):
    """mtime=0 and filename="" because gzip otherwise writes the current time and the
    file's own name into its header, so two runs of the same pipeline would produce
    different bytes and git would report a change that is not really there."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    return path.stat().st_size


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run `python {SCRIPT_NAME} prepare` first.")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def human_size(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.0f}{unit}" if unit == "B" else f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024


def save_result(cfg, tag, payload):
    path = Path(cfg.results_dir) / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


# %% stage 1: preprocessing
BR_TAG = re.compile(r"<\s*br\s*/?\s*>", re.I)
SHORT_TAG = re.compile(r"<[^>]{1,40}>")

# A word piece is shorter than a word, so a word count understates the tokenizer's count.
# 1.3 is the usual rule of thumb for English, used only for the truncation estimate below.
PIECES_PER_WORD = 1.3


def clean_text(text):
    """Undo HTML escaping, drop <br> and other stray tags, collapse whitespace.

    The corpus is scraped 2005-era web text, so it carries &#xd;, &amp; and <br />
    markers plus literal newlines. None of that indicates the topic, and inside a
    128-token window it is pure waste.
    """
    text = html.unescape(text or "")
    text = BR_TAG.sub(" ", text)
    text = SHORT_TAG.sub(" ", text)     # a long <...> run is probably real text, so leave it
    return " ".join(text.split())       # collapses newlines, tabs and runs of spaces


def join_fields(row, clean=True):
    """Build the single input string for one example."""
    parts = []
    for field in TEXT_FIELDS:
        value = row.get(field) or ""
        value = clean_text(value) if clean else " ".join(value.split())
        if value:
            parts.append(value)
    return " ".join(parts)


def download_raw(cfg):
    """Fetch the dataset and save the exact rows we will use, uncleaned."""
    from datasets import load_dataset

    print(f"downloading {HF_DATASET} ...")
    data = load_dataset(HF_DATASET)
    names = data["train"].features["topic"].names
    if names != CLASS_NAMES:
        # if the upstream label order ever changed, every saved result would silently
        # mean something different, so this has to be an error rather than a warning
        raise SystemExit(f"label order changed upstream!\n  expected {CLASS_NAMES}\n  got {names}")

    train, test = data["train"], data["test"]
    if cfg.train_size and cfg.train_size < len(train):
        train = train.shuffle(seed=cfg.seed).select(range(cfg.train_size))
    if cfg.test_size and cfg.test_size < len(test):
        test = test.shuffle(seed=cfg.seed).select(range(cfg.test_size))

    keep = list(TEXT_FIELDS) + ["topic"]
    for name, split in (("train", train), ("test", test)):
        rows = [{k: row[k] for k in keep} for row in split]
        size = write_jsonl(Path(cfg.data_dir) / "raw" / f"{name}.jsonl.gz", rows)
        print(f"  {cfg.data_dir}/raw/{name}.jsonl.gz  {len(rows):,} rows  {human_size(size)}")


def clean_split(rows, clean=True):
    """Join and clean the text, then drop what is unusable.

    Nothing here looks at the label, so cleaning cannot leak label information
    into the text.
    """
    kept, seen = [], set()
    dropped_empty = dropped_duplicate = 0
    raw_chars = clean_chars = 0

    for row in rows:
        raw_chars += sum(len(row.get(f) or "") for f in TEXT_FIELDS)
        text = join_fields(row, clean)
        if not text:
            dropped_empty += 1
            continue
        if text in seen:            # the corpus genuinely contains repeated questions
            dropped_duplicate += 1
            continue
        seen.add(text)
        clean_chars += len(text)
        kept.append({"text": text, "label": int(row["topic"])})

    return kept, {"dropped_empty": dropped_empty, "dropped_duplicate": dropped_duplicate,
                  "raw_chars": raw_chars, "clean_chars": clean_chars}


def stratified_split(rows, n_held, seed):
    """Hold out about n_held rows with the class balance preserved.

    Done by hand rather than with sklearn so the behaviour is obvious: shuffle
    within each class, then take a proportional slice of each.
    """
    by_class = {}
    for i, row in enumerate(rows):
        by_class.setdefault(row["label"], []).append(i)

    rng = random.Random(seed)
    held = []
    for label in sorted(by_class):
        idx = by_class[label][:]
        rng.shuffle(idx)
        held.extend(idx[:round(n_held * len(idx) / len(rows))])

    held_set = set(held)
    return ([r for i, r in enumerate(rows) if i not in held_set],
            [rows[i] for i in sorted(held_set)])


def split_stats(rows):
    words = np.array([len(r["text"].split()) for r in rows])
    chars = np.array([len(r["text"]) for r in rows])
    counts = Counter(r["label"] for r in rows)
    pieces = words * PIECES_PER_WORD
    return {
        "rows": len(rows),
        "class_counts": {CLASS_NAMES[k]: counts.get(k, 0) for k in range(len(CLASS_NAMES))},
        "support": [counts.get(k, 0) for k in range(len(CLASS_NAMES))],
        "words": {"mean": float(words.mean()), "median": float(np.median(words)),
                  "p95": float(np.percentile(words, 95)), "max": int(words.max())},
        "chars": {"mean": float(chars.mean()), "median": float(np.median(chars))},
        # share of examples a given max_len would cut, estimated from word counts
        "estimated_truncated": {str(limit): float((pieces > limit).mean())
                                for limit in (128, 256, 512)},
    }


def stage_prepare(cfg, from_raw=False):
    """Stage 1: Hugging Face -> data/raw (untouched) -> data/clean (what we train on).

    The two steps are separate on purpose: `--from_raw` re-runs only the cleaning, on
    the saved raw files, with no network, and produces byte-identical output.
    """
    raw_dir = Path(cfg.data_dir) / "raw"
    clean_dir = Path(cfg.data_dir) / "clean"
    if not from_raw:
        download_raw(cfg)
    elif not (raw_dir / "train.jsonl.gz").exists():
        raise SystemExit(f"--from_raw needs {raw_dir}/train.jsonl.gz; run prepare without it.")

    print("\ncleaning ...")
    report, splits = {}, {}
    for name in ("train", "test"):
        rows = read_jsonl(raw_dir / f"{name}.jsonl.gz")
        cleaned, info = clean_split(rows, cfg.clean_html)
        splits[name], report[name] = cleaned, info
        shrink = 100 * (1 - info["clean_chars"] / max(info["raw_chars"], 1))
        print(f"  {name}: {len(rows):,} -> {len(cleaned):,} rows "
              f"({info['dropped_empty']:,} empty, {info['dropped_duplicate']:,} duplicate), "
              f"text {shrink:.1f}% smaller after cleaning")

    train, val = stratified_split(splits["train"], cfg.val_size, cfg.seed)
    test = splits["test"]

    # Train/test overlap would quietly inflate the test score, so measure it.
    overlap = len(set(r["text"] for r in train) & set(r["text"] for r in test))
    print(f"\n  train/test text overlap: {overlap} examples"
          + ("  <- investigate before trusting the test score" if overlap else "  (clean)"))

    print()
    sizes = {}
    for name, rows in (("train", train), ("val", val), ("test", test)):
        sizes[name] = write_jsonl(clean_dir / f"{name}.jsonl.gz", rows)
        print(f"  {clean_dir}/{name}.jsonl.gz  {len(rows):,} rows  {human_size(sizes[name])}")

    stats = {
        "source": HF_DATASET, "class_names": CLASS_NAMES,
        "text_fields_concatenated": list(TEXT_FIELDS),
        "config": dataclasses.asdict(cfg), "cleaning": report,
        "train_test_overlap": overlap, "file_bytes": sizes,
        "splits": {n: split_stats(r) for n, r in
                   (("train", train), ("val", val), ("test", test))},
    }
    stats["metric_choice"] = metric_choice(stats["splits"]["test"]["support"])
    (clean_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    ts = stats["splits"]["test"]
    print(f"\n  test set: {min(ts['support']):,}-{max(ts['support']):,} examples per class "
          f"-> primary metric: {stats['metric_choice']['primary']}")
    print(f"  median length {ts['words']['median']:.0f} words; at max_len=128 about "
          f"{ts['estimated_truncated']['128'] * 100:.0f}% of examples are truncated, "
          f"at 256 about {ts['estimated_truncated']['256'] * 100:.0f}%")
    print(f"\nwrote {clean_dir}/stats.json  ({human_size(sum(sizes.values()))} of clean data)")
    return stats


def load_prepared(cfg):
    """Read the three splits written by `prepare`."""
    clean_dir = Path(cfg.data_dir) / "clean"
    splits = [read_jsonl(clean_dir / f"{n}.jsonl.gz") for n in ("train", "val", "test")]
    if cfg.limit_train:
        splits[0] = splits[0][:cfg.limit_train]
    return splits


# %% metrics
def confusion_matrix(true_labels, predictions, n_classes=len(CLASS_NAMES)):
    """Rows are the true class, columns the predicted class."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true, predicted in zip(true_labels, predictions):
        matrix[true, predicted] += 1
    return matrix


def metrics_from_confusion(matrix):
    """Accuracy, per-class precision/recall/F1, and the three F1 averages.

    All derived from the one confusion matrix, so the numbers cannot disagree.
    """
    matrix = np.asarray(matrix)
    correct = np.diag(matrix)
    predicted_per_class = matrix.sum(axis=0)
    actual_per_class = matrix.sum(axis=1)

    # clamping the denominator at 1 only ever matters for a class that never appears
    precision = correct / np.maximum(predicted_per_class, 1)
    recall = correct / np.maximum(actual_per_class, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    accuracy = float(correct.sum() / matrix.sum())
    return {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        # with one label per example, micro-F1 equals accuracy; reported so the equality
        # is visible rather than something a reader has to take on trust
        "micro_f1": accuracy,
        "weighted_f1": float((f1 * actual_per_class).sum() / actual_per_class.sum()),
        "precision": precision.tolist(), "recall": recall.tolist(),
        "f1": f1.tolist(), "support": actual_per_class.tolist(),
    }


def metric_choice(support):
    """Which metric should lead, judged from the actual class counts.

    A function rather than a sentence in the README, so the justification comes from
    the data we really have and changes if the data does.
    """
    support = np.asarray(support)
    imbalance = support.max() / max(support.min(), 1)
    balanced = bool(imbalance < 1.5)
    reason = (f"The test set has {support.min():,}-{support.max():,} examples per class "
              f"(imbalance ratio {imbalance:.2f}). ")
    if balanced:
        reason += ("That is effectively balanced, so accuracy is not inflated by a majority "
                   "class and is the right headline metric. It is also what every published "
                   "result on this benchmark reports, which is what makes comparison possible. "
                   "Macro-F1 is reported alongside it as a safeguard: because it weights every "
                   "class equally, a model that quietly abandoned one class would show up there "
                   "while accuracy stayed respectable.")
    else:
        reason += ("That is skewed enough that accuracy would reward predicting the common "
                   "classes, so macro-F1 leads: it weights every class equally and a neglected "
                   "class drags it down. Accuracy is still reported for comparability.")
    reason += (" Micro-F1 is identical to accuracy in single-label classification, so it adds "
               "nothing here. AUC is not used because this is a 10-class decision reported at a "
               "fixed operating point, not a ranking problem with a tunable threshold.")
    return {"primary": "accuracy" if balanced else "macro-F1", "balanced": balanced,
            "imbalance_ratio": float(imbalance), "justification": reason}


# %% reports
def print_per_class(results):
    print(f"\n{'class':<26}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>8}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<26}{results['precision'][i] * 100:7.1f}{results['recall'][i] * 100:7.1f}"
              f"{results['f1'][i] * 100:7.1f}{results['support'][i]:8d}")


def print_confusions(matrix, top=8):
    """The mistakes actually made, biggest first.

    More useful than the raw 10x10 grid: it names the pairs the model cannot separate,
    and on this dataset those are mostly genuinely ambiguous questions.
    """
    matrix = np.asarray(matrix)
    mistakes = []
    for true in range(len(CLASS_NAMES)):
        for predicted in range(len(CLASS_NAMES)):
            if true != predicted and matrix[true, predicted]:
                share = matrix[true, predicted] / matrix[true].sum()
                mistakes.append((int(matrix[true, predicted]), share, true, predicted))
    mistakes.sort(reverse=True)

    print("\nmost common mistakes (true -> predicted)")
    for count, share, true, predicted in mistakes[:top]:
        print(f"  {count:5d}  {share * 100:4.1f}% of {CLASS_NAMES[true]:<24} -> "
              f"{CLASS_NAMES[predicted]}")


def fmt(mean, std):
    if mean is None:
        return "--"
    return f"{mean:.2f}" + (f" ± {std:.2f}" if std else "")


def load_results(cfg):
    rows = []
    for path in sorted(Path(cfg.results_dir).glob("*.json")):
        if path.name.startswith("benchmark"):
            continue
        payload = json.loads(path.read_text())
        if "metrics" in payload:
            rows.append(payload)
    return rows


def group_runs(rows):
    """Collapse the seeds of one configuration into a single row with mean and spread."""
    groups = {}
    for row in rows:
        key = (row["method"], row["architecture"], row["model"], row.get("max_len"))
        groups.setdefault(key, []).append(row)

    summary = []
    for (method, architecture, model, max_len), runs in groups.items():
        acc = np.array([r["metrics"]["accuracy"] for r in runs]) * 100
        mf1 = np.array([r["metrics"]["macro_f1"] for r in runs]) * 100
        summary.append({
            "method": method, "architecture": architecture, "model": model,
            "family": "own", "max_len": max_len, "kind": runs[0].get("kind", "unknown"),
            "n_train": runs[0].get("n_train"),
            "seeds": sorted(r["seed"] for r in runs),
            "accuracy_mean": float(acc.mean()),
            "accuracy_std": float(acc.std(ddof=1)) if len(runs) > 1 else None,
            "macro_f1_mean": float(mf1.mean()),
            "macro_f1_std": float(mf1.std(ddof=1)) if len(runs) > 1 else None,
            "train_minutes_mean": float(np.mean([r.get("train_minutes", 0) for r in runs])),
        })
    return sorted(summary, key=lambda r: -r["accuracy_mean"])


def build_markdown(summary, stats, failures=()):
    lines = ["# Benchmark: Yahoo! Answers topic classification", ""]
    if stats:
        s = stats["splits"]
        lines += [
            f"- **Dataset**: `{stats['source']}`, {len(stats['class_names'])} classes",
            f"- **Train / val / test**: {s['train']['rows']:,} / {s['val']['rows']:,} / "
            f"{s['test']['rows']:,}",
            f"- **Fields concatenated**: {', '.join(stats['text_fields_concatenated'])}",
            f"- **Train/test text overlap**: {stats['train_test_overlap']} examples",
            "", "## Evaluation metric", "",
            f"**Primary metric: {stats['metric_choice']['primary']}.** "
            + stats["metric_choice"]["justification"], "",
        ]

    lines += [
        "## Results", "",
        "`±` is the standard deviation over random seeds. Rows marked *cited* are published "
        "numbers, not our runs: they used the full 1.4M training examples (and up to 512 "
        "tokens for the BERT rows), so they are an upper reference rather than a "
        "like-for-like comparison.", "",
        "| # | Method | Architecture | Model | Accuracy (%) | Macro-F1 (%) | Source |",
        "|---|--------|--------------|-------|--------------|--------------|--------|",
    ]
    combined = [dict(r, sort_key=r["accuracy_mean"]) for r in summary]
    combined += [dict(r, family="cited", sort_key=r["accuracy"]) for r in LITERATURE]
    combined.sort(key=lambda r: -r["sort_key"])

    for i, row in enumerate(combined, 1):
        if row["family"] == "own":
            n = len(row["seeds"])
            source = (f"**this work** ({n} seed{'s' if n > 1 else ''}"
                      + (f", max_len {row['max_len']}" if row["max_len"] else "") + ")")
            acc = fmt(row["accuracy_mean"], row["accuracy_std"])
            mf1 = fmt(row["macro_f1_mean"], row["macro_f1_std"])
        else:
            source, acc, mf1 = f"*cited* -- {row['source']}", f"{row['accuracy']:.2f}", "--"
        lines.append(f"| {i} | {row['method']} | {row['architecture']} | `{row['model']}` | "
                     f"{acc} | {mf1} | {source} |")

    lines += ["", "## Our runs in detail", "",
              "| Method | Model | Train rows | max_len | Seeds | Accuracy (%) | Macro-F1 (%) "
              "| Train (min) |",
              "|--------|-------|-----------:|--------:|-------|--------------|--------------"
              "|-------------|"]
    for row in summary:
        lines.append(f"| {row['method']} | `{row['model']}` | {row['n_train']:,} | "
                     f"{row['max_len'] or '--'} | {', '.join(map(str, row['seeds']))} | "
                     f"{fmt(row['accuracy_mean'], row['accuracy_std'])} | "
                     f"{fmt(row['macro_f1_mean'], row['macro_f1_std'])} | "
                     f"{row['train_minutes_mean']:.1f} |")

    if failures:
        lines += ["", f"> Incomplete: {', '.join(failures)} did not finish.", ""]
    lines += ["", f"Regenerate with `python {SCRIPT_NAME} benchmark --report_only`.", ""]
    return "\n".join(lines)


# %% stage 2a: classical baselines
def baseline_specs(max_features, seed):
    """The vanilla, pre-neural methods. They set the floor a transformer has to clear,
    and the majority-class row shows what a model that learned nothing would score."""
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    def tfidf(ngram_range=(1, 2), analyzer="word"):
        return TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer,
                               max_features=max_features, min_df=2, sublinear_tf=True,
                               lowercase=True, strip_accents="unicode")

    return {
        "majority": ("Majority class (no learning)", "trivial", "most_frequent",
                     lambda: DummyClassifier(strategy="most_frequent")),
        "tfidf_nb": ("TF-IDF + Complement Naive Bayes", "Naive Bayes",
                     "tfidf-word-1-2gram + ComplementNB",
                     lambda: make_pipeline(tfidf(), ComplementNB())),
        # lbfgs, not liblinear: liblinear is binary-only and newer scikit-learn refuses
        # multiclass with it rather than quietly doing one-vs-rest. A ConvergenceWarning
        # at this scale is harmless for a baseline.
        "tfidf_logreg": ("TF-IDF + logistic regression", "linear",
                         "tfidf-word-1-2gram + LogisticRegression",
                         lambda: make_pipeline(tfidf(), LogisticRegression(
                             solver="lbfgs", max_iter=400, C=4.0, tol=1e-3, random_state=seed))),
        "tfidf_svm": ("TF-IDF + linear SVM", "linear", "tfidf-word-1-2gram + LinearSVC",
                      lambda: make_pipeline(tfidf(), LinearSVC(C=0.5, random_state=seed))),
        "tfidf_char_svm": ("Character TF-IDF + linear SVM", "linear",
                           "tfidf-char-3-5gram + LinearSVC",
                           lambda: make_pipeline(tfidf((3, 5), "char_wb"),
                                                 LinearSVC(C=0.5, random_state=seed))),
    }


def stage_baselines(cfg, max_features=300_000):
    """Stage 2a. scikit-learn only, so this runs on a CPU in about a minute."""
    train, _, test = load_prepared(cfg)
    print(f"train {len(train):,} | test {len(test):,} | {len(CLASS_NAMES)} classes")
    texts, labels = [r["text"] for r in train], [r["label"] for r in train]
    test_texts, test_labels = [r["text"] for r in test], [r["label"] for r in test]

    best = None
    for key, (method, architecture, model_name, build) in baseline_specs(
            max_features, cfg.seed).items():
        print(f"\n--- {method}")
        model = build()
        started = time.time()
        model.fit(texts, labels)
        minutes = (time.time() - started) / 60

        matrix = confusion_matrix(test_labels, model.predict(test_texts))
        metrics = metrics_from_confusion(matrix)
        print(f"test accuracy {metrics['accuracy'] * 100:.2f}  "
              f"macro-f1 {metrics['macro_f1'] * 100:.2f}  (fit in {minutes:.1f} min)")

        payload = {"method": method, "architecture": architecture, "model": model_name,
                   "family": "own", "kind": "classical", "seed": cfg.seed,
                   "n_train": len(train), "n_test": len(test), "max_len": None,
                   "train_minutes": minutes, "metrics": metrics,
                   "confusion": matrix.tolist(), "class_names": CLASS_NAMES}
        save_result(cfg, f"{key}_seed{cfg.seed}", payload)
        if best is None or metrics["accuracy"] > best[0]["accuracy"]:
            best = (metrics, matrix, method)

    metrics, matrix, method = best
    print(f"\nbest classical baseline: {method} at {metrics['accuracy'] * 100:.2f}% accuracy")
    print_per_class(metrics)
    print_confusions(matrix)
    return best


# %% stage 2b: the model, optimizer and mixed precision
def build_model(model_id):
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(CLASS_NAMES), id2label=dict(enumerate(CLASS_NAMES)),
        label2id={name: i for i, name in enumerate(CLASS_NAMES)})


def layer_depth(name, n_layers):
    """How deep a parameter sits: 0 for embeddings, 1..n for encoder layers, n+1 for the head."""
    match = re.search(r"\.layer\.(\d+)\.", name)
    if match:
        return int(match.group(1)) + 1
    if "embeddings" in name or "encoder.LayerNorm" in name:
        return 0
    return n_layers + 1


def make_optimizer(model, cfg):
    """Two standard tricks: no weight decay on biases and LayerNorms, and -- if
    layer_decay is below 1 -- a smaller learning rate the closer a layer is to the input.

    Lower layers hold general lexical and syntactic features worth preserving; upper
    layers are the ones that need to specialise.
    """
    import torch
    n_layers = model.config.num_hidden_layers
    groups = {}
    for name, param in model.named_parameters():
        depth = layer_depth(name, n_layers)
        lr = cfg.lr * cfg.layer_decay ** (n_layers + 1 - depth)
        decay = 0.0 if name.endswith("bias") or "LayerNorm.weight" in name else cfg.weight_decay
        groups.setdefault((lr, decay), []).append(param)

    param_groups = [{"params": params, "lr": lr, "weight_decay": decay}
                    for (lr, decay), params in groups.items()]
    # betas and eps follow the RoBERTa paper rather than the PyTorch defaults
    return torch.optim.AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.98), eps=1e-6)


def lr_scale(step, warmup, total_steps, kind="linear"):
    """The learning-rate multiplier at a given step: 0 -> 1 over warmup, then down to 0.

    Warmup matters because the classification head starts random: its early gradients
    are noise, and taking full-size steps on noise damages the pretrained weights.
    Decaying to zero at the end lets the model settle into a minimum instead of
    bouncing around it.
    """
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    if kind == "cosine":
        return 0.5 * (1 + math.cos(math.pi * progress))
    return max(0.0, 1 - progress)


def make_scheduler(optimizer, cfg, total_steps):
    import torch
    warmup = int(cfg.warmup_ratio * total_steps)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_scale(step, warmup, total_steps, cfg.scheduler))


def pick_precision(cfg, device):
    """Pick the dtype for autocast.

    bf16 keeps fp32's exponent range, so gradients cannot underflow and no loss scaling
    is needed. fp16 has a narrower range and needs a GradScaler.
    """
    import torch
    if device.type != "cuda" or cfg.precision == "fp32":
        return torch.float32
    if cfg.precision == "bf16" or (cfg.precision == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


class TextDataset:
    """Holds already-cleaned text. Tokenizing happens in the collate function, so each
    batch is padded to its own longest member instead of every example to max_len."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]["text"], self.rows[i]["label"]


class Collator:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        import torch
        texts, labels = zip(*batch)
        encoded = self.tokenizer(list(texts), truncation=True, max_length=self.max_len,
                                 padding=True, pad_to_multiple_of=8, return_tensors="pt")
        encoded["labels"] = torch.tensor(labels)
        return encoded


def make_loader(rows, cfg, tokenizer, shuffle, device):
    from torch.utils.data import DataLoader
    return DataLoader(TextDataset(rows),
                      batch_size=cfg.batch_size if shuffle else cfg.batch_size * 2,
                      shuffle=shuffle, num_workers=cfg.num_workers,
                      collate_fn=Collator(tokenizer, cfg.max_len),
                      pin_memory=device.type == "cuda")


def evaluate(model, loader, device, amp_dtype):
    import torch
    import torch.nn.functional as F
    model.eval()
    all_true, all_pred, loss_total, n_seen = [], [], 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_dtype != torch.float32):
                logits = model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
            logits = logits.float()
            # summing then dividing once gives a true per-example mean; averaging
            # per-batch means would over-weight the smaller final batch
            loss_total += F.cross_entropy(logits, batch["labels"], reduction="sum").item()
            n_seen += len(batch["labels"])
            all_pred.append(logits.argmax(dim=-1).cpu())
            all_true.append(batch["labels"].cpu())

    matrix = confusion_matrix(torch.cat(all_true).numpy(), torch.cat(all_pred).numpy())
    results = metrics_from_confusion(matrix)
    results["loss"] = loss_total / n_seen
    return results, matrix


# %% training
def save_atomically(obj, path):
    """Write to a temporary name then rename. Renaming is atomic, so a crash mid-write
    cannot leave a half-written checkpoint we would later resume from."""
    import torch
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_state(path, cfg, device):
    import torch
    if not (cfg.resume and path.exists()):
        return None
    # our own file, holding plain Python objects, so weights_only has to be off
    state = torch.load(path, map_location=device, weights_only=False)
    for key in ("model", "batch_size", "grad_accum", "max_len", "epochs", "seed"):
        if state["config"][key] != getattr(cfg, key):
            raise SystemExit(
                f"{path} was saved with {key}={state['config'][key]!r} but this run uses "
                f"{getattr(cfg, key)!r}. Either match it, or use a different --runs_dir.")
    return state


def train(model, optimizer, scheduler, scaler, loaders, cfg, device, amp_dtype, out):
    """One epoch at a time: train on every batch, score the validation split, and keep
    the weights if they are the best so far."""
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    train_loader, val_loader = loaders
    best_path, state_path = out / "best.pt", out / "state.pt"
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)

    start_epoch, best_accuracy, history, step = 1, -1.0, [], 0
    state = load_state(state_path, cfg, device)
    if state:
        model.load_state_dict(state["model_weights"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_accuracy, history, step = state["best_accuracy"], state["history"], state["step"]
        print(f"carrying on after epoch {state['epoch']}, best val accuracy so far "
              f"{best_accuracy * 100:.2f}")

    started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        loss_total, n_batches = 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)

        for i, batch in enumerate(progress, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_dtype != torch.float32):
                logits = model(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"]).logits
            # the loss is computed in fp32 even under autocast: the softmax is where
            # reduced precision actually hurts
            loss = F.cross_entropy(logits.float(), batch["labels"],
                                   label_smoothing=cfg.label_smoothing)
            scaler.scale(loss / cfg.grad_accum).backward()
            loss_total += loss.item()
            n_batches += 1

            # with grad_accum > 1 we only update every Nth batch (and on the last one)
            if i % cfg.grad_accum and i != len(train_loader):
                continue

            scaler.unscale_(optimizer)      # undo fp16 loss scaling before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg.log_every == 0:
                tqdm.write(f"  step {step}/{steps_per_epoch * cfg.epochs}  "
                           f"loss {loss_total / n_batches:.4f}  "
                           f"lr {max(scheduler.get_last_lr()):.2e}")
                loss_total, n_batches = 0.0, 0

        progress.close()

        val, _ = evaluate(model, val_loader, device, amp_dtype)
        history.append({"epoch": epoch, "step": step, "val_loss": val["loss"],
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"]})
        is_best = val["accuracy"] > best_accuracy
        if is_best:
            best_accuracy = val["accuracy"]
            save_atomically(model.state_dict(), best_path)
        print(f"epoch {epoch}: val loss {val['loss']:.4f}  accuracy "
              f"{val['accuracy'] * 100:.2f}  macro-f1 {val['macro_f1'] * 100:.2f}"
              + ("  <- best" if is_best else ""))

        save_atomically({"model_weights": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                         "epoch": epoch, "step": step, "best_accuracy": best_accuracy,
                         "history": history, "config": dataclasses.asdict(cfg)}, state_path)

    return best_accuracy, history, (time.time() - started) / 60, best_path


def stage_train(cfg):
    """Stage 2b: fine-tune one pretrained transformer on the cleaned data."""
    import torch
    from transformers import AutoTokenizer

    set_seed(cfg.seed)
    architecture, model_id = MODELS.get(cfg.model, ("unknown", cfg.model))
    tag = f"{cfg.model}_seed{cfg.seed}"
    out = Path(cfg.runs_dir) / tag
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = pick_precision(cfg, device)
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"{cfg.model} ({architecture}) on {hardware}, seed {cfg.seed}, "
          f"precision {str(amp_dtype).replace('torch.', '')}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_rows, val_rows, test_rows = load_prepared(cfg)
    print(f"train {len(train_rows):,} | val {len(val_rows):,} | test {len(test_rows):,} | "
          f"{len(CLASS_NAMES)} classes")
    print("sample:", train_rows[0]["text"][:160])

    loaders = (make_loader(train_rows, cfg, tokenizer, True, device),
               make_loader(val_rows, cfg, tokenizer, False, device))
    test_loader = make_loader(test_rows, cfg, tokenizer, False, device)

    model = build_model(model_id).to(device)
    optimizer = make_optimizer(model, cfg)
    total_steps = math.ceil(len(loaders[0]) / cfg.grad_accum) * cfg.epochs
    scheduler = make_scheduler(optimizer, cfg, total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{n_params:.0f}M parameters, {total_steps} optimizer steps, "
          f"{len(optimizer.param_groups)} parameter groups")

    best_accuracy, history, minutes, best_path = train(
        model, optimizer, scheduler, scaler, loaders, cfg, device, amp_dtype, out)
    print(f"\ntrained for {minutes:.1f} min, best val accuracy {best_accuracy * 100:.2f}")

    # score the best epoch's weights on the held-out test set, exactly once
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test, matrix = evaluate(model, test_loader, device, amp_dtype)
    print(f"\ntest accuracy {test['accuracy'] * 100:.2f}  "
          f"(error {100 - test['accuracy'] * 100:.2f})  macro-f1 {test['macro_f1'] * 100:.2f}")
    print_per_class(test)
    print_confusions(matrix)

    payload = {"method": f"{architecture} fine-tuned", "architecture": architecture,
               "model": model_id, "family": "own", "kind": "transformer", "seed": cfg.seed,
               "n_train": len(train_rows), "n_test": len(test_rows), "max_len": cfg.max_len,
               "train_minutes": minutes, "best_val_accuracy": best_accuracy,
               "epochs_done": history[-1]["epoch"] if history else 0, "history": history,
               "metrics": test, "confusion": matrix.tolist(), "class_names": CLASS_NAMES,
               "config": dataclasses.asdict(cfg), "n_parameters_millions": round(n_params, 1)}
    print(f"\nwrote {save_result(cfg, tag, payload)}")
    return payload


# %% stage 3: the sweep and the table
def run_stage(cfg, stage, overrides, label):
    """Run one stage of the sweep and say whether it worked.

    As a separate process when we are a real script: one failure then cannot take the
    whole sweep down, and a crash cannot corrupt the parent. In a notebook there is no
    script file to re-invoke, so fall back to calling the stage in-process.
    """
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
    started = time.time()
    stages = {"baselines": stage_baselines, "train": stage_train}

    if SCRIPT_PATH.exists():
        command = [sys.executable, str(SCRIPT_PATH), stage,
                   "--data_dir", cfg.data_dir, "--results_dir", cfg.results_dir,
                   "--runs_dir", cfg.runs_dir]
        for key, value in overrides.items():
            command += [f"--{key}", str(value)]
        failed = subprocess.run(command).returncode
        reason = f"exit code {failed}" if failed else None
    else:
        try:
            stages[stage](dataclasses.replace(cfg, **overrides))
            reason = None
        except Exception as error:            # keep the sweep going after one bad run
            reason = f"{type(error).__name__}: {error}"

    minutes = (time.time() - started) / 60
    if reason:
        print(f"!! {label} failed ({reason}) after {minutes:.1f} min")
        return False
    print(f"-- {label} done in {minutes:.1f} min")
    return True


def stage_benchmark(cfg, models, seeds, quick=False, report_only=False):
    """Stage 3: run whatever is missing, then build the table from results/."""
    results_dir = Path(cfg.results_dir)
    failures = []

    if not report_only:
        if list(results_dir.glob("tfidf_*.json")):
            print("baselines already in results/, skipping")
        elif not run_stage(cfg, "baselines", {"seed": cfg.seed}, "classical baselines"):
            failures.append("baselines")

        for model in models:
            for seed in (seeds[:1] if quick else seeds):
                if (results_dir / f"{model}_seed{seed}.json").exists():
                    print(f"{model}_seed{seed} already done, skipping")
                    continue
                if not run_stage(cfg, "train",
                                 {"model": model, "seed": seed, "max_len": cfg.max_len,
                                  "epochs": cfg.epochs}, f"{model}, seed {seed}"):
                    failures.append(f"{model}_seed{seed}")

    rows = load_results(cfg)
    if not rows:
        raise SystemExit(f"no result files in {cfg.results_dir}/ -- run the sweep first.")
    summary = group_runs(rows)

    stats_path = Path(cfg.data_dir) / "clean" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if stats is None:
        print("note: stats.json missing, the table will omit the dataset section")

    out_md = results_dir / "benchmark.md"
    out_md.write_text(build_markdown(summary, stats, failures))
    (results_dir / "benchmark.json").write_text(json.dumps(
        {"runs": summary, "literature": LITERATURE, "incomplete": failures,
         "metric_choice": stats["metric_choice"] if stats else None}, indent=2))

    print(f"\n{'=' * 70}")
    for row in summary:
        print(f"{row['accuracy_mean']:6.2f}  {row['method']:<40} "
              f"{len(row['seeds'])} seed(s)")
    print(f"{'=' * 70}\nwrote {out_md} and {results_dir / 'benchmark.json'}")
    if failures:
        print(f"these did not finish: {', '.join(failures)}")
    return summary


# %% self test
def selftest():
    """Check the parts that need neither a GPU nor the dataset. Run it after any edit."""
    n = 0

    def ok(condition, label):
        nonlocal n
        assert condition, f"FAILED: {label}"
        n += 1

    ok(clean_text("a &amp; b") == "a & b", "html entities unescaped")
    ok(clean_text("one<br />two<BR>three") == "one two three", "br tags, any form")
    ok(clean_text("a&#xd;<br>b") == "a b", "carriage-return entity")
    ok(clean_text("x\ny\tz   w") == "x y z w", "whitespace collapsed")
    ok(clean_text(None) == "" and clean_text("") == "", "None and empty are safe")
    ok("<" in clean_text("if a <" + "x" * 60 + "> b"), "long angle-bracket run kept")
    ok(join_fields({"question_title": "t", "question_content": "", "best_answer": "a"})
       == "t a", "empty field dropped")

    cm = confusion_matrix([0, 0, 1, 1, 2, 2, 2], [0, 1, 1, 1, 2, 2, 0], 3)
    ok(cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 2]], "confusion matrix")
    m = metrics_from_confusion(cm)
    ok(abs(m["accuracy"] - 5 / 7) < 1e-12, "accuracy")
    ok(abs(m["precision"][0] - 0.5) < 1e-12, "precision is column-wise")
    ok(abs(m["recall"][1] - 1.0) < 1e-12, "recall is row-wise")
    ok(m["micro_f1"] == m["accuracy"], "micro-F1 equals accuracy")
    ok(m["support"] == [2, 2, 3], "support")
    gave_up = metrics_from_confusion(np.array([[5, 0, 0], [5, 0, 0], [5, 0, 0]]))
    ok(gave_up["macro_f1"] < gave_up["accuracy"], "macro-F1 punishes abandoned classes")
    ok(not any(np.isnan(gave_up["f1"])), "no nan for an unpredicted class")

    ok(metric_choice([6000] * 10)["primary"] == "accuracy", "balanced -> accuracy")
    ok(metric_choice([10000] + [200] * 9)["primary"] == "macro-F1", "skewed -> macro-F1")
    ok(metric_choice([5, 0])["imbalance_ratio"] == 5.0, "no divide-by-zero on empty class")

    warm, total = 60, 1000
    ok(lr_scale(0, warm, total) == 0.0, "schedule starts at zero")
    ok(abs(lr_scale(warm, warm, total) - 1.0) < 1e-12, "peaks at the end of warmup")
    ok(abs(lr_scale(530, warm, total) - 0.5) < 1e-12, "half way down at the midpoint")
    ok(lr_scale(total, warm, total) == 0.0 and lr_scale(1200, warm, total) == 0.0,
       "ends at zero and stays there")
    curve = [lr_scale(s, warm, total) for s in range(warm, total + 1)]
    ok(all(a >= b for a, b in zip(curve, curve[1:])), "decay is monotone")
    ok(abs(lr_scale(530, warm, total, "cosine") - 0.5) < 1e-12, "cosine halves at the midpoint")
    ok(lr_scale(0, 0, 100) == 1.0, "zero warmup is safe")

    ok(layer_depth("roberta.embeddings.word_embeddings.weight", 12) == 0, "embeddings depth")
    ok(layer_depth("deberta.encoder.rel_embeddings.weight", 12) == 0, "deberta embeddings")
    ok(layer_depth("roberta.encoder.layer.0.attention.self.query.weight", 12) == 1, "layer 0")
    ok(layer_depth("deberta.encoder.layer.11.output.LayerNorm.bias", 12) == 12, "layer 11")
    ok(layer_depth("classifier.out_proj.bias", 12) == 13, "head is deepest")

    ok(MODELS["bert-base"] == ("BERT", "bert-base-uncased"),
       "architecture BERT kept distinct from model bert-base-uncased")
    ok(len(LITERATURE) >= 10 and all(60 < r["accuracy"] < 85 for r in LITERATURE),
       "literature table populated and plausible")

    rows = [{"text": f"t{i}", "label": i % 10} for i in range(200)]
    kept, held = stratified_split(rows, 50, 0)
    ok(len(kept) + len(held) == 200 and abs(len(held) - 50) <= 5, "split sizes")
    ok(len(set(Counter(r["label"] for r in held).values())) == 1, "split keeps classes balanced")
    cleaned, info = clean_split(
        [{"question_title": "a", "question_content": "", "best_answer": "", "topic": 1},
         {"question_title": "a", "question_content": "", "best_answer": "", "topic": 1},
         {"question_title": "<br/>", "question_content": "", "best_answer": "", "topic": 2}])
    ok(len(cleaned) == 1, "duplicate and empty rows dropped")
    ok(info["dropped_duplicate"] == 1 and info["dropped_empty"] == 1, "drops counted")

    import tempfile
    tmp = Path(tempfile.mkdtemp())
    data = [{"text": "üñïçödé, \"quoted\"\nnewline", "label": 3}]
    write_jsonl(tmp / "a.jsonl.gz", data)
    write_jsonl(tmp / "b.jsonl.gz", data)
    ok(read_jsonl(tmp / "a.jsonl.gz") == data, "jsonl round-trip")
    ok((tmp / "a.jsonl.gz").read_bytes() == (tmp / "b.jsonl.gz").read_bytes(),
       "same content gives same bytes")

    fake = [{"method": "RoBERTa fine-tuned", "architecture": "RoBERTa", "model": "roberta-base",
             "max_len": 128, "seed": s, "n_train": 100, "train_minutes": 1.0,
             "metrics": {"accuracy": a, "macro_f1": a - 0.01}}
            for s, a in ((42, 0.76), (1, 0.74), (2, 0.75))]
    grouped = group_runs(fake)
    ok(len(grouped) == 1 and grouped[0]["seeds"] == [1, 2, 42], "seeds collapsed into one row")
    ok(abs(grouped[0]["accuracy_mean"] - 75.0) < 1e-9, "mean across seeds")
    ok(abs(grouped[0]["accuracy_std"] - float(np.std([76, 74, 75], ddof=1))) < 1e-9, "sample std")
    ok(group_runs(fake[:1])[0]["accuracy_std"] is None, "one seed reports no std")
    ok(fmt(75.0, None) == "75.00" and fmt(75.0, 1.0) == "75.00 ± 1.00", "number formatting")
    md = build_markdown(grouped, None, ["bert-base_seed1"])
    ok("**this work**" in md and "*cited*" in md, "own runs separated from citations")
    ok("bert-base_seed1 did not finish" in md, "incomplete runs flagged")

    cfg, extras = parse_cli(["train", "--lr", "3e-5", "--no-clean_html", "--epochs", "3"])
    ok((cfg.lr, cfg.clean_html, cfg.epochs) == (3e-5, False, 3), "flags parsed")
    ok(extras["stage"] == "train", "stage parsed")
    for field in dataclasses.fields(Config):
        d = field.default
        if isinstance(d, bool):
            argv, want = [f"--{'no-' if d else ''}{field.name}"], (not d)
        elif isinstance(d, int):
            argv, want = [f"--{field.name}", "7"], 7
        elif isinstance(d, float):
            argv, want = [f"--{field.name}", "0.25"], 0.25
        else:
            argv, want = [f"--{field.name}", "zz"], "zz"
        if field.name == "smoke":
            continue    # smoke rewrites other fields, so it is checked separately
        got, _ = parse_cli(["train"] + argv)
        ok(getattr(got, field.name) == want, f"--{field.name} is a flag")
    smoked, _ = parse_cli(["train", "--smoke"])
    ok(smoked.train_size == 2000 and smoked.epochs == 1, "--smoke shrinks the run")
    bench, extras = parse_cli(["benchmark", "--models", "bert-base", "--seeds", "1", "2",
                               "--quick"])
    ok(extras["models"] == ["bert-base"] and extras["seeds"] == [1, 2] and extras["quick"],
       "benchmark flags parsed")

    import shutil
    shutil.rmtree(tmp)
    print(f"selftest: all {n} checks passed")
    return n


# %% entry point
def main(argv=None):
    cfg, extras = parse_cli(argv)
    stage = extras.get("stage") or "all"

    if stage == "selftest":
        return selftest()
    if stage == "prepare":
        return stage_prepare(cfg)
    if stage == "baselines":
        return stage_baselines(cfg)
    if stage == "train":
        return stage_train(cfg)
    if stage == "benchmark":
        return stage_benchmark(cfg, extras["models"], extras["seeds"],
                               extras.get("quick", False), extras.get("report_only", False))
    if stage == "all":
        if not (Path(cfg.data_dir) / "clean" / "train.jsonl.gz").exists():
            stage_prepare(cfg)
        else:
            print(f"{cfg.data_dir}/clean already prepared, skipping stage 1")
        return stage_benchmark(cfg, ["bert-base", "roberta-base", "deberta-v3-base"],
                               [42, 1, 2], quick=cfg.smoke)
    raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main()
