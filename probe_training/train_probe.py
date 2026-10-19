#!/usr/bin/env python3
"""
Linear probe classifier on hidden states.

Two probe types (--probe_type):

  solvability (default)
      Label: 0=solvable, 1=unsolvable  (token_solvability in npz files)
      Data:  cot_generation/outputs/<model>/<style>/<solvability>/

  verbalization
      Label: 0=solved, 1=unsolvable  (judge verbal_annotation field)
      Drops examples with verbal_annotation=unparseable.
      Data:  cot_generation/outputs + judge_outputs/<model>/<style>/<solvability>/

Results/checkpoints go to probe_training/solvability/ or probe_training/verbalization/
depending on --probe_type.

Usage:
    python train_probe.py --model qwen3_4b_instruct --layer 18
    python train_probe.py --model qwen3_4b_instruct --all_layers --probe_type verbalization
"""

import os
import re
import json
import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get("KV_ROOT", Path(__file__).resolve().parents[1]))  # repo root (override with $KV_ROOT)
OUTPUTS_ROOT  = REPO_ROOT / "cot_generation/outputs"
JUDGE_ROOT    = REPO_ROOT / "cot_generation/judge_outputs"
CLASSIF_ROOT  = REPO_ROOT / "probe_training"

# Styles excluded from ALL probe types — no CoT, hidden states reflect
# output token prediction rather than reasoning, and labels are noisy.
EXCLUDED_STYLES = {"solvability_only"}

# Additional styles excluded from verbalization only — the prompt itself
# strongly biases the verbalization label, so the probe would measure
# prompt-following rather than internal solvability representation.
EXCLUDED_STYLES_VERBAL = EXCLUDED_STYLES | {"unsolvable_conditioned"}

# Set in main() from args. For thinking models the npz files and judge files
# carry a trace suffix; for non-thinking models the suffix is empty.
#   npz:   ex000000_layer18.npz   (non-thinking)  /  ex000000_thinking_layer18.npz
#   judge: judge_metadata_0_1102.jsonl            /  judge_metadata_0_1102_thinking.jsonl
TRACE = "response"          # 'response' | 'thinking'
THINKING_MODEL = False      # whether this model has separate thinking/response traces


def npz_name(ex_idx: int, layer: int) -> str:
    if THINKING_MODEL:
        return f"ex{ex_idx:06d}_{TRACE}_layer{layer}.npz"
    return f"ex{ex_idx:06d}_layer{layer}.npz"


def judge_glob() -> str:
    if THINKING_MODEL:
        return f"judge_metadata_*_{TRACE}.jsonl"
    return "judge_metadata_*.jsonl"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True,
                        help="Model subfolder name under OUTPUTS_ROOT, e.g. qwen3_4b_instruct")
    parser.add_argument("--layer", type=int, default=None,
                        help="Single layer index to train on.")
    parser.add_argument("--all_layers", action="store_true",
                        help="Train a classifier for every available layer.")
    parser.add_argument("--tokens_per_style", type=int, default=50_000,
                        help="Max tokens to sample from each (style, solvability) bucket. "
                             "Controls balance across prompting styles.")
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Fraction of problems held out for validation (problem-level split).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iter", type=int, default=1000,
                        help="Max iterations for LogisticRegression solver.")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength for LogisticRegression.")
    parser.add_argument("--tokens_per_problem", type=int, default=20,
                        help="Tokens sampled per problem, spread across the full sequence depth. "
                             "Set to 0 to fall back to --tokens_per_style subsampling only.")
    parser.add_argument("--probe_type", type=str, default="solvability",
                        choices=["solvability", "verbalization"],
                        help="solvability: label=problem solvability. "
                             "verbalization: label=model verbal output (solved vs unsolvable).")
    parser.add_argument("--train_styles", type=str, default=None,
                        help="Comma-separated styles used for TRAINING only "
                             "(e.g. 'standard'). Validation/per-style AUC still span "
                             "all styles. Default: train on all styles.")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="Subdir inserted under checkpoints/ and results/ to "
                             "keep this run separate (e.g. 'version2').")
    parser.add_argument("--judge_version", type=str, default=None,
                        help="Judge-output version subdir, e.g. 'version2'. "
                             "Default: top-level judge_outputs (v1).")
    parser.add_argument("--trace", type=str, default="response",
                        choices=["response", "thinking"],
                        help="For thinking models: which trace's npz + judge labels "
                             "to use. Non-thinking models only have 'response'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_latest_metadata(d: Path) -> Optional[List[Dict]]:
    candidates = list(d.glob("metadata_0_*.json"))
    if not candidates:
        return None
    def end_idx(p):
        m = re.search(r"metadata_0_(\d+)\.json", p.name)
        return int(m.group(1)) if m else 0
    return json.load(open(max(candidates, key=end_idx)))


def problem_level_split(meta: List[Dict], val_frac: float, seed: int) -> Tuple[set, set]:
    """Return (train_example_idxs, val_example_idxs) split at problem level."""
    all_idxs = [ex["example_idx"] for ex in meta]
    rng = random.Random(seed)
    shuffled = all_idxs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return set(shuffled[n_val:]), set(shuffled[:n_val])


def sample_tokens_from_problem(
    h: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n token positions spread across sequence depth via stratified binning.

    Divides the sequence into n equal-width bins and draws one token uniformly
    from each bin, ensuring coverage of early/middle/late positions rather than
    random clustering. Returns all indices if sequence shorter than n.
    """
    T = len(h)
    if T <= n:
        return np.arange(T)
    bin_edges = np.linspace(0, T, n + 1).astype(int)
    return np.array([rng.integers(bin_edges[i], bin_edges[i + 1]) for i in range(n)])


def load_tokens_for_style(
    model_dir: Path,
    style: str,
    solvability: str,
    layer: int,
    val_frac: float,
    seed: int,
    tokens_per_problem: int,
    tokens_per_style: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load hidden states + labels for one (style, solvability) bucket.

    Per-problem: sample `tokens_per_problem` tokens spread across sequence depth
    (stratified binning so early/middle/late positions all represented).
    Per-bucket:  cap at `tokens_per_style` tokens as a secondary safety limit.
    Returns (X_train, y_train, X_val, y_val).
    """
    d = model_dir / style / solvability
    meta = load_latest_metadata(d)
    if meta is None:
        return np.empty((0, 0)), np.empty(0), np.empty((0, 0)), np.empty(0)

    train_idxs, val_idxs = problem_level_split(meta, val_frac, seed)

    X_train, y_train = [], []
    X_val,   y_val   = [], []

    for ex in meta:
        ex_idx = ex["example_idx"]
        npz_path = d / npz_name(ex_idx, layer)
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        h   = data["hidden_states"].astype(np.float32)  # [n_tokens, hidden_dim]
        sol = data["token_solvability"].astype(np.int8)  # [n_tokens]

        if tokens_per_problem > 0:
            rng = np.random.default_rng(seed + ex_idx)
            sel = sample_tokens_from_problem(h, tokens_per_problem, rng)
            h, sol = h[sel], sol[sel]

        if ex_idx in train_idxs:
            X_train.append(h)
            y_train.append(sol)
        else:
            X_val.append(h)
            y_val.append(sol)

    def concat_and_cap(Xs, ys, max_tok, rng_seed):
        if not Xs:
            return np.empty((0, 1)), np.empty(0, dtype=np.int8)
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        if max_tok > 0 and len(X) > max_tok:
            rng = np.random.default_rng(rng_seed)
            idx = rng.choice(len(X), size=max_tok, replace=False)
            X, y = X[idx], y[idx]
        return X, y

    X_tr, y_tr = concat_and_cap(X_train, y_train, tokens_per_style, seed)
    X_v,  y_v  = concat_and_cap(X_val,   y_val,   tokens_per_style, seed + 1)
    return X_tr, y_tr, X_v, y_v


def load_judge_labels(judge_dir: Path) -> Dict[int, int]:
    """Load verbal_annotation labels from a judge JSONL file.

    Returns {example_idx: label} where label is 0=solved, 1=unsolvable.
    Examples with verbal_annotation='unparseable' are excluded (not in dict).
    """
    label_map = {"solved": 0, "unsolvable": 1}
    f = next(judge_dir.glob(judge_glob()), None)
    if f is None:
        return {}
    result = {}
    for line in open(f):
        ex = json.loads(line)
        ann = ex.get("verbal_annotation")
        if ann in label_map:
            result[ex["example_idx"]] = label_map[ann]
    return result


def load_tokens_for_style_verbal(
    model_dir: Path,
    judge_dir: Path,
    style: str,
    solvability: str,
    layer: int,
    val_frac: float,
    seed: int,
    tokens_per_problem: int,
    tokens_per_style: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like load_tokens_for_style but labels come from judge verbal_annotation.

    Label: 0=solved, 1=verbalized_as_unsolvable.
    Examples with verbal_annotation=unparseable are dropped.
    """
    d = model_dir / style / solvability
    meta = load_latest_metadata(d)
    if meta is None:
        return np.empty((0, 0)), np.empty(0), np.empty((0, 0)), np.empty(0)

    jd = judge_dir / style / solvability
    labels = load_judge_labels(jd)
    if not labels:
        return np.empty((0, 0)), np.empty(0), np.empty((0, 0)), np.empty(0)

    # Only use examples that have a judge label
    meta_filtered = [ex for ex in meta if ex["example_idx"] in labels]
    if not meta_filtered:
        return np.empty((0, 0)), np.empty(0), np.empty((0, 0)), np.empty(0)

    train_idxs, val_idxs = problem_level_split(meta_filtered, val_frac, seed)

    X_train, y_train = [], []
    X_val,   y_val   = [], []

    for ex in meta_filtered:
        ex_idx = ex["example_idx"]
        npz_path = d / npz_name(ex_idx, layer)
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        h = data["hidden_states"].astype(np.float32)
        label = np.full(len(h), labels[ex_idx], dtype=np.int8)

        if tokens_per_problem > 0:
            rng = np.random.default_rng(seed + ex_idx)
            sel = sample_tokens_from_problem(h, tokens_per_problem, rng)
            h, label = h[sel], label[sel]

        if ex_idx in train_idxs:
            X_train.append(h)
            y_train.append(label)
        else:
            X_val.append(h)
            y_val.append(label)

    def concat_and_cap(Xs, ys, max_tok, rng_seed):
        if not Xs:
            return np.empty((0, 1)), np.empty(0, dtype=np.int8)
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        if max_tok > 0 and len(X) > max_tok:
            rng = np.random.default_rng(rng_seed)
            idx = rng.choice(len(X), size=max_tok, replace=False)
            X, y = X[idx], y[idx]
        return X, y

    X_tr, y_tr = concat_and_cap(X_train, y_train, tokens_per_style, seed)
    X_v,  y_v  = concat_and_cap(X_val,   y_val,   tokens_per_style, seed + 1)
    return X_tr, y_tr, X_v, y_v



def collect_data(
    model_dir: Path,
    layer: int,
    tokens_per_problem: int,
    tokens_per_style: int,
    val_frac: float,
    seed: int,
    train_styles: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect and pool data from all available (style, solvability) buckets.

    If train_styles is given, only those styles contribute to the TRAIN set;
    val (held-out problems) is collected from all styles regardless.
    """
    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []

    styles = sorted(p.name for p in model_dir.iterdir() if p.is_dir() and p.name not in EXCLUDED_STYLES)
    for style in styles:
        use_train = train_styles is None or style in train_styles
        for solvability in ("solvable", "unsolvable"):
            d = model_dir / style / solvability
            if not d.exists():
                continue
            logger.info(f"  Loading {style}/{solvability} layer {layer} "
                        f"(train={'yes' if use_train else 'no'}) ...")
            X_tr, y_tr, X_v, y_v = load_tokens_for_style(
                model_dir, style, solvability, layer,
                val_frac, seed, tokens_per_problem, tokens_per_style,
            )
            if len(X_v) > 0:
                all_X_val.append(X_v)
                all_y_val.append(y_v)
            if use_train and len(X_tr) > 0:
                all_X_train.append(X_tr)
                all_y_train.append(y_tr)
                logger.info(f"    train={len(X_tr):,}  val={len(X_v):,}  "
                            f"unsolvable_frac={y_tr.mean():.2f}")
            else:
                logger.info(f"    train=(excluded)  val={len(X_v):,}")

    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_val   = np.concatenate(all_X_val,   axis=0)
    y_val   = np.concatenate(all_y_val,   axis=0)
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    C: float,
    max_iter: int,
    seed: int,
) -> Tuple[Pipeline, Dict]:
    logger.info(f"  Train: {X_train.shape}  class balance: {y_train.mean():.3f}")
    logger.info(f"  Val:   {X_val.shape}    class balance: {y_val.mean():.3f}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )),
    ])
    pipe.fit(X_train, y_train)

    y_pred    = pipe.predict(X_val)
    y_prob    = pipe.predict_proba(X_val)[:, 1]
    acc       = accuracy_score(y_val, y_pred)
    auc       = roc_auc_score(y_val, y_prob)
    report    = classification_report(y_val, y_pred, target_names=["solvable", "unsolvable"])
    cm        = confusion_matrix(y_val, y_pred).tolist()

    metrics = {
        "val_accuracy": acc,
        "val_roc_auc":  auc,
        "classification_report": report,
        "confusion_matrix": cm,
        "n_train": int(len(X_train)),
        "n_val":   int(len(X_val)),
        "train_class_balance": float(y_train.mean()),
        "val_class_balance":   float(y_val.mean()),
    }

    logger.info(f"  Val accuracy: {acc:.4f}  ROC-AUC: {auc:.4f}")
    logger.info(f"\n{report}")
    return pipe, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_data_verbal(
    model_dir: Path,
    judge_dir: Path,
    layer: int,
    tokens_per_problem: int,
    tokens_per_style: int,
    val_frac: float,
    seed: int,
    train_styles: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect verbalization-labelled data from all (style, solvability) buckets.

    If train_styles is given, only those styles contribute to the TRAIN set;
    val is collected from all styles regardless.
    """
    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []

    styles = sorted(p.name for p in model_dir.iterdir() if p.is_dir() and p.name not in EXCLUDED_STYLES_VERBAL)
    for style in styles:
        use_train = train_styles is None or style in train_styles
        for solvability in ("solvable", "unsolvable"):
            d = model_dir / style / solvability
            if not d.exists():
                continue
            logger.info(f"  Loading verbal {style}/{solvability} layer {layer} "
                        f"(train={'yes' if use_train else 'no'}) ...")
            X_tr, y_tr, X_v, y_v = load_tokens_for_style_verbal(
                model_dir, judge_dir, style, solvability, layer,
                val_frac, seed, tokens_per_problem, tokens_per_style,
            )
            if len(X_v) > 0:
                all_X_val.append(X_v)
                all_y_val.append(y_v)
            if use_train and len(X_tr) > 0:
                all_X_train.append(X_tr)
                all_y_train.append(y_tr)
                logger.info(f"    train={len(X_tr):,}  val={len(X_v):,}  "
                            f"verbalized_unsolvable_frac={y_tr.mean():.2f}")
            else:
                logger.info(f"    train=(excluded)  val={len(X_v):,}")

    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_val   = np.concatenate(all_X_val,   axis=0)
    y_val   = np.concatenate(all_y_val,   axis=0)
    return X_train, y_train, X_val, y_val



def eval_per_style(
    pipe: Pipeline,
    model_dir: Path,
    layer: int,
    val_frac: float,
    seed: int,
    tokens_per_problem: int,
    probe_type: str = "solvability",
    judge_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate a trained probe on each prompt style's val set.

    For each style, combines solvable + unsolvable val splits so both classes
    are present. Uses the correct label source per probe_type:
      solvability:   token_solvability from npz
      verbalization: judge verbal_annotation (skips styles with no judge file)
    Returns {style_name: auc}.
    """
    results = {}
    styles = sorted(p.name for p in model_dir.iterdir() if p.is_dir() and p.name not in EXCLUDED_STYLES_VERBAL)
    for style in styles:
        Xs, ys = [], []
        for solvability in ("solvable", "unsolvable"):
            d = model_dir / style / solvability
            if not d.exists():
                continue
            if probe_type == "verbalization":
                if judge_dir is None:
                    continue
                _, _, X_v, y_v = load_tokens_for_style_verbal(
                    model_dir, judge_dir, style, solvability, layer,
                    val_frac, seed, tokens_per_problem, tokens_per_style=0,
                )
            else:
                _, _, X_v, y_v = load_tokens_for_style(
                    model_dir, style, solvability, layer,
                    val_frac, seed, tokens_per_problem, tokens_per_style=0,
                )
            if len(X_v) > 0:
                Xs.append(X_v)
                ys.append(y_v)
        if not Xs:
            continue
        X_val = np.concatenate(Xs, axis=0)
        y_val = np.concatenate(ys, axis=0)
        if len(X_val) < 10 or len(np.unique(y_val)) < 2:
            continue
        y_prob = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        results[style] = round(auc, 4)
        logger.info(f"    {style}: AUC={auc:.4f}  n={len(X_val)}")
    return results


def run_layer(args, layer: int):
    model_dir = OUTPUTS_ROOT / args.model
    if not model_dir.exists():
        raise FileNotFoundError(f"Model output dir not found: {model_dir}")

    probe_root = CLASSIF_ROOT / args.probe_type
    tag = (args.run_tag,) if args.run_tag else ()
    ckpt_dir   = probe_root.joinpath("checkpoints", *tag, args.model)
    res_dir    = probe_root.joinpath("results",     *tag, args.model)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True,  exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {args.model}  |  Layer: {layer}  |  Probe: {args.probe_type}")
    logger.info(f"{'='*60}")

    train_styles = (set(s.strip() for s in args.train_styles.split(",") if s.strip())
                    if args.train_styles else None)
    if train_styles:
        logger.info(f"Training restricted to styles: {sorted(train_styles)} "
                    f"(validation spans all styles)")

    if args.probe_type == "solvability":
        X_train, y_train, X_val, y_val = collect_data(
            model_dir, layer,
            tokens_per_problem=args.tokens_per_problem,
            tokens_per_style=args.tokens_per_style,
            val_frac=args.val_frac,
            seed=args.seed,
            train_styles=train_styles,
        )
    else:
        judge_dir = JUDGE_ROOT / args.model
        if not judge_dir.exists():
            raise FileNotFoundError(f"Judge output dir not found: {judge_dir}")
        X_train, y_train, X_val, y_val = collect_data_verbal(
            model_dir, judge_dir, layer,
            tokens_per_problem=args.tokens_per_problem,
            tokens_per_style=args.tokens_per_style,
            val_frac=args.val_frac,
            seed=args.seed,
            train_styles=train_styles,
        )
    logger.info(f"Total train: {len(X_train):,}  val: {len(X_val):,}")

    pipe, metrics = train_probe(
        X_train, y_train, X_val, y_val,
        C=args.C, max_iter=args.max_iter, seed=args.seed,
    )

    # Per-style AUC on validation set
    logger.info("  Per-style validation AUC:")
    per_style_auc = eval_per_style(
        pipe, model_dir, layer,
        val_frac=args.val_frac, seed=args.seed,
        tokens_per_problem=args.tokens_per_problem,
        probe_type=args.probe_type,
        judge_dir=JUDGE_ROOT / args.model if args.probe_type == "verbalization" else None,
    )
    metrics["per_style_auc"] = per_style_auc

    # Save checkpoint
    ckpt_path = ckpt_dir / f"probe_layer{layer}.joblib"
    joblib.dump(pipe, ckpt_path)
    logger.info(f"Saved probe -> {ckpt_path}")

    # Save results
    res_path = res_dir / f"probe_layer{layer}_results.json"
    metrics["layer"] = layer
    metrics["model"] = args.model
    metrics["probe_type"] = args.probe_type
    metrics["train_styles"] = sorted(train_styles) if train_styles else "all"
    metrics["tokens_per_problem"] = args.tokens_per_problem
    metrics["tokens_per_style"] = args.tokens_per_style
    metrics["val_frac"] = args.val_frac
    metrics["C"] = args.C
    with open(res_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved results -> {res_path}")

    return metrics, res_dir


def main():
    global TRACE, THINKING_MODEL, JUDGE_ROOT
    args = parse_args()
    np.random.seed(args.seed)

    model_dir = OUTPUTS_ROOT / args.model
    if not model_dir.exists():
        raise FileNotFoundError(f"{model_dir} not found")

    # Detect thinking model by presence of *_thinking_layer*.npz files.
    THINKING_MODEL = any(model_dir.rglob("ex000000_thinking_layer*.npz"))
    TRACE = args.trace
    if not THINKING_MODEL and args.trace == "thinking":
        raise ValueError(f"{args.model} is not a thinking model (no _thinking npz).")
    if args.judge_version:
        JUDGE_ROOT = JUDGE_ROOT / args.judge_version
    logger.info(f"thinking_model={THINKING_MODEL}  trace={TRACE}  "
                f"judge_root={JUDGE_ROOT}")

    # Discover available layers from npz files for the active trace
    glob_pat = (f"ex000000_{TRACE}_layer*.npz" if THINKING_MODEL
                else "ex000000_layer*.npz")
    sample_npzs = list(model_dir.rglob(glob_pat))
    available_layers = sorted(set(
        int(re.search(r"layer(\d+)", p.name).group(1))
        for p in sample_npzs
    ))
    logger.info(f"Available layers: {available_layers}")

    if args.all_layers:
        layers = available_layers
    elif args.layer is not None:
        if args.layer not in available_layers:
            raise ValueError(f"Layer {args.layer} not in available layers {available_layers}")
        layers = [args.layer]
    else:
        raise ValueError("Specify --layer <N> or --all_layers")

    all_metrics = {}
    res_dir = None
    for layer in layers:
        metrics, res_dir = run_layer(args, layer)
        all_metrics[layer] = metrics

    # Reload ALL layer results from disk (other parallel jobs may have finished)
    if res_dir is None:
        tag = (args.run_tag,) if args.run_tag else ()
        res_dir = CLASSIF_ROOT.joinpath(args.probe_type, "results", *tag, args.model)
    all_metrics_full = {}
    for f in sorted(res_dir.glob("probe_layer*_results.json"),
                    key=lambda p: int(re.search(r"layer(\d+)", p.name).group(1))):
        d = json.load(open(f))
        all_metrics_full[d["layer"]] = d

    # Summary table (log only current run's layers)
    logger.info("\n" + "="*60)
    logger.info("SUMMARY (this run)")
    logger.info("="*60)
    logger.info(f"{'Layer':>8}  {'Accuracy':>10}  {'ROC-AUC':>10}")
    for layer in sorted(all_metrics):
        m = all_metrics[layer]
        logger.info(f"{layer:>8}  {m['val_accuracy']:>10.4f}  {m['val_roc_auc']:>10.4f}")

    # Save combined summary JSON (all layers found on disk)
    summary_path = res_dir / "probe_summary.json"
    with open(summary_path, "w") as f:
        json.dump({str(l): m for l, m in all_metrics_full.items()}, f, indent=2)
    logger.info(f"\nSummary saved -> {summary_path} ({len(all_metrics_full)} layers)")

    # Save human-readable TXT summary (always uses all layers found on disk)
    txt_path = res_dir / "probe_summary.txt"
    sorted_layers = sorted(all_metrics_full)
    all_style_keys = sorted({k for m in all_metrics_full.values() for k in m.get("per_style_auc", {})})
    with open(txt_path, "w") as f:
        f.write(f"Linear Probe Results — {args.model}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"  tokens_per_problem={args.tokens_per_problem}  "
                f"val_frac={args.val_frac}  C={args.C}\n\n")
        # Overall table
        f.write(f"{'Layer':>8}  {'Accuracy':>10}  {'ROC-AUC (overall)':>20}\n")
        f.write("-" * 44 + "\n")
        for layer in sorted_layers:
            m = all_metrics_full[layer]
            f.write(f"{layer:>8}  {m['val_accuracy']:>10.4f}  {m['val_roc_auc']:>20.4f}\n")
        f.write("\n")
        # Per-style AUC table
        if all_style_keys:
            col_w = 32
            f.write("Per-style Validation AUC\n")
            f.write("-" * (10 + col_w * len(sorted_layers)) + "\n")
            header = f"{'Style/Solvability':<30}" + "".join(f"{str(l):>{col_w}}" for l in sorted_layers)
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for key in all_style_keys:
                row = f"{key:<30}"
                for layer in sorted_layers:
                    val = all_metrics_full[layer].get("per_style_auc", {}).get(key)
                    row += f"{val:>{col_w}.4f}" if val is not None else f"{'N/A':>{col_w}}"
                f.write(row + "\n")
    logger.info(f"TXT summary saved -> {txt_path}")

    logger.info(f"Results in {res_dir}")


if __name__ == "__main__":
    main()
