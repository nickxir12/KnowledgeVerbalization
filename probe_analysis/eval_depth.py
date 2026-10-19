#!/usr/bin/env python3
"""
Evaluate probe classifiers across sequence depth bins.

--probe_type solvability (default): K probe, ground-truth solvability labels.
--probe_type verbalization: V probe, judge verbal_annotation labels
(0="solved", 1="unsolvable"; unparseable dropped).

For each problem, the sequence is divided into N equal-width depth bins
(default 10: 0-10%, 10-20%, ..., 90-100%). The token at the centre of
each bin is passed through the saved probe for that layer.  AUC is
computed per bin, giving a curve of "how decodable is solvability at
this point in the model's output?"

Memory-efficient: processes one layer at a time, so peak RAM is
  n_problems × 1 token × hidden_dim  (not n_layers × n_tokens).

Outputs one PNG per prompt style + one combined PNG (all styles pooled).
solvability_only is excluded (no reasoning trace).

Usage:
    python eval_depth.py --model qwen3_4b_instruct
    python eval_depth.py --model llama_31_8b_instruct --n_bins 10
    python eval_depth.py --model qwen3_4b_instruct --probe_type verbalization
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get("KV_ROOT", Path(__file__).resolve().parents[1]))  # repo root (override with $KV_ROOT)
OUTPUTS_ROOT = REPO_ROOT / "cot_generation/outputs"
JUDGE_ROOT   = REPO_ROOT / "cot_generation/judge_outputs"
CKPT_ROOT    = REPO_ROOT / "probe_training/solvability/checkpoints"      # re-pointed in main() per --probe_type
RESULTS_ROOT = REPO_ROOT / "probe_training/solvability/results"

EXCLUDED_STYLES = {"solvability_only", "unsolvable_conditioned"}

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Model name, e.g. qwen3_4b_instruct")
    parser.add_argument("--probe_type", type=str, default="solvability",
                        choices=["solvability", "verbalization"],
                        help="solvability: K probe vs ground truth; "
                             "verbalization: V probe vs judge labels.")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of depth bins (default 10 = every 10%%)")
    parser.add_argument("--min_tokens", type=int, default=20,
                        help="Skip sequences shorter than this many tokens.")
    parser.add_argument("--version", type=str, default=None,
                        help="Checkpoint/results version subdir, e.g. version2.")
    parser.add_argument("--trace", type=str, default="",
                        choices=["", "response", "thinking"],
                        help="Trace infix in npz filenames for thinking models "
                             "(e.g. 'response' -> ex000000_response_layer1.npz). "
                             "Leave empty for non-thinking models.")
    parser.add_argument("--output_model", type=str, default=None,
                        help="Model name for cot_generation/outputs if different from --model "
                             "(e.g. Qwen3-4b-thinking when --model is thinking/Qwen3-4b-thinking).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_latest_metadata(d: Path) -> Optional[List[Dict]]:
    candidates = list(d.glob("metadata_0_*.json"))
    if not candidates:
        return None
    def end_idx(p):
        m = re.search(r"metadata_0_(\d+)\.json", p.name)
        return int(m.group(1)) if m else 0
    return json.load(open(max(candidates, key=end_idx)))


def load_judge_labels(judge_dir: Path) -> Dict[int, int]:
    """Load verbal_annotation labels. Returns {example_idx: label}.
    label: 0=solved, 1=verbalized_as_unsolvable. Unparseable excluded.
    """
    label_map = {"solved": 0, "unsolvable": 1}
    f = next(judge_dir.glob("judge_metadata_*.jsonl"), None)
    if f is None:
        return {}
    result = {}
    for line in open(f):
        ex = json.loads(line)
        ann = ex.get("verbal_annotation")
        if ann in label_map:
            result[ex["example_idx"]] = label_map[ann]
    return result


def centre_of_bin(T: int, bin_idx: int, n_bins: int) -> int:
    lo = int(T * bin_idx / n_bins)
    hi = int(T * (bin_idx + 1) / n_bins)
    hi = max(hi, lo + 1)
    return min((lo + hi) // 2, T - 1)


def load_probe(model: str, layer: int):
    p = CKPT_ROOT / model / f"probe_layer{layer}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Probe not found: {p}")
    return joblib.load(p)


def discover_layers(model: str) -> List[int]:
    return sorted(
        int(re.search(r"layer(\d+)", p.name).group(1))
        for p in (CKPT_ROOT / model).glob("probe_layer*.joblib")
    )


def discover_styles(model_dir: Path) -> List[str]:
    return sorted(
        p.name for p in model_dir.iterdir()
        if p.is_dir() and p.name not in EXCLUDED_STYLES
    )


# ---------------------------------------------------------------------------
# Core: collect one layer's depth bins across a list of styles
# ---------------------------------------------------------------------------

def collect_one_layer(
    model_dir: Path,
    styles: List[str],
    layer: int,
    n_bins: int,
    min_tokens: int,
    trace: str = "",
    judge_dir: Optional[Path] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load the centre token from each depth bin for every problem across
    `styles`, for a single `layer`.  Returns {bin_idx: (X [N,D], y [N])}.

    If `judge_dir` is given, labels come from the judge (verbalization);
    otherwise from the ground-truth solvability split.

    Processes one npz at a time — peak RAM is O(n_problems × hidden_dim).
    """
    bins: Dict[int, Tuple[List, List]] = {b: ([], []) for b in range(n_bins)}

    for style in styles:
        for solvability in ("solvable", "unsolvable"):
            d = model_dir / style / solvability
            if not d.exists():
                continue
            meta = load_latest_metadata(d)
            if meta is None:
                continue
            labels = None
            if judge_dir is not None:
                labels = load_judge_labels(judge_dir / style / solvability)
                if not labels:
                    logger.warning(f"  No judge labels: {judge_dir / style / solvability}")
                    continue
            label = 1 if solvability == "unsolvable" else 0
            n_loaded = 0

            for ex in meta:
                ex_idx = ex["example_idx"]
                infix = f"_{trace}" if trace else ""
                if labels is not None:
                    if ex_idx not in labels:
                        continue
                    label = labels[ex_idx]
                npz_path = d / f"ex{ex_idx:06d}{infix}_layer{layer}.npz"
                if not npz_path.exists():
                    continue

                # Load only the hidden states for this one layer
                h = np.load(npz_path)["hidden_states"].astype(np.float32)
                T = h.shape[0]
                if T < min_tokens:
                    continue

                for b in range(n_bins):
                    tok_idx = centre_of_bin(T, b, n_bins)
                    bins[b][0].append(h[tok_idx])   # shape: [hidden_dim]
                    bins[b][1].append(label)

                del h   # free immediately
                n_loaded += 1

            logger.info(f"    layer {layer} | {style}/{solvability}: "
                        f"{n_loaded} problems")

    # Consolidate
    out = {}
    for b in range(n_bins):
        Xs, ys = bins[b]
        if Xs:
            out[b] = (np.stack(Xs), np.array(ys, dtype=np.int8))
        else:
            out[b] = (np.empty((0, 1)), np.empty(0, dtype=np.int8))
    return out


# ---------------------------------------------------------------------------
# AUC curve from pre-collected bin data + probe
# ---------------------------------------------------------------------------

def compute_auc_curve(
    bin_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    probe,
    n_bins: int,
) -> List[Optional[float]]:
    aucs = []
    for b in range(n_bins):
        X, y = bin_data[b]
        if len(X) == 0 or len(np.unique(y)) < 2:
            aucs.append(None)
            continue
        probs = probe.predict_proba(X)[:, 1]
        aucs.append(float(roc_auc_score(y, probs)))
    return aucs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_LAYER_COLORS  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']
_LAYER_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', 'p']


def plot_depth_curves(
    curves: Dict[int, List[Optional[float]]],
    layers: List[int],
    n_bins: int,
    title: str,
    out_path: Path,
):
    bin_centres = [100 * (b + 0.5) / n_bins for b in range(n_bins)]
    tick_labels = [f"{int(100*b/n_bins)}–{int(100*(b+1)/n_bins)}%"
                   for b in range(n_bins)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='chance (0.5)')

    for i, layer in enumerate(layers):
        col = _LAYER_COLORS[i % len(_LAYER_COLORS)]
        mkr = _LAYER_MARKERS[i % len(_LAYER_MARKERS)]
        ys  = curves[layer]
        xs_valid = [x for x, y in zip(bin_centres, ys) if y is not None]
        ys_valid = [y for y in ys if y is not None]
        ax.plot(xs_valid, ys_valid, marker=mkr, color=col, linewidth=2,
                markersize=7, label=f'Layer {layer}')
        for x, y in zip(xs_valid, ys_valid):
            ax.annotate(f'{y:.3f}', (x, y),
                        textcoords='offset points', xytext=(0, 7),
                        ha='center', fontsize=7, color=col)

    ax.set_xticks(bin_centres)
    ax.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=8)
    ax.set_xlabel('Sequence depth bin', fontsize=10)
    ax.set_ylabel('ROC-AUC', fontsize=10)
    ax.set_ylim(0.3, 1.0)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    global CKPT_ROOT, RESULTS_ROOT, JUDGE_ROOT
    verbal = args.probe_type == "verbalization"
    if verbal:
        CKPT_ROOT    = REPO_ROOT / "probe_training/verbalization/checkpoints"
        RESULTS_ROOT = REPO_ROOT / "probe_training/verbalization/results"
    if args.version:
        CKPT_ROOT    = CKPT_ROOT    / args.version
        RESULTS_ROOT = RESULTS_ROOT / args.version
        JUDGE_ROOT   = JUDGE_ROOT   / args.version

    output_model = args.output_model or args.model
    model_dir = OUTPUTS_ROOT / output_model
    if not model_dir.exists():
        raise FileNotFoundError(f"Model output dir not found: {model_dir}")

    judge_dir = None
    if verbal:
        judge_dir = JUDGE_ROOT / output_model
        if not judge_dir.exists():
            raise FileNotFoundError(f"Judge output dir not found: {judge_dir}")

    layers = discover_layers(args.model)
    styles = discover_styles(model_dir)
    logger.info(f"Model:   {args.model}")
    logger.info(f"Layers:  {layers}")
    logger.info(f"Styles:  {styles}")
    logger.info(f"Bins:    {args.n_bins}")

    out_dir = RESULTS_ROOT / args.model / "depth_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # curves[style_key][layer] = [auc_bin0, ..., auc_binN]
    # style_key is a style name or "combined"
    all_results: Dict[str, Dict[int, List[Optional[float]]]] = {}

    # ── Per-layer loop (memory-efficient) ───────────────────────────────────
    # For each layer, collect data for all styles in one pass, then compute
    # per-style and combined AUC curves before freeing the layer's data.

    # Accumulate combined bin data across layers (only centre tokens, small)
    combined_bins: Dict[int, Dict[int, Tuple[List, List]]] = {
        layer: {b: ([], []) for b in range(args.n_bins)}
        for layer in layers
    }
    per_style_bins: Dict[str, Dict[int, Dict[int, Tuple[List, List]]]] = {
        style: {layer: {b: ([], []) for b in range(args.n_bins)}
                for layer in layers}
        for style in styles
    }

    for layer in layers:
        logger.info(f"\n=== Layer {layer} ===")
        probe = load_probe(args.model, layer)

        # Per-style
        for style in styles:
            logger.info(f"  Style: {style}")
            bin_data = collect_one_layer(
                model_dir, [style], layer, args.n_bins, args.min_tokens, args.trace,
                judge_dir=judge_dir,
            )
            # Store centre vectors for later combined computation
            for b in range(args.n_bins):
                X, y = bin_data[b]
                if len(X) > 0:
                    combined_bins[layer][b][0].append(X)
                    combined_bins[layer][b][1].append(y)
                    per_style_bins[style][layer][b][0].append(X)
                    per_style_bins[style][layer][b][1].append(y)

            aucs = compute_auc_curve(bin_data, probe, args.n_bins)
            all_results.setdefault(style, {})[layer] = aucs
            logger.info(f"    AUC: " +
                        " ".join(f"{a:.3f}" if a else " — " for a in aucs))
            del bin_data

        del probe

    # ── Combined AUC ────────────────────────────────────────────────────────
    logger.info("\n=== Combined (all styles) ===")
    for layer in layers:
        probe = load_probe(args.model, layer)
        bin_data = {}
        for b in range(args.n_bins):
            Xs, ys = combined_bins[layer][b]
            if Xs:
                bin_data[b] = (np.concatenate(Xs), np.concatenate(ys))
            else:
                bin_data[b] = (np.empty((0, 1)), np.empty(0, dtype=np.int8))
        aucs = compute_auc_curve(bin_data, probe, args.n_bins)
        all_results.setdefault("combined", {})[layer] = aucs
        logger.info(f"  Layer {layer}: " +
                    " ".join(f"{a:.3f}" if a else " — " for a in aucs))
        del probe, bin_data

    # ── Save JSONs and plots ─────────────────────────────────────────────────
    for key, curves in all_results.items():
        json_path = out_dir / f"depth_auc_{key}.json"
        with open(json_path, "w") as f:
            json.dump({str(l): curves[l] for l in layers}, f, indent=2)

        label = key if key == "combined" else f"prompt style: {key}"
        plot_depth_curves(
            curves, layers, args.n_bins,
            title=f"{args.model} — {'verbalization probe ' if verbal else ''}depth AUC\n{label}",
            out_path=out_dir / f"depth_auc_{key}.png",
        )

    logger.info(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
