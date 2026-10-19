#!/usr/bin/env python3
"""
For one model, run K (solvability) and V (verbalization) probes on hidden states
from `standard` and `unsolvable_biased` prompts, on both `solvable` and
`unsolvable` splits. Report what each probe predicts and how the prompt shifts
predictions, per layer.

Outputs:
    combined_analysis/<version>/<model>/vector_prediction_prompt_effect/
        layer<L>/predictions.json   - per-example K/V scores and predictions
        layer<L>/summary.json       - prediction rates and shifts
        all_layers_summary.json     - per-layer roll-up
"""
import os
import re
import json
import argparse
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(os.environ.get("KV_ROOT", Path(__file__).resolve().parents[1]))  # repo root (override with $KV_ROOT)
OUTPUTS_ROOT  = REPO_ROOT / "cot_generation/outputs"
JUDGE_ROOT    = REPO_ROOT / "cot_generation/judge_outputs"
SOL_CKPT      = REPO_ROOT / "probe_training/solvability/checkpoints"
VERB_CKPT     = REPO_ROOT / "probe_training/verbalization/checkpoints"
ANALYSIS_ROOT = REPO_ROOT / "probe_analysis/combined_analysis"

STYLES = ["standard", "standard_plus_solvability",
          "unsolvable_aware", "unsolvable_biased", "unsolvable_conditioned"]
SPLITS = ["solvable", "unsolvable"]
# unsolvable_conditioned only exists for the unsolvable split (by design).
STYLE_SPLITS = {style: SPLITS for style in STYLES}
STYLE_SPLITS["unsolvable_conditioned"] = ["unsolvable"]
BASELINE_STYLE = "standard"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--all_layers", action="store_true")
    p.add_argument("--tokens_per_problem", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", type=str, default=None)
    p.add_argument("--judge_model", type=str, default=None)
    p.add_argument("--output_model", type=str, default=None)
    p.add_argument("--trace", type=str, default="", choices=["", "response", "thinking"])
    return p.parse_args()


def load_latest_metadata(d: Path) -> Optional[List[Dict]]:
    cands = list(d.glob("metadata_0_*.json"))
    if not cands:
        return None
    def k(p):
        m = re.search(r"metadata_0_(\d+)\.json", p.name)
        return int(m.group(1)) if m else 0
    return json.load(open(max(cands, key=k)))


def load_judge_labels(judge_dir: Path) -> Dict[int, int]:
    label_map = {"solved": 0, "unsolvable": 1}
    f = next(judge_dir.glob("judge_metadata_*.jsonl"), None)
    if f is None:
        return {}
    out = {}
    for line in open(f):
        ex = json.loads(line)
        ann = ex.get("verbal_annotation")
        if ann in label_map:
            out[ex["example_idx"]] = label_map[ann]
    return out


def sample_tokens(T: int, n: int, rng: np.random.Generator) -> np.ndarray:
    if T <= n:
        return np.arange(T)
    edges = np.linspace(0, T, n + 1).astype(int)
    return np.array([rng.integers(edges[i], edges[i + 1]) for i in range(n)])


def score_and_predict(h_raw: np.ndarray, pipe) -> Tuple[float, int]:
    """Returns (mean signed score, majority prediction over tokens)."""
    scaler = pipe.named_steps["scaler"]
    clf    = pipe.named_steps["clf"]
    h_std  = scaler.transform(h_raw)
    scores = h_std @ clf.coef_[0].astype(np.float64) + clf.intercept_[0]
    preds = (scores > 0).astype(int)
    return float(scores.mean()), int(round(preds.mean()))


def discover_layers(model: str) -> List[int]:
    sol  = {int(re.search(r"layer(\d+)", p.name).group(1))
            for p in (SOL_CKPT / model).glob("probe_layer*.joblib")}
    verb = {int(re.search(r"layer(\d+)", p.name).group(1))
            for p in (VERB_CKPT / model).glob("probe_layer*.joblib")}
    return sorted(sol & verb)


def collect(
    model_dir: Path, judge_dir: Path, style: str, split: str, layer: int,
    sol_pipe, verb_pipe, tokens_per_problem: int, seed: int, trace: str,
) -> Dict[int, Dict]:
    d  = model_dir / style / split
    jd = judge_dir / style / split
    meta = load_latest_metadata(d)
    if meta is None:
        return {}
    judge_labels = load_judge_labels(jd)
    infix = f"_{trace}" if trace else ""
    out = {}
    for ex in meta:
        i = ex["example_idx"]
        npz = d / f"ex{i:06d}{infix}_layer{layer}.npz"
        if not npz.exists():
            continue
        h = np.load(npz)["hidden_states"].astype(np.float32)
        if len(h) == 0:
            continue
        rng = np.random.default_rng(seed + i)
        sel = sample_tokens(len(h), tokens_per_problem, rng)
        h_sel = h[sel]
        K_score, K_pred = score_and_predict(h_sel, sol_pipe)
        V_score, V_pred = score_and_predict(h_sel, verb_pipe)
        out[i] = {
            "K_score": K_score, "K_pred": K_pred,
            "V_score": V_score, "V_pred": V_pred,
            "verbal_label": judge_labels.get(i, -1),
        }
    return out


def analyse_layer(model: str, layer: int, model_dir: Path, judge_dir: Path,
                  tokens_per_problem: int, seed: int, out_dir: Path, trace: str):
    logger.info(f"=== Layer {layer} ===")
    sol_pipe  = joblib.load(SOL_CKPT  / model / f"probe_layer{layer}.joblib")
    verb_pipe = joblib.load(VERB_CKPT / model / f"probe_layer{layer}.joblib")

    cells = {}
    for style in STYLES:
        for split in STYLE_SPLITS[style]:
            cells[(style, split)] = collect(
                model_dir, judge_dir, style, split, layer,
                sol_pipe, verb_pipe, tokens_per_problem, seed, trace,
            )

    # Per-cell prediction rates (= % of problems predicted "unsolvable")
    summary = {"layer": layer, "cells": {}, "shifts": {}}
    for (style, split), data in cells.items():
        n = len(data)
        if n == 0:
            summary["cells"][f"{style}/{split}"] = {"n": 0}
            continue
        K_pos = sum(d["K_pred"] for d in data.values()) / n
        V_pos = sum(d["V_pred"] for d in data.values()) / n
        K_mean = sum(d["K_score"] for d in data.values()) / n
        V_mean = sum(d["V_score"] for d in data.values()) / n
        summary["cells"][f"{style}/{split}"] = {
            "n": n,
            "K_pred_unsv_rate": K_pos, "V_pred_unsv_rate": V_pos,
            "K_score_mean": K_mean,   "V_score_mean": V_mean,
        }

    # Paired shifts (standard -> each other style) per split where both exist
    for style in STYLES:
        if style == BASELINE_STYLE:
            continue
        summary["shifts"][style] = {}
        for split in STYLE_SPLITS[style]:
            a = cells.get((BASELINE_STYLE, split), {})
            b = cells.get((style, split), {})
            shared = sorted(set(a) & set(b))
            if not shared:
                summary["shifts"][style][split] = {"n_shared": 0}
                continue
            dK_pred = np.mean([b[i]["K_pred"] - a[i]["K_pred"] for i in shared])
            dV_pred = np.mean([b[i]["V_pred"] - a[i]["V_pred"] for i in shared])
            dK_score = np.mean([b[i]["K_score"] - a[i]["K_score"] for i in shared])
            dV_score = np.mean([b[i]["V_score"] - a[i]["V_score"] for i in shared])
            K_flip_to_unsv = sum(1 for i in shared if a[i]["K_pred"] == 0 and b[i]["K_pred"] == 1) / len(shared)
            V_flip_to_unsv = sum(1 for i in shared if a[i]["V_pred"] == 0 and b[i]["V_pred"] == 1) / len(shared)
            K_flip_to_solv = sum(1 for i in shared if a[i]["K_pred"] == 1 and b[i]["K_pred"] == 0) / len(shared)
            V_flip_to_solv = sum(1 for i in shared if a[i]["V_pred"] == 1 and b[i]["V_pred"] == 0) / len(shared)
            summary["shifts"][style][split] = {
                "n_shared": len(shared),
                "dK_pred_rate": float(dK_pred), "dV_pred_rate": float(dV_pred),
                "dK_score_mean": float(dK_score), "dV_score_mean": float(dV_score),
                "K_flip_to_unsv": K_flip_to_unsv, "V_flip_to_unsv": V_flip_to_unsv,
                "K_flip_to_solv": K_flip_to_solv, "V_flip_to_solv": V_flip_to_solv,
            }

    layer_dir = out_dir / f"layer{layer}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    with open(layer_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # Per-example details (smaller, optional)
    with open(layer_dir / "predictions.json", "w") as f:
        json.dump({
            f"{style}/{split}": cells[(style, split)]
            for style in STYLES for split in STYLE_SPLITS[style]
        }, f, indent=2)
    logger.info(f"  Saved -> {layer_dir}")
    return summary


def main():
    args = parse_args()
    global SOL_CKPT, VERB_CKPT, JUDGE_ROOT, ANALYSIS_ROOT
    if args.version:
        SOL_CKPT      = SOL_CKPT      / args.version
        VERB_CKPT     = VERB_CKPT     / args.version
        JUDGE_ROOT    = JUDGE_ROOT    / args.version
        ANALYSIS_ROOT = ANALYSIS_ROOT / args.version

    output_model = args.output_model or args.model
    judge_model  = args.judge_model  or args.model
    model_dir = OUTPUTS_ROOT / output_model
    judge_dir = JUDGE_ROOT   / judge_model

    for style in STYLES:
        for split in STYLE_SPLITS[style]:
            if not (model_dir / style / split).exists():
                raise FileNotFoundError(f"{style}/{split} missing under {model_dir}")
            if not (judge_dir / style / split).exists():
                raise FileNotFoundError(f"judge {style}/{split} missing under {judge_dir}")

    available = discover_layers(args.model)
    if not available:
        raise RuntimeError("No layers with both probes available.")
    if args.all_layers:
        layers = available
    elif args.layer is not None:
        if args.layer not in available:
            raise ValueError(f"layer {args.layer} not in {available}")
        layers = [args.layer]
    else:
        raise ValueError("Specify --layer or --all_layers.")

    out_dir = ANALYSIS_ROOT / args.model / "vector_prediction_prompt_effect"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Model: {args.model}  layers: {layers}")
    all_summaries = {}
    for L in layers:
        all_summaries[L] = analyse_layer(
            args.model, L, model_dir, judge_dir,
            args.tokens_per_problem, args.seed, out_dir, args.trace,
        )
    with open(out_dir / "all_layers_summary.json", "w") as f:
        json.dump({str(L): s for L, s in all_summaries.items()}, f, indent=2)
    logger.info(f"Cross-layer summary -> {out_dir / 'all_layers_summary.json'}")


if __name__ == "__main__":
    main()
