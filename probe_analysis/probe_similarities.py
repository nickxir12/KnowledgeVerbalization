#!/usr/bin/env python3
"""
Probe similarity analysis: compare K (solvability) and V (verbalization) probe
directions across models, layers, and prompt styles.

Analyses performed
------------------
1. K-V cosine similarity per (model, layer)
   - How aligned are the knowledge and verbalization probes in the same model?

2. Cross-model K-K and V-V cosine similarity
   - Do different models learn the same probe direction?

3. Per-prompt, per-layer K and V probe scores
   - Mean ± std of raw probe projections split by: all unsolvable,
     correctly-abstaining (judge: "unsolvable"), fabricating (judge: "solved")

4. Correctly-abstaining vs. fabricating shift in K and V
   - Mean probe score difference, Welch t-test, Cohen's d

5. K-V agreement per example (abstain vs. fabricate)
   - For each example, compute Pearson r between token-level K and V projections.
     r > 0 means K and V agree (both signal unsolvable or both signal solvable).
     r < 0 means they disagree (faithfulness failure: model "knows" but doesn't "say").
   - Compare mean r between abstaining and fabricating examples per style/layer.

Outputs (under --out_dir / probes_similarities)
------------------------------------------------
  kv_cosine/          cosine similarity plots + txt
  cross_model/        K-K and V-V heatmaps per layer + txt
  prompt_scores/      per-prompt mean-score plots + txt
  abstain_vs_fab/     abstain vs fabricate plots + txt (scores + agreement)
  summary.txt         human-readable digest of all results

Usage
-----
    python probe_similarities.py --model llama_31_8b_instruct --version version2
    python probe_similarities.py --model thinking/Qwen3-4b-thinking --version version2 \\
        --output_model Qwen3-4b-thinking --judge_model Qwen3-4b-thinking --trace response
    python probe_similarities.py --all_models --version version2
"""

import re
import os
import json
import argparse
import logging
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get("KV_ROOT", Path(__file__).resolve().parents[1]))  # repo root (override with $KV_ROOT)
BASE = REPO_ROOT
OUTPUTS_ROOT  = BASE / "cot_generation" / "outputs"
JUDGE_ROOT    = BASE / "cot_generation" / "judge_outputs"
SOL_CKPT      = BASE / "probe_training" / "solvability"   / "checkpoints"
VERB_CKPT     = BASE / "probe_training" / "verbalization" / "checkpoints"
RESULTS_ROOT  = BASE / "probe_analysis" / "probes_similarities"

EXCLUDED_STYLES = {"solvability_only", "unsolvable_conditioned"}

# Canonical model configs: (probe_model, output_model, judge_model, trace, label)
#   probe_model:  folder under checkpoints/<version>/ holding probe_layer*.joblib.
#                 For reasoning models, the response-trace probes live at the top
#                 level (e.g. "Qwen3-4b-thinking") and the thinking-trace probes
#                 live under "thinking/" (e.g. "thinking/Qwen3-4b-thinking").
#   trace:        NPZ filename infix selecting which trace's activations to load
#                 ("" for plain instruct models, else "response" | "thinking").
#                 MUST match the probe_model's trace.
#   label:        output subfolder name under each analysis dir (must be unique).
MODEL_CONFIGS = [
    # probe_model                              output_model                    judge_model                     trace        label
    ("llama_31_8b_instruct",                   "llama_31_8b_instruct",         "llama_31_8b_instruct",         "",          "llama_8b"),
    ("qwen3_4b_instruct",                      "qwen3_4b_instruct",            "qwen3_4b_instruct",            "",          "qwen3_4b"),
    ("qwen3_30b_instruct",                     "qwen3_30b_instruct",           "qwen3_30b_instruct",           "",          "qwen3_30b"),
    ("gemma-4-31B-it",                         "gemma-4-31B-it",               "gemma-4-31B-it",               "",          "gemma_4_31b"),
    # Reasoning models — response trace (probes at top level)
    ("deepseek_r1_distill_llama8b",            "deepseek_r1_distill_llama8b",  "deepseek_r1_distill_llama8b",  "response",  "deepseek_8b_response"),
    ("Qwen3-4b-thinking",                      "Qwen3-4b-thinking",            "Qwen3-4b-thinking",            "response",  "qwen3_4b_thinking_response"),
    ("Qwen3-30b-thinking",                     "Qwen3-30b-thinking",           "Qwen3-30b-thinking",           "response",  "qwen3_30b_thinking_response"),
    # Reasoning models — thinking trace (probes under thinking/)
    ("thinking/deepseek_r1_distill_llama8b",   "deepseek_r1_distill_llama8b",  "deepseek_r1_distill_llama8b",  "thinking",  "deepseek_8b_thinking"),
    ("thinking/Qwen3-4b-thinking",             "Qwen3-4b-thinking",            "Qwen3-4b-thinking",            "thinking",  "qwen3_4b_thinking_thinking"),
    ("thinking/Qwen3-30b-thinking",            "Qwen3-30b-thinking",           "Qwen3-30b-thinking",           "thinking",  "qwen3_30b_thinking_thinking"),
]

# Map probe_model -> output folder label (for callers that only have probe_model).
LABEL_BY_PROBE = {cfg[0]: cfg[4] for cfg in MODEL_CONFIGS}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_name(probe_model: str) -> str:
    """Output subfolder label for a probe model. Falls back to a sanitized name."""
    if probe_model in LABEL_BY_PROBE:
        return LABEL_BY_PROBE[probe_model]
    return probe_model.replace("thinking/", "T_").replace("/", "_")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def load_probe_weights(ckpt_root: Path, model: str, layer: int) -> Optional[np.ndarray]:
    p = ckpt_root / model / f"probe_layer{layer}.joblib"
    if not p.exists():
        return None
    pipe = joblib.load(p)
    return pipe.named_steps["clf"].coef_[0].astype(np.float64)


def load_probe_pipeline(ckpt_root: Path, model: str, layer: int):
    p = ckpt_root / model / f"probe_layer{layer}.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


def discover_layers(model: str, version: Optional[str]) -> List[int]:
    root = SOL_CKPT
    if version:
        root = root / version
    return sorted(
        int(re.search(r"layer(\d+)", p.name).group(1))
        for p in (root / model).glob("probe_layer*.joblib")
    )


def discover_styles(model_dir: Path) -> List[str]:
    return sorted(
        p.name for p in model_dir.iterdir()
        if p.is_dir() and p.name not in EXCLUDED_STYLES
    )


def load_judge_labels(judge_dir: Path, style: str, split: str) -> Dict[int, str]:
    """Returns {example_idx: verbal_annotation} for the given split."""
    d = judge_dir / style / split
    if not d.exists():
        return {}
    f = next(d.glob("judge_metadata_*.jsonl"), None)
    if f is None:
        return {}
    result = {}
    for line in open(f):
        ex = json.loads(line)
        ann = ex.get("verbal_annotation")
        if ann:
            result[ex["example_idx"]] = ann
    return result


def load_latest_metadata(d: Path) -> Optional[List[Dict]]:
    candidates = list(d.glob("metadata_0_*.json"))
    if not candidates:
        return None
    def end_idx(p):
        m = re.search(r"metadata_0_(\d+)\.json", p.name)
        return int(m.group(1)) if m else 0
    return json.load(open(max(candidates, key=end_idx)))


def project_pipe(h: np.ndarray, pipe) -> np.ndarray:
    """Standardize then project with classifier weight vector. Returns [T]."""
    scaler = pipe.named_steps["scaler"]
    clf    = pipe.named_steps["clf"]
    h_std  = scaler.transform(h.astype(np.float32))
    return (h_std @ clf.coef_[0].astype(np.float64))


def sample_tokens(T: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if T <= n:
        return np.arange(T)
    edges = np.linspace(0, T, n + 1).astype(int)
    return np.array([rng.integers(edges[i], edges[i + 1]) for i in range(n)])


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 +
                          (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


# ---------------------------------------------------------------------------
# Analysis 1: K-V cosine similarity within a model
# ---------------------------------------------------------------------------

def analysis_kv_cosine(
    model: str, layers: List[int], version: Optional[str], out_dir: Path
) -> Dict:
    """Cosine similarity between K and V weight vectors per layer."""
    sol_root  = SOL_CKPT  / (version or "") if version else SOL_CKPT
    verb_root = VERB_CKPT / (version or "") if version else VERB_CKPT
    if version:
        sol_root  = SOL_CKPT  / version
        verb_root = VERB_CKPT / version

    results = {}
    for layer in layers:
        w_k = load_probe_weights(sol_root,  model, layer)
        w_v = load_probe_weights(verb_root, model, layer)
        if w_k is None or w_v is None:
            results[layer] = None
            continue
        results[layer] = cosine(w_k, w_v)

    # Plot
    valid_layers = [l for l in layers if results[l] is not None]
    sims = [results[l] for l in valid_layers]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(valid_layers)), sims, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(valid_layers)))
    ax.set_xticklabels([f"L{l}" for l in valid_layers])
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"K–V probe cosine similarity\n{short_name(model)}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "results.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()

    # Text
    lines = [f"K-V cosine similarity — {model}", "=" * 60]
    for l in valid_layers:
        lines.append(f"  Layer {l:3d}:  {results[l]:+.6f}")
    lines.append("")
    txt_path = out_dir / "results.txt"
    txt_path.write_text("\n".join(lines))
    logger.info(f"  K-V cosine saved: {plot_path}")
    return results


# ---------------------------------------------------------------------------
# Analysis 2: Cross-model K-K and V-V cosine similarity
# ---------------------------------------------------------------------------

def _collect_weights_all_models(
    configs: List[Tuple], layers_by_model: Dict[str, List[int]],
    ckpt_root: Path
) -> Dict[str, Dict[int, Optional[np.ndarray]]]:
    """weights[probe_model][layer] = weight vector or None."""
    weights = {}
    for cfg in configs:
        probe_model = cfg[0]
        weights[probe_model] = {}
        for layer in layers_by_model.get(probe_model, []):
            weights[probe_model][layer] = load_probe_weights(ckpt_root, probe_model, layer)
    return weights


def analysis_cross_model(
    configs: List[Tuple], layers_by_model: Dict[str, List[int]],
    version: Optional[str], out_dir: Path
):
    """Heatmaps of pairwise K-K and V-V similarity for each layer."""
    sol_root  = SOL_CKPT  / version if version else SOL_CKPT
    verb_root = VERB_CKPT / version if version else VERB_CKPT

    probe_models = [c[0] for c in configs]
    labels = [short_name(m) for m in probe_models]

    k_weights  = _collect_weights_all_models(configs, layers_by_model, sol_root)
    v_weights  = _collect_weights_all_models(configs, layers_by_model, verb_root)

    # Collect all unique layers across models
    all_layers = sorted({l for m in probe_models for l in layers_by_model.get(m, [])})

    txt_lines = ["Cross-model probe cosine similarities", "=" * 70]

    for layer in all_layers:
        n = len(probe_models)
        k_mat = np.full((n, n), np.nan)
        v_mat = np.full((n, n), np.nan)

        for i, m1 in enumerate(probe_models):
            for j, m2 in enumerate(probe_models):
                wk1 = k_weights[m1].get(layer)
                wk2 = k_weights[m2].get(layer)
                wv1 = v_weights[m1].get(layer)
                wv2 = v_weights[m2].get(layer)
                if wk1 is not None and wk2 is not None:
                    k_mat[i, j] = cosine(wk1, wk2)
                if wv1 is not None and wv2 is not None:
                    v_mat[i, j] = cosine(wv1, wv2)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, mat, title in [
            (axes[0], k_mat, "K probe (solvability)"),
            (axes[1], v_mat, "V probe (verbalization)"),
        ]:
            masked = np.ma.masked_invalid(mat)
            im = ax.imshow(masked, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color="black")
            ax.set_title(f"{title}\nLayer {layer}", fontsize=10)

        fig.suptitle(f"Cross-model cosine similarity — Layer {layer}", fontsize=12)
        plt.tight_layout()
        plot_path = out_dir / f"cross_model_layer{layer}.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()

        # Text block for this layer
        txt_lines.append(f"\nLayer {layer}")
        txt_lines.append("  K (solvability) pairwise cosine:")
        hdr = "  " + " " * 20 + "  ".join(f"{l:>20s}" for l in labels)
        txt_lines.append(hdr)
        for i, m in enumerate(probe_models):
            row = f"  {labels[i]:>20s}  " + "  ".join(
                f"{k_mat[i, j]:>20.4f}" if not np.isnan(k_mat[i, j]) else f"{'N/A':>20s}"
                for j in range(n)
            )
            txt_lines.append(row)
        txt_lines.append("  V (verbalization) pairwise cosine:")
        txt_lines.append(hdr)
        for i, m in enumerate(probe_models):
            row = f"  {labels[i]:>20s}  " + "  ".join(
                f"{v_mat[i, j]:>20.4f}" if not np.isnan(v_mat[i, j]) else f"{'N/A':>20s}"
                for j in range(n)
            )
            txt_lines.append(row)

    txt_path = out_dir / "cross_model_summary.txt"
    txt_path.write_text("\n".join(txt_lines))
    logger.info(f"  Cross-model summary saved: {txt_path}")


# ---------------------------------------------------------------------------
# Analysis 3: Per-prompt, per-layer probe scores
# ---------------------------------------------------------------------------

def _collect_probe_scores_for_style(
    model_dir: Path,
    judge_dir: Path,
    style: str,
    split: str,
    layer: int,
    sol_pipe,
    verb_pipe,
    trace: str,
    tokens_per_problem: int = 20,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """Returns {'K': [...], 'V': [...]} for all problems in this style/split."""
    d  = model_dir / style / split
    meta = load_latest_metadata(d)
    if meta is None or not d.exists():
        return {"K": [], "V": []}

    Ks, Vs = [], []
    for ex in meta:
        ex_idx = ex["example_idx"]
        infix  = f"_{trace}" if trace else ""
        npz_p  = d / f"ex{ex_idx:06d}{infix}_layer{layer}.npz"
        if not npz_p.exists():
            continue
        h = np.load(npz_p)["hidden_states"].astype(np.float32)
        T = len(h)
        if T == 0:
            continue
        sel = sample_tokens(T, tokens_per_problem, seed + ex_idx)
        h_sel = h[sel]
        Ks.append(float(project_pipe(h_sel, sol_pipe).mean()))
        Vs.append(float(project_pipe(h_sel, verb_pipe).mean()))

    return {"K": Ks, "V": Vs}


def analysis_prompt_scores(
    probe_model: str, output_model: str, judge_model: str, trace: str,
    layers: List[int], version: Optional[str],
    out_dir: Path, tokens_per_problem: int = 20
) -> Dict:
    sol_root  = SOL_CKPT  / version if version else SOL_CKPT
    verb_root = VERB_CKPT / version if version else VERB_CKPT

    model_dir = OUTPUTS_ROOT / output_model
    styles    = discover_styles(model_dir)

    results: Dict[str, Dict[int, Dict]] = {}   # [style][layer] -> stats

    txt_lines = [
        f"Per-prompt probe scores — {probe_model}",
        "=" * 70,
    ]

    for style in styles:
        results[style] = {}
        txt_lines.append(f"\nStyle: {style}")
        for layer in layers:
            sol_pipe  = load_probe_pipeline(sol_root,  probe_model, layer)
            verb_pipe = load_probe_pipeline(verb_root, probe_model, layer)
            if sol_pipe is None or verb_pipe is None:
                continue

            style_stats = {}
            for split in ("solvable", "unsolvable"):
                sc = _collect_probe_scores_for_style(
                    model_dir, JUDGE_ROOT / (version or "") / judge_model if version
                    else JUDGE_ROOT / judge_model,
                    style, split, layer, sol_pipe, verb_pipe, trace, tokens_per_problem
                )
                Ks = np.array(sc["K"])
                Vs = np.array(sc["V"])
                style_stats[split] = {
                    "n": len(Ks),
                    "K_mean": float(Ks.mean()) if len(Ks) else float("nan"),
                    "K_std":  float(Ks.std())  if len(Ks) else float("nan"),
                    "V_mean": float(Vs.mean()) if len(Vs) else float("nan"),
                    "V_std":  float(Vs.std())  if len(Vs) else float("nan"),
                }

            results[style][layer] = style_stats
            txt_lines.append(
                f"  Layer {layer:3d} | "
                f"sol  K={style_stats.get('solvable',  {}).get('K_mean', float('nan')):.3f}±"
                f"{style_stats.get('solvable',  {}).get('K_std',  float('nan')):.3f}  "
                f"V={style_stats.get('solvable',  {}).get('V_mean', float('nan')):.3f}±"
                f"{style_stats.get('solvable',  {}).get('V_std',  float('nan')):.3f}  |  "
                f"unsol K={style_stats.get('unsolvable', {}).get('K_mean', float('nan')):.3f}±"
                f"{style_stats.get('unsolvable', {}).get('K_std',  float('nan')):.3f}  "
                f"V={style_stats.get('unsolvable', {}).get('V_mean', float('nan')):.3f}±"
                f"{style_stats.get('unsolvable', {}).get('V_std',  float('nan')):.3f}"
            )

    # Save txt
    txt_path = out_dir / "results.txt"
    txt_path.write_text("\n".join(txt_lines))

    # Plots: for each layer, bar chart over styles (K and V side-by-side)
    for layer in layers:
        fig, axes = plt.subplots(1, 2, figsize=(max(8, len(styles) * 1.5), 5))
        x = np.arange(len(styles))
        width = 0.35
        colors = {"solvable": "steelblue", "unsolvable": "tomato"}

        for ax, probe_key, probe_label in [
            (axes[0], "K_mean", "K (solvability probe)"),
            (axes[1], "V_mean", "V (verbalization probe)"),
        ]:
            std_key = probe_key.replace("mean", "std")
            for offset, split in enumerate(("solvable", "unsolvable")):
                means = [results[s].get(layer, {}).get(split, {}).get(probe_key, float("nan"))
                         for s in styles]
                stds  = [results[s].get(layer, {}).get(split, {}).get(std_key, 0.0)
                         for s in styles]
                means = np.array(means, dtype=float)
                stds  = np.array(stds,  dtype=float)
                ax.bar(x + offset * width - width / 2, means, width, yerr=stds,
                       label=split, color=colors[split], alpha=0.8, capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(styles, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel("Mean probe projection")
            ax.set_title(f"{probe_label}\nLayer {layer}")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(f"{short_name(probe_model)} — per-prompt probe scores (L{layer})", fontsize=11)
        plt.tight_layout()
        p = out_dir / f"layer{layer}.png"
        plt.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"  Prompt scores saved: {out_dir}")
    return results


# ---------------------------------------------------------------------------
# Analysis 4: Abstain vs. Fabricate
# ---------------------------------------------------------------------------

def analysis_abstain_vs_fabricate(
    probe_model: str, output_model: str, judge_model: str, trace: str,
    layers: List[int], version: Optional[str],
    out_dir: Path, tokens_per_problem: int = 20, seed: int = 42,
    kv_cosine_res: Optional[Dict] = None,
) -> Dict:
    """
    For each (style, layer): compare K and V probe scores between
    - correctly abstaining examples (verbal_annotation == 'unsolvable')
    - fabricating examples       (verbal_annotation == 'solved')
    on unsolvable problems only.
    """
    sol_root  = SOL_CKPT  / version if version else SOL_CKPT
    verb_root = VERB_CKPT / version if version else VERB_CKPT

    judge_root_model = (JUDGE_ROOT / version / judge_model) if version else (JUDGE_ROOT / judge_model)
    model_dir        = OUTPUTS_ROOT / output_model
    styles           = discover_styles(model_dir)

    all_results: Dict[str, Dict[int, Dict]] = {}
    kv_cosine_res = kv_cosine_res or {}
    txt_lines = [
        f"Abstain vs. Fabricate probe score comparison — {probe_model}",
        "=" * 70,
        "(unsolvable problems only; abstain = judge says 'unsolvable', fab = 'solved')",
        "cos(K,V) = cosine similarity between the K and V probe weight vectors at that layer.",
        "",
    ]

    for style in styles:
        all_results[style] = {}
        txt_lines.append(f"Style: {style}")

        judge_labels = load_judge_labels(judge_root_model, style, "unsolvable")
        d = model_dir / style / "unsolvable"
        if not d.exists():
            txt_lines.append("  (directory missing)\n")
            continue

        meta = load_latest_metadata(d)
        if meta is None:
            txt_lines.append("  (no metadata)\n")
            continue

        # Bucket example indices by judge label
        abstain_idxs = {ex["example_idx"] for ex in meta
                        if judge_labels.get(ex["example_idx"]) == "unsolvable"}
        fab_idxs     = {ex["example_idx"] for ex in meta
                        if judge_labels.get(ex["example_idx"]) == "solved"}

        logger.info(f"  {style}/unsolvable: {len(abstain_idxs)} abstain, {len(fab_idxs)} fab")
        txt_lines.append(f"  Abstaining: {len(abstain_idxs)}  |  Fabricating: {len(fab_idxs)}")

        if len(abstain_idxs) < 3 or len(fab_idxs) < 3:
            txt_lines.append("  (too few samples, skipping)\n")
            continue

        for layer in layers:
            sol_pipe  = load_probe_pipeline(sol_root,  probe_model, layer)
            verb_pipe = load_probe_pipeline(verb_root, probe_model, layer)
            if sol_pipe is None or verb_pipe is None:
                continue

            K_abstain, V_abstain, R_abstain = [], [], []
            K_fab,     V_fab,     R_fab     = [], [], []

            for ex in meta:
                ex_idx = ex["example_idx"]
                infix  = f"_{trace}" if trace else ""
                npz_p  = d / f"ex{ex_idx:06d}{infix}_layer{layer}.npz"
                if not npz_p.exists():
                    continue
                h = np.load(npz_p)["hidden_states"].astype(np.float32)
                T = len(h)
                if T == 0:
                    continue
                sel = sample_tokens(T, tokens_per_problem, seed + ex_idx)
                h_sel = h[sel]
                K_tok = project_pipe(h_sel, sol_pipe)   # shape [n_tokens]
                V_tok = project_pipe(h_sel, verb_pipe)  # shape [n_tokens]
                K_val = float(K_tok.mean())
                V_val = float(V_tok.mean())

                # Per-example K-V agreement: Pearson r across token projections.
                # r > 0: K and V agree token-by-token (faithful).
                # r < 0: K and V disagree (model "knows" but doesn't "say").
                if len(K_tok) >= 2:
                    r_val = float(np.corrcoef(K_tok, V_tok)[0, 1])
                    if np.isnan(r_val):
                        r_val = 0.0
                else:
                    r_val = 0.0

                if ex_idx in abstain_idxs:
                    K_abstain.append(K_val)
                    V_abstain.append(V_val)
                    R_abstain.append(r_val)
                elif ex_idx in fab_idxs:
                    K_fab.append(K_val)
                    V_fab.append(V_val)
                    R_fab.append(r_val)

            Ka = np.array(K_abstain)
            Kf = np.array(K_fab)
            Va = np.array(V_abstain)
            Vf = np.array(V_fab)
            Ra = np.array(R_abstain)
            Rf = np.array(R_fab)

            layer_res = {}
            for probe_key, a_arr, f_arr in [("K", Ka, Kf), ("V", Va, Vf)]:
                if len(a_arr) < 2 or len(f_arr) < 2:
                    layer_res[probe_key] = None
                    continue
                t_res = stats.ttest_ind(a_arr, f_arr, equal_var=False)
                d_val = cohen_d(a_arr, f_arr)
                layer_res[probe_key] = {
                    "abstain_mean": float(a_arr.mean()),
                    "abstain_std":  float(a_arr.std()),
                    "fab_mean":     float(f_arr.mean()),
                    "fab_std":      float(f_arr.std()),
                    "delta":        float(a_arr.mean() - f_arr.mean()),
                    "t_stat":       float(t_res.statistic),
                    "p_value":      float(t_res.pvalue),
                    "cohen_d":      d_val,
                    "n_abstain":    len(a_arr),
                    "n_fab":        len(f_arr),
                }
                txt_lines.append(
                    f"  Layer {layer:3d}  {probe_key}: "
                    f"abstain={a_arr.mean():.3f}±{a_arr.std():.3f}  "
                    f"fab={f_arr.mean():.3f}±{f_arr.std():.3f}  "
                    f"Δ={a_arr.mean()-f_arr.mean():.3f}  "
                    f"t={t_res.statistic:.3f}  p={t_res.pvalue:.4f}  d={d_val:.3f}"
                )

            # cos(K,V): geometric alignment of the two probe directions at this layer
            cos_kv = kv_cosine_res.get(layer)
            layer_res["cos_KV"] = float(cos_kv) if cos_kv is not None else None
            if cos_kv is not None:
                txt_lines.append(f"  Layer {layer:3d}  cos(K,V): {cos_kv:+.4f}")

            # K-V agreement (Pearson r per example)
            if len(Ra) >= 2 and len(Rf) >= 2:
                t_r = stats.ttest_ind(Ra, Rf, equal_var=False)
                d_r = cohen_d(Ra, Rf)
                layer_res["KV_agreement"] = {
                    "abstain_mean": float(Ra.mean()),
                    "abstain_std":  float(Ra.std()),
                    "fab_mean":     float(Rf.mean()),
                    "fab_std":      float(Rf.std()),
                    "delta":        float(Ra.mean() - Rf.mean()),
                    "t_stat":       float(t_r.statistic),
                    "p_value":      float(t_r.pvalue),
                    "cohen_d":      d_r,
                    "n_abstain":    len(Ra),
                    "n_fab":        len(Rf),
                }
                sig = "**" if t_r.pvalue < 0.01 else ("*" if t_r.pvalue < 0.05 else "")
                txt_lines.append(
                    f"  Layer {layer:3d}  KV_agreement(r): "
                    f"abstain={Ra.mean():.3f}±{Ra.std():.3f}  "
                    f"fab={Rf.mean():.3f}±{Rf.std():.3f}  "
                    f"Δ={Ra.mean()-Rf.mean():.3f}  "
                    f"t={t_r.statistic:.3f}  p={t_r.pvalue:.4f}  d={d_r:.3f}  {sig}"
                )
            else:
                layer_res["KV_agreement"] = None

            all_results[style][layer] = layer_res
        txt_lines.append("")

    # Save txt
    txt_path = out_dir / "results.txt"
    txt_path.write_text("\n".join(txt_lines))

    # Plot: per style, across layers — delta for K, V, and KV agreement
    for style in styles:
        if not all_results.get(style):
            continue
        layer_list = sorted(l for l in all_results[style] if all_results[style][l])
        if not layer_list:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, pk, title in [
            (axes[0], "K",            "K (solvability probe)\nΔ abstain−fab"),
            (axes[1], "V",            "V (verbalization probe)\nΔ abstain−fab"),
            (axes[2], "KV_agreement", "K-V agreement (Pearson r)\nΔ abstain−fab"),
        ]:
            valid = [(l, all_results[style][l][pk]) for l in layer_list
                     if all_results[style][l] and all_results[style][l].get(pk)]
            if not valid:
                ax.set_visible(False)
                continue
            lys, stats_list = zip(*valid)
            deltas = [s["delta"] for s in stats_list]
            pvals  = [s["p_value"] for s in stats_list]
            bar_colors = ["darkgreen" if d > 0 else "crimson" for d in deltas]
            ax.bar(range(len(lys)), deltas, color=bar_colors, edgecolor="white", alpha=0.8)
            for i, pv in enumerate(pvals):
                if pv < 0.05:
                    ax.text(i, deltas[i] + (0.02 if deltas[i] >= 0 else -0.04),
                            "*", ha="center", fontsize=12, color="black")
            ax.set_xticks(range(len(lys)))
            ax.set_xticklabels([f"L{l}" for l in lys])
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_ylabel("Δ (abstain − fab)")
            ax.set_title(f"{title}\n{style}", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(f"{short_name(probe_model)} — abstain vs. fabricate (* p<0.05)", fontsize=11)
        plt.tight_layout()
        p = out_dir / f"{style}.png"
        plt.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close()

    # Summary cross-style plot: delta at midpoint layer per style (K, V, KV agreement)
    fig, axes = plt.subplots(1, 3, figsize=(max(12, len(styles) * 2), 5))
    x = np.arange(len(styles))
    mid_layer = layers[len(layers) // 2] if layers else None

    for ax, pk, title in [
        (axes[0], "K",            "K probe (solvability)"),
        (axes[1], "V",            "V probe (verbalization)"),
        (axes[2], "KV_agreement", "K-V agreement (Pearson r)"),
    ]:
        if mid_layer is None:
            ax.set_visible(False)
            continue
        deltas, sig = [], []
        for style in styles:
            sr = all_results.get(style, {}).get(mid_layer, {})
            if sr and sr.get(pk):
                deltas.append(sr[pk]["delta"])
                sig.append(sr[pk]["p_value"] < 0.05)
            else:
                deltas.append(float("nan"))
                sig.append(False)

        bar_colors = ["darkgreen" if (not np.isnan(d) and d > 0) else "crimson"
                      for d in deltas]
        ax.bar(x, deltas, color=bar_colors, edgecolor="white", alpha=0.8)
        for i, (d, s) in enumerate(zip(deltas, sig)):
            if s and not np.isnan(d):
                ax.text(i, d + (0.02 if d >= 0 else -0.04), "*",
                        ha="center", fontsize=12, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(styles, rotation=35, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Δ (abstain − fab)")
        ax.set_title(f"{title}\n(layer {mid_layer})", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{short_name(probe_model)} — abstain vs. fab by prompt style (* p<0.05)", fontsize=11)
    plt.tight_layout()
    p = out_dir / "cross_style.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"  Abstain-vs-fab saved: {out_dir}")
    return all_results


# ---------------------------------------------------------------------------
# Master summary
# ---------------------------------------------------------------------------

def write_master_summary(
    kv_results: Dict[str, Dict],
    abstain_results: Dict[str, Dict],
    probe_models: List[str],
    version: Optional[str],
    out_dir: Path,
):
    lines = [
        "=" * 70,
        "PROBE SIMILARITIES — MASTER SUMMARY",
        f"Version: {version or 'default'}",
        "=" * 70,
        "",
        "1. K-V COSINE SIMILARITY (higher = probes more aligned)",
        "-" * 70,
    ]
    for m in probe_models:
        r = kv_results.get(m, {})
        lines.append(f"  {short_name(m):35s}: " +
                     "  ".join(f"L{l}:{v:+.3f}" if v is not None else f"L{l}:N/A"
                               for l, v in sorted(r.items())))
    lines.extend([
        "",
        "2. ABSTAIN vs. FABRICATE — HIGHLIGHT (K, V, KV-agreement delta per layer)",
        "   KV_agreement = Pearson r between token-level K and V projections per example.",
        "   Δ > 0 means abstaining examples show MORE agreement/higher score than fabricating.",
        "-" * 70,
    ])
    for m in probe_models:
        ar = abstain_results.get(m, {})
        for style, layer_data in ar.items():
            for layer, probe_data in sorted(layer_data.items()):
                if not probe_data:
                    continue
                kd  = probe_data.get("K")
                vd  = probe_data.get("V")
                rd  = probe_data.get("KV_agreement")
                if not kd or not vd:
                    continue
                sig_k = "**" if kd["p_value"] < 0.01 else ("*" if kd["p_value"] < 0.05 else "")
                sig_v = "**" if vd["p_value"] < 0.01 else ("*" if vd["p_value"] < 0.05 else "")
                sig_r = ""
                r_str = "N/A"
                if rd:
                    sig_r = "**" if rd["p_value"] < 0.01 else ("*" if rd["p_value"] < 0.05 else "")
                    r_str = f"{rd['delta']:+.3f}{sig_r}"
                lines.append(
                    f"  {short_name(m):25s}  {style:30s}  L{layer:3d}  "
                    f"ΔK={kd['delta']:+.3f}{sig_k:2s}  ΔV={vd['delta']:+.3f}{sig_v:2s}  "
                    f"ΔR={r_str}"
                )
    lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines))
    logger.info(f"Master summary: {out_dir / 'summary.txt'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None,
                   help="Probe model path (e.g. llama_31_8b_instruct or "
                        "thinking/Qwen3-4b-thinking). Omit for --all_models.")
    p.add_argument("--output_model", default=None,
                   help="Model name under cot_generation/outputs (if different from --model).")
    p.add_argument("--judge_model", default=None,
                   help="Model name under judge_outputs (if different from --model).")
    p.add_argument("--trace", default="",
                   choices=["", "response", "thinking"],
                   help="NPZ filename infix for thinking models.")
    p.add_argument("--all_models", action="store_true",
                   help="Run for all canonical model configurations.")
    p.add_argument("--version", default=None,
                   help="Checkpoint/judge version subdir (train_probe --run_tag); default none.")
    p.add_argument("--tokens_per_problem", type=int, default=20,
                   help="Tokens sampled per problem for probe projection.")
    p.add_argument("--out_dir", type=Path, default=RESULTS_ROOT,
                   help="Output root (default: …/probe_analysis/probes_similarities).")
    p.add_argument("--skip_cross_model", action="store_true",
                   help="Skip cross-model similarity (useful when running single model).")
    return p.parse_args()


def run_for_config(cfg: Tuple, version: str, out_root: Path, tokens_per_problem: int):
    probe_model, output_model, judge_model, trace = cfg[0], cfg[1], cfg[2], cfg[3]
    label = cfg[4] if len(cfg) > 4 else short_name(probe_model)
    logger.info(f"\n{'='*60}")
    logger.info(f"Model: {probe_model}  (trace={trace or 'none'}, label={label})")
    logger.info(f"{'='*60}")

    layers = discover_layers(probe_model, version)
    if not layers:
        logger.warning(f"No layers found for {probe_model}, skipping.")
        return {}, {}

    logger.info(f"Layers: {layers}")

    # K-V cosine
    kv_out = out_root / "kv_cosine" / label
    kv_out.mkdir(parents=True, exist_ok=True)
    kv_res = analysis_kv_cosine(probe_model, layers, version, kv_out)

    # Prompt scores
    ps_out = out_root / "prompt_scores" / label
    ps_out.mkdir(parents=True, exist_ok=True)
    analysis_prompt_scores(
        probe_model, output_model, judge_model, trace,
        layers, version, ps_out, tokens_per_problem
    )

    # Abstain vs. fabricate (kv_res supplies the per-layer cos(K,V) column)
    af_out = out_root / "abstain_vs_fab" / label
    af_out.mkdir(parents=True, exist_ok=True)
    af_res = analysis_abstain_vs_fabricate(
        probe_model, output_model, judge_model, trace,
        layers, version, af_out, tokens_per_problem,
        kv_cosine_res=kv_res,
    )

    return kv_res, af_res


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_models:
        configs = MODEL_CONFIGS
    elif args.model:
        probe_model  = args.model
        output_model = args.output_model or args.model.split("/")[-1] if "/" in args.model else args.model
        judge_model  = args.judge_model  or output_model
        configs = [(probe_model, output_model, judge_model, args.trace)]
    else:
        raise ValueError("Specify --model or --all_models.")

    kv_all      = {}
    abstain_all = {}

    for cfg in configs:
        kv_res, af_res = run_for_config(cfg, args.version, args.out_dir, args.tokens_per_problem)
        kv_all[cfg[0]]      = kv_res
        abstain_all[cfg[0]] = af_res

    # Cross-model similarity (all models together)
    if not args.skip_cross_model:
        cm_out = args.out_dir / "cross_model"
        cm_out.mkdir(parents=True, exist_ok=True)
        layers_by_model = {c[0]: discover_layers(c[0], args.version) for c in configs}
        analysis_cross_model(configs, layers_by_model, args.version, cm_out)

    # Master summary
    write_master_summary(
        kv_all, abstain_all, [c[0] for c in configs],
        args.version, args.out_dir
    )

    logger.info(f"\nAll results in: {args.out_dir}")


if __name__ == "__main__":
    main()