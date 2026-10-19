#!/usr/bin/env python3
"""
K-V agreement analysis: per-example Pearson r between token-level K and V
probe projections, compared between abstaining vs. fabricating examples.

For each example, K_tok[t] = w_K · h_std[t] and V_tok[t] = w_V · h_std[t].
Pearson r across tokens:
  r > 0  → K and V agree token-by-token (faithful: model "knows" and "says")
  r < 0  → K and V disagree (faithfulness failure: model "knows" but doesn't "say")

Compares mean r between:
  - abstaining examples  (judge: verbal_annotation == "unsolvable")
  - fabricating examples (judge: verbal_annotation == "solved")

Outputs under <out_dir>/kv_agreement/
  kv_agreement_<model>.txt              — per style/layer stats + Welch t-test
  kv_agreement_<model>_<style>.png      — delta across layers per style
  kv_agreement_<model>_cross_style.png  — cross-style summary at midpoint layer

Usage:
    python probe_kv_agreement.py --model llama_31_8b_instruct --version version2
    python probe_kv_agreement.py --all_models --version version2
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(os.environ.get("KV_ROOT", Path(__file__).resolve().parents[1]))  # repo root (override with $KV_ROOT)
BASE         = REPO_ROOT
OUTPUTS_ROOT = BASE / "cot_generation" / "outputs"
JUDGE_ROOT   = BASE / "cot_generation" / "judge_outputs"
SOL_CKPT     = BASE / "probe_training" / "solvability"   / "checkpoints"
VERB_CKPT    = BASE / "probe_training" / "verbalization" / "checkpoints"
RESULTS_ROOT = BASE / "probe_analysis" / "probes_similarities"

EXCLUDED_STYLES = {"solvability_only", "unsolvable_conditioned"}

# (probe_model, output_model, judge_model, trace, label). Reasoning models have
# two probe sets: response trace (top-level folder) and thinking trace (under
# "thinking/"). trace MUST match the probe_model's folder.
MODEL_CONFIGS = [
    ("llama_31_8b_instruct",                  "llama_31_8b_instruct",         "llama_31_8b_instruct",         "",          "llama_8b"),
    ("qwen3_4b_instruct",                     "qwen3_4b_instruct",            "qwen3_4b_instruct",            "",          "qwen3_4b"),
    ("qwen3_30b_instruct",                    "qwen3_30b_instruct",           "qwen3_30b_instruct",           "",          "qwen3_30b"),
    ("gemma-4-31B-it",                        "gemma-4-31B-it",               "gemma-4-31B-it",               "",          "gemma_4_31b"),
    ("deepseek_r1_distill_llama8b",           "deepseek_r1_distill_llama8b",  "deepseek_r1_distill_llama8b",  "response",  "deepseek_8b_response"),
    ("Qwen3-4b-thinking",                     "Qwen3-4b-thinking",            "Qwen3-4b-thinking",            "response",  "qwen3_4b_thinking_response"),
    ("Qwen3-30b-thinking",                    "Qwen3-30b-thinking",           "Qwen3-30b-thinking",           "response",  "qwen3_30b_thinking_response"),
    ("thinking/deepseek_r1_distill_llama8b",  "deepseek_r1_distill_llama8b",  "deepseek_r1_distill_llama8b",  "thinking",  "deepseek_8b_thinking"),
    ("thinking/Qwen3-4b-thinking",            "Qwen3-4b-thinking",            "Qwen3-4b-thinking",            "thinking",  "qwen3_4b_thinking_thinking"),
    ("thinking/Qwen3-30b-thinking",           "Qwen3-30b-thinking",           "Qwen3-30b-thinking",           "thinking",  "qwen3_30b_thinking_thinking"),
]

LABEL_BY_PROBE = {cfg[0]: cfg[4] for cfg in MODEL_CONFIGS}


def short_name(m: str) -> str:
    if m in LABEL_BY_PROBE:
        return LABEL_BY_PROBE[m]
    return m.replace("thinking/", "T_").replace("/", "_")


def discover_layers(model: str, version: Optional[str]) -> List[int]:
    root = SOL_CKPT / version if version else SOL_CKPT
    return sorted(
        int(re.search(r"layer(\d+)", p.name).group(1))
        for p in (root / model).glob("probe_layer*.joblib")
    )


def discover_styles(model_dir: Path) -> List[str]:
    return sorted(p.name for p in model_dir.iterdir()
                  if p.is_dir() and p.name not in EXCLUDED_STYLES)


def load_pipeline(ckpt_root: Path, model: str, layer: int):
    p = ckpt_root / model / f"probe_layer{layer}.joblib"
    return joblib.load(p) if p.exists() else None


def load_latest_metadata(d: Path) -> Optional[List[Dict]]:
    candidates = list(d.glob("metadata_0_*.json"))
    if not candidates:
        return None
    def end_idx(p):
        m = re.search(r"metadata_0_(\d+)\.json", p.name)
        return int(m.group(1)) if m else 0
    return json.load(open(max(candidates, key=end_idx)))


def load_judge_labels(judge_dir: Path, style: str, split: str) -> Dict[int, str]:
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


def project(h: np.ndarray, pipe) -> np.ndarray:
    h_std = pipe.named_steps["scaler"].transform(h.astype(np.float32))
    return h_std @ pipe.named_steps["clf"].coef_[0].astype(np.float64)


def sample_tokens(T: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if T <= n:
        return np.arange(T)
    edges = np.linspace(0, T, n + 1).astype(int)
    return np.array([rng.integers(edges[i], edges[i + 1]) for i in range(n)])


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a)-1)*a.std(ddof=1)**2 + (len(b)-1)*b.std(ddof=1)**2)
                     / (len(a)+len(b)-2))
    return float((a.mean() - b.mean()) / pooled) if pooled else float("nan")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def compute_kv_agreement(
    probe_model: str, output_model: str, judge_model: str, trace: str,
    layers: List[int], version: Optional[str],
    out_dir: Path, tokens_per_problem: int = 20, seed: int = 42,
) -> Dict:
    sol_root  = SOL_CKPT  / version if version else SOL_CKPT
    verb_root = VERB_CKPT / version if version else VERB_CKPT
    judge_dir = (JUDGE_ROOT / version / judge_model) if version else (JUDGE_ROOT / judge_model)
    model_dir = OUTPUTS_ROOT / output_model
    styles    = discover_styles(model_dir)

    all_results: Dict[str, Dict[int, Optional[Dict]]] = {}
    txt_lines = [
        f"K-V Agreement (Pearson r per example) — {probe_model}",
        "=" * 70,
        "r > 0: K and V agree token-by-token (faithful).",
        "r < 0: K and V disagree (model 'knows' but doesn't 'say' — faithfulness failure).",
        "Δ = mean_r(abstain) − mean_r(fabricate)",
        "Significance: Welch t-test on Fisher-z(r) [p], Mann-Whitney U [MWU_p]; d = Cohen's d on z.",
        "  ** p<0.01   * p<0.05",
        "",
    ]

    for style in styles:
        all_results[style] = {}
        txt_lines.append(f"Style: {style}")

        judge_labels = load_judge_labels(judge_dir, style, "unsolvable")
        d = model_dir / style / "unsolvable"
        if not d.exists():
            txt_lines.append("  (directory missing)\n")
            continue

        meta = load_latest_metadata(d)
        if meta is None:
            txt_lines.append("  (no metadata)\n")
            continue

        abstain_idxs = {ex["example_idx"] for ex in meta
                        if judge_labels.get(ex["example_idx"]) == "unsolvable"}
        fab_idxs     = {ex["example_idx"] for ex in meta
                        if judge_labels.get(ex["example_idx"]) == "solved"}

        logger.info(f"  {style}: {len(abstain_idxs)} abstain, {len(fab_idxs)} fab")
        txt_lines.append(f"  Abstaining: {len(abstain_idxs)}  |  Fabricating: {len(fab_idxs)}")

        if len(abstain_idxs) < 3 or len(fab_idxs) < 3:
            txt_lines.append("  (too few samples, skipping)\n")
            continue

        for layer in layers:
            sol_pipe  = load_pipeline(sol_root,  probe_model, layer)
            verb_pipe = load_pipeline(verb_root, probe_model, layer)
            if sol_pipe is None or verb_pipe is None:
                continue

            R_abstain, R_fab = [], []

            for ex in meta:
                ex_idx = ex["example_idx"]
                infix  = f"_{trace}" if trace else ""
                npz_p  = d / f"ex{ex_idx:06d}{infix}_layer{layer}.npz"
                if not npz_p.exists():
                    continue
                h = np.load(npz_p)["hidden_states"].astype(np.float32)
                T = len(h)
                if T < 2:
                    continue

                sel   = sample_tokens(T, tokens_per_problem, seed + ex_idx)
                h_sel = h[sel]
                K_tok = project(h_sel, sol_pipe)
                V_tok = project(h_sel, verb_pipe)

                r = float(np.corrcoef(K_tok, V_tok)[0, 1])
                if np.isnan(r):
                    r = 0.0

                if ex_idx in abstain_idxs:
                    R_abstain.append(r)
                elif ex_idx in fab_idxs:
                    R_fab.append(r)

            Ra = np.array(R_abstain)
            Rf = np.array(R_fab)

            if len(Ra) < 2 or len(Rf) < 2:
                all_results[style][layer] = None
                continue

            # Fisher z-transform stabilizes the variance of Pearson r before testing.
            Za = np.arctanh(np.clip(Ra, -0.999999, 0.999999))
            Zf = np.arctanh(np.clip(Rf, -0.999999, 0.999999))
            t_res = stats.ttest_ind(Za, Zf, equal_var=False)          # Welch on z
            u_res = stats.mannwhitneyu(Ra, Rf, alternative="two-sided")  # rank test (assumption-free)
            d_val = cohen_d(Za, Zf)                                    # effect size on z
            sig   = "**" if t_res.pvalue < 0.01 else ("*" if t_res.pvalue < 0.05 else "")

            layer_res = {
                "abstain_mean": float(Ra.mean()),
                "abstain_std":  float(Ra.std()),
                "fab_mean":     float(Rf.mean()),
                "fab_std":      float(Rf.std()),
                "delta":        float(Ra.mean() - Rf.mean()),
                "t_stat":       float(t_res.statistic),   # Welch t on Fisher-z
                "p_value":      float(t_res.pvalue),
                "mwu_stat":     float(u_res.statistic),   # Mann-Whitney U
                "mwu_p":        float(u_res.pvalue),
                "cohen_d":      d_val,
                "n_abstain":    len(Ra),
                "n_fab":        len(Rf),
            }
            all_results[style][layer] = layer_res
            logger.info(f"    L{layer}  r_abstain={Ra.mean():.3f}±{Ra.std():.3f}  "
                        f"r_fab={Rf.mean():.3f}±{Rf.std():.3f}  "
                        f"Δ={Ra.mean()-Rf.mean():.3f}  p={t_res.pvalue:.2e}  "
                        f"pU={u_res.pvalue:.2e}  d={d_val:.3f} {sig}")
            txt_lines.append(
                f"  Layer {layer:3d}: "
                f"r_abstain={Ra.mean():.3f}±{Ra.std():.3f}  "
                f"r_fab={Rf.mean():.3f}±{Rf.std():.3f}  "
                f"Δ={Ra.mean()-Rf.mean():.3f}  "
                f"t={t_res.statistic:.3f}  p={t_res.pvalue:.2e}  "
                f"MWU_p={u_res.pvalue:.2e}  d={d_val:.3f}  {sig}"
            )
        txt_lines.append("")

    # ── Save txt ────────────────────────────────────────────────────────────
    txt_path = out_dir / "results.txt"
    txt_path.write_text("\n".join(txt_lines))

    # ── Per-style plots: delta across layers ────────────────────────────────
    for style in styles:
        layer_data = {l: v for l, v in all_results.get(style, {}).items() if v}
        if not layer_data:
            continue
        lys    = sorted(layer_data)
        deltas = [layer_data[l]["delta"]   for l in lys]
        pvals  = [layer_data[l]["p_value"] for l in lys]
        r_a    = [layer_data[l]["abstain_mean"] for l in lys]
        r_f    = [layer_data[l]["fab_mean"]     for l in lys]
        r_a_s  = [layer_data[l]["abstain_std"]  for l in lys]
        r_f_s  = [layer_data[l]["fab_std"]      for l in lys]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: mean r per group across layers
        ax = axes[0]
        x  = np.arange(len(lys))
        ax.bar(x - 0.18, r_a, 0.35, yerr=r_a_s, label="abstain", color="steelblue",
               alpha=0.8, capsize=3)
        ax.bar(x + 0.18, r_f, 0.35, yerr=r_f_s, label="fabricate", color="tomato",
               alpha=0.8, capsize=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in lys])
        ax.set_ylabel("Mean Pearson r (K-V per example)")
        ax.set_title(f"K-V agreement by group\n{style}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Right: delta (abstain - fab) with significance stars
        ax = axes[1]
        bar_cols = ["darkgreen" if d > 0 else "crimson" for d in deltas]
        ax.bar(x, deltas, color=bar_cols, edgecolor="white", alpha=0.8)
        for i, pv in enumerate(pvals):
            if pv < 0.05:
                ax.text(i, deltas[i] + (0.01 if deltas[i] >= 0 else -0.02),
                        "*", ha="center", fontsize=12, color="black")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in lys])
        ax.set_ylabel("Δ r (abstain − fab)")
        ax.set_title(f"Δ K-V agreement (* p<0.05)\n{style}")
        ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(f"{short_name(probe_model)} — K-V agreement: abstain vs. fabricate", fontsize=11)
        plt.tight_layout()
        p = out_dir / f"{style}.png"
        plt.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close()

    # ── Cross-style summary plot at midpoint layer ───────────────────────────
    mid_layer = layers[len(layers) // 2] if layers else None
    if mid_layer is not None:
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(styles) * 1.8), 5))
        x = np.arange(len(styles))

        # Left: mean r for each group per style
        ax = axes[0]
        r_a_vals = [all_results.get(s, {}).get(mid_layer, {}) or {} for s in styles]
        means_a  = [v.get("abstain_mean", float("nan")) for v in r_a_vals]
        means_f  = [v.get("fab_mean",     float("nan")) for v in r_a_vals]
        stds_a   = [v.get("abstain_std",  0.0)          for v in r_a_vals]
        stds_f   = [v.get("fab_std",      0.0)          for v in r_a_vals]
        ax.bar(x - 0.18, means_a, 0.35, yerr=stds_a, label="abstain",
               color="steelblue", alpha=0.8, capsize=3)
        ax.bar(x + 0.18, means_f, 0.35, yerr=stds_f, label="fabricate",
               color="tomato", alpha=0.8, capsize=3)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(styles, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Mean Pearson r")
        ax.set_title(f"K-V agreement per style\n(layer {mid_layer})")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Right: delta per style
        ax = axes[1]
        deltas = [v.get("delta",   float("nan")) for v in r_a_vals]
        pvals  = [v.get("p_value", 1.0)          for v in r_a_vals]
        bar_cols = ["darkgreen" if (not np.isnan(d) and d > 0) else "crimson"
                    for d in deltas]
        ax.bar(x, deltas, color=bar_cols, edgecolor="white", alpha=0.8)
        for i, (d, pv) in enumerate(zip(deltas, pvals)):
            if pv < 0.05 and not np.isnan(d):
                ax.text(i, d + (0.01 if d >= 0 else -0.02), "*",
                        ha="center", fontsize=12, color="black")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(styles, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Δ r (abstain − fab)")
        ax.set_title(f"Δ K-V agreement by prompt style (* p<0.05)\n(layer {mid_layer})")
        ax.grid(True, axis="y", alpha=0.3)

        fig.suptitle(f"{short_name(probe_model)} — K-V agreement cross-style", fontsize=11)
        plt.tight_layout()
        p = out_dir / "cross_style.png"
        plt.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"  Saved -> {out_dir}")
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--output_model", default=None)
    p.add_argument("--judge_model",  default=None)
    p.add_argument("--trace", default="", choices=["", "response", "thinking"])
    p.add_argument("--all_models", action="store_true")
    p.add_argument("--version", default=None)
    p.add_argument("--tokens_per_problem", type=int, default=20)
    p.add_argument("--out_dir", type=Path,
                   default=RESULTS_ROOT / "kv_agreement")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_models:
        configs = MODEL_CONFIGS
    elif args.model:
        om = args.output_model or (args.model.split("/")[-1] if "/" in args.model else args.model)
        jm = args.judge_model or om
        configs = [(args.model, om, jm, args.trace, short_name(args.model))]
    else:
        raise ValueError("Specify --model or --all_models.")

    for cfg in configs:
        probe_model, output_model, judge_model, trace = cfg[0], cfg[1], cfg[2], cfg[3]
        label = cfg[4] if len(cfg) > 4 else short_name(probe_model)
        logger.info(f"\n{'='*60}\nModel: {probe_model}  (trace={trace or 'none'}, label={label})\n{'='*60}")
        layers = discover_layers(probe_model, args.version)
        if not layers:
            logger.warning(f"No layers found for {probe_model}, skipping.")
            continue
        model_out = args.out_dir / label
        model_out.mkdir(parents=True, exist_ok=True)
        compute_kv_agreement(
            probe_model, output_model, judge_model, trace,
            layers, args.version, model_out, args.tokens_per_problem,
        )

    logger.info(f"\nAll results in: {args.out_dir}")


if __name__ == "__main__":
    main()