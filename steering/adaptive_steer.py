#!/usr/bin/env python3
"""
Steering experiment: causally test whether closing the K-V gap rescues
unfaithful abstention failures.

One invocation = ONE (model, layer, positions) config, swept over conditions
and alpha multipliers. Submit many parallel jobs across the grid.

Conditions (unit-normalized directions in raw hidden-state space):
  KAPPA   push along w_V_raw_hat, per-sample magnitude = m*alpha_unit*gap_norm
          (gap_norm in [0,1] = positive K-V gap relative to population p90)
  V       push along w_V_raw_hat by m*alpha_unit
  K       push along w_K_raw_hat by m*alpha_unit
  Random  push along a fixed random unit vector by m*alpha_unit

alpha_unit = mean residual-stream norm at the steered layer.

Populations: all_fab (judge=solved) and unfaithful (baseline high_K_low_V).
Positions: all | last20 | first50.

Outputs: steering/<model>/layer{L}/<positions[_suffix]>/
  config.json, baseline_classifications.jsonl (cached at layer root),
  per_run_results.jsonl, summary.json, dose_response_*.png
A post-hoc LLM judge pass is run via steering_judge.py if --judge_model is set.
"""

import os
import re
import json
import argparse
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
try:
    import joblib            # needs sklearn; absent in the gemma4 env
except ImportError:
    joblib = None
import torch
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None               # in-script plotting skipped; merge_alpha_plot.py does it
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(os.environ.get("KV_ROOT", Path(__file__).resolve().parents[1]))  # repo root (override with $KV_ROOT)
MODELS_ROOT   = REPO_ROOT / "models"
OUTPUTS_ROOT  = REPO_ROOT / "cot_generation/outputs"
JUDGE_ROOT    = REPO_ROOT / "cot_generation/judge_outputs"
SOL_CKPT      = REPO_ROOT / "probe_training/solvability/checkpoints"
VERB_CKPT     = REPO_ROOT / "probe_training/verbalization/checkpoints"
STEERING_ROOT = REPO_ROOT / "steering/results"
SCRIPTS_DIR   = Path(__file__).parent


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--positions", type=str, default="all",
                   choices=["all", "last20", "first50"])
    p.add_argument("--conditions", type=str, default="KAPPA,V,K,Random", help="Subset of {KAPPA,V,K,Random,ACT_V,ACT_GAP,ACT_GAP_orthK,KV,KV_gateK,K_gateK,K_gateK_soft,KV_gateK_soft,V_gateGap_soft,K_gateGap_soft,KV_gateGap_soft}.")
    p.add_argument("--alphas", type=str, default="0.5,1.0,2.0,4.0",
                   help="Multipliers of the magnitude unit (see --magnitude_policy).")
    p.add_argument("--magnitude_policy", type=str, default="resid_norm",
                   choices=["resid_norm", "proj_std"],
                   help="What 'm' multiplies. resid_norm (default, unchanged): m * mean "
                        "residual-stream norm -- same for all conditions. proj_std: m * the "
                        "std of the baseline residual's projection onto that condition's unit "
                        "steering direction, i.e. push in units of the natural variation ALONG "
                        "the direction. Use proj_std for models whose residual norm dwarfs the "
                        "along-direction variation (e.g. gemma: norm~283 but proj std~0.09), "
                        "where resid_norm scaling over-steers into degeneration at any m.")
    p.add_argument("--out_suffix", type=str, default=None,
                   help="Suffix appended to positions subdir, e.g. 'a0.5', to keep "
                        "per-alpha jobs from clobbering each other.")
    p.add_argument("--n_problems", type=int, default=150)
    p.add_argument("--restrict_idxs", type=str, default=None,
                   help="Path to a JSON list of example_idx to restrict candidates to "
                        "(e.g. the probe validation split).")
    p.add_argument("--no_fabricated_filter", action="store_true",
                   help="Steer ALL problems with a question, not only baseline-fabricated "
                        "('solved') ones. With --restrict_idxs, steers the full fixed set. "
                        "Baseline abstention rate is then measured over all of them.")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refresh_baseline", action="store_true")
    p.add_argument("--judge_model", type=str, default=None,
                   help="If set, run post-hoc LLM judge (steering_judge.py) after generation.")
    p.add_argument("--judge_tp", type=int, default=4)
    p.add_argument("--judge_gpu_mem_util", type=float, default=0.85,
                   help="vLLM gpu_memory_utilization for the in-process judge. Lower "
                        "(e.g. 0.80) for larger generation models whose freed CUDA cache "
                        "leaves less headroom; the 0.95 default can fail to start.")
    p.add_argument("--probe_version", type=str, default="", help="Checkpoint subdir (the --run_tag used in train_probe.py); empty = none.")
    p.add_argument("--judge_version", type=str, default="", help="Judge-output subdir; empty = none.")
    p.add_argument("--layer_tag", type=str, default=None)
    p.add_argument("--solvability", type=str, default="unsolvable",
                   choices=["unsolvable", "solvable"],
                   help="unsolvable -> baseline-fabricated (rescue target); "
                        "solvable -> baseline-solved (specificity / false-abstention control). "
                        "Keeps separate baseline cache + results from the default unsolvable runs.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gate_beta", type=float, default=0.0, help="Additive floor on the ACT gate so steering never fully vanishes.")
    p.add_argument("--log_kl", action="store_true", help="Log KL(steered || unsteered) per run (one extra forward).")
    return p.parse_args()


def find_decoder_layers(model):
    """Return the ModuleList of decoder blocks, across architectures.

    Plain CausalLM (Qwen/Llama): model.model.layers
    Multimodal wrappers (Gemma4ForConditionalGeneration): the text decoder is
    nested, e.g. model.model.language_model.layers or model.language_model.layers.
    We probe the known candidate paths and return the first ModuleList found, so
    the steering hook attaches to the actual transformer blocks (matching the
    hidden_states[L] indexing the probes were trained on).
    """
    import torch.nn as nn
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.model.layers,
    ]
    for get in candidates:
        try:
            layers = get(model)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise AttributeError(
        "Could not locate decoder layer ModuleList; add this model's path to "
        "find_decoder_layers(). Inspect with: [n for n,_ in model.named_modules() if n.endswith('layers')]")


def resolve_num_layers(model):
    """num_hidden_layers, falling back to text_config for multimodal configs."""
    n = getattr(model.config, "num_hidden_layers", None)
    if n is None and hasattr(model.config, "text_config"):
        n = getattr(model.config.text_config, "num_hidden_layers", None)
    return n


# ---------------------------------------------------------------------------
# Probe / direction helpers
# ---------------------------------------------------------------------------

class _NpzScaler:
    def __init__(self, mean, var): self.mean_ = mean; self.var_ = var
class _NpzClf:
    def __init__(self, coef, intercept): self.coef_ = coef[None, :]; self.intercept_ = np.array([intercept])
class _NpzProbe:
    """Minimal stand-in for a (StandardScaler, LogisticRegression) Pipeline, built
    from a plain .npz of {mean,var,coef,intercept}. Lets the steering code load
    probes in environments without sklearn/joblib (e.g. the gemma4 env), exposing
    the same .named_steps['scaler'|'clf'] interface the helpers below use."""
    def __init__(self, npz):
        self.named_steps = {
            "scaler": _NpzScaler(npz["mean"].astype(np.float64), npz["var"].astype(np.float64)),
            "clf":    _NpzClf(npz["coef"].astype(np.float64), float(npz["intercept"])),
        }


def load_probe(joblib_path: Path):
    """Load a probe as joblib Pipeline if sklearn/joblib available, else from the
    sibling .npz (pre-extracted params). Prefer .npz when joblib is absent."""
    npz_path = joblib_path.with_suffix(".npz")
    if joblib is None:
        if not npz_path.exists():
            raise FileNotFoundError(
                f"joblib/sklearn unavailable and no npz fallback at {npz_path}. "
                f"Pre-extract it in an sklearn env (mean/var/coef/intercept).")
        return _NpzProbe(np.load(npz_path))
    return joblib.load(joblib_path)


def load_probe_direction_raw(pipe) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (w_std, w_raw, w_raw_hat) for a (StandardScaler, LR) Pipeline."""
    scaler = pipe.named_steps["scaler"]
    clf    = pipe.named_steps["clf"]
    w_std  = clf.coef_[0].astype(np.float64)
    sigma  = np.sqrt(scaler.var_.astype(np.float64))
    w_raw  = w_std / sigma
    w_raw_hat = w_raw / np.linalg.norm(w_raw)
    return w_std, w_raw, w_raw_hat


def project_K_V(h_raw: np.ndarray, sol_pipe, verb_pipe) -> Tuple[float, float]:
    sc_K = sol_pipe.named_steps["scaler"]
    sc_V = verb_pipe.named_steps["scaler"]
    w_K_std = sol_pipe.named_steps["clf"].coef_[0].astype(np.float64)
    w_V_std = verb_pipe.named_steps["clf"].coef_[0].astype(np.float64)
    h_K = (h_raw.astype(np.float64) - sc_K.mean_) / np.sqrt(sc_K.var_)
    h_V = (h_raw.astype(np.float64) - sc_V.mean_) / np.sqrt(sc_V.var_)
    return float((h_K @ w_K_std).mean()), float((h_V @ w_V_std).mean())

def probe_logit_params(pipe, device, dtype=torch.float32):
    """Pack (StandardScaler, LR) into tensors for on-device sigmoid(<w,z>+b)."""
    sc, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    mean      = torch.tensor(sc.mean_,               device=device, dtype=dtype)
    inv_sigma = torch.tensor(1.0 / np.sqrt(sc.var_), device=device, dtype=dtype)
    w         = torch.tensor(clf.coef_[0],           device=device, dtype=dtype)
    b         = torch.tensor(float(clf.intercept_[0]), device=device, dtype=dtype)
    return mean, inv_sigma, w, b


def probe_proba(h: torch.Tensor, params) -> torch.Tensor:
    """h: [B, D] -> [B] probability. Standardize then logistic."""
    mean, inv_sigma, w, b = params
    h = h.to(w.device, torch.float32)
    return torch.sigmoid(((h - mean) * inv_sigma) @ w + b)
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_latest_metadata(d: Path) -> Optional[List[Dict]]:
    cands = list(d.glob("metadata_0_*.json"))
    if not cands:
        return None
    def end_idx(p):
        m = re.search(r"metadata_0_(\d+)\.json", p.name)
        return int(m.group(1)) if m else 0
    return json.load(open(max(cands, key=end_idx)))


def load_judge_labels(judge_dir: Path) -> Dict[int, str]:
    f = next(judge_dir.glob("judge_metadata_*.jsonl"), None)
    if f is None:
        return {}
    out = {}
    for line in open(f):
        rec = json.loads(line)
        out[rec["example_idx"]] = rec.get("verbal_annotation", "unparseable")
    return out


def load_unsol_questions(model: str, solvability: str = "unsolvable") -> Dict[int, Dict]:
    d = OUTPUTS_ROOT / model / "standard" / solvability
    meta = load_latest_metadata(d)
    out = {}
    for ex in meta:
        out[ex["example_idx"]] = {
            "question":  ex["question"],
            "data_id":   ex.get("data_id"),
            "edit_type": ex.get("edit_type"),
        }
    return out


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_standard_prompt(tokenizer, question: str) -> str:
    user_content = (
        "Solve the following math problem step by step.\n"
        "Put your final answer within \\boxed{}.\n"
        f"Problem: {question}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True,
        tokenize=False,
    )


# ---------------------------------------------------------------------------
# Steering hook (pre-hook on the steered block; captures its own input)
# ---------------------------------------------------------------------------

class SteeringHook:
    def __init__(self):
        self.direction      = None
        self.alpha          = 0.0      # base magnitude = m * alpha_unit
        self.policy         = "none"   # token window: all | last20 | first50 | none
        self.gate           = "none"   # intensity: none | v | gap
        self.beta           = 0.0
        self.K_params       = None
        self.V_params       = None
        self.prompt_len     = 0
        self.expected_total = 0
        self.gen_step       = 0
        self.finished       = None     # [B] bool mask set by generate_batch each step
        self.captured_input = []
        self._gate_pK = []; self._gate_pV = []; self._gate_val = []

    def _intensity(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, D] arriving hidden -> [B] per-position push magnitude."""
        base = float(self.alpha)
        finished = self.finished
        active = (~finished) if finished is not None else torch.ones(h.shape[0], dtype=torch.bool, device=h.device)
        if self.gate == "none":
            # unit factor; final `g = base * g` below applies the magnitude once
            g = torch.ones((h.shape[0],), device=h.device, dtype=torch.float32)
            pV = pK = None
        elif self.gate == "kpos":
            # Binary gate on solvability: push at full alpha only when the model
            # "knows" the problem is solvable (pK > 0.5, i.e. K-logit > 0), else 0.
            # No magnitude scaling -- isolates the effect of *gating* vs always-on.
            pK = probe_proba(h, self.K_params)
            g = (pK > 0.5).float() + self.beta
            g = g.clamp(min=0.0)
            pV = None
        elif self.gate == "ksoft":
            # Soft gate on solvability: scale push *continuously* by pK instead of a
            # hard pK>0.5 step. Borderline-K tokens (e.g. pK~0.45) the hard gate slams
            # to 0 still get a proportional (0.45x) nudge, while solvable problems
            # (pK~0) stay near-zero -> keeps specificity but rescues close-K cases.
            pK = probe_proba(h, self.K_params)
            g = pK + self.beta
            g = g.clamp(min=0.0)
            pV = None
        elif self.gate == "gapsoft":
            # Soft gate on the K-V *gap*: push proportional to how far knowledge
            # outruns verbalization, clamp(pK-pV, 0, 1). This is the direct causal
            # analogue of the correlation finding (fabrication = K leads, V lags):
            # it fires exactly in the misalignment regime and no-ops when K and V
            # already agree (gap<=0) -- including solvable problems (low pK -> gap<=0
            # -> no false alarm). Direction is chosen per-condition (V, K, or KV).
            pK = probe_proba(h, self.K_params)
            pV = probe_proba(h, self.V_params)
            g = (pK - pV).clamp(min=0.0, max=1.0) + self.beta
            g = g.clamp(min=0.0)
        else:
            pV = probe_proba(h, self.V_params)
            if self.gate == "v":
                g = (1.0 - pV) + self.beta; pK = None
            elif self.gate == "gap":
                pK = probe_proba(h, self.K_params); g = pK * (1.0 - pV) + self.beta
            else:
                g = torch.ones_like(pV); pK = None
            g = g.clamp(min=0.0)
        # store FULL per-position arrays so stats can be sliced per-sequence later
        if pV is not None: self._gate_pV.append(pV.detach().cpu().numpy())
        if pK is not None: self._gate_pK.append(pK.detach().cpu().numpy())
        self._gate_val.append(g.detach().cpu().numpy())
        g = base * g
        if finished is not None and finished.shape[0] == g.shape[0]:
            g = g.clone(); g[finished] = 0.0
        return g

    def __call__(self, module, args, kwargs):
        if args:
            hidden = args[0]; hidden_is_arg = True
        else:
            hidden = kwargs["hidden_states"]; hidden_is_arg = False

        if self.policy == "none" or self.alpha == 0.0 or self.direction is None:
            self.captured_input.append(hidden.detach().cpu().float())
            return None

        dirv = self.direction.to(hidden.dtype).to(hidden.device)
        T = hidden.shape[1]
        modified = False
        if T == 1:
            if self._should_apply(self.gen_step):
                scale = self._intensity(hidden[:, 0, :]).to(hidden.dtype)
                if torch.any(scale != 0):
                    delta = (scale[:, None] * dirv[None, :]).to(hidden.dtype)
                    hidden = hidden + delta.unsqueeze(1)
                    modified = True
        else:
            gen_start = max(self.prompt_len, 0)
            if gen_start < T:
                positions = [pos for li, pos in enumerate(range(gen_start, T))
                             if self._should_apply(li)]
                if positions:
                    idxs = torch.tensor(positions, device=hidden.device)
                    hsel = hidden[:, idxs, :]
                    scale = self._intensity(hsel.reshape(-1, hsel.shape[-1])).reshape(hsel.shape[:-1])
                    if torch.any(scale != 0):
                        mod = hidden.clone()
                        scale = scale.to(hidden.dtype)
                        mod[:, idxs, :] = hsel + (scale[..., None] * dirv).to(hidden.dtype)
                        hidden = mod
                        modified = True

        self.captured_input.append(hidden.detach().cpu().float())
        if not modified:
            return None
        if hidden_is_arg:
            return ((hidden,) + tuple(args[1:]), kwargs)
        nk = dict(kwargs); nk["hidden_states"] = hidden
        return (args, nk)

    def _should_apply(self, gen_step: int) -> bool:
        if self.policy == "all":
            return True
        if self.policy == "last20":
            if self.expected_total <= 0:
                return False
            return gen_step >= int(self.expected_total * 0.80)
        if self.policy == "first50":
            return gen_step < 50
        return False

    def seq_gate_stats(self, b: int, n_steps: int):
        """Per-sequence gate stats: index b, first n_steps decode steps.
        Arrays were stored per forward call ([B] each)."""
        def m(arrs):
            vals = [a[b] for a in arrs[:n_steps]]
            return float(np.mean(vals)) if vals else 0.0
        return {"gate_pK_mean": m(self._gate_pK),
                "gate_pV_mean": m(self._gate_pV),
                "gate_val_mean": m(self._gate_val)}

    def reset(self):
        self.captured_input = []
        self.finished = None
        self._gate_pK = []; self._gate_pV = []; self._gate_val = []

# ---------------------------------------------------------------------------
# Generation (manual decode loop with KV cache; per-step gen_step tracking)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_with_steering(model, tokenizer, prompt, hook,
                           max_new_tokens, direction, alpha, policy, expected_total, gate="none", beta=0.0, K_params=None, V_params=None,
                           collect_logits=False):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = enc["input_ids"].shape[1]

    hook.direction      = direction
    hook.alpha          = float(alpha)
    hook.policy         = policy
    hook.gate=gate
    hook.beta = float(beta)
    hook.K_params = K_params
    hook.V_params = V_params
    hook.prompt_len     = prompt_len
    hook.expected_total = expected_total
    hook.gen_step       = 0
    hook.reset()

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    past_kv = None
    eos_id = tokenizer.eos_token_id
    gen_ids = []
    steered_logits = [] if collect_logits else None
    for step in range(max_new_tokens):
        hook.gen_step = step
        if past_kv is None:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        else:
            out = model(input_ids=input_ids[:, -1:], attention_mask=attention_mask,
                        past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        last_logits = out.logits[:, -1, :]
        if collect_logits:
            steered_logits.append(last_logits.detach().to("cpu", torch.float16).squeeze(0))
        next_id = last_logits.argmax(dim=-1, keepdim=True)
        gen_ids.append(int(next_id))
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype,
                                        device=attention_mask.device)], dim=1)
        if next_id.item() == eos_id:
            break

    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    cap = hook.captured_input
    if len(cap) > 1:
        gen_h = torch.cat(cap[1:], dim=1).squeeze(0).numpy()
    elif cap:
        gen_h = np.empty((0, cap[0].shape[-1]), dtype=np.float32)
    else:
        gen_h = np.empty((0, model.config.hidden_size), dtype=np.float32)

    n_steps = len(gen_ids)
    return {"text": text, "n_gen": n_steps, "gen_hiddens": gen_h,
            "prompt_len": prompt_len, "full_ids": input_ids,
            "steered_logits": steered_logits,
            "gate_stats": hook.seq_gate_stats(0, n_steps)}

@torch.no_grad()
def generate_batch(model, tokenizer, prompts, hook, max_new_tokens,
                   direction, alpha, gate, beta, K_params, V_params):
    """Batched greedy decode for policy='all'. Left-pads, shares gen_step,
    captures per-sequence hidden at the steered layer. Returns list of dicts."""
    hook.direction=direction; hook.alpha=float(alpha); hook.policy="all"
    hook.gate=gate; hook.beta=float(beta); hook.K_params=K_params; hook.V_params=V_params
    hook.expected_total=max_new_tokens

    tokenizer.padding_side="left"
    enc=tokenizer(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)
    ids=enc["input_ids"]; am=enc["attention_mask"]
    # prevent prefill from steering prompt positions (decode steps use T==1 path)
    hook.prompt_len=ids.shape[1]
    B=ids.shape[0]; eos=tokenizer.eos_token_id
    hook.reset()
    past=None
    gen=[[] for _ in range(B)]
    finished=torch.zeros(B, dtype=torch.bool, device=ids.device)
    # per-seq captured hidden of the generated tokens (decode steps)
    cap=[[] for _ in range(B)]
    for step in range(max_new_tokens):
        hook.gen_step=step
        hook.finished=finished.clone()
        if past is None:
            out=model(input_ids=ids, attention_mask=am, use_cache=True)
        else:
            out=model(input_ids=ids[:, -1:], attention_mask=am, past_key_values=past, use_cache=True)
        past=out.past_key_values
        # capture this step's steered input (skip prefill step 0)
        if step>0 and hook.captured_input:
            h=hook.captured_input[-1]            # [B,1,D]
            for b in range(B):
                if not finished[b]: cap[b].append(h[b,0].numpy())
        nxt=out.logits[:,-1,:].argmax(-1, keepdim=True)
        for b in range(B):
            if not finished[b]:
                gen[b].append(int(nxt[b]))
                if int(nxt[b])==eos: finished[b]=True
        ids=torch.cat([ids,nxt],1)
        am=torch.cat([am, torch.ones((B,1),dtype=am.dtype,device=am.device)],1)
        if bool(finished.all()): break
    res=[]
    for b in range(B):
        gh=np.stack(cap[b]) if cap[b] else np.empty((0,model.config.hidden_size),np.float32)
        n_steps=len(gen[b])
        res.append({"text": tokenizer.decode(gen[b], skip_special_tokens=True),
                    "n_gen": n_steps, "gen_hiddens": gh,
                    "gate_stats": hook.seq_gate_stats(b, n_steps)})
    return res


@torch.no_grad()
def mean_kl_vs_unsteered(model, hook, full_ids, prompt_len, steered_logits) -> float:
    if not steered_logits:
        return 0.0
    saved = (hook.policy, hook.alpha, hook.gate, hook.direction)
    hook.policy, hook.alpha, hook.gate, hook.direction = "none", 0.0, "none", None
    out = model(input_ids=full_ids, use_cache=False)
    hook.policy, hook.alpha, hook.gate, hook.direction = saved
    base_logits = out.logits[0, prompt_len - 1:-1, :]
    G = min(len(steered_logits), base_logits.shape[0])
    kls = []
    for t in range(G):
        ps = torch.log_softmax(steered_logits[t].to(base_logits.device).float(), -1)
        pb = torch.log_softmax(base_logits[t].float(), -1)
        kls.append(float((ps.exp() * (ps - pb)).sum()))
    return float(np.mean(kls)) if kls else 0.0

def classify_baseline(K, V, K_med, V_med):
    hi_K = K >= K_med; hi_V = V >= V_med
    if hi_K and not hi_V: return "high_K_low_V"
    if hi_K and     hi_V: return "high_K_high_V"
    if not hi_K and not hi_V: return "low_K_low_V"
    return "low_K_high_V"


# ---------------------------------------------------------------------------
# Aggregation + plotting (uses judge_label if present, else heur_label)
# ---------------------------------------------------------------------------

def label_of(r):
    return r.get("judge_label", r.get("heur_label", "unparseable"))


def aggregate(baseline_recs, runs):
    quad = {r["example_idx"]: r["quadrant"] for r in runs}
    pops = {"all_fab": lambda q: True, "unfaithful": lambda q: q == "high_K_low_V"}
    buckets = {}
    for r in runs:
        q = quad.get(r["example_idx"])
        for pop, keep in pops.items():
            if keep(q):
                buckets.setdefault((pop, r["condition"], r["alpha_label"]), []).append(label_of(r))
    summary = {"abstention_rates": {}, "baseline_abstention_rate": {}}
    for (pop, cond, lbl), labels in buckets.items():
        n = len(labels); n_abs = sum(1 for l in labels if l == "unsolvable")
        summary["abstention_rates"].setdefault(pop, {}).setdefault(cond, {})[lbl] = \
            {"n": n, "n_abstain": n_abs, "rate": n_abs / n if n else 0.0}
    for pop, recs in {"all_fab": baseline_recs,
                      "unfaithful": [r for r in baseline_recs if r["quadrant"] == "high_K_low_V"]}.items():
        n = len(recs)
        n_abs = sum(1 for r in recs if label_of(r) == "unsolvable")
        summary["baseline_abstention_rate"][pop] = {"n": n, "n_abstain": n_abs,
                                                     "rate": n_abs / n if n else 0.0}
    return summary


def plot_dose_response(summary, cfg, out_dir, runs):
    from collections import defaultdict
    colors  = {"V": "#1f77b4", "K": "#ff7f0e", "Random": "gray", "KAPPA": "#2ca02c",
               "ACT_V": "#9467bd", "ACT_GAP": "#d62728", "ACT_GAP_orthK": "#8c564b",
               "KV": "#17becf", "KV_gateK": "#e377c2"}
    markers = {"V": "o", "K": "s", "Random": "x", "KAPPA": "^",
               "ACT_V": "D", "ACT_GAP": "P", "ACT_GAP_orthK": "*",
               "KV": "v", "KV_gateK": "h"}
    quad = {r["example_idx"]: r["quadrant"] for r in runs}
    pop_keep = {"all_fab": lambda q: True, "unfaithful": lambda q: q == "high_K_low_V"}
    for pop in ("all_fab", "unfaithful"):
        groups = defaultdict(list)
        for r in runs:
            if pop_keep[pop](quad.get(r["example_idx"])):
                # x = multiplier parsed from alpha_label ("0.5xalpha_unit" -> 0.5)
                mult = float(r["alpha_label"].split("x")[0])
                groups[(r["condition"], r["alpha_label"])].append(
                    (mult, 1 if label_of(r) == "unsolvable" else 0))
        if not groups:
            continue
        per_cond = defaultdict(list)
        for (cond, lbl), rows in groups.items():
            per_cond[cond].append((rows[0][0],
                                   float(np.mean([x[1] for x in rows]))))
        fig, ax = plt.subplots(figsize=(8, 5))
        for cond in cfg.get("conditions", ("KAPPA", "V", "K", "Random")):
            pts = sorted((a, r) for a, r in per_cond.get(cond, []) if a > 0)
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker=markers.get(cond, "o"), color=colors.get(cond, "k"),
                        linewidth=2, markersize=8, label=cond)
        bl = summary["baseline_abstention_rate"].get(pop, {})
        if "rate" in bl:
            ax.axhline(bl["rate"], color="black", linestyle=":", linewidth=1,
                       label=f"baseline={bl['rate']:.2f}")
        ax.set_xlabel("steering multiplier  (× residual-stream norm)")
        ax.set_ylabel("abstention rate")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{cfg['model']} L{cfg['layer']} {cfg['positions']} — {pop}  "
                     f"N={bl.get('n','?')}, cos(K,V)={cfg.get('cos_KV', 0):+.2f}")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        f = out_dir / f"dose_response_{pop}.png"
        plt.savefig(str(f), dpi=140, bbox_inches="tight"); plt.close()
        logger.info(f"Saved -> {f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pos_dir = args.positions + (f"_{args.out_suffix}" if args.out_suffix else "")
    ldir = f"layer{args.layer}" + (f"_{args.layer_tag}" if args.layer_tag else "")
    out_dir = STEERING_ROOT / args.model / ldir / pos_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_root = STEERING_ROOT / args.model / ldir
    logger.info(f"Output dir: {out_dir}")

    # Probes & directions
    sol_pipe  = load_probe(SOL_CKPT  / args.probe_version / args.model / f"probe_layer{args.layer}.joblib")
    verb_pipe = load_probe(VERB_CKPT / args.probe_version / args.model / f"probe_layer{args.layer}.joblib")
    _,       _,       w_K_hat = load_probe_direction_raw(sol_pipe)
    w_V_std, w_V_raw, w_V_hat = load_probe_direction_raw(verb_pipe)
    sigma_V = np.sqrt(verb_pipe.named_steps["scaler"].var_.astype(np.float64))
    ΔV_per_α = float(np.dot(w_V_std, w_V_hat / sigma_V))
    cos_KV = float(np.dot(w_K_hat, w_V_hat))
    logger.info(f"cos(K,V)={cos_KV:+.3f}  ΔV/α={ΔV_per_α:.4f}")
    w_V_perp = w_V_hat - np.dot(w_V_hat, w_K_hat) * w_K_hat
    nperp = np.linalg.norm(w_V_perp)
    w_V_perp = w_V_perp / nperp if nperp > 1e-8 else w_V_hat.copy()
    logger.info(f"||V⟂K||={nperp:.3f} cos(V⟂K,K)={float(np.dot(w_V_perp, w_K_hat)):+.1e}")
    # Common K/V direction: equal-weight sum of the two unit probe directions,
    # renormalized to unit. Used by KV (always-on) and KV_gateK (gated on pK>0.5).
    w_KV = w_K_hat + w_V_hat
    w_KV_hat = w_KV / np.linalg.norm(w_KV)
    logger.info(f"||K+V||={np.linalg.norm(w_KV):.3f} "
                f"cos(KV,K)={float(np.dot(w_KV_hat, w_K_hat)):+.3f} "
                f"cos(KV,V)={float(np.dot(w_KV_hat, w_V_hat)):+.3f}")
    rand_vec = rng.standard_normal(w_V_hat.shape).astype(np.float64)
    rand_vec /= np.linalg.norm(rand_vec)

    # Model
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    logger.info(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODELS_ROOT / args.model), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(MODELS_ROOT / args.model), torch_dtype=dtype, device_map="auto",
            attn_implementation=args.attn, trust_remote_code=True)
    except (ValueError, KeyError) as e:
        # Multimodal checkpoints (e.g. Gemma4ForConditionalGeneration) are not
        # registered for AutoModelForCausalLM. Fall back to the generic AutoModel,
        # which loads the full wrapper; find_decoder_layers() locates the text tower.
        logger.warning(f"AutoModelForCausalLM failed ({e}); falling back to AutoModel.")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            str(MODELS_ROOT / args.model), torch_dtype=dtype, device_map="auto",
            attn_implementation=args.attn, trust_remote_code=True)
    model.eval()
    decoder_layers = find_decoder_layers(model)
    n_blocks = resolve_num_layers(model) or len(decoder_layers)
    if args.layer >= len(decoder_layers):
        raise ValueError(f"layer {args.layer} >= n_blocks {len(decoder_layers)}; can't pre-hook.")
    logger.info(f"Loaded. n_blocks={n_blocks}")

    hook = SteeringHook()
    pre_handle = decoder_layers[args.layer].register_forward_pre_hook(hook, with_kwargs=True)
    direction_K = torch.from_numpy(w_K_hat)
    direction_V = torch.from_numpy(w_V_hat)
    direction_R = torch.from_numpy(rand_vec)
    direction_Vperp = torch.from_numpy(w_V_perp)
    direction_KV = torch.from_numpy(w_KV_hat)
    steered_block = decoder_layers[args.layer]
    gate_device = next(steered_block.parameters()).device
    K_params = probe_logit_params(sol_pipe,  gate_device)
    V_params = probe_logit_params(verb_pipe, gate_device)

    def cond_spec(cond):
        return {
            "KAPPA":         (direction_V,     "none"),
            "V":             (direction_V,     "none"),
            "K":             (direction_K,     "none"),
            "Random":        (direction_R,     "none"),
            "ACT_V":         (direction_V,     "v"),
            "ACT_GAP":       (direction_V,     "gap"),
            "ACT_GAP_orthK": (direction_Vperp, "gap"),
            # New ablation: common K/V direction (normalized K_hat + V_hat).
            "KV":            (direction_KV,    "none"),   # always-on, fixed alpha
            "KV_gateK":      (direction_KV,    "kpos"),   # gated: push only when pK>0.5
            # Improvement sweep: steer along the stronger pure-K direction, gated on K.
            "K_gateK":       (direction_K,     "kpos"),   # K dir, hard pK>0.5 gate
            "V_gateK":        (direction_V,     "kpos"),   # V dir, hard pK>0.5 gate
            "K_gateK_soft":  (direction_K,     "ksoft"),  # K dir, soft (pK-scaled) gate
            "KV_gateK_soft": (direction_KV,    "ksoft"),  # KV dir, soft (pK-scaled) gate
            # K-V gap gate (clamp(pK-pV,0,1)): push exactly where knowledge leads
            # verbalization -- the causal test of the K/V misalignment->fabrication story.
            "V_gateGap_soft":  (direction_V,   "gapsoft"),  # PRIMARY: surgical gap-close on V
            "K_gateGap_soft":  (direction_K,   "gapsoft"),  # same gate, knowledge axis
            "KV_gateGap_soft": (direction_KV,  "gapsoft"),  # same gate, combined axis
        }.get(cond, (None, None))
    # Problems (judge=solved on the chosen pool):
    #   unsolvable -> baseline fabricated (the rescue target)
    #   solvable   -> baseline correctly solved (specificity / false-abstention control)
    judge_labels = load_judge_labels(JUDGE_ROOT / args.judge_version / args.model / "standard" / args.solvability)
    questions = load_unsol_questions(args.model, args.solvability)
    if args.no_fabricated_filter:
        cand = sorted(i for i in questions if i in judge_labels)
        logger.info(f"no_fabricated_filter: all {len(cand)} problems (baseline rate measured over these)")
    else:
        cand = sorted(i for i, lbl in judge_labels.items() if lbl == "solved" and i in questions)
    # Optionally restrict to a fixed set of example_idx (e.g. the probe VALIDATION split),
    # so steering is evaluated only on held-out problems the probes were not trained on.
    if args.restrict_idxs:
        keep = set(json.load(open(args.restrict_idxs)))
        before = len(cand)
        cand = [i for i in cand if i in keep]
        logger.info(f"restrict_idxs={args.restrict_idxs}: {before} -> {len(cand)} candidates")
    rng.shuffle(cand)
    cand = cand[:args.n_problems]
    _desc = "baseline-fabricated unsolvable" if args.solvability == "unsolvable" else "baseline-solved solvable"
    logger.info(f"Selected {len(cand)} {_desc} problems")

    # Baseline pass (cached at layer root so all alpha jobs share it)
    baseline_path = layer_root / "baseline_classifications.jsonl"
    proj_std_path = layer_root / "proj_std.json"   # sidecar for proj_std magnitude policy
    _proj_samples = {"K": [], "V": [], "KV": []}    # raw projections, filled during baseline pass
    baseline_recs: List[Dict] = []
    if baseline_path.exists() and not args.refresh_baseline:
        baseline_recs = [json.loads(l) for l in open(baseline_path)]
        cached = {r["example_idx"] for r in baseline_recs}
        to_run = [i for i in cand if i not in cached]
        if to_run:
            logger.info(f"Cache missing {len(to_run)} problems; extending.")
    else:
        to_run = cand
    if to_run:
        logger.info(f"Baseline pass for {len(to_run)} problems ...")
        for i, ex_idx in enumerate(to_run):
            q = questions[ex_idx]
            r = generate_with_steering(model, tokenizer,
                                       build_standard_prompt(tokenizer, q["question"]), hook,
                                       args.max_new_tokens, None, 0.0, "none", args.max_new_tokens)
            if r["gen_hiddens"].shape[0] > 0:
                K, V = project_K_V(r["gen_hiddens"], sol_pipe, verb_pipe)
                resid = float(np.linalg.norm(r["gen_hiddens"], axis=1).mean())
                # raw-space projection of each token onto the unit steering directions,
                # accumulated to estimate the natural along-direction std (proj_std policy).
                h = r["gen_hiddens"].astype(np.float64)
                _proj_samples["K"].append(h @ w_K_hat)
                _proj_samples["V"].append(h @ w_V_hat)
                _proj_samples["KV"].append(h @ w_KV_hat)
            else:
                K, V, resid = 0.0, 0.0, 0.0
            baseline_recs.append({
                "example_idx": ex_idx, "data_id": q["data_id"], "edit_type": q["edit_type"],
                "K_baseline": K, "V_baseline": V, "n_gen": r["n_gen"], "resid_norm": resid,
                "judge_label": judge_labels[ex_idx],
                "heur_label_baseline": "solved",  # filled by LLM judge later via baseline judge
                "text_tail": r["text"][-400:],
            })
            if (i + 1) % 20 == 0:
                logger.info(f"  baseline {i+1}/{len(to_run)}")
        with open(baseline_path, "w") as f:
            for r in baseline_recs:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Saved baseline -> {baseline_path}")

    sel = set(cand)
    baseline_recs = [r for r in baseline_recs if r["example_idx"] in sel]
    Ks = np.array([r["K_baseline"] for r in baseline_recs])
    Vs = np.array([r["V_baseline"] for r in baseline_recs])
    K_med, V_med = float(np.median(Ks)), float(np.median(Vs))
    resid_norms = [r["resid_norm"] for r in baseline_recs if r["resid_norm"] > 0]
    alpha_unit = float(np.mean(resid_norms)) if resid_norms else 1.0

    # proj_std magnitude unit: std of baseline residual's projection onto each unit
    # steering direction. Computed from fresh baseline hiddens if available; else read
    # from the sidecar (so cached-baseline alpha jobs can still use proj_std policy).
    proj_std = {}
    if any(len(v) for v in _proj_samples.values()):
        for k, chunks in _proj_samples.items():
            allp = np.concatenate(chunks) if chunks else np.array([1.0])
            proj_std[k] = float(np.std(allp))
        try:
            with open(proj_std_path, "w") as f: json.dump(proj_std, f)
        except OSError: pass
    elif proj_std_path.exists():
        proj_std = json.load(open(proj_std_path))
    if args.magnitude_policy == "proj_std" and not proj_std:
        raise RuntimeError(
            "magnitude_policy=proj_std but no baseline hiddens were captured and no "
            f"{proj_std_path} sidecar exists. Run once with --refresh_baseline (or an "
            "uncached baseline) so proj_std can be measured.")
    # per-condition magnitude unit
    def magnitude_unit(cond):
        if args.magnitude_policy == "resid_norm":
            return alpha_unit
        # proj_std: pick the std for the direction this condition steers along
        if cond in ("KV", "KV_gateK", "KV_gateK_soft", "KV_gateGap_soft"):
            return proj_std.get("KV", 1.0)
        if cond in ("V", "ACT_V", "ACT_GAP", "ACT_GAP_orthK", "KAPPA",
                    "V_gateK", "V_gateGap_soft"):
            return proj_std.get("V", 1.0)
        return proj_std.get("K", 1.0)   # K, K_gateK, K_gateK_soft, K_gateGap_soft, Random
    for r in baseline_recs:
        r["quadrant"] = classify_baseline(r["K_baseline"], r["V_baseline"], K_med, V_med)
    n_unf = sum(1 for r in baseline_recs if r["quadrant"] == "high_K_low_V")
    pos_gaps = [r["K_baseline"] - r["V_baseline"] for r in baseline_recs
                if (r["K_baseline"] - r["V_baseline"]) > 0]
    gap_denom = float(np.percentile(pos_gaps, 90)) if pos_gaps else 1.0
    logger.info(f"K_med={K_med:+.3f} V_med={V_med:+.3f} alpha_unit={alpha_unit:.3f} "
                f"gap_denom(p90)={gap_denom:.3f} unfaithful={n_unf}/{len(baseline_recs)}")

    conditions = [c for c in args.conditions.split(",") if c]
    alphas_fixed = [float(x) for x in args.alphas.split(",")]
    cfg = {"model": args.model, "layer": args.layer, "positions": pos_dir,
           "conditions": conditions, "alphas_fixed": alphas_fixed,
           "n_problems": len(baseline_recs), "n_unfaithful": n_unf,
           "max_new_tokens": args.max_new_tokens, "K_median": K_med, "V_median": V_med,
           "alpha_unit": alpha_unit, "gap_denom_p90": gap_denom,
           "magnitude_policy": args.magnitude_policy, "proj_std": proj_std,
           "cos_KV": cos_KV, "delta_V_per_alpha": ΔV_per_α}
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Steering loop
    total = len(conditions) * len(alphas_fixed) * len(baseline_recs)
    logger.info(f"Total steered generations: {total}")
    runs_path = out_dir / "per_run_results.jsonl"
    rf = open(runs_path, "w")
    done = 0
    BS = getattr(args, "batch_size", 16)
    for cond in conditions:
        direction, gate = cond_spec(cond)
        if direction is None:
            continue
        cond_unit = magnitude_unit(cond)
        for m in alphas_fixed:
            for s in range(0, len(baseline_recs), BS):
                chunk = baseline_recs[s:s+BS]
                prompts = [build_standard_prompt(tokenizer, questions[b["example_idx"]]["question"])
                           for b in chunk]
                # magnitude = m * (mean residual norm | along-direction std), per policy
                alpha = m * cond_unit
                rs = generate_batch(model, tokenizer, prompts, hook, args.max_new_tokens,
                                    direction, alpha, gate, args.gate_beta, K_params, V_params)
                for br, r in zip(chunk, rs):
                    q = questions[br["example_idx"]]
                    if r["gen_hiddens"].shape[0] > 0:
                        K_s, V_s = project_K_V(r["gen_hiddens"], sol_pipe, verb_pipe)
                    else:
                        K_s, V_s = 0.0, 0.0
                    gs = r["gate_stats"]
                    rf.write(json.dumps({
                        "example_idx": br["example_idx"], "data_id": q["data_id"],
                        "edit_type": br.get("edit_type"),
                        "quadrant": br["quadrant"], "condition": cond, "gate": gate,
                        "alpha": alpha, "alpha_label": f"{m}xalpha_unit",
                        "position": args.positions, "K_steer": K_s, "V_steer": V_s,
                        "K_baseline": br["K_baseline"], "V_baseline": br["V_baseline"],
                        "gate_pK_mean": gs["gate_pK_mean"], "gate_pV_mean": gs["gate_pV_mean"],
                        "gate_val_mean": gs["gate_val_mean"], "kl_vs_unsteered": None,
                        "n_gen": r["n_gen"], "heur_label": "",
                        "text_tail": r["text"][-400:],
                    }) + "\n")
                done += len(chunk)
                rf.flush()
                logger.info(f"  {done}/{total}  ({cond} a={m})")
    rf.close()
    pre_handle.remove()

    # Optional LLM judge pass
    if args.judge_model:
        logger.info("Running LLM judge pass ...")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cmd = [sys.executable, str(SCRIPTS_DIR / "steering_judge.py"),
               "--run_results", str(runs_path),
               "--baseline", str(baseline_path),
               "--judge_model", args.judge_model,
               "--tensor_parallel_size", str(args.judge_tp),
               # Lower than the 0.95 default: the just-freed generation model's CUDA
               # cache is not always fully returned to the OS by empty_cache(), so the
               # in-process judge can see slightly less free memory (e.g. 59.3/63.4 GiB)
               # than 0.95 demands and refuse to start. Configurable for big gen models.
               "--gpu_memory_utilization", str(args.judge_gpu_mem_util)]
        env = dict(os.environ); env["VLLM_USE_V1"] = "0"
        subprocess.run(cmd, check=True, env=env)
        judged = runs_path.with_name("per_run_results_judged.jsonl")
        if judged.exists():
            runs = [json.loads(l) for l in open(judged)]
        else:
            runs = [json.loads(l) for l in open(runs_path)]
    else:
        runs = [json.loads(l) for l in open(runs_path)]

    summary = aggregate(baseline_recs, runs)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if plt is not None:
        plot_dose_response(summary, cfg, out_dir, runs)
    else:
        logger.info("matplotlib unavailable; skipping in-script plot "
                    "(use merge_alpha_plot.py for figures)")
    logger.info(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
