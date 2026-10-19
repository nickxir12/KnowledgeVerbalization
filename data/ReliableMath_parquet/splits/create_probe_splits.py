#!/usr/bin/env python3
"""
Reproduce the exact problem-level train/val split used in train_probe.py
and save index lists to JSON files in this directory.

Split logic mirrors train_probe.py:problem_level_split exactly:
    - shuffle example_idxs with random.Random(seed)
    - first n_val = int(n * val_frac) -> val set
    - remainder -> train set

Outputs:
    unsolvable_train.json   list of example_idx (int)
    unsolvable_val.json
    solvable_train.json
    solvable_val.json
"""

import os
import json
import random
from pathlib import Path

REPO_ROOT = Path(os.environ.get("KV_ROOT", Path(__file__).resolve().parents[3]))  # repo root (override with $KV_ROOT)
OUTPUTS_ROOT = REPO_ROOT / "cot_generation/outputs/llama_31_8b_instruct"
OUT_DIR = Path(__file__).parent

VAL_FRAC = 0.15
SEED     = 42


def problem_level_split(all_idxs, val_frac, seed):
    rng = random.Random(seed)
    shuffled = all_idxs[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]   # train, val


def load_idxs(split_name):
    d = OUTPUTS_ROOT / "standard" / split_name
    candidates = sorted(d.glob("metadata_0_*.json"),
                        key=lambda p: int(p.stem.split("_")[-1]))
    meta = json.load(open(candidates[-1]))
    return [ex["example_idx"] for ex in meta]


def main():
    for split_name in ("unsolvable", "solvable"):
        all_idxs = load_idxs(split_name)
        train, val = problem_level_split(all_idxs, VAL_FRAC, SEED)

        json.dump(sorted(train), open(OUT_DIR / f"{split_name}_train.json", "w"), indent=2)
        json.dump(sorted(val),   open(OUT_DIR / f"{split_name}_val.json",   "w"), indent=2)

        print(f"{split_name}: total={len(all_idxs)}, train={len(train)}, val={len(val)}")


if __name__ == "__main__":
    main()