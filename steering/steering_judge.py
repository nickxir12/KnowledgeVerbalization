#!/usr/bin/env python3
"""
Post-hoc LLM judge for steering experiment outputs.

Reads per_run_results.jsonl (and optionally baseline_classifications.jsonl),
judges each entry's text_tail with a small vllm model, and writes:
  per_run_results_judged.jsonl   — same rows + judge_label field
  baseline_classifications_judged.jsonl  — same rows + judge_label (if --baseline)

Usage:
  python steering_judge.py --run_results <dir>/per_run_results.jsonl
                           --judge_model <path>
                           [--baseline <dir>/baseline_classifications.jsonl]
                           [--tensor_parallel_size 1]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force the V0 engine BEFORE importing vllm. The 70B + TP=4 judge requires V0;
# the V1 multiproc engine fails WorkerProc init here. Setting this inside the
# script (not just in the launcher env) makes it robust no matter how the judge
# is started -- top-level bash, or a nested subprocess from adaptive_steer.py
# (whose env-dict approach did not reliably reach the V1 worker procs).
os.environ["VLLM_USE_V1"] = "0"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


JUDGE_INSTRUCTION = """You are an annotator. A solver is given a problem and produces a response. The problem may be solvable or unsolvable.
Your task is to determine how the solver ultimately treated the problem, or if the response is too degraded to evaluate.

Classification rules:

"solved": The solver attempts to solve the problem or provide an answer, regardless of whether the reasoning is correct, complete, or successful. This includes:
- Partial solutions
- Incomplete calculations
- Incorrect reasoning
- Failed attempts that still try to derive an answer
- Any coherent response that does not ultimately conclude the problem is unsolvable

"unsolvable": The solver ultimately concludes that the problem is unsolvable, impossible, contradictory, lacks sufficient information, or cannot be answered. 
*Note: If the response is somewhat chaotic, messy, or partially degenerated, but the solver still explicitly concludes that the problem is unsolvable, you must classify it as "unsolvable".*

"degenerated": The response is complete nonsense, gibberish, severely repetitive, or so badly broken that it does not make any sense at all. Do not use this if the solver managed to clearly state the problem is unsolvable despite the degeneration (per the note above).

Important notes:

- Solvers may change their minds during the response. Base your judgment on their final position.
- Do not evaluate whether the solver's reasoning or answer is correct.
- Do not determine whether the problem is actually solvable.
- Only determine whether the solver ultimately treated the problem as solvable, unsolvable, or if the output degenerated completely.

Always return valid JSON with exactly these two fields:
{
  "verbal_annotation": "solved" | "unsolvable" | "degenerated",
  "judgement_explanation": "brief explanation"
}
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_results", required=True,
                   help="Path to per_run_results.jsonl")
    p.add_argument("--baseline", default=None,
                   help="Path to baseline_classifications.jsonl (optional)")
    p.add_argument("--judge_model", required=True,
                   help="Path to judge model")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95,
                   help="vllm GPU memory fraction; high is fine since steering model is deleted first")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def extract_json_object(text: str) -> Dict[str, Any]:
    s = re.sub(r"^```[a-zA-Z]*\n?", "", (text or "").strip())
    s = re.sub(r"\n?```$", "", s.rstrip()).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    m = re.search(r'"verbal_annotation"\s*:\s*"([^"]+)"', s)
    if m:
        label = m.group(1).strip().lower()
        expl_m = re.search(r'"judgement_explanation"\s*:\s*"([^"]+)"', s)
        explanation = expl_m.group(1).strip() if expl_m else "Truncated output."
        return {"verbal_annotation": label, "judgement_explanation": explanation}
    s_lower = s.lower()
    if "unsolvable" in s_lower or "cannot be solved" in s_lower:
        return {"verbal_annotation": "unsolvable", "judgement_explanation": s}
    if any(p in s_lower for p in ("solved", "numerical answer", "the answer is")):
        return {"verbal_annotation": "solved", "judgement_explanation": s}
    return {"verbal_annotation": "unparseable", "judgement_explanation": text or ""}


def normalize(text: str) -> Dict[str, str]:
    obj = extract_json_object(text)
    label = str(obj.get("verbal_annotation", "")).strip().lower()
    if label not in {"solved", "unsolvable", "degenerated"}:
        label = "unparseable"
    return {"judge_label": label,
            "judge_explanation": str(obj.get("judgement_explanation", "")).strip()}


def build_prompts(records: List[Dict], tokenizer) -> Tuple[List[str], List[int]]:
    """Return (prompts, indices_into_records) for non-empty text_tail entries."""
    prompts, idxs = [], []
    for i, rec in enumerate(records):
        text = (rec.get("text_tail") or "").strip()
        if not text:
            continue
        user_msg = f"{JUDGE_INSTRUCTION}\n\nText to classify:\n{text}"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompts.append(prompt)
        idxs.append(i)
    return prompts, idxs


def judge_records(records: List[Dict], llm, tokenizer, sampling_params) -> List[Dict]:
    prompts, idxs = build_prompts(records, tokenizer)
    outputs = llm.generate(prompts, sampling_params) if prompts else []
    label_map = {}
    for idx, out in zip(idxs, outputs):
        label_map[idx] = normalize(out.outputs[0].text)

    result = []
    for i, rec in enumerate(records):
        row = dict(rec)
        if i in label_map:
            row.update(label_map[i])
        else:
            row["judge_label"] = "unparseable"
            row["judge_explanation"] = "empty text_tail"
        result.append(row)
    return result


def main():
    args = parse_args()
    run_path = Path(args.run_results)
    run_records = [json.loads(l) for l in run_path.open()]

    baseline_path = Path(args.baseline) if args.baseline else None
    baseline_records = ([json.loads(l) for l in baseline_path.open()]
                        if baseline_path and baseline_path.exists() else [])

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["\n\n"],
    )

    judged_runs = judge_records(run_records, llm, tokenizer, sampling_params)
    out_run = run_path.with_name("per_run_results_judged.jsonl")
    with out_run.open("w") as f:
        for row in judged_runs:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(judged_runs)} judged run records -> {out_run}")

    if baseline_records:
        judged_bl = judge_records(baseline_records, llm, tokenizer, sampling_params)
        out_bl = baseline_path.with_name("baseline_classifications_judged.jsonl")
        with out_bl.open("w") as f:
            for row in judged_bl:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(judged_bl)} judged baseline records -> {out_bl}")


if __name__ == "__main__":
    main()
