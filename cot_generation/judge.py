#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


JUDGE_INSTRUCTION = """You are an annotator. A solver is given a problem and produces a response. The problem may be solvable or unsolvable.
Your task is to determine how the solver ultimately treated the problem.

Classification rules:

"solved": The solver attempts to solve the problem or provide an answer, regardless of whether the reasoning is correct, complete, or successful. This includes:
Partial solutions
Incomplete calculations
Incorrect reasoning
Failed attempts that still try to derive an answer
Any response that does not ultimately conclude the problem is unsolvable

"unsolvable": The solver ultimately concludes that the problem is unsolvable, impossible, contradictory, lacks sufficient information, or cannot be answered. If the solver reaches this conclusion and does not later return to solving the problem, classify it as "unsolvable".

Important notes:

Solvers may change their minds during the response. Base your judgment on their final position.
Do not evaluate whether the solver's reasoning or answer is correct.
Do not determine whether the problem is actually solvable.
Only determine whether the solver ultimately treated the problem as solvable (and attempted to solve it) or as unsolvable.

Always return valid JSON with exactly these two fields:
{
  "verbal_annotation": "solved" | "unsolvable",
  "judgement_explanation": "brief explanation"
}

"""



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge_model", required=True)

    p.add_argument(
        "--judge_part",
        choices=["response", "thinking", "both"],
        default="response",
    )

    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max_model_len", type=int, default=None)

    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--use_chat_template", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


def extract_json_object(text: str) -> Dict[str, Any]:
    s = (text or "").strip()

    # strip opening code fence (closing fence may be absent if output was truncated)
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s.rstrip())
    s = s.strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    # try to find a complete {...} object
    m = re.search(r"\{.*?\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    # output was truncated mid-JSON — extract verbal_annotation directly
    m = re.search(r'"verbal_annotation"\s*:\s*"([^"]+)"', s)
    if m:
        label = m.group(1).strip().lower()
        expl_m = re.search(r'"judgement_explanation"\s*:\s*"([^"]+)"', s)
        explanation = expl_m.group(1).strip() if expl_m else "Truncated output."
        return {"verbal_annotation": label, "judgement_explanation": explanation}

    # no JSON at all — model returned plain prose, infer label from semantics
    s_lower = s.lower()

    # explicit "unsolvable" verdict wins (e.g. "the solver concludes it is unsolvable")
    if "unsolvable" in s_lower or "cannot be solved" in s_lower or "impossible to solve" in s_lower:
        return {"verbal_annotation": "unsolvable", "judgement_explanation": s}

    # any indication the solver tried to solve → "solved" (per the rule: attempted = solved)
    solved_phrases = (
        "solved",
        "attempt to solve",
        "attempts to solve",
        "attempting to solve",
        "tries to solve",
        "trying to solve",
        "provides a numerical answer",
        "gives a numerical answer",
        "indicating an attempt",
        "the answer is",
    )
    if any(p in s_lower for p in solved_phrases):
        return {"verbal_annotation": "solved", "judgement_explanation": s}

    # model refused to classify (e.g. "no text to classify") — treat as unsolvable
    if any(phrase in s_lower for phrase in ("no conclusion", "no text", "there is no", "cannot classify", "no attempt")):
        return {"verbal_annotation": "unsolvable", "judgement_explanation": s}

    # nothing matched — return sentinel so the pipeline never crashes
    return {"verbal_annotation": "unparseable", "judgement_explanation": text or ""}


def normalize_judge_output(text: str) -> Dict[str, str]:
    obj = extract_json_object(text)

    label = str(obj.get("verbal_annotation", "")).strip().lower()
    explanation = str(obj.get("judgement_explanation", "")).strip()

    if label not in {"solved", "unsolvable", "unparseable"}:
        label = "unparseable"

    if not explanation:
        explanation = "The judge did not provide an explanation."

    return {
        "verbal_annotation": label,
        "judgement_explanation": explanation,
    }


def build_prompt(tokenizer, text_to_judge: str, use_chat_template: bool) -> str:
    user_message = f"{JUDGE_INSTRUCTION}\n\nText to classify:\n{text_to_judge}"

    if use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True,
            tokenize=False,
        )

    return user_message + "\nAnnotation:"


def get_parts(record: Dict, judge_part: str) -> List[Tuple[str, str]]:
    parts = []

    if judge_part in {"response", "both"}:
        parts.append(("response", record.get("response_text") or ""))

    if judge_part in {"thinking", "both"}:
        parts.append(("thinking", record.get("thinking_text") or ""))

    return parts


def empty_result() -> Dict[str, str]:
    return {
        "verbal_annotation": "empty",
        "judgement_explanation": "No text was available to classify.",
    }


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = json.loads(input_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    prompts: List[str] = []
    prompt_keys: List[Tuple[int, str]] = []
    labels_by_key: Dict[Tuple[int, str], Dict[str, str]] = {}

    for i, record in enumerate(records):
        for part_name, text_to_judge in get_parts(record, args.judge_part):
            key = (i, part_name)
            text_to_judge = (text_to_judge or "").strip()

            if not text_to_judge:
                labels_by_key[key] = empty_result()
                continue

            label = None
            prompts.append(
                build_prompt(
                    tokenizer=tokenizer,
                    text_to_judge=text_to_judge,
                    use_chat_template=args.use_chat_template,
                )
            )
            prompt_keys.append(key)

    if prompts:
        llm_kwargs = {
            "model": args.judge_model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype,
            "trust_remote_code": True,
        }

        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len

        llm = LLM(**llm_kwargs)

        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stop=["\n\n"],
        )

        outputs = llm.generate(prompts, sampling_params)

        for key, output in zip(prompt_keys, outputs):
            raw = output.outputs[0].text
            labels_by_key[key] = normalize_judge_output(raw)

    if args.judge_part == "both":
        response_path = output_path.with_name(output_path.stem + "_response.jsonl")
        thinking_path = output_path.with_name(output_path.stem + "_thinking.jsonl")

        with response_path.open("w", encoding="utf-8") as f_resp:
            for i, record in enumerate(records):
                result = labels_by_key.get((i, "response"), empty_result())
                out = {
                    "example_idx": record.get("example_idx"),
                    "data_id": record.get("data_id"),
                    "prompt_style": record.get("prompt_style"),
                    "judged_part": "response",
                    "text": record.get("response_text") or "",
                    "verbal_annotation": result["verbal_annotation"],
                    "judgement_explanation": result["judgement_explanation"],
                }
                f_resp.write(json.dumps(out, ensure_ascii=False) + "\n")

        with thinking_path.open("w", encoding="utf-8") as f_think:
            for i, record in enumerate(records):
                result = labels_by_key.get((i, "thinking"), empty_result())
                out = {
                    "example_idx": record.get("example_idx"),
                    "data_id": record.get("data_id"),
                    "prompt_style": record.get("prompt_style"),
                    "judged_part": "thinking",
                    "text": record.get("thinking_text") or "",
                    "verbal_annotation": result["verbal_annotation"],
                    "judgement_explanation": result["judgement_explanation"],
                }
                f_think.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"Wrote response annotations to {response_path}")
        print(f"Wrote thinking annotations to {thinking_path}")

    else:
        part_name = args.judge_part
        text_field = "response_text" if part_name == "response" else "thinking_text"

        with output_path.open("w", encoding="utf-8") as f:
            for i, record in enumerate(records):
                result = labels_by_key.get((i, part_name), empty_result())
                out = {
                    "example_idx": record.get("example_idx"),
                    "data_id": record.get("data_id"),
                    "prompt_style": record.get("prompt_style"),
                    "judged_part": part_name,
                    "text": record.get(text_field) or "",
                    "verbal_annotation": result["verbal_annotation"],
                    "judgement_explanation": result["judgement_explanation"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"Wrote {part_name} annotations to {output_path}")


if __name__ == "__main__":
    main()
