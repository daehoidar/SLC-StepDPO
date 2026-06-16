"""
data_pipeline/5_evaluate.py

Step DPO / Full-Step DPO 공용 평가 스크립트.

평가 지표:
1. GSM8K-ko final answer accuracy
2. Step-level math accuracy (GPT-4o judge)
3. Persona consistency (GPT-4o judge)

Usage (Step DPO):
    python data_pipeline/5_evaluate.py \\
        --model checkpoints/bc_stepdpo \\
        --test-set data/test.jsonl \\
        --output checkpoints/bc_stepdpo/eval_results.json

Usage (Full-Step DPO):
    python data_pipeline/5_evaluate.py \\
        --model checkpoints/fullstepdpo \\
        --test-set data/test.jsonl \\
        --output checkpoints/fullstepdpo/eval_results.json

--flip-stats: Step DPO 전용 선택 인자. 3_5_analyze_flip_rate.py 산출 파일 경로.
              생략 시 해당 통계는 결과에서 제외됨.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI  # noqa: E402

# vLLM이 없는 환경(Mac M-series 등)에선 transformers fallback 사용.
try:
    from vllm import LLM, SamplingParams  # type: ignore  # noqa: E402
    _VLLM_AVAILABLE = True
except ImportError:
    from inference_backend import (  # noqa: E402
        TransformersLLM as LLM,
        TransformersSamplingParams as SamplingParams,
    )
    _VLLM_AVAILABLE = False

from judge_prompts import (  # noqa: E402
    STEP_JUDGE_SYSTEM, STEP_JUDGE_USER_TEMPLATE, build_step_judge_kwargs,
)
from utils import load_personas, parse_steps  # noqa: E402
from openai_client import make_openai_client  # noqa: E402


def normalize_answer(s: str) -> str:
    return re.sub(r"[,\s]", "", s.strip().lower())


def extract_final_answer(text: str) -> str:
    m = re.search(r"final answer[:\s]+(.+?)(?:\n|$)", text.lower())
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def generate(llm: LLM, problems: list[dict], personas: list[dict]) -> list[dict]:
    prompts, meta = [], []
    for p in problems:
        for pers in personas:
            prompts.append(f"{pers['tag']}\nProblem: {p['problem']}\nSolution:\n")
            meta.append({"problem": p, "persona": pers})
    sp = SamplingParams(temperature=0.0, max_tokens=800)
    outputs = llm.generate(prompts, sp)
    results = []
    for m, o in zip(meta, outputs):
        text = o.outputs[0].text
        results.append({
            "problem_id": m["problem"]["problem_id"],
            "problem": m["problem"]["problem"],
            "ground_truth": m["problem"]["ground_truth"],
            "persona": m["persona"],
            "solution_text": text,
            "steps": parse_steps(text),
            "predicted_answer": extract_final_answer(text),
        })
    return results


def metric_final_accuracy(results: list[dict]) -> float:
    correct = sum(
        normalize_answer(r["predicted_answer"]) == normalize_answer(r["ground_truth"])
        for r in results
    )
    return correct / max(1, len(results))


def metric_step_judge(client: OpenAI, results: list[dict]) -> dict:
    """Step-level: math accuracy + persona consistency 한꺼번에 측정."""
    n_accept, n_math_err, n_persona_err, n_total = 0, 0, 0, 0
    for r in results:
        if not r["steps"]:
            continue
        pers = r["persona"]
        sys_p = STEP_JUDGE_SYSTEM.format(**build_step_judge_kwargs(pers))
        user_p = STEP_JUDGE_USER_TEMPLATE.format(
            problem=r["problem"],
            ground_truth=r["ground_truth"],
            persona_tag=pers["tag"],
            solution_with_steps="\n".join(f"[{i+1}] {s}" for i, s in enumerate(r["steps"])),
        )
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": sys_p},
                    {"role": "user", "content": user_p},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            out = json.loads(resp.choices[0].message.content)
            for s in out.get("steps", []):
                lbl = s.get("label")
                if lbl == "acceptable":
                    n_accept += 1
                elif lbl == "reject_math":
                    n_math_err += 1
                elif lbl == "reject_persona":
                    n_persona_err += 1
                n_total += 1
        except Exception:
            continue
    return {
        "step_accept_rate": n_accept / max(1, n_total),
        "step_math_err_rate": n_math_err / max(1, n_total),
        "step_persona_err_rate": n_persona_err / max(1, n_total),
        "n_steps_judged": n_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--flip-stats", default=None,
                        help="data/flip_stats.json — 학습 데이터의 flip 통계")
    parser.add_argument("--personas-path", default="data/personas.json")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    client = make_openai_client()
    llm = LLM(model=args.model, dtype="bfloat16")
    personas = load_personas(args.personas_path)
    problems = [json.loads(l) for l in open(args.test_set, encoding="utf-8")]

    print(f"Generating {len(problems) * len(personas)} solutions...")
    results = generate(llm, problems, personas)

    print("Computing metrics...")
    final_acc = metric_final_accuracy(results)
    step_metrics = metric_step_judge(client, results)

    metrics = {
        "final_answer_accuracy": final_acc,
        **step_metrics,
    }

    # flip 통계 정보가 있으면 로그
    if args.flip_stats and Path(args.flip_stats).exists():
        with open(args.flip_stats) as f:
            metrics["training_data_flip_stats"] = json.load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "n_results": len(results)},
                  f, ensure_ascii=False, indent=2)

    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        elif isinstance(v, int):
            print(f"  {k:30s}: {v}")
    print(f"→ Full results in {args.output}")


if __name__ == "__main__":
    main()
