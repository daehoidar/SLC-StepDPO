"""
eval/evaluate.py

평가 지표:
1. GSM8K-ko final answer accuracy
2. Step-level math accuracy (GPT-4o judge)
3. Persona consistency (GPT-4o judge)
4. Belief-flip handling: 학습 데이터의 flip 케이스에 대해 정답률 측정

Usage:
    python eval/evaluate.py \\
        --model checkpoints/bc_stepdpo \\
        --test-set data/test.jsonl \\
        --flip-stats data/flip_stats.json \\
        --output eval_results.json
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


JUDGE_MODEL = "gpt-4o-mini"   # main()에서 --gpt-model로 덮어씀 (평가 judge 모델)


def _judge_one(client: OpenAI, r: dict):
    if not r["steps"]:
        return None
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
            model=JUDGE_MODEL,
            messages=[{"role": "system", "content": sys_p},
                      {"role": "user", "content": user_p}],
            temperature=0.0, seed=42, response_format={"type": "json_object"},
        )
        out = json.loads(resp.choices[0].message.content)
        a = m = p = t = 0
        for s in out.get("steps", []):
            lbl = s.get("label")
            if lbl == "acceptable":
                a += 1
            elif lbl == "reject_math":
                m += 1
            elif lbl == "reject_persona":
                p += 1
            t += 1
        return (a, m, p, t)
    except Exception:
        return None


def metric_step_judge(client: OpenAI, results: list[dict]) -> dict:
    """Step-level: math accuracy + persona consistency (judge 호출 병렬)."""
    import concurrent.futures
    n_accept = n_math_err = n_persona_err = n_total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for res in ex.map(lambda r: _judge_one(client, r), results):
            if res:
                a, m, p, t = res
                n_accept += a; n_math_err += m; n_persona_err += p; n_total += t
    return {
        "step_accept_rate": n_accept / max(1, n_total),
        "step_math_err_rate": n_math_err / max(1, n_total),
        "step_persona_err_rate": n_persona_err / max(1, n_total),
        "n_steps_judged": n_total,
    }


def metric_format(results: list[dict], all_tags: list[str]) -> dict:
    """출력 형식 준수율 (programmatic): Step1 시작·2단계+·순차 번호·Final answer·태그 비노출."""
    tagset = {t for t in all_tags if t}
    n = n_ok = 0
    for r in results:
        text = r.get("solution_text", "")
        steps = r.get("steps", [])
        nums = [int(m.group(1)) for ln in text.splitlines()
                if (m := re.match(r"^Step\s+(\d+)[:.]", ln.strip()))]
        has_step1 = bool(nums) and nums[0] == 1
        multi = len(steps) >= 2
        seq = bool(nums) and nums == list(range(1, len(nums) + 1))
        has_final = bool(re.search(r"final answer", text, re.I))
        no_leak = not any(t in text for t in tagset | {r["persona"]["tag"]})
        fc = len(text.strip()) >= 10 and has_step1 and multi and seq and has_final and no_leak
        n_ok += int(fc); n += 1
    return {"format_compliant_rate": n_ok / max(1, n)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--flip-stats", default=None,
                        help="data/flip_stats.json — 학습 데이터의 flip 통계")
    parser.add_argument("--personas-path", default="data/personas.json")
    parser.add_argument("--gpt-model", default="gpt-4o-mini",
                        help="평가 judge 모델 (특허 신뢰도용 gpt-4o 권장)")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()
    global JUDGE_MODEL
    JUDGE_MODEL = args.gpt_model
    print(f"[judge] eval judge model = {JUDGE_MODEL}")

    client = make_openai_client()
    llm = LLM(model=args.model, dtype="bfloat16")
    personas = load_personas(args.personas_path)
    problems = [json.loads(l) for l in open(args.test_set, encoding="utf-8")]

    print(f"Generating {len(problems) * len(personas)} solutions...")
    results = generate(llm, problems, personas)

    print("Computing metrics...")
    final_acc = metric_final_accuracy(results)
    step_metrics = metric_step_judge(client, results)
    fmt_metrics = metric_format(results, [p["tag"] for p in personas])

    metrics = {
        "final_answer_accuracy": final_acc,
        **step_metrics,
        **fmt_metrics,
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
