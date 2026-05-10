"""GPT-4o로 페르소나별 정답 해설을 합성하여 SFT 시드를 만든다.

입력: data_pipeline/output/seed_problems.jsonl  (0_seed_problems.py 산출물)
출력: data_pipeline/output/sft_data.jsonl

특징:
- AsyncOpenAI + 세마포어로 동시 호출 제한
- 호출 단위 지수 백오프 재시도 (RateLimit, Timeout, ServerError)
- 출력 jsonl을 append 모드로 즉시 기록 -> 중단 시 재개 가능
- (problem_id, persona) 키 기준으로 처리 완료 항목 자동 스킵
- 비용 가드: --max-cost USD 초과 시 신규 호출 중단

환경변수:
    OPENAI_API_KEY  (필수)
    OPENAI_BASE_URL (선택; 프록시 사용 시)

사용 예:
    export OPENAI_API_KEY=sk-...
    python data_pipeline/1_synthesize_sft.py --concurrency 8 --max-cost 80
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from personas import all_personas, render_system_prompt  # noqa: E402

# 가격은 변동 가능. 사용 전 https://openai.com/api/pricing 확인.
PRICE_INPUT_PER_1M = 2.50   # USD per 1M input tokens (gpt-4o, 2024-09 기준)
PRICE_OUTPUT_PER_1M = 10.00

MODEL = "gpt-4o"
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.3


def build_prompt_table():
    """페르소나 id -> system prompt 매핑."""
    return {p["id"]: render_system_prompt(p) for p in all_personas()}


def make_key(row: dict) -> str:
    return f"{row['problem_id']}::{row['persona']}"


def load_processed_keys(path: Path) -> set:
    if not path.exists():
        return set()
    keys = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                keys.add(make_key(row))
            except Exception:
                continue
    return keys


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_INPUT_PER_1M / 1e6
            + completion_tokens * PRICE_OUTPUT_PER_1M / 1e6)


async def call_one(client, sem, system_prompt, user_msg, retries=4):
    delay = 2.0
    for attempt in range(retries):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                return resp
            except Exception as e:
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
    raise RuntimeError("unreachable")


async def main_async(args):
    from openai import AsyncOpenAI

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        sys.exit(f"[error] input not found: {in_path}")

    seed_rows = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            seed_rows.append(json.loads(line))
    print(f"[load] {len(seed_rows)} seed rows from {in_path}")

    processed = load_processed_keys(out_path)
    todo = [r for r in seed_rows if make_key(r) not in processed]
    print(f"[resume] already done: {len(processed)},  todo: {len(todo)}")
    if args.limit:
        todo = todo[: args.limit]
        print(f"[limit] truncate todo to {len(todo)}")

    system_prompts = build_prompt_table()
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    sem = asyncio.Semaphore(args.concurrency)

    state = {"cost": 0.0, "done": 0, "fail": 0, "stopped": False}
    out_f = open(out_path, "a", encoding="utf-8")
    out_lock = asyncio.Lock()

    t0 = time.time()

    async def run_one(row):
        if state["stopped"]:
            return
        if state["cost"] >= args.max_cost:
            state["stopped"] = True
            return
        sys_p = system_prompts[row["persona"]]
        user_msg = f"{row['persona']}\n{row['question']}\n\n위 문제를 단계별로 풀어주세요."
        # 페르소나 태그를 user 메시지 헤더에 포함 (학습/추론 시점 입력 형태와 일치)
        user_msg = f"<{row['persona']}>\n{row['question']}\n\n위 문제를 단계별로 풀어주세요."
        try:
            resp = await call_one(client, sem, sys_p, user_msg)
        except Exception as e:
            state["fail"] += 1
            print(f"[fail] {make_key(row)}: {type(e).__name__}: {e}")
            return
        usage = resp.usage
        cost = estimate_cost(usage.prompt_tokens, usage.completion_tokens)
        state["cost"] += cost
        state["done"] += 1
        record = {
            **row,
            "gpt4o_solution": resp.choices[0].message.content,
            "model": MODEL,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_usd": round(cost, 6),
        }
        async with out_lock:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
        if state["done"] % 20 == 0:
            elapsed = time.time() - t0
            rate = state["done"] / max(elapsed, 1e-3)
            print(f"[progress] done={state['done']} fail={state['fail']} "
                  f"cost=${state['cost']:.2f} rate={rate:.2f}/s")

    await asyncio.gather(*(run_one(r) for r in todo))
    out_f.close()

    elapsed = time.time() - t0
    print(f"\n[summary] done={state['done']} fail={state['fail']} "
          f"cost=${state['cost']:.2f} elapsed={elapsed:.0f}s")
    if state["stopped"]:
        print(f"[note] stopped early due to --max-cost ({args.max_cost})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(REPO_ROOT / "data_pipeline" / "output" / "seed_problems.jsonl"))
    ap.add_argument("--output", default=str(REPO_ROOT / "data_pipeline" / "output" / "sft_data.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-cost", type=float, default=80.0,
                    help="누적 비용(USD) 상한. 초과 시 신규 호출 중단.")
    ap.add_argument("--limit", type=int, default=0,
                    help="0이면 전체. 디버그용 호출 개수 제한.")
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("[error] OPENAI_API_KEY 환경변수를 먼저 설정하세요.")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
