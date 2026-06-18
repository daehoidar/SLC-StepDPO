"""data_pipeline/1_synthesize_sft.py

Stage 1: GPT-4o로 페르소나별 정답 풀이 합성 → SFT 데이터.

0_seed_problems.py가 만든 seed_problems.jsonl(각 행 = problem+persona 쌍)을
입력으로 받고, 각 행마다 --solutions-per-row 개의 풀이를 합성한다.

Output format (JSONL):
    {
      "problem_id": "gsm8k_train_42",
      "problem": "...",
      "ground_truth": "5/6",
      "persona_id": "elem_low",
      "persona_tag": "<초등-하위권>",
      "solution_text": "Step 1: ...\nStep 2: ...",
      "steps": ["Step 1: ...", "Step 2: ..."],
      "augmentation_type": "GSM_AnsAug"
    }

Usage:
    python data_pipeline/1_synthesize_sft.py \
        --seed-problems data_pipeline/output/seed_problems.jsonl \
        --solutions-per-row 5 \
        --output data_pipeline/output/sft_data.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import openai
from openai import OpenAI  # noqa: E402
from judge_prompts import (  # noqa: E402
    GENERATOR_SYSTEM, GENERATOR_USER_TEMPLATE, build_generator_kwargs,
)
from utils import load_personas, parse_steps  # noqa: E402
from openai_client import make_openai_client  # noqa: E402
from persona_verifier import PersonaVerifier  # noqa: E402


def load_seed_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def generate_one_solution(
    client: OpenAI, seed_row: dict, persona: dict,
    model: str = "gpt-4o", max_retries: int = 3,
    verifier: PersonaVerifier | None = None,
) -> dict | None:
    sys_prompt = GENERATOR_SYSTEM.format(**build_generator_kwargs(persona))
    user_prompt = GENERATOR_USER_TEMPLATE.format(
        problem=seed_row["question"],
        ground_truth=seed_row.get("gt_answer", ""),
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                max_tokens=800,
            )
            
            solution_text = resp.choices[0].message.content
            steps = parse_steps(solution_text)
            if len(steps) < 2:
                continue

            # PersonaVerifier로 forbidden term 위반 스텝 필터링
            if verifier is not None:
                n_violations = sum(
                    1 for i, s in enumerate(steps)
                    if verifier.verify_step(s, persona, prefix=steps[:i]).verdict == "reject_persona"
                )
                if n_violations > 0:
                    continue  # 위반 있으면 재생성

            return {
                "problem_id": seed_row["problem_id"],
                "problem": seed_row["question"],
                "ground_truth": seed_row.get("gt_answer", ""),
                "persona_id": persona["id"],
                "persona_tag": persona["tag"],
                "solution_text": solution_text,
                "steps": steps,
                "augmentation_type": seed_row.get("augmentation_type"),
            }
        except openai.AuthenticationError as e:
            print(f"[Fatal Error] API Key / Auth failed: {e}")
            raise  # 인증 에러 등 치명적 에러는 즉시 중단
        except Exception as e:
            print(f"[retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)
            
    # 최대 재시도 후에도 실패한 경우 추적을 위해 남김
    return {
        "_error": True, 
        "problem_id": seed_row.get("problem_id"), 
        "persona_id": persona.get("id")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-problems",
        default=str(REPO_ROOT / "data_pipeline" / "output" / "seed_problems.jsonl"),
        help="0_seed_problems.py 산출물 jsonl",
    )
    parser.add_argument("--solutions-per-row", type=int, default=5,
                        help="(문제, 페르소나) 한 쌍당 합성할 풀이 개수")
    parser.add_argument("--personas-path", default=str(REPO_ROOT / "personas.json"))
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "data_pipeline" / "output" / "sft_data.jsonl"),
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="0이면 전체. 디버그/부트스트랩용 호출 수 제한.")
    parser.add_argument("--verify", action="store_true",
                        help="PersonaVerifier로 forbidden term 위반 풀이 재생성 (권장)")
    args = parser.parse_args()

    client = make_openai_client()
    verifier = PersonaVerifier(
        stage_b_client=None,
        stage_c_client=client,
        stage_c_model=args.model,
        enable_stage_b=False,
        enable_stage_c=True,
    ) if args.verify else None
    if verifier:
        print("[verify] PersonaVerifier 활성 — forbidden term 위반 풀이 재생성")
    personas = {p["id"]: p for p in load_personas(args.personas_path)}
    seed_rows = load_seed_rows(Path(args.seed_problems))
    print(f"[load] {len(seed_rows)} seed rows, {len(personas)} personas")

    out_path = Path(args.output)
    error_path = out_path.parent / "errors.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Resume 로직: 기존 output에서 (problem_id, persona_id)별 카운트 파악
    done_counts: dict[tuple[str, str], int] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("problem_id", ""), r.get("persona_id", ""))
                done_counts[key] = done_counts.get(key, 0) + 1
        print(f"[resume] {sum(done_counts.values())} existing solutions across "
              f"{len(done_counts)} (problem, persona) keys")

    # (seed_row, persona) × (solutions_per_row - already_done) 작업 큐
    tasks = []
    for row in seed_rows:
        pers = personas.get(row["persona"])
        if pers is None:
            continue
        key = (row["problem_id"], pers["id"])
        need = args.solutions_per_row - done_counts.get(key, 0)
        if need <= 0:
            continue
        for _ in range(need):
            tasks.append((row, pers))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[tasks] {len(tasks)} new calls "
          f"(~ ${len(tasks) * 0.002:.2f} estimated cost)")
    if not tasks:
        print("Nothing to do. Output already complete.")
        return

    n_success = 0
    n_errors = 0

    # append 모드 — 기존 데이터 보존하고 뒤에 이어 쓰기
    with open(out_path, "a", encoding="utf-8") as fout, \
         open(error_path, "a", encoding="utf-8") as ferr, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(generate_one_solution, client, row, pers, args.model, verifier)
            for row, pers in tasks
        ]
        for i, fut in enumerate(as_completed(futures)):
            result = fut.result()
            if result is not None:
                if result.get("_error"):
                    ferr.write(json.dumps(result, ensure_ascii=False) + "\n")
                    n_errors += 1
                else:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    n_success += 1
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(tasks)}] success: {n_success}, errors: {n_errors}")

    print(f"Done. {n_success}/{len(tasks)} -> {out_path}")
    if n_errors > 0:
        print(f"Failed {n_errors} tasks. Check {error_path} for details.")


if __name__ == "__main__":
    main()
