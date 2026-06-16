"""data_pipeline/6_check_format.py

태스크 1: SFT 모델 출력이 *원하는 형식*을 따르는지 결정론적으로 체크.

원하는 형식(= SFT 데이터의 형식, 100% 준수):
  - "Step 1:"부터 시작
  - "Step N:" 마커가 순차 번호(1,2,3,...)로 step을 제대로 분리 (>= 2 step)
  - "Final answer:" 포함
  - 프롬프트의 persona tag(<elem_low> 등)를 출력에 그대로 흘리지 않음
  - 비어있지 않음

GPT-4o judge(태스크 2)와 무관하게 *형식*만 본다. 부가로, 정답(ground_truth)
exact-match 정확도도 결정론적으로 같이 계산해 참고용으로 보고한다.

두 가지 실행 모드:
  (1) 추론 + 분석 (기본): 모델을 로드해 test-set 각 (problem, persona) 행에 대해
      해당 persona_tag로 1개 풀이를 생성 → generations.jsonl 저장 → 형식 분석.
  (2) 분석 전용: --generations 로 이미 생성된 jsonl을 받아 형식 지표만 계산.
      (서버 GPU 생성 / 로컬 분석 분리에 유용)

Usage (추론+분석, 서버):
    python data_pipeline/6_check_format.py \
        --test-set data_pipeline/output/sft_test_eval60.jsonl \
        --base-model Qwen/Qwen3-1.7B \
        --adapter checkpoints/sft_qwen3_1.7b \
        --personas-path personas.json \
        --out-generations data_pipeline/output/format_generations.jsonl \
        --out-report data_pipeline/output/format_report.json

Usage (분석 전용, 로컬):
    python data_pipeline/6_check_format.py \
        --generations data_pipeline/output/format_generations.jsonl \
        --personas-path personas.json \
        --out-report data_pipeline/output/format_report.json
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import load_personas, parse_steps  # noqa: E402


# ─────────────────────────── 형식 분석 ────────────────────────────────────

def _step_numbers(text: str) -> list[int]:
    """줄 시작의 "Step N:" 마커 번호만 수집.

    본문 중 "from Step 1." 같은 *인라인 참조*는 세지 않는다 (parse_steps와 동일
    기준의 line-anchored 매칭).
    """
    nums = []
    for line in text.split("\n"):
        m = re.match(r"^Step\s+(\d+)[:.]", line.strip())
        if m:
            nums.append(int(m.group(1)))
    return nums


def normalize_answer(s: str) -> str:
    return re.sub(r"[,\s$]", "", (s or "").strip().lower())


def extract_final_answer(text: str) -> str:
    m = re.search(r"final answer[:\s]+(.+?)(?:\n|$)", text.lower())
    if m:
        # 원문에서 같은 위치를 다시 잘라 대소문자/기호 보존
        start = m.start(1)
        raw = text[start:start + (m.end(1) - m.start(1))]
        return raw.strip()
    nums = re.findall(r"-?\d+(?:/\d+)?(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def analyze_one(text: str, persona_tag: str, all_tags: list[str],
                ground_truth: str | None) -> dict:
    """한 생성 결과의 형식 체크 결과(불리언 지표 묶음) 반환."""
    steps = parse_steps(text)
    nums = _step_numbers(text)

    not_empty = len(text.strip()) >= 10
    has_step1 = bool(nums) and nums[0] == 1
    multi_step = len(steps) >= 2
    sequential = bool(nums) and nums == list(range(1, len(nums) + 1))
    has_final = bool(re.search(r"final answer", text, re.I))
    # persona tag 누수: 어떤 persona tag라도 출력에 등장하면 위반
    tag_leak = any(t and t in text for t in set(all_tags) | {persona_tag})

    fully_compliant = (
        not_empty and has_step1 and multi_step and sequential
        and has_final and not tag_leak
    )

    pred = extract_final_answer(text)
    answer_correct = (
        ground_truth is not None
        and normalize_answer(pred) == normalize_answer(ground_truth)
    )

    return {
        "n_steps": len(steps),
        "step_markers": len(nums),
        "not_empty": not_empty,
        "has_step1": has_step1,
        "multi_step": multi_step,
        "sequential_numbering": sequential,
        "has_final_answer": has_final,
        "no_tag_leak": not tag_leak,
        "fully_compliant": fully_compliant,
        "predicted_answer": pred,
        "answer_correct": answer_correct,
    }


_BOOL_METRICS = [
    "not_empty", "has_step1", "multi_step", "sequential_numbering",
    "has_final_answer", "no_tag_leak", "fully_compliant", "answer_correct",
]


def _rate(records: list[dict], key: str) -> float:
    return sum(bool(r[key]) for r in records) / max(1, len(records))


def aggregate(records: list[dict]) -> dict:
    by_persona: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_persona[r["persona_id"]].append(r)

    def block(recs: list[dict]) -> dict:
        out = {"n": len(recs)}
        for k in _BOOL_METRICS:
            out[k] = round(_rate(recs, k), 4)
        out["avg_n_steps"] = round(
            sum(r["n_steps"] for r in recs) / max(1, len(recs)), 2)
        return out

    return {
        "overall": block(records),
        "per_persona": {pid: block(by_persona[pid]) for pid in sorted(by_persona)},
    }


def print_table(agg: dict) -> None:
    cols = [
        ("fully_compliant", "compliant"),
        ("has_step1", "step1"),
        ("multi_step", ">=2step"),
        ("sequential_numbering", "seq#"),
        ("has_final_answer", "final"),
        ("no_tag_leak", "no-leak"),
        ("answer_correct", "acc"),
    ]
    header = f"{'persona':<12}{'n':>4}  " + "".join(f"{c[1]:>10}" for c in cols) + f"{'avg_steps':>11}"
    print("=" * len(header))
    print("형식 준수 리포트 (형식 지표는 비율, 1.0 = 100% 준수)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for pid, b in agg["per_persona"].items():
        row = f"{pid:<12}{b['n']:>4}  " + "".join(f"{b[c[0]]:>10.3f}" for c in cols) + f"{b['avg_n_steps']:>11.2f}"
        print(row)
    print("-" * len(header))
    o = agg["overall"]
    row = f"{'OVERALL':<12}{o['n']:>4}  " + "".join(f"{o[c[0]]:>10.3f}" for c in cols) + f"{o['avg_n_steps']:>11.2f}"
    print(row)
    print("=" * len(header))


# ─────────────────────────── Markdown 리포트 ──────────────────────────────

# 원하는 출력 형식 명세 (SFT 데이터가 따르는 규칙)
DESIRED_FORMAT_SPEC = """\
풀이는 아래 형식을 따라야 한다 (합성 SFT 데이터 100% 준수):

1. **"Step 1:"부터 시작**한다.
2. 각 단계를 **"Step N:" 마커로 분리**하고, 번호는 **1, 2, 3, … 순차**여야 한다 (최소 2단계).
3. 마지막에 **"Final answer: <정답>"** 을 포함한다.
4. 프롬프트의 **persona tag(`<elem_low>` 등)를 출력에 그대로 노출하지 않는다.**

예시:
```
Step 1: <첫 단계 설명과 계산>.
Step 2: <다음 단계>.
Step 3: <마지막 계산>.
Final answer: 42
```
"""

_CHECK_LABELS = [
    ("has_step1", "Step 1로 시작"),
    ("multi_step", "단계 2개 이상 분리"),
    ("sequential_numbering", "번호 순차(1,2,3,…)"),
    ("has_final_answer", "Final answer 포함"),
    ("no_tag_leak", "persona tag 미노출"),
]


def _md_table(agg: dict) -> str:
    cols = [c[0] for c in _CHECK_LABELS] + ["fully_compliant", "answer_correct", "avg_n_steps"]
    head = ["persona", "n"] + [
        {"has_step1": "step1", "multi_step": "≥2step",
         "sequential_numbering": "seq#", "has_final_answer": "final",
         "no_tag_leak": "no-leak", "fully_compliant": "**준수**",
         "answer_correct": "acc", "avg_n_steps": "avg_steps"}[c] for c in cols
    ]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]

    def row(name, b):
        vals = [name, str(b["n"])]
        for c in cols:
            v = b[c]
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        return "| " + " | ".join(vals) + " |"

    for pid, b in agg["per_persona"].items():
        lines.append(row(pid, b))
    lines.append(row("**OVERALL**", agg["overall"]))
    return "\n".join(lines)


def write_markdown(agg: dict, records: list[dict], gen_rows: list[dict],
                   path: str, model_desc: str) -> None:
    o = agg["overall"]
    n = o["n"]

    # 체크별 실패 건수 (어디서 깨지는지)
    fail_counts = {k: sum(1 for r in records if not r[k]) for k, _ in _CHECK_LABELS}

    out = []
    out.append("# SFT 출력 형식 준수 리포트\n")
    out.append(f"- 모델: `{model_desc}`")
    out.append(f"- 평가 샘플 수: **{n}** (페르소나별 균형)")
    out.append(f"- **형식 완전 준수율(fully_compliant): {o['fully_compliant']:.1%}**")
    out.append(f"- (참고) 정답 정확도: {o['answer_correct']:.1%}, 평균 단계 수: {o['avg_n_steps']}")
    out.append("\n> 기준선: 합성 SFT 데이터(학습 타깃) 자체의 완전 준수율은 "
               "**1.000** (형식 완벽). 모델 점수는 이 1.0 대비 해석.\n")

    out.append("## 1. 원하는 출력 형식\n")
    out.append(DESIRED_FORMAT_SPEC)

    out.append("\n## 2. 페르소나별 준수율\n")
    out.append(_md_table(agg))

    out.append("\n## 3. 어디서 깨지는가 (체크별 실패 건수)\n")
    out.append("| 체크 항목 | 실패/전체 | 실패율 |")
    out.append("|---|---|---|")
    for k, label in _CHECK_LABELS:
        c = fail_counts[k]
        out.append(f"| {label} (`{k}`) | {c}/{n} | {c / max(1, n):.1%} |")

    # 예시 모음
    paired = list(zip(records, gen_rows))

    def fmt_example(rec: dict, g: dict, limit: int = 700) -> str:
        flags = " ".join(
            ("✅" if rec[k] else "❌") + k for k, _ in _CHECK_LABELS
        )
        txt = (g.get("generation") or "").strip()
        if len(txt) > limit:
            txt = txt[:limit] + " …(생략)"
        return (f"- **{g.get('persona_id')}** / `{g.get('problem_id')}` "
                f"(gt={g.get('ground_truth')})\n  {flags}\n\n  ```\n  "
                + txt.replace("\n", "\n  ") + "\n  ```")

    out.append("\n## 4. 준수 예시\n")
    ok = [(r, g) for r, g in paired if r["fully_compliant"]][:2]
    if ok:
        for r, g in ok:
            out.append(fmt_example(r, g))
    else:
        out.append("_완전 준수 샘플 없음._")

    out.append("\n## 5. 위반 예시 (체크별)\n")
    any_viol = False
    for k, label in _CHECK_LABELS:
        bad = [(r, g) for r, g in paired if not r[k]][:2]
        if not bad:
            continue
        any_viol = True
        out.append(f"### ❌ {label} 실패\n")
        for r, g in bad:
            out.append(fmt_example(r, g))
    if not any_viol:
        out.append("_위반 없음 — 모든 샘플이 모든 형식 체크를 통과._")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


# ─────────────────────────── 생성(추론) ───────────────────────────────────

def build_prompt(persona_tag: str, problem: str) -> str:
    """학습(2_train_sft.py)과 *동일한* 프롬프트 포맷이어야 분포 일치."""
    return f"{persona_tag}\nProblem: {problem}\nSolution:\n"


def generate_with_transformers(rows: list[dict], base_model: str,
                               adapter: str | None, max_new_tokens: int,
                               device_arg: str) -> list[str]:
    """transformers(+ 선택적 PEFT LoRA)로 그리디 생성. Mac/서버 공통 동작."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device_arg == "auto":
        if torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            device, dtype = "mps", torch.float32
        else:
            device, dtype = "cpu", torch.float32
    else:
        device = device_arg
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[gen] base={base_model} adapter={adapter} device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device).eval()

    stops = ["\nProblem:", "\nProblem :"]
    gens: list[str] = []
    for i, r in enumerate(rows):
        prompt = build_prompt(r["persona_tag"], r["problem"])
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        for s in stops:
            if s in text:
                text = text[:text.index(s)]
        gens.append(text.strip())
        if (i + 1) % 10 == 0:
            print(f"  generated {i + 1}/{len(rows)}")
    return gens


# ─────────────────────────── main ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    # 입력: test-set(추론) 또는 generations(분석전용) 중 하나
    ap.add_argument("--test-set", default=None,
                    help="추론+분석 모드: (problem, persona_tag) 행 jsonl")
    ap.add_argument("--generations", default=None,
                    help="분석 전용 모드: 이미 생성된 결과 jsonl")
    ap.add_argument("--personas-path", default="personas.json")
    # 모델 (추론 모드)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 경로 (옵션)")
    ap.add_argument("--max-new-tokens", type=int, default=800)
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--limit", type=int, default=0, help=">0이면 앞 N개만 (스모크용)")
    # 출력
    ap.add_argument("--out-generations",
                    default="data_pipeline/output/format_generations.jsonl")
    ap.add_argument("--out-report",
                    default="data_pipeline/output/format_report.json")
    ap.add_argument("--out-md", default=None,
                    help="사람이 읽기 쉬운 Markdown 리포트 경로 "
                         "(미지정 시 out-report의 .json→.md)")
    args = ap.parse_args()

    if not args.test_set and not args.generations:
        ap.error("--test-set (추론+분석) 또는 --generations (분석 전용) 중 하나는 필수")

    all_tags = [p["tag"] for p in load_personas(args.personas_path)]

    # ── 1) 생성 결과 확보 ──────────────────────────────────────────────
    if args.generations:
        gen_rows = [json.loads(l) for l in open(args.generations, encoding="utf-8")]
        print(f"[analyze-only] {args.generations}  rows={len(gen_rows)}")
    else:
        rows = [json.loads(l) for l in open(args.test_set, encoding="utf-8")]
        if args.limit > 0:
            rows = rows[:args.limit]
        print(f"[infer] test-set={args.test_set}  rows={len(rows)}")
        texts = generate_with_transformers(
            rows, args.base_model, args.adapter, args.max_new_tokens, args.device)
        gen_rows = []
        for r, t in zip(rows, texts):
            gen_rows.append({
                "problem_id": r["problem_id"],
                "persona_id": r["persona_id"],
                "persona_tag": r["persona_tag"],
                "problem": r["problem"],
                "ground_truth": r.get("ground_truth"),
                "generation": t,
            })
        Path(args.out_generations).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_generations, "w", encoding="utf-8") as f:
            for g in gen_rows:
                f.write(json.dumps(g, ensure_ascii=False) + "\n")
        print(f"[saved] generations -> {args.out_generations}")

    # ── 2) 형식 분석 ──────────────────────────────────────────────────
    records = []
    for g in gen_rows:
        a = analyze_one(g["generation"], g.get("persona_tag", ""), all_tags,
                        g.get("ground_truth"))
        a.update({"problem_id": g.get("problem_id"),
                  "persona_id": g.get("persona_id")})
        records.append(a)

    agg = aggregate(records)

    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump({"aggregate": agg, "per_sample": records}, f,
                  ensure_ascii=False, indent=2)

    # ── 3) 사람이 읽기 쉬운 Markdown 리포트 ────────────────────────────
    md_path = args.out_md or re.sub(r"\.json$", ".md", args.out_report)
    if md_path == args.out_report:
        md_path = args.out_report + ".md"
    model_desc = (
        f"{args.base_model}" + (f" + adapter {args.adapter}" if args.adapter else "")
        if not args.generations else f"(from {args.generations})"
    )
    write_markdown(agg, records, gen_rows, md_path, model_desc)

    print()
    print_table(agg)
    print(f"\n→ report(json): {args.out_report}")
    print(f"→ report(md):   {md_path}")
    if not args.generations:
        print(f"→ generations:  {args.out_generations}")


if __name__ == "__main__":
    main()
