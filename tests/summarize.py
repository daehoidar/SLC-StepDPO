"""tests/summarize.py — 각 테스트 phase 결과를 분석해 REPORT.md를 생성.

Usage:
    python tests/summarize.py sft_data  --out-dir tests/output/sft_data
    python tests/summarize.py sft_train --out-dir tests/output/sft_train --train-exit-code 0
    python tests/summarize.py pairs     --out-dir tests/output/pairs_full --mode full

각 phase별로 다른 통계를 산출하고 동일 이름 REPORT.md 작성.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# sft_data summary
# ---------------------------------------------------------------------------
def summarize_sft_data(out_dir: Path) -> str:
    seed_path = out_dir / "seed_problems.jsonl"
    sft_path = out_dir / "sft_data.jsonl"

    md = ["# Test Report — SFT Data (Stage 0+1)\n",
          "두 모드 공통 산출. Step-DPO·Full 학습 시 같은 sft_data.jsonl 사용.\n"]

    if not seed_path.exists():
        md.append(f"⚠️ {seed_path} 없음. Stage 0 실패.\n")
    else:
        seed = load_jsonl(seed_path)
        persona = Counter(r["persona"] for r in seed)
        aug = Counter(r.get("augmentation_type", "?") for r in seed)
        n_problems = len({r["problem_id"] for r in seed})

        md.append("## Stage 0 — seed_problems.jsonl\n")
        md.append(f"- **Total rows**: {len(seed)} (= {n_problems} problems × {len(persona)} personas)")
        md.append(f"- **Augmentation types**: " + ", ".join(f"{k}={v}" for k, v in sorted(aug.items())))
        md.append("- **Personas**: " + ", ".join(f"{k}={v}" for k, v in sorted(persona.items())))
        md.append("\n### Sample row\n```json")
        md.append(json.dumps(seed[0], ensure_ascii=False, indent=2))
        md.append("```\n")

    if not sft_path.exists():
        md.append(f"⚠️ {sft_path} 없음. Stage 1 실패.\n")
    else:
        sft = load_jsonl(sft_path)
        by_persona = Counter(r["persona_id"] for r in sft)
        step_counts = [len(r.get("steps", [])) for r in sft]
        avg_steps = sum(step_counts) / max(len(step_counts), 1)
        sol_chars = [len(r["solution_text"]) for r in sft]
        avg_chars = sum(sol_chars) / max(len(sol_chars), 1)

        md.append("## Stage 1 — sft_data.jsonl (GPT-4o 합성)\n")
        md.append(f"- **Generated rows**: {len(sft)}")
        md.append(f"- **By persona**: " + ", ".join(f"{k}={v}" for k, v in sorted(by_persona.items())))
        md.append(f"- **Avg steps per solution**: {avg_steps:.1f}")
        md.append(f"- **Avg solution length (chars)**: {avg_chars:.0f}")
        md.append("\n### Sample per persona (first 1 each)\n")
        seen = set()
        for r in sft:
            pid = r["persona_id"]
            if pid in seen:
                continue
            seen.add(pid)
            md.append(f"#### {r['persona_tag']}")
            md.append(f"**Problem**: {r['problem'][:120]}...")
            md.append(f"**Steps ({len(r.get('steps', []))})**:")
            for i, s in enumerate(r.get("steps", [])[:3]):
                md.append(f"  {i+1}. {s[:140]}")
            if len(r.get("steps", [])) > 3:
                md.append(f"  ... (+{len(r['steps']) - 3} more)")
            md.append("")

    md.append("\n## Pass/Fail Verdict")
    if seed_path.exists() and sft_path.exists():
        n = len(load_jsonl(sft_path))
        md.append(f"- ✅ Stage 0+1 completed, {n} SFT rows generated.")
        md.append("- 점검: 페르소나별 row 수 균등한가? 풀이 길이가 비현실적으로 짧지 않은가? \\boxed가 모든 풀이 끝에 있나?")
    else:
        md.append("- ❌ 산출 누락. 위 stage 로그 확인 필요.")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# sft_train summary
# ---------------------------------------------------------------------------
def summarize_sft_train(out_dir: Path, train_exit_code: int = 0) -> str:
    log_path = out_dir / "training_log.txt"
    ckpt_path = out_dir / "checkpoint"

    md = ["# Test Report — SFT Training (Stage 2)\n",
          "두 모드 공통 산출. 학습된 체크포인트가 π_ref로 Step-DPO·Full 양쪽에 사용됨.\n"]

    md.append(f"## 종료 상태\n- exit code: **{train_exit_code}** ({'✅ OK' if train_exit_code == 0 else '❌ FAIL'})")

    if log_path.exists():
        log = log_path.read_text(errors="ignore")
        loss_lines = re.findall(r"loss=([-\d.eE]+)", log)
        nan_inf = sum(1 for x in loss_lines if x.lower() in {"nan", "inf", "-inf"})
        md.append(f"\n## Training Log\n- log 라인: {len(log.splitlines())}")
        md.append(f"- loss 출력 횟수: {len(loss_lines)}")
        if loss_lines:
            first = loss_lines[0]
            last = loss_lines[-1]
            md.append(f"- first loss: `{first}`, last loss: `{last}`")
            md.append(f"- NaN/Inf 발생: **{nan_inf}** {'(❌ 학습 발산)' if nan_inf > 0 else '(✅ 정상)'}")
        # 마지막 10라인
        md.append("\n### log tail (last 10 lines)")
        md.append("```")
        md.extend(log.strip().splitlines()[-10:])
        md.append("```")
    else:
        md.append(f"\n⚠️ {log_path} 없음.")

    if ckpt_path.exists():
        files = list(ckpt_path.iterdir())
        sizes = [(f.name, f.stat().st_size) for f in files if f.is_file()]
        total_size = sum(s for _, s in sizes) / (1024 * 1024)
        adapter_present = any("adapter" in f.name.lower() for f in files)
        md.append("\n## Checkpoint")
        md.append(f"- 경로: `{ckpt_path}`")
        md.append(f"- 총 크기: {total_size:.1f} MB")
        md.append(f"- adapter 파일 존재: **{'✅' if adapter_present else '❌'}**")
        md.append(f"- 파일 ({len(files)}개):")
        for n, s in sorted(sizes, key=lambda x: -x[1])[:10]:
            md.append(f"  - {n} ({s/(1024*1024):.2f} MB)")
    else:
        md.append("\n⚠️ checkpoint 디렉토리 없음.")

    md.append("\n## Pass/Fail Verdict")
    pass_train = (train_exit_code == 0) and ckpt_path.exists() and any("adapter" in f.name.lower() for f in ckpt_path.iterdir() if f.is_file()) if ckpt_path.exists() else False
    if pass_train:
        md.append("- ✅ SFT 학습 파이프라인 sanity OK. 다음: `bash tests/run_pairs.sh full`")
    else:
        md.append("- ❌ 학습 또는 저장 실패. log tail + OOM·tokenizer 호환 확인.")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# pairs summary
# ---------------------------------------------------------------------------
def summarize_pairs(out_dir: Path, mode: str) -> str:
    pairs_path = out_dir / "preference_pairs.jsonl"
    flip_stats_path = out_dir / "flip_stats.json"

    md = [f"# Test Report — Preference Pairs (Stage 3) — `{mode}` 모드\n"]
    if mode == "step_dpo":
        md.append("`data_pipeline_stepdpo/`의 2단(first-error → rectify) 파이프라인. "
                  "출력은 `step_pair`만 (belief_flip은 정의상 생성되지 않음). "
                  "스키마는 BC-StepDPO 학습(Proposition 2)에 그대로 들어간다.\n")
    else:
        md.append("`data_pipeline/3_build_pairs.py`의 Type-1 + Type-2 동시 빌드. "
                  "flip rate가 (A7)의 empirical 증거.\n")

    if not pairs_path.exists():
        md.append(f"⚠️ {pairs_path} 없음. Stage 3 실패.\n")
        return "\n".join(md) + "\n"

    pairs = load_jsonl(pairs_path)
    type_counts = Counter(p.get("pair_type", "?") for p in pairs)
    reject_dist = Counter(p.get("reject_type", "?") for p in pairs if p.get("pair_type") == "step_pair")
    flip_matrix = defaultdict(int)
    for p in pairs:
        if p.get("pair_type") == "belief_flip_pair":
            flip_matrix[(p.get("persona_id"), p.get("flip_persona_id"))] += 1

    md.append("## 산출 개요\n")
    md.append(f"- **Total pairs**: {len(pairs)}")
    md.append(f"- **By pair_type**: " + ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items())))
    md.append(f"- **Type-1 reject_type 분포**: " + ", ".join(f"{k}={v}" for k, v in sorted(reject_dist.items())))

    n_t2 = type_counts.get("belief_flip_pair", 0)
    n_t1 = type_counts.get("step_pair", 0)

    md.append("\n## Mode 별 사용 가능 페어\n")
    if mode == "step_dpo":
        md.append(f"- 학습에 사용되는 step_pair: **{n_t1}**개.")
        md.append(f"- `(problem × persona)` 중 SFT 모델이 실패한 (= first-error가 발견된) "
                  f"케이스 수와 일치해야 함.")
        if n_t2 > 0:
            md.append(f"- ⚠️ Step-DPO 모드인데 belief_flip_pair가 {n_t2}개 검출됨 — "
                      f"파이프라인 라우팅 점검 필요.")
    else:
        md.append(f"- Full Step-DPO 학습엔 Type-1 + Type-2 합 {n_t1 + n_t2}개 사용.")
        md.append(f"- **flip rate (Type-2/Total)**: {n_t2 / max(len(pairs), 1):.2%}")

    if flip_matrix:
        md.append("\n## Type-2 flip 매트릭스 (top 10)")
        sorted_flips = sorted(flip_matrix.items(), key=lambda x: -x[1])[:10]
        for (cur, flip), cnt in sorted_flips:
            md.append(f"  - `{cur}` ⇄ `{flip}`: {cnt}")

    if flip_stats_path.exists():
        try:
            stats = json.loads(flip_stats_path.read_text())
            md.append("\n## flip_stats.json 핵심")
            md.append(f"```json\n{json.dumps({k: stats[k] for k in stats if k != 'flip_matrix'}, ensure_ascii=False, indent=2)}\n```")
        except Exception as e:
            md.append(f"\n⚠️ flip_stats.json 파싱 실패: {e}")

    md.append("\n## Sample pairs\n")
    for ptype in ("step_pair", "belief_flip_pair"):
        sample = next((p for p in pairs if p.get("pair_type") == ptype), None)
        if sample:
            md.append(f"### {ptype}")
            md.append("```json")
            md.append(json.dumps({k: v for k, v in sample.items() if k != "prefix_steps" or len(str(v)) < 500}, ensure_ascii=False, indent=2)[:1200])
            md.append("```\n")

    md.append("## Pass/Fail Verdict")
    if mode == "step_dpo":
        ok = n_t1 > 0
        md.append(f"- {'✅' if ok else '❌'} Type-1 페어 {n_t1}개 확보 ({'학습 가능' if ok else 'Stage 2 SFT 모델 출력이 페르소나 분기 안 되었거나 K 너무 작음'})")
    else:
        ok = (n_t1 > 0) and (n_t2 > 0)
        md.append(f"- {'✅' if ok else '⚠️'} Type-1 {n_t1} + Type-2 {n_t2}")
        if n_t2 == 0:
            md.append("  - **flip rate=0 경고**: (A7) 검증 불가. persona vocab guide 강화 + K 늘림 + judge 프롬프트 점검.")

    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["sft_data", "sft_train", "pairs"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["step_dpo", "full"], default="full",
                    help="pairs phase에서만 사용")
    ap.add_argument("--train-exit-code", type=int, default=0,
                    help="sft_train phase에서만 사용")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "sft_data":
        md = summarize_sft_data(out_dir)
    elif args.phase == "sft_train":
        md = summarize_sft_train(out_dir, args.train_exit_code)
    elif args.phase == "pairs":
        md = summarize_pairs(out_dir, args.mode)
    else:
        sys.exit(f"unknown phase: {args.phase}")

    report = out_dir / "REPORT.md"
    report.write_text(md, encoding="utf-8")
    print(f"[summarize:{args.phase}] -> {report}")
    # 콘솔에도 짧게 출력
    print("=" * 60)
    print(md[:2000])
    if len(md) > 2000:
        print(f"... ({len(md) - 2000} chars more in {report})")


if __name__ == "__main__":
    main()
