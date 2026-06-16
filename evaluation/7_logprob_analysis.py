"""data_pipeline/7_logprob_analysis.py

태스크 2 (B): win/lose 선호쌍에 대해 모델이 부여하는 *로그확률 차이*가
실제로 존재하는지(=선호 신호가 모델에 반영돼 있는지)를 통계적으로 검정·정리.

각 쌍에 대해 (teacher forcing) 다음을 계산:
    logp(win)  = sum_t log π(win_token_t  | prompt_win,  prev tokens)
    logp(lose) = sum_t log π(lose_token_t | prompt_lose, prev tokens)
    Δ_sum  = logp(win) - logp(lose)                  (DPO 손실이 쓰는 양)
    Δ_mean = logp(win)/len_win - logp(lose)/len_lose (길이 보정, 공정 비교)

지표/검정:
    - win-rate          : Δ>0 인 쌍 비율 (모델이 win을 더 선호한 비율)
    - paired t-test     : H0: E[Δ]=0  (정규근사 fallback 포함)
    - Wilcoxon signed-rank: 비모수 검정 (scipy 있을 때)
    - Cohen's d_z       : 효과크기, 95% CI
    pair_type(type1_math / type2_belief)별로 분리해 보고.

입력 스키마(일반형, make_pilot_pairs.py / 실제 build_pairs 모두 호환):
    {"pair_id","pair_type","persona_id",
     "win": {"persona_tag","problem","prefix_steps":[...],"step"},
     "lose":{...}}
  레거시 스키마({persona_tag,prefix_steps,step_win,step_lose})도 자동 변환.

Usage:
    python data_pipeline/7_logprob_analysis.py \
        --pairs data_pipeline/output/pilot_pairs.jsonl \
        --base-model Qwen/Qwen3-1.7B \
        --adapter checkpoints/sft_qwen3_1.7b \
        --model-label "SFT(pi_ref)" \
        --out-report data_pipeline/output/logprob_report.json \
        --out-md data_pipeline/output/logprob_report.md \
        --plot-dir data_pipeline/output/logprob_plots
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

try:
    from scipy import stats as _scipy_stats  # noqa: E402
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ─────────────────────────── 입력 정규화 ──────────────────────────────────

def normalize_pair(p: dict) -> dict:
    """일반형/레거시 스키마를 일반형(win/lose context)으로 통일."""
    if "win" in p and "lose" in p:
        return p
    # 레거시: persona_tag + prefix_steps + step_win/step_lose
    ctx = {"persona_tag": p.get("persona_tag", ""),
           "problem": p.get("problem", ""),
           "prefix_steps": p.get("prefix_steps", [])}
    return {
        "pair_id": p.get("pair_id", ""),
        "pair_type": p.get("pair_type", "step_pair"),
        "persona_id": p.get("persona_id", ""),
        "win":  {**ctx, "step": p["step_win"]},
        "lose": {**ctx, "step": p["step_lose"]},
        "meta": {k: p.get(k) for k in ("problem_id", "ground_truth") if k in p},
    }


def build_prompt(ctx: dict) -> str:
    """학습 프롬프트 형식과 동일: <persona>\\nProblem: ...\\nSolution:\\n[+prefix]."""
    head = f"{ctx['persona_tag']}\nProblem: {ctx['problem']}\nSolution:\n"
    prefix = ctx.get("prefix_steps") or []
    if prefix:
        return head + "\n\n".join(prefix) + "\n\n"
    return head


# ─────────────────────────── 모델 / logp ──────────────────────────────────

def load_model(base_model: str, adapter: str | None, device_arg: str):
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

    print(f"[load] base={base_model} adapter={adapter} device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device).eval()
    return model, tok, device


def seq_logp(model, tok, device, prompt: str, continuation: str):
    """continuation 토큰들의 (sum log-prob, 토큰수) 반환. teacher forcing."""
    import torch
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tok(prompt + continuation, add_special_tokens=False)["input_ids"]
    n_prompt = len(prompt_ids)
    if len(full_ids) <= n_prompt:
        return None, 0
    input_ids = torch.tensor([full_ids], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits  # [1, T, V]
        logp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:]                       # [1, T-1]
        tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # [T-1]
    cont_lp = tok_lp[n_prompt - 1:]                       # continuation 부분만
    return float(cont_lp.sum().item()), int(cont_lp.numel())


# ─────────────────────────── 통계 ─────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def paired_stats(delta: np.ndarray) -> dict:
    """Δ 배열에 대한 검정 묶음. scipy 없으면 정규근사."""
    n = len(delta)
    mean = float(np.mean(delta))
    sd = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else float("nan")
    win_rate = float(np.mean(delta > 0))
    cohens_dz = mean / sd if sd > 0 else float("nan")
    ci95 = (mean - 1.96 * se, mean + 1.96 * se) if se == se else (float("nan"),) * 2

    if _HAS_SCIPY and n > 1 and sd > 0:
        t_stat, t_p = _scipy_stats.ttest_1samp(delta, 0.0)
        try:
            w_stat, w_p = _scipy_stats.wilcoxon(delta)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
        t_stat, t_p, w_p = float(t_stat), float(t_p), float(w_p)
        test_kind = "scipy"
    else:
        # 정규근사 (one-sample z on Δ)
        z = mean / se if se and se == se else float("nan")
        t_stat = z
        t_p = 2 * (1 - _norm_cdf(abs(z))) if z == z else float("nan")
        w_p = float("nan")
        test_kind = "normal_approx(no scipy)"

    return {
        "n": n,
        "mean_delta": mean,
        "sd_delta": sd,
        "se_delta": se,
        "win_rate": win_rate,
        "t_stat": t_stat,
        "t_pvalue": t_p,
        "wilcoxon_pvalue": w_p,
        "cohens_dz": cohens_dz,
        "ci95_low": ci95[0],
        "ci95_high": ci95[1],
        "test_kind": test_kind,
    }


# ─────────────────────────── 리포트 ───────────────────────────────────────

def _fmt_p(p: float) -> str:
    if p != p:
        return "n/a"
    return "<1e-4" if p < 1e-4 else f"{p:.4g}"


def write_markdown(blocks: dict, records: list[dict], path: str,
                   model_label: str, plot_files: dict) -> None:
    out = []
    out.append("# win/lose 로그확률 차이 분석 (태스크 2-B)\n")
    out.append(f"- 모델: `{model_label}`")
    out.append(f"- 총 쌍 수: **{len(records)}**")
    out.append("\n각 쌍에서 Δ = logp(win) − logp(lose). "
               "**Δ>0 이면 모델이 win을 더 선호**한다는 뜻.\n")
    out.append("> 귀무가설 H0: E[Δ]=0 (선호 신호 없음). "
               "p값이 작고 win-rate가 0.5보다 크게 벗어나면 신호가 실재.\n")

    metrics = [
        ("metric", "Δ 종류"),
    ]
    out.append("## 결과 요약\n")
    header = ("| 그룹 | n | win-rate | mean Δ | 95% CI | t/z | p (t/z) | "
              "p (Wilcoxon) | Cohen dz |")
    sep = "|" + "|".join(["---"] * 9) + "|"
    for variant, vlabel in [("mean", "per-token(길이보정)"), ("sum", "sum(DPO와 동일)")]:
        out.append(f"### Δ 종류: {vlabel}\n")
        out.append(header)
        out.append(sep)
        for grp in ["overall"] + [k for k in blocks if k not in ("overall",)]:
            s = blocks[grp][variant]
            ci = f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}]"
            out.append(
                f"| {grp} | {s['n']} | {s['win_rate']:.1%} | {s['mean_delta']:.3f} | "
                f"{ci} | {s['t_stat']:.2f} | {_fmt_p(s['t_pvalue'])} | "
                f"{_fmt_p(s['wilcoxon_pvalue'])} | {s['cohens_dz']:.2f} |")
        out.append("")
        kind = blocks["overall"][variant]["test_kind"]
        out.append(f"_검정 방식: {kind}_\n")

    if plot_files:
        out.append("## 분포 그림\n")
        for grp, fp in plot_files.items():
            out.append(f"- {grp}: `{fp}`")
        out.append("")

    # 해석 가이드
    out.append("## 해석\n")
    o = blocks["overall"]["mean"]
    sig = (o["t_pvalue"] == o["t_pvalue"]) and o["t_pvalue"] < 0.05
    direction = "win을 유의하게 더 선호" if (sig and o["mean_delta"] > 0) else \
                ("lose를 더 선호(!)" if (sig and o["mean_delta"] < 0) else "유의한 차이 없음")
    out.append(f"- 전체(per-token): win-rate {o['win_rate']:.1%}, "
               f"mean Δ {o['mean_delta']:.3f}, p={_fmt_p(o['t_pvalue'])} → **{direction}**.")
    out.append("- type1_math = 수학 정오 신호 / type2_belief = 페르소나 belief-flip 신호. "
               "두 신호 모두 Δ>0·p작음이면, DPO 학습이 잡아낼 선호가 데이터에 실재함을 의미.")

    # 예시
    out.append("\n## 예시 (Δ per-token 기준 정렬)\n")
    recs = sorted([r for r in records if r["delta_mean"] == r["delta_mean"]],
                  key=lambda r: r["delta_mean"], reverse=True)
    def show(r):
        return (f"- `{r['pair_type']}` Δmean={r['delta_mean']:.3f} "
                f"(logp win {r['logp_win_mean']:.3f} vs lose {r['logp_lose_mean']:.3f}) "
                f"persona={r['persona_id']}")
    out.append("**가장 win-선호:**")
    for r in recs[:3]:
        out.append(show(r))
    out.append("\n**가장 lose-선호(이상치 점검용):**")
    for r in recs[-3:]:
        out.append(show(r))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


# ─────────────────────────── main ─────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model-label", default="model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-report", default="data_pipeline/output/logprob_report.json")
    ap.add_argument("--out-md", default="data_pipeline/output/logprob_report.md")
    ap.add_argument("--plot-dir", default="data_pipeline/output/logprob_plots")
    args = ap.parse_args()

    pairs = [normalize_pair(json.loads(l)) for l in open(args.pairs, encoding="utf-8")]
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print(f"[pairs] {args.pairs}  n={len(pairs)}")

    model, tok, device = load_model(args.base_model, args.adapter, args.device)

    records = []
    for i, p in enumerate(pairs):
        lw_sum, lw_n = seq_logp(model, tok, device, build_prompt(p["win"]), p["win"]["step"])
        ll_sum, ll_n = seq_logp(model, tok, device, build_prompt(p["lose"]), p["lose"]["step"])
        if lw_sum is None or ll_sum is None or lw_n == 0 or ll_n == 0:
            continue
        lw_mean, ll_mean = lw_sum / lw_n, ll_sum / ll_n
        records.append({
            "pair_id": p.get("pair_id", ""),
            "pair_type": p.get("pair_type", "?"),
            "persona_id": p.get("persona_id", ""),
            "logp_win_sum": lw_sum, "logp_lose_sum": ll_sum,
            "logp_win_mean": lw_mean, "logp_lose_mean": ll_mean,
            "len_win": lw_n, "len_lose": ll_n,
            "delta_sum": lw_sum - ll_sum,
            "delta_mean": lw_mean - ll_mean,
        })
        if (i + 1) % 25 == 0:
            print(f"  scored {i + 1}/{len(pairs)}")

    if not records:
        print("[error] 점수 매겨진 쌍이 없습니다.")
        sys.exit(1)

    # 그룹별(overall + pair_type별) 통계
    groups = defaultdict(list)
    for r in records:
        groups["overall"].append(r)
        groups[r["pair_type"]].append(r)

    blocks = {}
    for grp, recs in groups.items():
        d_sum = np.array([r["delta_sum"] for r in recs])
        d_mean = np.array([r["delta_mean"] for r in recs])
        blocks[grp] = {"sum": paired_stats(d_sum), "mean": paired_stats(d_mean)}

    # 히스토그램
    plot_files = {}
    if _HAS_MPL:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)
        for grp, recs in groups.items():
            d = np.array([r["delta_mean"] for r in recs])
            fig, axx = plt.subplots(figsize=(6, 4))
            axx.hist(d, bins=30, color="#4c72b0", alpha=0.85)
            axx.axvline(0, color="red", linestyle="--", linewidth=1)
            axx.set_title(f"Δ per-token logp (win-lose): {grp} (n={len(d)})")
            axx.set_xlabel("Δ = logp(win) - logp(lose)")
            axx.set_ylabel("count")
            fp = str(Path(args.plot_dir) / f"delta_hist_{grp}.png")
            fig.tight_layout(); fig.savefig(fp, dpi=120); plt.close(fig)
            plot_files[grp] = fp

    # 저장
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        json.dump({"model_label": args.model_label,
                   "n_pairs_scored": len(records),
                   "stats": blocks, "per_pair": records},
                  f, ensure_ascii=False, indent=2)
    write_markdown(blocks, records, args.out_md, args.model_label, plot_files)

    # 콘솔 요약
    print("\n" + "=" * 70)
    print(f"win/lose 로그확률 차이 — {args.model_label}  (scored {len(records)})")
    print("=" * 70)
    for grp in ["overall"] + [g for g in blocks if g != "overall"]:
        s = blocks[grp]["mean"]
        print(f"[{grp:13s}] n={s['n']:4d}  win-rate={s['win_rate']:.1%}  "
              f"meanΔ(per-tok)={s['mean_delta']:+.3f}  p={_fmt_p(s['t_pvalue'])}  "
              f"dz={s['cohens_dz']:.2f}")
    print(f"\n→ report(json): {args.out_report}")
    print(f"→ report(md):   {args.out_md}")
    if plot_files:
        print(f"→ plots:        {args.plot_dir}/")
    if not _HAS_SCIPY:
        print("[note] scipy 미설치 → 정규근사 검정 사용. 정확한 t/Wilcoxon은 "
              "`pip install scipy` 후 재실행.")


if __name__ == "__main__":
    main()
