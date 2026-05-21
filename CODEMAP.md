# CODEMAP — 파일별 한 줄 요약

레포 전체 코드/설정/문서 파일을 *한 줄로* 정리. 각 항목 아래 *부연*은 필요한 한도에서.

## 📄 문서

### `README.md`
> BC-StepDPO 방법론·손실식·페르소나·repo 구조의 일반 소개.

페이퍼-style overview. 손실 수식 + ablation grid 표 + 의존성 + 참고문헌. 첫 진입점.

### `PIPELINE.md`
> Stage 0~5 전체 데이터 파이프라인을 **Step-DPO**와 **Full Step-DPO** 두 모드로 나눠 설명.

흐름도, 분기점(Stage 3), 두 모드에서 같은 코드/다른 toggle임을 명시. 팀원이 *실행 흐름*을 이해할 때 보는 문서.

### `CODEMAP.md`
> 본 파일. 모든 코드/설정/문서 파일의 한 줄 요약.

---

## ⚙️ 페르소나·교육과정 정의

### `personas.json`
> **활성** 페르소나 정의 6종 (영어). 메타 + 화법 + forbidden/preferred terms + 교육과정 evidence.

`utils.load_personas`로 모든 stage에서 로드됨. 영어 입출력 파이프라인용. 한국어 원본은 `personas_ko.json`에 백업.

### `personas_ko.json`
> 영어 변환 이전의 원본 한국어 페르소나 정의 (백업).

한국어 데이터셋(KMMLU-Math, AIHub 등)으로 복귀할 때 `mv personas_ko.json personas.json`.

### `curriculum/achievement_standards_2022.json`
> 한국 2022 개정 수학과 성취기준 254개 (학년군 + 영역 + 진술).

`derive_persona_evidence.py`가 이걸로 페르소나의 `exemplar_standards` + `term_evidence`를 자동 주입. 외부 의존 없는 정적 reference.

### `derive_persona_evidence.py`
> personas.json을 교육과정과 cross-reference해 evidence를 자동 enrich.

forbidden/preferred 어휘의 첫 도입 학년·코드 자동 매칭 + consistency check. ⚠️ 영어 personas.json에 다시 돌리면 한국어로 덮어씀 — 주의.

---

## 🔧 공통 유틸·인프라

### `utils.py`
> 공용 헬퍼: `load_personas`·`parse_steps`·`extract_gsm8k_answer`.

3줄짜리 작은 모듈. 거의 모든 stage에서 import.

### `inference_backend.py`
> vLLM 미지원 환경(Mac M-series)을 위한 transformers fallback wrapper.

`TransformersLLM` / `TransformersSamplingParams`가 vLLM의 동명 클래스 인터페이스를 흉내냄. `3_build_pairs.py`·`5_evaluate.py`가 try/except import로 자동 fallback.

### `judge_prompts.py`
> GPT-4o judge 3종(generator·step judge·cross-belief check) prompt + 포매팅 헬퍼.

페르소나·교육과정 evidence를 system prompt에 자동 주입. 영어 응답 강제.

### `bc_stepdpo_loss.py`
> BC-StepDPO 손실 함수 (Proposition 2)와 step_logprob 헬퍼.

prefix 토큰을 step_mask로 가리고 step 토큰의 log p_θ·log p_ref만 합산. 단일 손실로 Type-1·Type-2 동시 처리.

---

## 🛤 데이터 파이프라인 (Stage 0~5)

### `data_pipeline/0_seed_problems.py`
> MetaMathQA-40K에서 GSM_ 계열만 + query dedupe → 6 페르소나 복제 배정 (seed_problems.jsonl).

AnsAug의 같은 query 중복 제거가 핵심. easy+medium 버킷만 사용.

### `data_pipeline/1_synthesize_sft.py`
> 각 (problem, persona) 쌍에 GPT-4o로 페르소나 조건 풀이 N개 합성 → sft_data.jsonl.

ThreadPoolExecutor 동시 호출. judge_prompts.GENERATOR_SYSTEM을 시스템 프롬프트로 사용.

### `data_pipeline/2_train_sft.py`
> sft_data.jsonl로 base LLM(Qwen3-0.6B)을 LoRA SFT → π_ref 모델.

Accelerate + transformers + peft. 두 모드 공통의 reference 모델 생성.

### `data_pipeline/3_build_pairs.py`
> π_ref로 K샘플 + GPT-4o judge·cross-belief check → Type-1 + Type-2 preference pair (preference_pairs.jsonl).

Mac에선 vLLM이 transformers로 자동 fallback. 본 framework의 데이터 면 핵심.

### `data_pipeline/3_5_analyze_flip_rate.py`
> preference_pairs.jsonl 스캔해 Type-2 flip rate·페르소나별 flip 매트릭스 집계 → flip_stats.json + 콘솔 표.

Proposition 3 검증 산출물. Full 모드 핵심 지표, Step-DPO 모드엔 의미 없음.

### `data_pipeline/4_train_bc_stepdpo.py`
> π_ref + preference_pairs.jsonl + bc_stepdpo_loss → BC-StepDPO 학습 (config의 toggle로 모드 결정).

`disable_step_mask`·`disable_belief_token`·`disable_type2` 3개 toggle로 Vanilla DPO / Step-DPO / Conditional DPO / BC-StepDPO 4종을 같은 코드로.

### `data_pipeline/5_evaluate.py`
> 학습 모델로 페르소나×문제 풀이 생성 → final acc + step judge + persona consistency + flip handling 측정.

벤치마크 vLLM/transformers 호환. 두 모드 비교용 동일 지표.

### `data_pipeline/run_full_pipeline.sh`
> Stage 0~5 일괄 실행 orchestrator (Linux+CUDA 권장).

Mac에선 Stage 0~2까지만 정상. Stage 3·5는 transformers fallback 사용 가능하나 느림.

---

## ⚙️ 학습 설정

### `configs/default.yaml`
> Full Step-DPO (BC-StepDPO) 기본 학습 설정 — toggle 3개 모두 OFF (즉 belief·type2 다 사용).

상단 주석에 6 ablation 조건의 toggle 조합 표 있음.

### `configs/step_dpo.yaml`
> Step-DPO 모드 preset — `disable_belief_token: true, disable_type2: true` (다른 값은 default와 동일).

원본 Lai et al. Step-DPO와 동등한 학습용.

---

## 🧪 테스트 하니스 (tests/)

### `tests/README.md`
> 3 phase × 2 mode sanity test의 사용법 + 비용·시간 추정 + 트러블슈팅.

처음 보는 사람도 5분 안에 phase A→B→C 일괄 실행 가능.

### `tests/run_sft_data.sh`
> Phase A: Stage 0+1 소규모 실행 후 summarize.py로 REPORT.md 자동 생성.

기본 N=2, sols=2. GPT-4o ~24회.

### `tests/run_sft_train.sh`
> Phase B: Stage 2 1-epoch sanity 학습 + REPORT.md.

epochs/batch_size override를 임시 yaml로 처리. NaN check 포함.

### `tests/run_pairs.sh`
> Phase C: Stage 3 + 3.5를 모드(`step_dpo`|`full`) 인자로 실행 + REPORT.md.

체크포인트가 없으면 base model로 fallback. K=2 default.

### `tests/summarize.py`
> 각 phase의 산출물을 분석해 REPORT.md를 자동 작성하는 generator.

`phase`(sft_data/sft_train/pairs)별 다른 통계. pass/fail verdict + 트러블슈팅 힌트 자동 추가.

---

## 🎨 데모 (예정)

### `app.py`
> 학습된 모델 데모 진입점 (TODO 스텁).

미구현. Gradio 기반 페르소나 선택 + 풀이 생성 데모 UI 예정.

---

## 디렉토리별 한 줄 정리

| 디렉토리 | 한 줄 |
|---|---|
| `/` | 핵심 모듈 + 페르소나 + 설정 + 손실 |
| `configs/` | 학습 yaml 설정 (default = Full, step_dpo = Step-DPO) |
| `curriculum/` | 한국 2022 개정 수학 교육과정 (페르소나 evidence 소스) |
| `data_pipeline/` | Stage 0~5 모든 데이터 처리·학습·평가 스크립트 |
| `tests/` | phase별 sanity test + REPORT.md 자동 생성 |
| `outputs/` | 학습 산출 체크포인트 (gitignored) |
| `checkpoints/` | 동일 (gitignored) |
| `data_pipeline/output/` | 파이프라인 jsonl 산출물 (gitignored) |
| `tests/output/` | 테스트 산출물 + REPORT.md (gitignored) |

---

## 🔗 의존 그래프 (요약)

```
                      ┌──────────────────┐
                      │ utils.py         │
                      │ judge_prompts.py │ ─┐
                      │ personas.json    │ │
                      │ inference_backend│ │
                      └──────────────────┘ │
                                           ↓
       0_seed → 1_synth → 2_train_sft → 3_build_pairs → 3_5_analyze
                              │                ↓
                              └────────→ 4_train_bc_stepdpo (uses bc_stepdpo_loss.py)
                                                ↓
                                         5_evaluate
```

테스트는 위 흐름의 각 화살표를 *최소 규모로* 한 번씩 검증.
