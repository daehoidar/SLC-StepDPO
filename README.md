# Persona-Step-DPO

2022 개정 교육과정 기반 다차원 페르소나 수학 튜터 sLLM 학습 파이프라인.
GDPO와 Step-DPO를 결합한 이중 제어 아키텍처로 Qwen3-0.6B를 미세조정한다.

Repo: https://github.com/daehoidar/Persona-Step-DPO

## 개요

기존 교육용 LLM의 두 가지 한계, 학습자 수준에 무관한 획일적 해설과 다단계
추론에서의 논리 오류를, 동시에 해결하는 것을 목표로 한다.

- 페르소나 6종: 연령 3종(초등, 중등, 고등) x 난이도 2종(상위권, 하위권)
- GDPO: 페르소나 조건부 화법 분포 정렬
- Step-DPO: 추론 스텝 단위 정합성 정렬
- 베이스 모델: Qwen3-0.6B
- 학습 데이터: GSM8K 기반 페르소나 합성 데이터

## 디렉토리 구조

```
Persona-Step-DPO/
  README.md
  requirements.txt
  personas.py                         페르소나 6종 정의 + system prompt 빌더
  curriculum/
    achievement_standards_2022.json   2022 개정 수학과 성취기준 구조화 자료
  data_pipeline/
    0_seed_problems.py                GSM8K 샘플링 + 난이도 버킷 배정
    1_synthesize_sft.py               GPT-4o로 페르소나별 정답 해설 합성
    2_run_sft.sh                      Qwen3-0.6B SFT 실행 스크립트
    3_collect_errors.sh               참조 모델로 오답 수집
    4_locate_error.py                 GPT-4o로 첫 오류 스텝 식별
    5_prepare_correction.py           오류 직전 prefix 추출
    6_rectify.sh                      참조 모델로 정답 재샘플링
    7_build_step_pairs.py             step_pair (단계 쌍) 생성
    8_build_belief_pairs.py           belief_pair (페르소나 쌍) 생성
    9_merge.py                        step_pair + belief_pair 병합
  configs/                            학습용 yaml 설정
  evaluation/                         정답 매칭 유틸 (Step-DPO에서 차용)
  train.py                            GDPO + Step-DPO 통합 Trainer
  eval_math_persona.py                페르소나 입력을 지원하는 평가 스크립트
  app.py                              데모 인터페이스
```

## 의존성

```
pip install -r requirements.txt
```

## 실행 순서

1. SFT 시드 데이터 생성

   ```
   python data_pipeline/0_seed_problems.py --per-persona 1500 --seed 42
   export OPENAI_API_KEY=sk-...
   python data_pipeline/1_synthesize_sft.py --concurrency 8 --max-cost 80
   ```

2. 참조 모델 SFT

   ```
   bash data_pipeline/2_run_sft.sh
   ```

3. 페르소나 조건부 오답 수집과 첫 오류 스텝 식별

   ```
   bash data_pipeline/3_collect_errors.sh
   python data_pipeline/4_locate_error.py
   ```

4. 정답 재샘플링과 step_pair 빌드

   ```
   python data_pipeline/5_prepare_correction.py
   bash data_pipeline/6_rectify.sh
   python data_pipeline/7_build_step_pairs.py
   ```

5. belief_pair 빌드와 최종 병합

   ```
   python data_pipeline/8_build_belief_pairs.py
   python data_pipeline/9_merge.py
   ```

6. GDPO + Step-DPO 학습

   ```
   accelerate launch train.py configs/persona_step_dpo.yaml
   ```

## 데이터 출처

- GSM8K (Cobbe et al., 2021): https://huggingface.co/datasets/openai/gsm8k
- 2022 개정 수학과 성취수준: 교육부 고시 제2022-33호 부속 자료. 원본 hwp는
  본 레포에 포함하지 않으며, `curriculum/achievement_standards_2022.json`은
  원본에서 추출한 텍스트만 담고 있다.

## 참고 문헌

- Lai et al. Step-DPO: Step-wise Preference Optimization for Long-chain
  Reasoning of LLMs. arXiv:2406.18629, 2024.
- Yao et al. No Preference Left Behind: Group Distributional Preference
  Optimization. ICLR 2025. arXiv:2412.20299.
- Rafailov et al. Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model. NeurIPS 2023.
