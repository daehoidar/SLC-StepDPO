#!/usr/bin/env bash
# 참조 모델 pi_SFT로 페르소나 조건부 풀이를 대량 샘플링하여 정/오 라벨링한다.
# Step-DPO 원본 step1.sh의 페르소나 변형.
#
# 입력: SFT된 모델 (outputs/pi_sft_qwen3_0.6b)
#       문제 jsonl (페르소나 6종 x rep=2로 펼침)
# 출력: data_pipeline/output/predictions/*.json
#
# TODO: eval_math_persona.py 작성 후 vLLM 추론 호출.

set -e
echo "TODO: implement error collection"
