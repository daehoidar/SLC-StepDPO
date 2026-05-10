#!/usr/bin/env bash
# correction_inputs.jsonl의 prefix 뒤를 모델이 자기 자신으로 이어쓰며 정답 후보 생성.
# Step-DPO 원본 step3.sh의 페르소나 변형.
#
# 입력: data_pipeline/output/correction_inputs.jsonl
# 출력: data_pipeline/output/corrections/*.json
#
# TODO: eval_math_persona.py를 --prompt qwen3-persona-prefix 모드로 호출.

set -e
echo "TODO: implement self-rectification"
