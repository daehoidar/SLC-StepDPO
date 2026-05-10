"""step_pair와 belief_pair를 병합하여 최종 학습 데이터를 만든다.

입력: data_pipeline/output/step_pairs.jsonl + belief_pairs.jsonl
출력: data_pipeline/output/train.jsonl

각 행은 type 필드(step_pair | belief_pair)를 보존한다.
학습 단계에서 type을 분기 신호로 사용하여 두 손실(L_step, L_belief)을 가중 합산한다.
"""
