"""GDPO + Step-DPO 통합 Trainer 진입점.

train.jsonl의 type 필드(step_pair | belief_pair)를 분기 신호로 두고,
두 손실 (L_step, L_belief)을 가중 합산하여 학습한다.

TODO: Step-DPO 레포의 stepdpo_trainer.py와 trl DPOTrainer를 베이스로 구현.
"""
