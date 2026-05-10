"""정답 재샘플링 결과로 step_pair 데이터를 빌드한다.

입력: data_pipeline/output/correction_inputs.jsonl + corrections/*.json
출력: data_pipeline/output/step_pairs.jsonl

각 행 형식:
    {"type": "step_pair", "persona": "초등-하위권", "prompt": "...",
     "prefix": "Let's think step by step.\\nStep 1: ...\\nStep 2:",
     "chosen": " 정답 첫 스텝", "rejected": " 오답 첫 스텝", "answer": "..."}

TODO: Step-DPO 레포의 generate_dataset.py를 베이스로 작성.
chosen/rejected는 첫 스텝만 비교 (split('\\nStep ')[0]).
"""
