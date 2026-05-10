"""오류 스텝 직전까지의 prefix를 추출하여 재샘플링용 입력 jsonl을 만든다.

입력: data_pipeline/output/located_errors/*.json
출력: data_pipeline/output/correction_inputs.jsonl

TODO: Step-DPO 레포의 prepare_for_correction.py를 베이스로 작성.
persona 필드를 보존하도록 한 줄 추가.
"""
