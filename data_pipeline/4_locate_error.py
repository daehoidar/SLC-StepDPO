"""GPT-4o로 첫 오류 스텝의 위치를 식별한다 (Step-DPO 원본 step2와 동일 로직 + 페르소나 인지).

입력: 3_collect_errors.sh 산출물 (predictions/*.json, result=False만 필터링)
출력: data_pipeline/output/located_errors/{idx}.json

TODO: Step-DPO 레포의 locate_error_by_gpt4.py를 베이스로 작성.
페르소나 태그를 GPT-4o 프롬프트에 한 줄 추가하여 인지시키되,
첫 오류 스텝 식별 자체는 수학 정합성 판정이므로 로직은 동일하게 유지.
"""
