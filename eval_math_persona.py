"""페르소나 조건부 입력을 지원하는 vLLM 평가/추론 스크립트.

Step-DPO 레포의 eval_math.py에 다음 prompt 템플릿 분기를 추가한 변형:
  - qwen3-persona-step:   <persona> 태그 + Step 1 강제 시작
  - qwen3-persona-prefix: <persona> 태그 + 주어진 prefix 이어쓰기

TODO: eval_math.py를 베이스로 작성. evaluation/ 모듈의 정답 매칭 함수 재사용.
"""
