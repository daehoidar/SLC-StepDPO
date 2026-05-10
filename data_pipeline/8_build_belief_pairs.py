"""페르소나 교차로 belief_pair 데이터를 빌드한다.

입력: 1_synthesize_sft.py 산출물(sft_data.jsonl) 또는 3단계의 정답 풀이 풀
출력: data_pipeline/output/belief_pairs.jsonl

각 행 형식:
    {"type": "belief_pair", "persona": "초등-하위권", "prompt": "...",
     "chosen": "(타겟 페르소나 톤의 정답 풀이 전체)",
     "rejected": "(다른 페르소나 톤의 정답 풀이 전체)"}

원리:
- 같은 문제 q에 대해 6개 페르소나의 정답 풀이가 모이면,
  각 페르소나 p에 대해 chosen=p, rejected=p'(p와 다른) 페어를 구성한다.
- 두 풀이 모두 정답(수학적 정합)이어야 한다. 화법 차이만 신호로 남김.
- 비교 단위는 풀이 전체 (step_pair와 달리 첫 스텝만 자르지 않음).
"""
