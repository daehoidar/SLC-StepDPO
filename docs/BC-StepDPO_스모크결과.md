# BC-StepDPO 파이프라인 스모크 결과

> SFT 모델 출력 → GPT-4o judge → 선호쌍 데이터 → BC-StepDPO 학습까지 소량(문제 1~2개)으로 검증한 결과.

## 1. 파이프라인 단계별 산출량

| 단계 | 산출 |
|---|---|
| ① 샘플링 (SFT 출력) | chains **12** (문제 1 × 페르소나 6) |
| ② persona judge | 라벨된 chain **10** |
| ③ 선호쌍 | 총 **1** (Type-1 step 1 / Type-2 belief 0) |
| reject 사유 | {'reject_math': 1} |

## 2. judge 시점 — 한 풀이의 step별 판정

persona verifier가 **각 step을 판정**하고, math judge가 그 위에 수학 오류를 본다. 아래는 한 샘플 chain의 step별 결과.

**문제** (elem_low, gt=413): If a hairstylist charges $5 for a normal haircut, $6 for a special haircut, and $8 for a trendy haircut, and he cuts 5 normal haircuts, 3 special haircuts, and 

| # | step (요약) | persona 판정 | stage | trigger |
|---|---|---|---|---|
| 1 | Step 1: Calculate the daily earnings of the hairstylist by a… | 🟢 ok | C | - |
| 2 | Step 2: Determine the weekly earnings by multiplying the dai… | 🟢 ok | C | - |

## 3. 생성된 선호쌍 예시

**Type-1 (수학/단일 step)** — `step_pair` / reject=reject_math (persona=elem_low)
- ✅ chosen: Step 1: Calculate the total amount earned per day from normal haircuts. He charges $5 per normal haircut and cuts 5 of them, so it's 5 * 5 = 25 dollars.
- ❌ rejected: Step 1: Calculate the daily earnings of the hairstylist by adding the charges for the three types of haircuts. The charges are $5 for a normal haircut, $6 for a special haircut, and $8 for a trendy ha

## 4. BC-StepDPO 학습 로그

_학습 로그 미지정 또는 없음. `--train-log logs/bc_pipe_<jobid>.out` 지정._

---
*소량 스모크 결과. 정상 동작 확인 후 전체 학습으로 확장.*
