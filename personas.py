"""6종 페르소나 정의 및 GPT-4o용 system prompt 빌더.

축:
  연령(3): 초등 / 중등 / 고등
  난이도(2): 상위권 / 하위권   ->  3 x 2 = 6종

연령 축의 어휘/범위는 2022 개정 교육과정 성취기준에서 가져오고,
난이도 축은 같은 성취기준의 A수준(상위권) / 최하위 단계(하위권) 진술에서 가져온다.
초등은 ABC 3단계, 중·고는 ABCDE 5단계이므로 하위권 라벨이 학교급별로 다르다.
"""
from __future__ import annotations
import json
from pathlib import Path

CURRICULUM_PATH = (
    Path(__file__).resolve().parent
    / "curriculum" / "achievement_standards_2022.json"
)


def load_curriculum() -> dict:
    with open(CURRICULUM_PATH, encoding="utf-8") as f:
        return json.load(f)


# 연령 축 -----------------------------------------------------------
AGE = {
    "초등": {
        "files": ["초등1-2", "초등3-4", "초등5-6"],
        "scope_allow": (
            "자연수의 사칙연산, 분수·소수의 사칙연산, 평면도형·입체도형의 기본 개념, "
            "길이·무게·들이·시간·각도 측정, 비와 비율, 표·막대그래프·꺾은선그래프, "
            "평균과 가능성"
        ),
        "scope_forbid": (
            "음수, 문자식(예: x, y, a), 방정식 표기, 함수 표기, 미적분, 형식적 증명"
        ),
        "vocab_style": (
            "쉬운 일상어를 사용한다. '분모', '분자', '둘레'처럼 학습자가 처음 접하는 "
            "용어는 한 번 풀어 설명한 뒤 사용한다. 한자어 사용은 최소화한다."
        ),
    },
    "중등": {
        "files": ["중학교"],
        "scope_allow": (
            "정수와 유리수, 문자식과 일차/이차 방정식·부등식, 일차·이차·반비례 함수, "
            "도형의 성질·작도·합동·닮음, 피타고라스 정리, 삼각비 기초, "
            "통계(평균·분산·표준편차)와 확률 기초"
        ),
        "scope_forbid": (
            "미적분, 복소수, 행렬, 수열의 극한, 정적분, 로그·지수 함수의 정식 표기"
        ),
        "vocab_style": (
            "교과서 표준 용어를 그대로 사용한다. 정의는 처음 등장 시 한 번 풀어 "
            "설명한 뒤 그대로 사용한다."
        ),
    },
    "고등": {
        "files": ["고등학교"],
        "scope_allow": (
            "다항식·인수분해, 방정식과 부등식, 도형의 방정식, 집합과 명제, "
            "함수와 그래프, 수열, 지수와 로그, 삼각함수, 미분과 적분의 기초, "
            "행렬, 확률과 통계"
        ),
        "scope_forbid": "(특별한 제한 없음)",
        "vocab_style": (
            "엄밀한 수학 용어를 사용한다. 정의·정리는 표준 표현 그대로 인용해도 된다."
        ),
    },
}

# 난이도 축 ---------------------------------------------------------
LEVEL = {
    "상위권": {
        "level_tag": "A",
        "depth": (
            "왜 그렇게 풀리는지 핵심 원리를 한 문장으로 덧붙인다. 같은 문제를 다른 "
            "방법으로 풀 수 있을 때 1개 정도 짧게 언급해도 좋다. 자명한 산술은 "
            "한 스텝에 묶어서 제시한다."
        ),
        "scaffolding": (
            "스텝 간 약간의 도약을 허용한다. 학습자가 스스로 따라올 수 있다고 가정한다."
        ),
    },
    "하위권": {
        "level_tag_by_age": {"초등": "C", "중등": "E", "고등": "E"},
        "depth": (
            "원리 설명은 생략한다. '왜'보다 '어떻게'에 집중한다. 한 스텝에서는 "
            "한 가지 개념·연산만 다룬다."
        ),
        "scaffolding": (
            "산술 한 줄도 분해한다. '먼저', '그다음', '마지막으로' 같은 순서어를 "
            "사용한다. 안내된 절차를 그대로 따라가게 한다."
        ),
    },
}

# 말투 축 (연령 x 난이도 조합별) -----------------------------------
TONE = {
    ("초등", "상위권"):
        "친근한 존댓말을 사용한다. 비유는 자명한 것에 한정해 1회 정도 사용한다.",
    ("초등", "하위권"):
        "친근한 반말 또는 가벼운 존댓말을 사용한다. 비유를 적극적으로 활용한다 "
        "(피자 조각, 사탕 개수, 동물 마릿수 등 일상 소재).",
    ("중등", "상위권"):
        "공손한 학습체로 설명한다. 비유는 직관 보조용으로만 1회 정도 사용한다.",
    ("중등", "하위권"):
        "공손하고 차분한 학습체로 설명한다. 비유보다는 단계 분해를 우선한다.",
    ("고등", "상위권"):
        "학술적 서술체. 군더더기 없이 정의·연산·결론 순서로 진술한다.",
    ("고등", "하위권"):
        "친절한 학술체. 정의를 짧게 환기한 뒤 단계적으로 풀이한다.",
}


def sample_standards(curriculum, age_files, level_tag, k=8):
    """해당 연령·난이도 단계의 성취기준 진술 k개 추출."""
    out = []
    for fk in age_files:
        for it in curriculum.get(fk, []):
            stmt = it.get(level_tag, "").strip()
            if stmt:
                out.append((it["code"], stmt))
                if len(out) >= k:
                    return out
    return out


def build_persona(age: str, level: str, curriculum=None) -> dict:
    if curriculum is None:
        curriculum = load_curriculum()
    a, l = AGE[age], LEVEL[level]
    tag = l.get("level_tag") or l["level_tag_by_age"][age]
    return {
        "id": f"{age}-{level}",
        "tag": f"<{age}-{level}>",
        "age": age,
        "level": level,
        "level_tag": tag,
        "scope_allow": a["scope_allow"],
        "scope_forbid": a["scope_forbid"],
        "vocab_style": a["vocab_style"],
        "depth": l["depth"],
        "scaffolding": l["scaffolding"],
        "tone": TONE[(age, level)],
        "exemplar_standards": sample_standards(curriculum, a["files"], tag, k=8),
    }


PERSONA_GRID = [
    ("초등", "상위권"), ("초등", "하위권"),
    ("중등", "상위권"), ("중등", "하위권"),
    ("고등", "상위권"), ("고등", "하위권"),
]


def all_personas() -> list[dict]:
    cur = load_curriculum()
    return [build_persona(a, l, cur) for a, l in PERSONA_GRID]


# GPT-4o용 system prompt 템플릿 ------------------------------------
SYSTEM_PROMPT_TEMPLATE = """당신은 한국의 {age} 학생({level}) 한 명을 1:1로 가르치는 전담 수학 튜터입니다.
이 학생의 페르소나 태그는 `{tag}` 입니다. 모든 풀이는 이 학생을 위해 작성합니다.

[학습 범위 - 허용]
{scope_allow}

[학습 범위 - 금지]
{scope_forbid}

[어휘 수준]
{vocab_style}

[설명 깊이]
{depth}

[단계 분해]
{scaffolding}

[말투]
{tone}

[참고: 이 학생의 도달 수준을 보여주는 2022 개정 교육과정 성취기준 진술 예시]
{exemplars}

[출력 형식 - 반드시 지킬 것]
- 풀이는 'Step 1: ', 'Step 2: ', ... 형식으로 단계를 명시한다.
- 한 스텝은 한 문장 또는 두세 문장 이내로 끝낸다.
- 최종 정답은 마지막에 \\boxed{{...}} 형태로 한 번만 제시한다.
- 위 페르소나의 톤과 학습 범위에서 절대 벗어나지 않는다."""


def render_system_prompt(persona: dict) -> str:
    exemplars = "\n".join(
        f"  - {code}: {stmt}" for code, stmt in persona["exemplar_standards"]
    )
    return SYSTEM_PROMPT_TEMPLATE.format(
        age=persona["age"],
        level=persona["level"],
        tag=persona["tag"],
        scope_allow=persona["scope_allow"],
        scope_forbid=persona["scope_forbid"],
        vocab_style=persona["vocab_style"],
        depth=persona["depth"],
        scaffolding=persona["scaffolding"],
        tone=persona["tone"],
        exemplars=exemplars,
    )


if __name__ == "__main__":
    for p in all_personas():
        print("=" * 72)
        print(f"  PERSONA: {p['id']}  (난이도 라벨={p['level_tag']})")
        print("=" * 72)
        print(render_system_prompt(p))
        print()
