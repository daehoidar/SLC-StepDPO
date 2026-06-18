"""
judge_prompts.py

BC-StepDPO를 위한 GPT-4o judge prompt + 포매팅 헬퍼.

단일 axis(belief-conditional) 라벨링으로 단순화됨. 각 step에 대해
"target persona 조건 하에서 이 step이 acceptable한가"를 판단.

수학 오류든 페르소나 drift든 모두 "이 belief 조건에서 부적합"으로 통합 라벨링.

본 버전은 2022 개정 수학과 교육과정 성취기준(achievement_standards_2022.json)을
근거로 한 evidence block을 모든 system prompt에 끼운다. 이 evidence는
derive_persona_evidence.py로 personas.json에 자동 주입된다.
"""
from __future__ import annotations


# ===== 포매팅 헬퍼 ===========================================================

def _format_exemplar_standards_block(persona: dict) -> str:
    """페르소나의 exemplar_standards를 prompt-ready 문자열로 변환."""
    exemplars = persona.get("exemplar_standards", [])
    if not exemplars:
        return "  (curriculum reference not available; run derive_persona_evidence.py first)"
    return "\n".join(
        f"  - {ex['code']} (level {ex['level_tag']}): {ex['statement']}"
        for ex in exemplars
    )


def _format_forbidden_with_evidence(persona: dict) -> str:
    """forbidden_terms를 도입 학년·근거 코드와 함께 포매팅."""
    evidence = persona.get("term_evidence", {})
    if not evidence:
        # fallback: 단순 리스트
        return "\n".join(f"  - {t}" for t in persona["forbidden_terms"])
    lines = []
    grade_band = persona.get("grade_band", "?")
    for term in persona["forbidden_terms"]:
        ev = evidence.get(term, {})
        intro = ev.get("first_introduced") or "?"
        code = ev.get("source_code") or "?"
        lines.append(
            f"  - {term}: introduced at {intro} {code} -> forbidden for {grade_band}"
        )
    return "\n".join(lines) if lines else "  (none)"


def _format_preferred_with_evidence(persona: dict) -> str:
    """preferred_terms를 도입 학년·근거 코드와 함께 포매팅."""
    evidence = persona.get("term_evidence", {})
    if not evidence:
        return ", ".join(persona["preferred_terms"])
    parts = []
    for term in persona["preferred_terms"]:
        ev = evidence.get(term, {})
        intro = ev.get("first_introduced")
        code = ev.get("source_code")
        if intro and code:
            parts.append(f"{term} ({intro} {code})")
        else:
            parts.append(term)
    return ", ".join(parts)


def build_generator_kwargs(persona: dict) -> dict:
    """GENERATOR_SYSTEM.format(**kwargs)에 그대로 넘길 수 있는 dict."""
    return {
        "persona_tag": persona["tag"],
        "grade_band": persona["grade_band"],
        "level": persona["level"],
        "vocabulary_guide": persona["vocabulary_guide"],
        "explanation_style": persona["explanation_style"],
        "exemplar_standards_block": _format_exemplar_standards_block(persona),
        "forbidden_terms_with_evidence": _format_forbidden_with_evidence(persona),
        "preferred_terms_with_evidence": _format_preferred_with_evidence(persona),
    }


def build_step_judge_kwargs(persona: dict) -> dict:
    """STEP_JUDGE_SYSTEM.format(**kwargs)용 dict."""
    return build_generator_kwargs(persona)  # 동일 슬롯 사용


def build_cross_belief_kwargs(
    step_text: str, prefix_text: str, problem: str,
    persona_a: dict, persona_b: dict,
) -> dict:
    """CROSS_BELIEF_CHECK_SYSTEM.format(**kwargs)용 dict."""
    return {
        "step_text": step_text,
        "prefix_text": prefix_text,
        "problem": problem,
        "persona_a_tag": persona_a["tag"],
        "persona_a_grade": persona_a["grade_band"],
        "persona_a_level": persona_a["level"],
        "persona_a_vocab": persona_a["vocabulary_guide"],
        "persona_a_forbidden_evidence": _format_forbidden_with_evidence(persona_a),
        "persona_b_tag": persona_b["tag"],
        "persona_b_grade": persona_b["grade_band"],
        "persona_b_level": persona_b["level"],
        "persona_b_vocab": persona_b["vocabulary_guide"],
        "persona_b_forbidden_evidence": _format_forbidden_with_evidence(persona_b),
    }


# ===== 페르소나별 풀이 생성용 prompt (Stage 1: SFT 데이터 합성) =================

GENERATOR_SYSTEM = """You are a math tutor for a student aligned to the Korean 2022 math curriculum. Respond entirely in English.

============================================================
TARGET STUDENT PROFILE
============================================================
Persona tag: {persona_tag}
Grade band: {grade_band}
Level: {level}
Vocabulary guide: {vocabulary_guide}
Explanation style: {explanation_style}

CURRICULUM REFERENCE (what this student CAN do, per 2022 Korean curriculum):
{exemplar_standards_block}

FORBIDDEN terms (introduced AFTER this persona's grade per 2022 Korean curriculum):
{forbidden_terms_with_evidence}

PREFERRED terms (with first-introduction grade): {preferred_terms_with_evidence}

The forbidden terms are forbidden because they are introduced AFTER this persona's grade level. NEVER use them. Stay within the curriculum scope shown above.

============================================================
STRICT NOTATION RULES BY GRADE BAND
============================================================
- "Elementary grades 3-4" / "Elementary grades 5-6":
  * NO LaTeX of any kind. NO `\\[ ... \\]`, NO `\\( ... \\)`, NO `\\times`,
    NO `\\div`, NO `\\frac`.
  * NO algebraic variables (no `x`, `y`, `c`, `s`, etc.).
  * NO "Let x = ..." style setup.
  * Use plain words and digits only. Example OK: "5 times 5 is 25" or
    "5 x 5 = 25". Example BAD: "\\(5 \\times 5 = 25\\)".

- "Middle school grades 1-3" (중1-3):
  * NO LaTeX. Use plain symbols ×, ÷, =, +, - in plain text only.
  * Algebraic variables (x, y) ALLOWED for setting up equations.
  * Example OK: "Let x be the number, then 2x = 12 so x = 6."
  * Example BAD: "\\(2x = 12\\)".

- "High school grades 1-3" (고1-3):
  * LaTeX ALLOWED and encouraged for non-trivial algebra.
  * Formal variable definitions, equation systems, brief justifications OK.

============================================================
REASONING DEPTH RULES BY LEVEL
============================================================
- level "below-average" (low):
  * ONE concrete arithmetic operation per step.
  * Prefer repeated addition over multiplication when grade allows
    (e.g., elementary: "200 + 200 = 400" instead of "200 × 2 = 400").
  * Anchor at least one step to a concrete metaphor (objects, money,
    distance, pieces) — matches the persona's explanation_style.
  * NO abstract setup, NO verification step at the end.

- level "above-average" (high):
  * May combine 2 related operations in one step (still 1-3 sentences).
  * For middle/high: define intermediate variables or named quantities
    when it clarifies.
  * Optionally include a brief verification at the end.
  * Tone slightly more formal.

============================================================
GROUND-TRUTH ANCHORING (HARD CONSTRAINT)
============================================================
The final numerical answer MUST exactly match the "Ground-truth final answer"
provided in the user message. The LAST LINE of your output MUST be exactly:

    Final answer: <the ground-truth string verbatim>

Do not add extra units, currency symbols, or rewordings that the ground-truth
doesn't have. If the ground-truth is "15", end with "Final answer: 15"
(not "Final answer: 15 items").

============================================================
OUTPUT FORMAT
============================================================
Step 1: ...
Step 2: ...
...
Final answer: <ground-truth verbatim>

- Total 3-6 steps. Each step is 1-3 sentences. No empty steps.
- Do NOT prefix the first step with any preamble like "To solve this..." —
  start directly with "Step 1:".
"""


GENERATOR_USER_TEMPLATE = """Problem: {problem}

Ground-truth final answer: {ground_truth}

Write a step-by-step solution for the target persona, strictly obeying all
notation rules, reasoning-depth rules, and the ground-truth anchoring rule
above. The last line MUST be exactly "Final answer: {ground_truth}"."""


# ===== Belief-conditional step judge (Stage 3: pair 구축) ====================

STEP_JUDGE_SYSTEM = """You are evaluating a step-by-step math solution under a SPECIFIC target persona.

Target persona: {persona_tag}
Grade band: {grade_band}
Level: {level}
Vocabulary guide: {vocabulary_guide}
Style guide: {explanation_style}

CURRICULUM LEVEL (what THIS persona has reached, per 2022 Korean curriculum):
{exemplar_standards_block}

FORBIDDEN terms (with first-introduction grade per curriculum):
{forbidden_terms_with_evidence}

PREFERRED terms: {preferred_terms_with_evidence}

For EACH step, output ONE of:
- "acceptable": The step is BOTH mathematically valid AND stays within the persona's curriculum scope.
- "reject_math": The step has a mathematical error (regardless of persona).
- "reject_persona": The step is mathematically valid BUT uses vocabulary/concepts NOT YET introduced at this persona's grade per the 2022 Korean curriculum.

When labeling "reject_persona", cite the specific forbidden term and the curriculum grade where it is introduced.

OUTPUT FORMAT (JSON only):
{{
  "steps": [
    {{"index": 1, "label": "acceptable", "reason": "..."}},
    {{"index": 2, "label": "reject_persona", "reason": "Uses '통분' which is introduced at 초등5-6 [6수01-08], but this persona is 초등 3-4학년."}},
    {{"index": 3, "label": "reject_math", "reason": "1/2 was incorrectly converted to 4/6."}}
  ],
  "first_reject_step": 2,
  "first_reject_type": "reject_persona"
}}

"first_reject_step": 1-indexed position of the first non-acceptable step (or null if all acceptable).
"first_reject_type": the label of that step.

IMPORTANT: A step labeled "reject_persona" is MATHEMATICALLY CORRECT but uses concepts not yet introduced at this persona's grade.
The same step might be "acceptable" under a different persona that has reached the relevant curriculum point.
"""


STEP_JUDGE_USER_TEMPLATE = """Problem: {problem}
Ground truth answer: {ground_truth}
Target persona: {persona_tag}

Solution to evaluate (numbered steps):
{solution_with_steps}

Evaluate each step. Remember: persona drift is NOT a math error; the same step under different persona may be acceptable."""


# ===== Cross-belief check (Stage 3: Type-2 belief-flip pair 생성) =============
#
# 같은 step이 두 페르소나에서 다른 라벨을 받는지 확인.
# 이 라벨 차이가 Type-2 pair의 정당성을 보장하며,
# 차이의 근거는 2022 개정 수학과 교육과정의 학년별 도입 시점이다.

CROSS_BELIEF_CHECK_SYSTEM = """You are comparing how a single math step would be evaluated under two different student personas, using the 2022 Korean math curriculum as the objective basis.

Step: "{step_text}"
Prefix context (previous steps): "{prefix_text}"
Problem: "{problem}"

Persona A: {persona_a_tag} ({persona_a_grade}, {persona_a_level})
- Vocabulary guide: {persona_a_vocab}
- Forbidden (with curriculum evidence):
{persona_a_forbidden_evidence}

Persona B: {persona_b_tag} ({persona_b_grade}, {persona_b_level})
- Vocabulary guide: {persona_b_vocab}
- Forbidden (with curriculum evidence):
{persona_b_forbidden_evidence}

For each persona, judge whether the step is acceptable:
- If the step uses a term forbidden for that persona (per curriculum evidence above), the step is NOT acceptable for that persona.
- A "flip" occurs when the same step is acceptable for one persona but not the other.
- The most common flip cause: a math term X is introduced at grade G in the curriculum; persona A is below G, persona B is at or above G.

OUTPUT FORMAT (JSON only):
{{
  "persona_a_acceptable": true,
  "persona_b_acceptable": false,
  "flip": true,
  "confidence": 0.95,
  "curriculum_basis": "[6수01-08]",
  "trigger_term": "통분",
  "explanation": "Step uses '통분', introduced at 초등5-6 [6수01-08]. Persona A (초등5-6+) has reached this; Persona B (초등3-4) has not."
}}

"flip" is true if and only if persona_a_acceptable != persona_b_acceptable.
"confidence": float 0.0–1.0. How certain you are the flip is grounded in the curriculum evidence.
  - 1.0: explicit forbidden term with a clear curriculum code.
  - 0.7–0.9: term is borderline or the curriculum code is inferred.
  - <0.7: ambiguous; the flip may be stylistic rather than curriculum-grounded.
"curriculum_basis": the curriculum code that justifies the flip, or null if not applicable.
"trigger_term": the specific term in the step that caused the flip, or null.
"""


CROSS_BELIEF_CHECK_USER_TEMPLATE = """Compare the step's acceptability across two personas."""
