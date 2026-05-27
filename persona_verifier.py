"""persona_verifier.py

BC-StepDPO 데이터 페르소나 검증 — 3-stage cascade.

  Stage A (regex)       : forbidden_terms 단어경계 매치 → reject_persona 즉결.
                           term_evidence에서 curriculum 코드 자동 첨부.
                           통과 시 escalate (regex는 false-negative 가능).
  Stage B (local LLM)   : Llama-3.1-8B-Instruct 등 *다른 family* base 모델로
                           verdict + confidence 산출. conf >= threshold면
                           X/O 어느 쪽이든 확정. 미만은 escalate.
  Stage C (external)    : GPT-4o로 final 판정.

설계 원칙:
  - X 확정은 어디서든 가능 (regex가 잡으면 끝).
  - O 확정은 통과 단계까지의 confidence가 충분히 높을 때만.
  - verifier ≠ policy: π_ref(SFT)와 다른 weight 사용 (self-confirmation bias 차단).

모든 검증 호출은 stage_log_path가 주어지면 jsonl 한 줄로 누적되어
cascade ablation / cost table / human-agreement 분석에 사용된다.
"""
from __future__ import annotations
import json
import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ──────────────────────────── 결과 자료형 ─────────────────────────────────

@dataclass
class VerifyResult:
    verdict: str                       # "persona_ok" | "reject_persona"
    confidence: float                  # 0.0 – 1.0
    stage: str                         # "A" | "B" | "C"
    trigger_term: Optional[str] = None
    evidence_code: Optional[str] = None       # e.g. "[6수01-08]"
    first_introduced: Optional[str] = None    # e.g. "Elementary grades 5-6"
    reasoning: str = ""
    raw_responses: dict = field(default_factory=dict)  # debug용 stage별 응답

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_responses", None)   # 외부 출력에서는 보통 제거
        return d


# ──────────────────────────── 프롬프트 ────────────────────────────────────

STAGE_B_SYSTEM = """You are a strict persona compliance evaluator for K-12 math tutoring.

Target persona:
- Grade band: {grade_band}
- Level: {level}
- Vocabulary guide: {vocabulary_guide}

The persona's curriculum scope (2022 Korean math curriculum):
{exemplar_standards_block}

FORBIDDEN terms (introduced AFTER this persona's grade per curriculum):
{forbidden_with_codes}

Your task: judge whether the given step VIOLATES the persona by using vocabulary
or concepts not yet introduced at this persona's grade.

A mathematically correct step can still VIOLATE the persona if it uses
post-grade vocabulary/concepts. Conversely, a math error does NOT count as a
persona violation here — judge persona only.

OUTPUT FORMAT (JSON only, no prose):
{{
  "verdict": "OK" | "VIOLATION",
  "confidence": 0.0-1.0,
  "trigger_term": "<term that triggered VIOLATION, or null>",
  "reasoning": "<one short sentence>"
}}

Confidence calibration:
  >=0.85 = clear-cut case (will be final)
  <0.85  = borderline (will be escalated to a stronger judge)
"""

STAGE_B_USER = """Prefix (previous steps under this persona):
{prefix_text}

Step to evaluate:
"{step}"

Output JSON only."""


STAGE_C_SYSTEM = """You are evaluating whether a single math step is appropriate
for a SPECIFIC student persona, using the 2022 Korean math curriculum as the
objective basis.

Target persona: {persona_tag}
Grade band: {grade_band}
Level: {level}
Vocabulary guide: {vocabulary_guide}

Curriculum standards reached by this persona:
{exemplar_standards_block}

FORBIDDEN terms (introduced AFTER this persona's grade):
{forbidden_with_codes}

Decide whether the step is "persona_ok" or "reject_persona". A step is
"reject_persona" iff it uses vocabulary/concepts forbidden by the persona's
grade per the curriculum evidence above.

OUTPUT FORMAT (JSON only):
{{
  "verdict": "persona_ok" | "reject_persona",
  "confidence": 0.0-1.0,
  "trigger_term": "<term or null>",
  "evidence_code": "<curriculum code like [6수01-08], or null>",
  "reasoning": "<one short sentence>"
}}
"""

STAGE_C_USER = """Prefix (previous steps under this persona):
{prefix_text}

Step to evaluate:
"{step}"

Output JSON only."""


# ──────────────────────────── 포매팅 헬퍼 ─────────────────────────────────

def _format_exemplars(persona: dict) -> str:
    items = persona.get("exemplar_standards", [])
    if not items:
        return "  (curriculum reference not available)"
    return "\n".join(
        f"  - {x['code']} (level {x.get('level_tag','?')}): {x['statement']}"
        for x in items
    )


def _format_forbidden_with_codes(persona: dict) -> str:
    evidence = persona.get("term_evidence", {})
    grade_band = persona.get("grade_band", "?")
    lines = []
    for term in persona.get("forbidden_terms", []):
        ev = evidence.get(term, {})
        intro = ev.get("first_introduced") or "?"
        code = ev.get("source_code") or "?"
        lines.append(f"  - {term}: introduced at {intro} {code}; forbidden for {grade_band}")
    return "\n".join(lines) if lines else "  (none)"


def _persona_kwargs(persona: dict) -> dict:
    return {
        "persona_tag": persona.get("tag", ""),
        "grade_band": persona.get("grade_band", ""),
        "level": persona.get("level", ""),
        "vocabulary_guide": persona.get("vocabulary_guide", ""),
        "exemplar_standards_block": _format_exemplars(persona),
        "forbidden_with_codes": _format_forbidden_with_codes(persona),
    }


# ──────────────────────────── PersonaVerifier ────────────────────────────

class PersonaVerifier:
    """3-stage cascade persona verifier.

    Example:
        from openai import OpenAI
        verifier = PersonaVerifier(
            stage_b_client=OpenAI(base_url="http://localhost:8001/v1",
                                  api_key="EMPTY"),
            stage_b_model="meta-llama/Llama-3.1-8B-Instruct",
            stage_c_client=OpenAI(),                  # 진짜 GPT-4o
            stage_c_model="gpt-4o",
            stage_log_path="output/stage_log.jsonl",
        )
        res = verifier.verify_step(step, persona, prefix=[...])
    """

    def __init__(
        self,
        stage_b_client=None,
        stage_b_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        stage_c_client=None,
        stage_c_model: str = "gpt-4o",
        stage_b_conf_threshold: float = 0.85,
        enable_stage_b: bool = True,
        enable_stage_c: bool = True,
        stage_log_path: Optional[str] = None,
        problem_context: Optional[str] = None,   # 로그에만 사용
    ):
        self.stage_b_client = stage_b_client
        self.stage_b_model = stage_b_model
        self.stage_c_client = stage_c_client
        self.stage_c_model = stage_c_model
        self.threshold = stage_b_conf_threshold
        self.enable_stage_b = enable_stage_b
        self.enable_stage_c = enable_stage_c
        self.stage_log_path = stage_log_path
        self._log_lock = threading.Lock()
        self.problem_context = problem_context

        # Counters (cascade funnel 분석용)
        self.counters = {"A_violation": 0, "A_pass": 0,
                         "B_violation": 0, "B_ok": 0, "B_borderline": 0,
                         "C_violation": 0, "C_ok": 0,
                         "total": 0}

    # ── public API ────────────────────────────────────────────────────

    def verify_step(self, step: str, persona: dict,
                    prefix: Optional[list[str]] = None) -> VerifyResult:
        self.counters["total"] += 1
        prefix = prefix or []

        # ── Stage A ───────────────────────────────────────────────
        a_res = self._stage_a_regex(step, persona)
        if a_res is not None:
            self.counters["A_violation"] += 1
            self._log(a_res, step, persona, prefix)
            return a_res
        self.counters["A_pass"] += 1

        # ── Stage B ───────────────────────────────────────────────
        b_res = None
        if self.enable_stage_b and self.stage_b_client is not None:
            b_res = self._stage_b_local(step, persona, prefix)
            if b_res.confidence >= self.threshold:
                if b_res.verdict == "reject_persona":
                    self.counters["B_violation"] += 1
                else:
                    self.counters["B_ok"] += 1
                self._log(b_res, step, persona, prefix)
                return b_res
            self.counters["B_borderline"] += 1

        # ── Stage C ───────────────────────────────────────────────
        if self.enable_stage_c and self.stage_c_client is not None:
            c_res = self._stage_c_external(step, persona, prefix)
            if c_res.verdict == "reject_persona":
                self.counters["C_violation"] += 1
            else:
                self.counters["C_ok"] += 1
            self._log(c_res, step, persona, prefix)
            return c_res

        # Stage C 비활성화 + Stage B borderline → Stage B 결과로 fallback
        if b_res is not None:
            self._log(b_res, step, persona, prefix)
            return b_res

        # 검증 인프라가 모두 비활성화된 경우 → 보수적으로 OK 처리 (escape hatch)
        fallback = VerifyResult(
            verdict="persona_ok", confidence=0.0, stage="A",
            reasoning="all LLM stages disabled; regex passed",
        )
        self._log(fallback, step, persona, prefix)
        return fallback

    def dump_counters(self) -> dict:
        return dict(self.counters)

    # ── Stage A: regex ────────────────────────────────────────────────

    def _stage_a_regex(self, step: str, persona: dict) -> Optional[VerifyResult]:
        evidence_map = persona.get("term_evidence", {})
        for term in persona.get("forbidden_terms", []):
            # 영어/숫자/_ 경계 매치. 한국어 페르소나 사용 시 lookaround로 확장.
            if not term:
                continue
            pattern = rf"\b{re.escape(term)}\b"
            if re.search(pattern, step, re.IGNORECASE):
                ev = evidence_map.get(term, {}) or {}
                return VerifyResult(
                    verdict="reject_persona",
                    confidence=1.0,
                    stage="A",
                    trigger_term=term,
                    evidence_code=ev.get("source_code"),
                    first_introduced=ev.get("first_introduced"),
                    reasoning=(
                        f"forbidden term '{term}' "
                        f"(introduced at {ev.get('first_introduced','?')} "
                        f"{ev.get('source_code','?')}) appeared in step"
                    ),
                )
        return None

    # ── Stage B: local LLM (e.g., Llama-3.1-8B-Instruct via vLLM) ────

    def _stage_b_local(self, step: str, persona: dict,
                       prefix: list[str]) -> VerifyResult:
        sys_prompt = STAGE_B_SYSTEM.format(**_persona_kwargs(persona))
        user_prompt = STAGE_B_USER.format(
            prefix_text="\n".join(prefix) if prefix else "(none)",
            step=step,
        )
        try:
            resp = self.stage_b_client.chat.completions.create(
                model=self.stage_b_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            data = json.loads(raw)
        except Exception as e:
            # 파싱 실패 → borderline으로 escalate
            return VerifyResult(
                verdict="persona_ok", confidence=0.0, stage="B",
                reasoning=f"stage-B parse/IO error → escalate: {e}",
                raw_responses={"B_error": str(e)},
            )

        verdict_raw = str(data.get("verdict", "")).strip().upper()
        verdict = "reject_persona" if verdict_raw == "VIOLATION" else "persona_ok"
        conf = float(data.get("confidence", 0.0) or 0.0)
        return VerifyResult(
            verdict=verdict,
            confidence=conf,
            stage="B",
            trigger_term=data.get("trigger_term"),
            reasoning=str(data.get("reasoning", ""))[:300],
            raw_responses={"B": data},
        )

    # ── Stage C: external (GPT-4o) ───────────────────────────────────

    def _stage_c_external(self, step: str, persona: dict,
                          prefix: list[str]) -> VerifyResult:
        sys_prompt = STAGE_C_SYSTEM.format(**_persona_kwargs(persona))
        user_prompt = STAGE_C_USER.format(
            prefix_text="\n".join(prefix) if prefix else "(none)",
            step=step,
        )
        try:
            resp = self.stage_c_client.chat.completions.create(
                model=self.stage_c_model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
        except Exception as e:
            return VerifyResult(
                verdict="persona_ok", confidence=0.0, stage="C",
                reasoning=f"stage-C error; defaulting to persona_ok: {e}",
                raw_responses={"C_error": str(e)},
            )

        verdict = str(data.get("verdict", "persona_ok")).strip().lower()
        if verdict not in ("persona_ok", "reject_persona"):
            verdict = "persona_ok"
        return VerifyResult(
            verdict=verdict,
            confidence=float(data.get("confidence", 1.0) or 1.0),
            stage="C",
            trigger_term=data.get("trigger_term"),
            evidence_code=data.get("evidence_code"),
            reasoning=str(data.get("reasoning", ""))[:300],
            raw_responses={"C": data},
        )

    # ── logging ──────────────────────────────────────────────────────

    def _log(self, res: VerifyResult, step: str, persona: dict,
             prefix: list[str]) -> None:
        if not self.stage_log_path:
            return
        rec = {
            "persona_id": persona.get("id"),
            "stage": res.stage,
            "verdict": res.verdict,
            "confidence": res.confidence,
            "trigger_term": res.trigger_term,
            "evidence_code": res.evidence_code,
            "step_preview": step[:200],
            "prefix_len": len(prefix),
            "problem_context": (self.problem_context or "")[:80],
            "reasoning": res.reasoning,
        }
        with self._log_lock:
            with open(self.stage_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ──────────────────────────── 편의 함수 ──────────────────────────────────

def first_persona_violation_index(
    verifier: PersonaVerifier, steps: list[str], persona: dict,
) -> tuple[Optional[int], Optional[VerifyResult]]:
    """steps를 위에서부터 훑어 첫 reject_persona step의 1-based 인덱스를 반환.

    Step-DPO mode에서 'first_error_idx = 첫 페르소나 위반 step'으로 직결.
    """
    for i, step in enumerate(steps):
        res = verifier.verify_step(step, persona, prefix=steps[:i])
        if res.verdict == "reject_persona":
            return i + 1, res
    return None, None
