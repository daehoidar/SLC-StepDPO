"""derive_persona_evidence.py

personas.json + achievement_standards_2022.json -> enriched personas.json

각 페르소나에 두 가지를 자동 추가한다:
  1. exemplar_standards: 페르소나 학년·난이도에 해당하는 성취기준 진술 8개
                         (영역별 stratified random sampling, seed=42)
  2. term_evidence:      forbidden_terms / preferred_terms 각각이 2022 개정 수학과
                         교육과정에서 처음 도입되는 학년군과 대표 코드

또한 consistency check를 돌려 다음을 경고한다:
  - forbidden인데 페르소나 학년 이하에 이미 도입된 어휘 (forbidden 정당성 약함)
  - preferred인데 페르소나 학년 이후에야 등장하는 어휘 (학년 부적합)
  - 교육과정에서 발견되지 않는 어휘 (출처 불명)

출력은 personas.json을 in-place로 덮어쓴다. seed 고정으로 재현 가능.

Usage:
    python derive_persona_evidence.py
"""
from __future__ import annotations
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CURRICULUM_PATH = REPO_ROOT / "curriculum" / "achievement_standards_2022.json"
PERSONAS_PATH = REPO_ROOT / "personas.json"

# 학년군 순서 (낮은 학년 -> 높은 학년)
GRADE_ORDER = ["초등1-2", "초등3-4", "초등5-6", "중학교", "고등학교"]
GRADE_IDX = {g: i for i, g in enumerate(GRADE_ORDER)}

# personas.json의 id를 curriculum의 학년군 키로 매핑
PERSONA_GRADE_FILE = {
    "elem_low":   "초등3-4",
    "elem_high":  "초등5-6",
    "mid_low":    "중학교",
    "mid_high":   "중학교",
    "high_low":   "고등학교",
    "high_high":  "고등학교",
}

# 성취기준 코드에서 영역 번호 추출
# 초·중: [2수01-01] [4수02-01] -> 01,02,03,04
# 고등: [10공수1-01-01]         -> 01
_DOMAIN_RX = re.compile(r"\[(?:\d+수|1\d[가-힣A-Z]+\d*-)(\d{2})")


def _domain_of(code: str) -> str:
    m = _DOMAIN_RX.match(code)
    return m.group(1) if m else "00"


def _level_tag_for(persona: dict) -> str:
    """페르소나 grade_band + level -> 진술 단계 라벨.

    초등: ABC 3단계   (상위권=A, 하위권=C)
    중·고: ABCDE 5단계 (상위권=A, 하위권=E)
    """
    grade = persona["grade_band"]
    if "초등" in grade:
        return "A" if persona["level"] == "상위권" else "C"
    return "A" if persona["level"] == "상위권" else "E"


def find_first_appearance(term: str, curriculum: dict) -> tuple[str | None, str | None]:
    """term이 처음 등장하는 학년군과 대표 코드 반환.

    각 항목의 body / A / B / C / D / E 모든 필드를 합쳐서 검색.
    1글자 단어는 substring 오탐 위험이 크므로 매칭에서 제외(None 반환).
    """
    if len(term) <= 1:
        return None, None
    for grade in GRADE_ORDER:
        for item in curriculum.get(grade, []):
            haystack = item.get("body", "")
            for lv in ("A", "B", "C", "D", "E"):
                haystack += " " + item.get(lv, "")
            if term in haystack:
                return grade, item["code"]
    return None, None


def sample_exemplar_standards(
    curriculum: dict, grade_file: str, level_tag: str, k: int = 8, seed: int = 42,
) -> list[dict]:
    """영역별 stratified random sampling으로 k개 진술 추출."""
    pool: list[tuple[str, str]] = []
    for it in curriculum.get(grade_file, []):
        stmt = it.get(level_tag, "").strip()
        if stmt:
            pool.append((it["code"], stmt))

    by_domain: dict[str, list[tuple[str, str]]] = {}
    for code, stmt in pool:
        by_domain.setdefault(_domain_of(code), []).append((code, stmt))

    rng = random.Random(seed)
    for d in by_domain:
        rng.shuffle(by_domain[d])
    domains = sorted(by_domain.keys())
    rng.shuffle(domains)

    out: list[tuple[str, str]] = []
    while len(out) < k and any(by_domain[d] for d in domains):
        for d in domains:
            if by_domain[d] and len(out) < k:
                out.append(by_domain[d].pop())

    return [
        {"code": c, "level_tag": level_tag, "statement": s}
        for c, s in out
    ]


def check_term_consistency(
    term: str, term_type: str, intro_grade: str | None, persona_grade: str,
) -> tuple[str, str | None]:
    """forbidden/preferred 어휘의 학년군 부합성 확인.

    반환: (status, message)
      status:
        - "ok"                       : 부합 (forbidden은 학년 이후, preferred는 학년에 도입)
        - "contradiction"            : 진짜 모순 (warning)
        - "out_of_curriculum"        : 교육과정 외 (학부 어휘 또는 구어; info)
      message: status가 ok가 아닐 때 사유 문자열, ok면 None
    """
    if intro_grade is None:
        # forbidden: 학부/외부 어휘로 간주 - 정상
        # preferred: 일상어/구어로 간주 - 정상
        return "out_of_curriculum", f"[{term_type}] '{term}': 교육과정 외 어휘 (학부 또는 구어로 간주)"
    pg = GRADE_IDX[persona_grade]
    ig = GRADE_IDX[intro_grade]
    if term_type == "forbidden" and ig <= pg:
        return "contradiction", (
            f"[forbidden] '{term}': {intro_grade}에 이미 도입됨 "
            f"(페르소나 학년 {persona_grade}과 같거나 이전). forbidden 정당성 약함."
        )
    if term_type == "preferred" and ig > pg:
        return "contradiction", (
            f"[preferred] '{term}': {intro_grade}에 등장 "
            f"(페르소나 학년 {persona_grade} 이후). preferred로 부적합."
        )
    return "ok", None


def enrich_persona(persona: dict, curriculum: dict) -> dict:
    pid = persona["id"]
    grade_file = PERSONA_GRADE_FILE[pid]
    level_tag = _level_tag_for(persona)

    # 메타 정보
    persona["curriculum_grade_file"] = grade_file
    persona["curriculum_level_tag"] = level_tag

    # 1) Exemplar standards
    persona["exemplar_standards"] = sample_exemplar_standards(
        curriculum, grade_file, level_tag, k=8, seed=42,
    )

    # 2) Term evidence
    evidence: dict[str, dict] = {}
    contradictions: list[str] = []
    out_of_curr: list[str] = []
    override_list = set(persona.get("expected_in_grade_but_restricted", []))

    for term in persona["forbidden_terms"]:
        intro, code = find_first_appearance(term, curriculum)
        status, msg = check_term_consistency(term, "forbidden", intro, grade_file)
        # sub-grade 명시적 제한이면 contradiction -> explicit_override로 강등
        if status == "contradiction" and term in override_list:
            status = "explicit_override"
            msg = None
        evidence[term] = {
            "type": "forbidden",
            "first_introduced": intro,
            "source_code": code,
            "status": status,
        }
        if status == "contradiction" and msg:
            contradictions.append(msg)
        elif status == "out_of_curriculum" and msg:
            out_of_curr.append(msg)

    for term in persona["preferred_terms"]:
        intro, code = find_first_appearance(term, curriculum)
        status, msg = check_term_consistency(term, "preferred", intro, grade_file)
        evidence[term] = {
            "type": "preferred",
            "first_introduced": intro,
            "source_code": code,
            "status": status,
        }
        if status == "contradiction" and msg:
            contradictions.append(msg)
        elif status == "out_of_curriculum" and msg:
            out_of_curr.append(msg)

    persona["term_evidence"] = evidence
    if contradictions:
        persona["term_contradictions"] = contradictions
    elif "term_contradictions" in persona:
        del persona["term_contradictions"]
    # 호환을 위해 옛 키도 정리
    if "term_consistency_warnings" in persona:
        del persona["term_consistency_warnings"]

    return persona, contradictions, out_of_curr


def main():
    if not CURRICULUM_PATH.exists():
        sys.exit(f"[error] not found: {CURRICULUM_PATH}")
    if not PERSONAS_PATH.exists():
        sys.exit(f"[error] not found: {PERSONAS_PATH}")

    with open(CURRICULUM_PATH, encoding="utf-8") as f:
        curriculum = json.load(f)
    with open(PERSONAS_PATH, encoding="utf-8") as f:
        personas_data = json.load(f)

    per_persona_info: list[tuple[dict, list[str], list[str]]] = []
    for persona in personas_data["personas"]:
        _, contras, ooc = enrich_persona(persona, curriculum)
        per_persona_info.append((persona, contras, ooc))

    # Save (overwrites personas.json)
    with open(PERSONAS_PATH, "w", encoding="utf-8") as f:
        json.dump(personas_data, f, ensure_ascii=False, indent=2)

    # Console summary
    print(f"Enriched personas written to: {PERSONAS_PATH}")
    print()
    total_contras, total_ooc = 0, 0
    for p, contras, ooc in per_persona_info:
        total_contras += len(contras)
        total_ooc += len(ooc)
        n_ex = len(p.get("exemplar_standards", []))
        n_ev = len(p.get("term_evidence", {}))
        print(
            f"[{p['id']:>10s}] grade={p['curriculum_grade_file']:>6s} "
            f"level_tag={p['curriculum_level_tag']}  "
            f"exemplars={n_ex}  evidence={n_ev}  "
            f"contradictions={len(contras)}  out_of_curr={len(ooc)}"
        )
        for w in contras:
            print(f"    [contradiction] {w}")
        for m in ooc:
            print(f"    [info] {m}")
    print()
    print(f"Total contradictions: {total_contras}   (action required)")
    print(f"Total out_of_curriculum: {total_ooc}   (info only — advanced/colloquial terms)")
    if total_contras > 0:
        print("=> personas.json의 forbidden/preferred 정의를 재검토하세요.")


if __name__ == "__main__":
    main()
