"""
보고서 품질 평가 에이전트 (Quality Evaluator / LLM Judge)
코드 기반 규칙 검사와 LLM Judge를 결합하여 4대 평가 기준(Groundedness, 중립성, 편향 통제, 관점 커버리지)을 심사한다.
"""

import re
from typing import List, Literal, Optional
from urllib.parse import urlparse

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from pydantic import BaseModel, Field

from agents.report_gen import report_inputs
from agents.synthesis import collect_references
from agents.worker_utils import PLANNER_MODEL
from graph.state import CriterionResult, MAX_REVISIONS, ReportReview, ReportState

URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")


def _urls(text: str) -> set:
    return {url.rstrip(".,;") for url in URL_PATTERN.findall(text)}


def check_reference_links(report: str, references: list[str]) -> tuple[bool, str]:
    """Groundedness(코드): 보고서에 나온 URL이 모두 실제 참고 자료 목록에 있는가."""
    unknown = sorted(_urls(report) - _urls("\n".join(references)))
    if unknown:
        return False, f"참고 자료 목록에 없는 URL: {', '.join(unknown)}"
    return True, "보고서의 URL이 모두 참고 자료 목록에 있음"


# '4. 관점별 평가'에 반드시 있어야 하는 소제목 (prompts/report_gen.txt의 출력 형식과 일치)
REQUIRED_PERSPECTIVES = ["기술성숙도 관점", "시장성 관점", "이해관계자 관점", "도메인 적용 관점", "종합의견"]


def _section(report: str, name: str) -> Optional[str]:
    """제목에 name이 들어간 절의 본문을 돌려준다 (같거나 더 높은 수준의 다음 제목 전까지).
    띄어쓰기 차이('기술 성숙도 관점')는 같은 것으로 본다."""
    key = re.sub(r"\s+", "", name)
    lines = report.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#") and key in re.sub(r"\s+", "", stripped):
            level = len(stripped) - len(stripped.lstrip("#"))
            body = []
            for nxt in lines[i + 1:]:
                s = nxt.lstrip()
                if s.startswith("#") and len(s) - len(s.lstrip("#")) <= level:
                    break
                body.append(nxt)
            return "\n".join(body)
    return None


def check_perspective_sections(report: str) -> tuple[bool, str]:
    """관점 커버리지(코드): 평가 관점 5개가 모두 소제목으로 있고, 명세가 '반드시' 요구한 두 가지
    (TRL이 공개 정보 기반 추정임을 명시, 종합의견에 상충 지점 명시)가 해당 절에 들어 있는가."""
    sections = {name: _section(report, name) for name in REQUIRED_PERSPECTIVES}
    problems = []
    missing = [name for name, body in sections.items() if body is None]
    if missing:
        problems.append(f"소제목이 없는 평가 관점: {', '.join(missing)}")

    trl = sections.get("기술성숙도 관점")
    if trl is not None:
        if "TRL" not in trl:
            problems.append("기술성숙도 관점에 TRL 등급이 없음")
        if not re.search(r"공개\s*정보", trl):
            problems.append("기술성숙도 관점에 '공개 정보 기반 추정'이라는 명시가 없음")
    overall = sections.get("종합의견")
    if overall is not None and "상충" not in overall:
        problems.append("종합의견에 관점 간 상충 지점이 명시되지 않음")

    if problems:
        return False, "; ".join(problems)
    return True, "평가 관점 5개가 모두 있고, TRL의 공개 정보 기반 추정 명시와 종합의견의 상충 지점 명시도 있음"


def check_source_diversity(references: list[str]) -> tuple[bool, str]:
    """편향 통제(코드): 근거가 단일 출처에 몰려 있지 않은가.
    원문은 파일 단위, 웹은 도메인 단위로 출처 종류를 센다."""
    papers, domains = set(), set()
    for ref in references:
        urls = _urls(ref)
        if urls:
            domains.update(urlparse(url).netloc for url in urls)
        else:
            papers.add(ref.split(" (p.")[0])
    total = len(papers) + len(domains)
    summary = f"출처 {total}종 (원문 {len(papers)}건, 웹 도메인 {len(domains)}개)"
    if total == 0:
        return False, "근거로 쓴 출처가 없음"
    if total == 1:
        return False, f"{summary} - 단일 출처에 의존"
    return True, summary


CRITERIA = {
    "Groundedness": "보고서의 주장(수치, 기업명, 사례, 평가 문장)이 [근거 자료]나 [참고 자료 목록]으로 추적되는가. "
    "근거 자료에 없는 수치·사례가 있거나, 출처를 알 수 없는 핵심 주장이 있으면 미달",
    "중립성": "특정 기술을 추천하거나 우열을 판정하는 표현이 없는가. '더 우수하다', '더 적합하다', "
    "'~을 권장한다', '승자' 같은 판정이 SUMMARY나 시사점을 포함해 어디에든 있으면 미달. "
    "관점별 특성과 차이를 병렬로 서술한 것은 통과",
    "편향 통제": "한 기술에 유리한 근거만 편중되지 않았는가. 두 기술 모두 강점과 한계가 함께 서술되고, "
    "근거 자료에 있는 불리한 내용이 빠지지 않았으며, 단일 출처에만 기대어 내린 결론이 없으면 통과. "
    "두 기술의 조사 깊이나 서술 분량이 다른 것 자체는 미달 사유가 아니다(기술마다 확인할 내용과 "
    "공개된 자료의 양이 다름). 근거의 '선택'이 한쪽에 유리하게 치우쳤는지만 본다",
    "관점 커버리지": "'4. 관점별 평가'의 다섯 관점이 각각 정해진 기준으로 다뤄졌는가. "
    "기술성숙도 관점: TRL 9단계 등급과 근거, 공개 정보 기반 추정임을 명시 / "
    "시장성 관점: 시장 규모·성장성, 상용화·채택 현황, 생태계 지지 / "
    "이해관계자 관점: 경쟁 기술 진영, 도입 기업·개발자, 투자 업계 / "
    "도메인 적용 관점: 선택한 도메인에서의 규모·비용·제약 / "
    "종합의견: 네 관점을 종합하고 관점 간 상충 지점을 명시. "
    "근거가 없는 세부 기준은 '근거 부족'으로 명시했으면 통과, 아예 빠졌거나 제목만 있으면 미달",
}


class CriterionJudgement(BaseModel):
    name: Literal["Groundedness", "중립성", "편향 통제", "관점 커버리지"]
    passed: bool = Field(description="기준을 충족하면 true")
    comment: str = Field(description="판정 근거. 미달이면 보고서의 어느 문장을 어떻게 고쳐야 하는지 구체적으로")


class ReviewOutput(BaseModel):
    criteria: List[CriterionJudgement] = Field(description="네 항목 각각에 대한 판정")


review_prompt = ChatPromptTemplate.from_template(
    """당신은 기술 평가 보고서의 품질을 검수하는 평가자입니다.
이 보고서의 목적은 두 기술의 우열을 가리는 것이 아니라, 여러 관점에서 각 기술의 특성을 정리하는 것입니다.
아래 [보고서]를 [근거 자료]와 대조해, [평가 항목] 네 가지를 각각 판정하세요.
보고서를 고쳐 쓰지 말고 판정과 사유만 내세요. 엄격하게 판정하되, 사유는 보고서의 구체적인 문장을 짚으세요.

[평가 항목]
{criteria}

[근거 자료 - 워커 조사 결과]
### 기술 조사(SW)
{tech_research_sw}

### 기술 조사(HW)
{tech_research_hw}

### 시장 평가
{market_eval}

### 이해관계자 평가
{stakeholder_eval}

### 도메인 평가
{domain_eval}

### 평가 종합
{synthesis}

[자료 한계]
{data_limited}

[참고 자료 목록]
{references}

[보고서]
{report}"""
)


def quality_eval(state: ReportState) -> dict:
    revision = state["revision"]
    report = state["report"]
    references = collect_references(state)
    print(f"\n==== [QUALITY EVAL] {revision}번째 보고서 ====")

    # 1. 코드로 확인할 수 있는 부분 (해당 항목의 LLM 판정과 둘 다 통과해야 통과)
    code_checks = {
        "Groundedness": check_reference_links(report, references),
        "편향 통제": check_source_diversity(references),
        "관점 커버리지": check_perspective_sections(report),
    }

    # 2. LLM 판정
    error = None
    judged = {}
    try:
        review_llm = init_chat_model(PLANNER_MODEL, model_provider="openai", temperature=0)
        reviewer = review_prompt | review_llm.with_structured_output(ReviewOutput)
        output = reviewer.invoke(
            {
                **report_inputs(state),
                "report": report,
                "criteria": "\n".join(f"- {name}: {rule}" for name, rule in CRITERIA.items()),
            }
        )
        judged = {c.name: c for c in output.criteria}
    except Exception as e:
        error = f"평가 LLM 호출 실패: {e}"

    # 3. 항목별로 코드 검사와 LLM 판정을 합친다
    criteria: list[CriterionResult] = []
    for name in CRITERIA:
        passed, comments = True, []
        if name in code_checks:
            code_ok, code_comment = code_checks[name]
            passed = passed and code_ok
            comments.append(f"[코드] {code_comment}")
        if name in judged:
            passed = passed and judged[name].passed
            comments.append(f"[LLM] {judged[name].comment}")
        else:  # 평가자가 빠뜨렸거나 호출이 실패한 항목은 통과로 치지 않는다
            passed = False
            comments.append("[LLM] 평가 결과 없음")
        criteria.append({"name": name, "passed": passed, "comment": " / ".join(comments)})

    failed = [c for c in criteria if not c["passed"]]
    passed = not failed and error is None
    review: ReportReview = {
        "passed": passed,
        "criteria": criteria,
        "feedback": "\n".join(f"- [{c['name']}] {c['comment']}" for c in failed),
        "error": error,
    }

    for c in criteria:
        print(f"  {'통과' if c['passed'] else '미달'}  {c['name']}: {c['comment']}")

    # 4. 다음 행동 결정 (라우팅 함수는 이 결과를 그대로 따른다)
    failed_names = ", ".join(c["name"] for c in failed)
    if passed:
        action, reason = "accept", f"{revision}번째 보고서가 모든 항목 통과"
    elif error:
        action, reason = "stop", f"{error} → 평가할 수 없어 현재 보고서로 종료"
    elif revision > MAX_REVISIONS:
        action, reason = "stop", f"재작성 상한({MAX_REVISIONS}회) 도달, 미달 항목: {failed_names}"
    else:
        action, reason = "revise", f"미달 항목: {failed_names}"
    print(f"  → {action}: {reason}")

    return {
        "review": review,
        "decisions": [
            {
                "node": "quality_eval",
                "action": action,
                "task_ids": [],
                "reason": reason,
            }
        ],
    }


def route_after_review(state: ReportState) -> str:
    """quality_eval이 남긴 마지막 결정에 따라 재작성(report_gen) 또는 종료."""
    return "report_gen" if state["decisions"][-1]["action"] == "revise" else END
