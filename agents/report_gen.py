"""
보고서 생성 에이전트 (Report Generator)
종합 결과와 각 관점별 조사 내용을 연결하여 최종 기술 평가 보고서를 작성하며,
품질 평가 피드백이 전달되면 지적 사항을 반영해 보고서를 개정(Revision)한다.
"""

import os
from pathlib import Path
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agents.report_fit import fit_report_to_pages
from agents.synthesis import collect_limitations, collect_references
from agents.worker_utils import PLANNER_MODEL
from graph.state import GraphState, ReportState

REPORT_PATH = "outputs/report_orchestrator_workers.md"

REVISION_SUFFIX = """

[재작성 지침]
- 중립성: 특정 기술을 추천하거나 일방적인 우열 판정(예: '더 우수하다', '더 적합하다', '승자', '~을 권장한다')을
  내리지 마세요. 두 기술의 장단점과 성격을 객관적으로 병렬 서술하세요.
- 편향 통제: 두 기술 모두 강점과 한계를 함께 쓰고, 한 출처에만 기대어 결론을 내리지 마세요.
  두 기술의 서술 분량을 억지로 맞출 필요는 없습니다. 자료가 있는 만큼 쓰고, 없는 부분은 "근거 부족"으로 두세요.
- 관점 커버리지: "4. 관점별 평가"는 기술성숙도 관점, 시장성 관점, 이해관계자 관점, 도메인 적용 관점,
  종합의견 다섯 소제목을 이름 그대로 모두 갖춰야 합니다. 조사 결과가 없는 관점은 빼지 말고 "근거 부족"이라고 쓰세요.

[이전 보고서]
{previous_report}

[품질 평가 피드백]
{feedback}
위 피드백이 "없음"이 아니면, 이전 보고서에서 지적된 항목을 모두 고쳐 보고서 전체를 다시 작성하세요.
지적되지 않은 부분은 유지하고, 위 [출력 형식]과 근거 사용 규칙은 그대로 지키세요.
"""

report_prompt = PromptTemplate.from_template(
    Path("prompts/report_gen.txt").read_text(encoding="utf-8") + REVISION_SUFFIX
)
report_llm = init_chat_model(PLANNER_MODEL, model_provider="openai", temperature=0)
report_chain = report_prompt | report_llm | StrOutputParser()


def results_of(state: GraphState, worker: str, target: Optional[str] = None) -> str:
    """한 워커(필요하면 한 기술)가 만든 결과들을 묶어 프롬프트의 고정 자리에 넣을 문자열로 만든다."""
    parts = [
        f"#### {task['perspective']}\n{state['results'][task_id]['content']}"
        for task_id, task in state["tasks"].items()
        if task["worker"] == worker
        and task["status"] == "done"
        and (target is None or task.get("target") == target)
    ]
    return "\n\n".join(parts) if parts else "조사 결과 없음"


def report_inputs(state: GraphState) -> dict:
    """프롬프트의 모든 채움 자리를 State에서 읽어 사전으로 만든다."""
    return {
        "tech_sw": state["tech_sw"],
        "tech_hw": state["tech_hw"],
        "domain": state["domain"],
        "tech_sw_research": results_of(state, "tech_research", "sw"),
        "tech_hw_research": results_of(state, "tech_research", "hw"),
        "market_eval": results_of(state, "market_eval"),
        "stakeholder_eval": results_of(state, "stakeholder_eval"),
        "domain_eval": results_of(state, "domain_eval"),
        "synthesis": state.get("synthesis") or "종합 결과 없음",
        "limitations": collect_limitations(state),
        "references": collect_references(state),
    }


def report_gen(state: ReportState) -> dict:
    """워커들의 조사 결과와 Synthesizer의 종합을 엮어 마크다운 보고서를 작성한다.
    품질 검사 피드백이 있으면 반영해 개정(Revision)한다."""
    revision = state.get("revision", 0) + 1
    review = state.get("review") or {}
    print(f"\n==== [REPORT GEN] {revision}번째 작성 ====")

    report = report_chain.invoke(
        {
            **report_inputs(state),
            "previous_report": state.get("report") or "없음",
            "feedback": review.get("feedback") or "없음",
        }
    )

    os.makedirs("outputs", exist_ok=True)
    # PDF가 10쪽을 넘으면 압축한 최종본으로 바꾼다 (md / PDF / State가 같은 내용이 되도록 PDF를 먼저 맞춤)
    report, pages = fit_report_to_pages(report, "outputs/report.pdf")
    print(f"outputs/report.pdf 저장 완료 ({pages}쪽)")

    # 기존 report.md와 오케스트레이터 전용 경로 둘 다 기록
    Path("outputs/report.md").write_text(report, encoding="utf-8")
    Path(REPORT_PATH).write_text(report, encoding="utf-8")
    print(f"{REPORT_PATH} 및 outputs/report.md 저장 완료")

    return {
        "report": report,
        "revision": revision,
    }
