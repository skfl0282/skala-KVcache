"""
LangGraph 기반 Orchestrator-Workers 에이전트 그래프 조립 모듈.

흐름 구조
---------
START
  -> orchestrator           (동적 플래닝: N개의 전문 Task 계획 수립)
  -> [Send() Fan-out]       (각 Task를 지정된 전문 워커로 병렬 디스패치)
       - tech_research (SW/HW 기술성숙도 RAG 조사)
       - market_eval   (시장성 웹검색)
       - stakeholder_eval (이해관계자 RAG + 웹검색)
       - domain_eval   (도메인 TCO RAG + 웹검색)
  -> synthesizer (Fan-in)   (Reducer로 병합된 결과 집계 및 상충 분석)
  -> report_gen             (최종 평가 보고서 작성 및 개정)
  -> quality_eval           (코드 규칙 + LLM Judge 4대 기준 심사)
       ├── [미달 & revision <= max] -> report_gen (피드백 반영 루프)
       └── [통과 OR revision > max] -> END (종료 보장)
"""

from typing import List
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.domain_eval import domain_eval
from agents.market_eval import market_eval
from agents.orchestrator import orchestrator
from agents.quality_eval import quality_eval, route_after_review
from agents.report_gen import report_gen
from agents.stakeholder_eval import stakeholder_eval
from agents.synthesis import synthesizer
from agents.tech_research import tech_research
from graph.state import GraphState, ReportState, WORKERS

WORKER_NODES = {
    "tech_research": tech_research,
    "market_eval": market_eval,
    "stakeholder_eval": stakeholder_eval,
    "domain_eval": domain_eval,
}


def assign_workers(state: GraphState) -> List[Send]:
    """Orchestrator가 수립한 task 목록을 기반으로 담당 워커 노드로 Dynamic Fan-out을 수행한다."""
    tasks = state.get("tasks", {})
    return [
        Send(
            task["worker"],
            {
                "task": task,
                "tech_sw": state["tech_sw"],
                "tech_hw": state["tech_hw"],
                "domain": state["domain"],
            },
        )
        for task in tasks.values()
    ]


def build_graph(checkpointer=None):
    """Orchestrator-Workers 전문 워커 그래프를 조립하여 컴파일된 그래프를 반환한다."""
    builder = StateGraph(ReportState)

    # 1. 노드 등록
    builder.add_node("orchestrator", orchestrator)
    for name, node in WORKER_NODES.items():
        builder.add_node(name, node)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("report_gen", report_gen)
    builder.add_node("quality_eval", quality_eval)

    # 2. 시작 및 동적 Fan-out 배선
    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", assign_workers, WORKERS)

    # 3. 워커 완료 후 synthesizer 집계 (Fan-in)
    for name in WORKER_NODES:
        builder.add_edge(name, "synthesizer")

    # 4. 종합 -> 보고서 작성 -> 품질 평가
    builder.add_edge("synthesizer", "report_gen")
    builder.add_edge("report_gen", "quality_eval")

    # 5. 품질 평가 조건부 루프 (Pass/Max -> END, Fail -> report_gen 재작성)
    builder.add_conditional_edges("quality_eval", route_after_review, ["report_gen", END])

    if checkpointer is None:
        checkpointer = MemorySaver()

    app = builder.compile(checkpointer=checkpointer)
    return app
