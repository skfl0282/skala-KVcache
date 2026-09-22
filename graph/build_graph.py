"""
설계산출물 D. Graph 흐름 설계(mermaid)를 LangGraph StateGraph로 옮긴 모듈.

흐름 요약
---------
START
  -> select_technology                (기술 선정)
  -> tech_research                    (기술 조사 RAG)
  -> [조건부, route_after_tech_research]
       sufficient   -> market_eval, stakeholder_eval, domain_eval 로 바로 fan-out
       insufficient -> web_search_supplement_tech
                          -> market_eval, stakeholder_eval, domain_eval 로 fan-out
  -> market_eval / stakeholder_eval / domain_eval   (병렬 3개 노드)
       (domain_eval 노드 내부에 자체 충분성 체크 + 웹검색 보완 루프 포함.
        이유는 아래 참고)
  -> synthesis   (["market_eval","stakeholder_eval","domain_eval"] 조인 -> fan-in)
  -> report_gen
  -> END

설계서 원안에서는 "도메인 평가 RAG" 뒤에도 별도의 "충분한가?" 분기 노드가
그래프 레벨로 그려져 있다. 다만 그 분기를 market/stakeholder 브랜치와
동일한 그래프 depth에서 별도 노드로 빼면 세 브랜치의 슈퍼스텝 수가
달라져 synthesis 노드의 fan-in(list-edge join) 타이밍이 어긋날 수 있다.
그래서 이 스켈레톤에서는 "충분한가? + 웹검색 보완"을
agents/domain_eval.py 노드 내부 로직(순차 호출)으로 접어 넣었다.
팀 논의 후 필요하면 domain 담당자가 별도 노드로 다시 분리해도 된다.
"""

from typing import Sequence

from langgraph.graph import END, START, StateGraph

from agents.domain_eval import domain_eval
from agents.market_eval import market_eval
from agents.report_gen import report_gen
from agents.stakeholder_eval import stakeholder_eval
from agents.synthesis import synthesis
from agents.tech_research import (
    route_after_tech_research,
    tech_research,
    web_search_supplement_tech,
)
from agents.tech_selector import select_technology
from graph.state import GraphState

EVAL_NODES = ["market_eval", "stakeholder_eval", "domain_eval"]


def route_tech_research_fanout(state: GraphState) -> Sequence[str]:
    """route_after_tech_research(sufficient/insufficient)의 결과를 실제 다음
    노드 목록으로 변환하는 fan-out 라우팅 함수.
    """
    if route_after_tech_research(state) == "sufficient":
        return EVAL_NODES
    return ["web_search_supplement_tech"]


def build_graph():
    """StateGraph를 조립해서 컴파일된 그래프를 반환한다.

    담당: 그래프 배선 담당자(팀 전체 합의 필요). 각 노드 함수의 내부 구현은
    agents/*.py 에서 담당자별로 채운다.
    """
    workflow = StateGraph(GraphState)

    # --- 노드 등록 ---
    workflow.add_node("select_technology", select_technology)
    workflow.add_node("tech_research", tech_research)
    workflow.add_node("web_search_supplement_tech", web_search_supplement_tech)
    workflow.add_node("market_eval", market_eval)
    workflow.add_node("stakeholder_eval", stakeholder_eval)
    workflow.add_node("domain_eval", domain_eval)
    workflow.add_node("synthesis", synthesis)
    workflow.add_node("report_gen", report_gen)

    # --- 선형 구간 ---
    workflow.add_edge(START, "select_technology")
    workflow.add_edge("select_technology", "tech_research")

    # --- 기술 조사 결과 충분성에 따라 바로 fan-out 하거나 웹검색 보완 후 fan-out ---
    workflow.add_conditional_edges(
        "tech_research",
        route_tech_research_fanout,
        ["web_search_supplement_tech", *EVAL_NODES],
    )
    workflow.add_edge("web_search_supplement_tech", "market_eval")
    workflow.add_edge("web_search_supplement_tech", "stakeholder_eval")
    workflow.add_edge("web_search_supplement_tech", "domain_eval")

    # --- Fan-in: 3개 노드 모두 끝나야 synthesis 실행 ---
    workflow.add_edge(EVAL_NODES, "synthesis")

    # --- 마무리 ---
    workflow.add_edge("synthesis", "report_gen")
    workflow.add_edge("report_gen", END)

    app = workflow.compile()
    return app
