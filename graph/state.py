"""
설계산출물 D. Graph 설계 - State 테이블을 코드로 옮긴 모듈.
"""

import operator
from typing import Annotated, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    tech_sw: Annotated[str, "SW 진영 선정 기술명 (입력)"]
    tech_hw: Annotated[str, "HW 진영 선정 기술명 (입력)"]
    domain: Annotated[str, "선택 도메인 (입력)"]

    tech_research_sw: Annotated[str, "SW 기술 조사 결과 (C, RAG)"]
    tech_research_hw: Annotated[str, "HW 기술 조사 결과 (C, RAG)"]

    market_eval: Annotated[str, "시장 평가 결과 (D, 웹검색) - Fan-out 전용 키"]
    stakeholder_eval: Annotated[str, "이해관계자 평가 결과 (E, RAG 및 웹검색) - Fan-out 전용 키"]
    domain_eval: Annotated[str, "도메인 평가 결과 (F, RAG 및 웹검색) - Fan-out 전용 키"]

    synthesis: Annotated[str, "평가 종합 결과 (G)"]
    report: Annotated[str, "최종 보고서 (H)"]

    data_limited: Annotated[List[str], "웹검색 보완 후에도 자료가 부족했던 노드 이름 기록"]
    # RAG/웹검색 노드가 실제로 인용한 출처 (report_gen의 REFERENCE 절 재료).
    # 각 노드는 자신이 실제로 참고한 출처만 append해서 반환하고, fan-out 노드
    # 간에는 operator.add로 리스트가 누적된다.
    references: Annotated[list, operator.add]
    messages: Annotated[list, add_messages]
