"""
보고서 생성 에이전트
담당: 정서영

설계서 A. Agent 정의 - "단계별 내용을 연결하여 보고서 생성" / RAG 미적용
설계서 E. 평가 보고서 - 목차(초안):
    SUMMARY (1/2페이지 이내)
    1. 분석 배경 / 2. 기술 선정 / 3. 기술 개요 / 4. 관점별 평가
    5. 시사점 / 6. 한계점 / REFERENCE
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from agents.prompt_utils import load_prompt
from graph.state import GraphState

MODEL_NAME = "gpt-5.6-terra"

report_gen_prompt = load_prompt("prompts/report_gen.txt")
llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
report_gen_chain = report_gen_prompt | llm | StrOutputParser()


def report_gen(state: GraphState):
    """synthesis와 이전 단계 결과들을 연결해 최종 보고서(report)를 write하고
    outputs/에 파일로 저장한다.
    """
    print("\n==== [REPORT GEN] ====\n")

    data_limited = state.get("data_limited", [])
    references = state.get("references", [])

    report = report_gen_chain.invoke(
        {
            "tech_research_sw": state.get("tech_research_sw", ""),
            "tech_research_hw": state.get("tech_research_hw", ""),
            "synthesis": state.get("synthesis", ""),
            "data_limited": ", ".join(data_limited) if data_limited else "없음",
            "references": "\n".join(f"- {ref}" for ref in references) if references else "없음",
        }
    )

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("outputs/report.md 저장 완료")

    return {
        "report": report,
        "messages": [("system", "[보고서 생성] outputs/report.md 저장 완료")],
    }
