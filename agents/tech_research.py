"""
기술 조사 에이전트 (RAG 적용)
담당: __________ (TODO: 담당자 배정)

설계서 A. Agent 정의 - "원문에서 기술 개요, 범위, 한계 추출"
설계서 D. Graph 설계 - tech_research_sw / tech_research_hw를 write하고,
충분한지 판단해서 부족하면 웹검색으로 보완한다.
"""

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from agents.prompt_utils import load_prompt
from graph.state import GraphState
from rag.pdf import build_tech_retrieval_chain, format_docs

MODEL_NAME = "gpt-4.1-mini"

# TODO: rag.pdf.build_tech_retrieval_chain() 은 원문 PDF(data/raw/)와 BGE-M3 임베딩
# 준비가 끝난 뒤 실제로 호출 가능. 그 전까지는 tech_research()에서 예외를 잡아 STUB으로 대체.
_tech_chain = None


def _get_tech_chain():
    global _tech_chain
    if _tech_chain is None:
        _tech_chain = build_tech_retrieval_chain()
    return _tech_chain


tech_research_prompt = load_prompt("prompts/tech_research.txt")
llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
tech_research_chain = tech_research_prompt | llm | StrOutputParser()


def tech_research(state: GraphState):
    """DeepSeek-V2(MLA), ITME 원문 PDF를 RAG로 검색해 기술 개요/범위/한계를
    추출하고 tech_research_sw, tech_research_hw를 write한다.
    """
    print("\n==== [TECH RESEARCH RAG] ====\n")
    tech_sw = state["tech_sw"]
    tech_hw = state["tech_hw"]

    try:
        retriever = _get_tech_chain().retriever
        sw_context = format_docs(retriever.invoke(tech_sw))
        hw_context = format_docs(retriever.invoke(tech_hw))
    except Exception as e:  # TODO: data/raw/ 원문 PDF 준비 전까지의 임시 예외처리
        print(f"[WARN] RAG 체인이 아직 준비되지 않았습니다: {e}")
        sw_context = hw_context = ""

    tech_research_sw = tech_research_chain.invoke(
        {"tech_name": tech_sw, "retrieved_chunks": sw_context}
    )
    tech_research_hw = tech_research_chain.invoke(
        {"tech_name": tech_hw, "retrieved_chunks": hw_context}
    )

    return {
        "tech_research_sw": tech_research_sw,
        "tech_research_hw": tech_research_hw,
        "messages": [("system", "[기술 조사 RAG] 완료")],
    }


# --- 충분성 체크 (CorrectiveRAG의 GradeDocuments 그레이더 패턴) ---
class GradeSufficiency(BaseModel):
    """조사 결과가 보고서 작성에 충분한지 평가하는 이진 점수"""

    binary_score: str = Field(description="충분하면 'yes', 부족하면 'no'")


_grader_llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
_sufficiency_grader = _grader_llm.with_structured_output(GradeSufficiency)

_sufficiency_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 기술 조사 결과가 보고서 작성에 충분한지 평가하는 채점자입니다. "
            "개요/범위/한계가 모두 구체적으로 서술되어 있으면 충분('yes'), "
            "그렇지 않으면 부족('no')으로 판정하세요.",
        ),
        ("human", "기술 조사 결과:\n\n{research_text}"),
    ]
)


def route_after_tech_research(state: GraphState) -> Literal["sufficient", "insufficient"]:
    """tech_research_sw/hw가 충분한지 판단하는 조건부 엣지 함수."""
    print("\n==== [CHECK TECH RESEARCH SUFFICIENCY] ====\n")
    chain = _sufficiency_prompt | _sufficiency_grader

    sw_score = chain.invoke({"research_text": state.get("tech_research_sw", "")})
    hw_score = chain.invoke({"research_text": state.get("tech_research_hw", "")})

    if sw_score.binary_score == "yes" and hw_score.binary_score == "yes":
        print("==== [DECISION: SUFFICIENT] ====")
        return "sufficient"
    print("==== [DECISION: INSUFFICIENT -> WEB SEARCH SUPPLEMENT] ====")
    return "insufficient"


# --- 웹검색 보완 ---
web_search_tool = TavilySearch(max_results=3)


def web_search_supplement_tech(state: GraphState):
    """기술 조사 결과가 부족할 때 웹검색으로 보완하고 data_limited에 기록한다."""
    print("\n==== [WEB SEARCH SUPPLEMENT: TECH RESEARCH] ====\n")

    sw_results = web_search_tool.invoke({"query": state["tech_sw"]})
    hw_results = web_search_tool.invoke({"query": state["tech_hw"]})

    data_limited = list(state.get("data_limited", []))
    data_limited.append("tech_research")

    return {
        "tech_research_sw": state.get("tech_research_sw", "") + f"\n[웹검색 보완]\n{sw_results}",
        "tech_research_hw": state.get("tech_research_hw", "") + f"\n[웹검색 보완]\n{hw_results}",
        "data_limited": data_limited,
        "messages": [("system", "[기술 조사] 웹검색 보완 수행")],
    }
