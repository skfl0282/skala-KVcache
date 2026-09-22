"""
기술 조사 에이전트 (RAG 적용)
담당: 장나리

설계서 A. Agent 정의 - "원문에서 기술 개요, 범위, 한계 추출"
설계서 D. Graph 설계 - tech_research_sw / tech_research_hw를 write한다.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from agents.prompt_utils import load_prompt
from graph.state import GraphState
from rag.pdf import (
    build_hw_comparison_retrieval_chain,
    build_sw_comparison_retrieval_chain,
    build_tech_retrieval_chain,
    format_docs,
)

MODEL_NAME = "gpt-5.6-luna"

_tech_chain = None
_sw_comparison_chain = None
_hw_comparison_chain = None


def _get_tech_chain():
    global _tech_chain
    if _tech_chain is None:
        _tech_chain = build_tech_retrieval_chain()
    return _tech_chain


def _get_sw_comparison_chain():
    """SW(DeepSeek-V2) 쪽에서 제외된 대조 기술(TurboQuant) 전용 RAG 체인."""
    global _sw_comparison_chain
    if _sw_comparison_chain is None:
        _sw_comparison_chain = build_sw_comparison_retrieval_chain()
    return _sw_comparison_chain


def _get_hw_comparison_chain():
    """HW(ITME) 쪽에서 제외된 대조 기술(InfiniGen) 전용 RAG 체인."""
    global _hw_comparison_chain
    if _hw_comparison_chain is None:
        _hw_comparison_chain = build_hw_comparison_retrieval_chain()
    return _hw_comparison_chain


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
    except Exception as e:  # RAG 체인 로딩 실패(PDF 누락, 인덱싱 오류 등) 시 빈 컨텍스트로 폴백
        print(f"[WARN] RAG 체인 호출 실패: {e}")
        sw_context = hw_context = ""

    # tech_sw/tech_hw 이름 자체는 대조 기술 원문에 등장하지 않으므로, 각 대조
    # 기술의 핵심 개념으로 직접 검색해야 논문 초록/핵심 아이디어 청크가 검색된다.
    try:
        sw_comparison_context = format_docs(
            _get_sw_comparison_chain().retriever.invoke("벡터 양자화 압축 기법의 핵심 아이디어")
        )
    except Exception as e:  # 대조 기술 원문(TurboQuant) 누락 시 비교 문장 생략
        print(f"[WARN] SW 대조 기술 RAG 체인 호출 실패: {e}")
        sw_comparison_context = ""

    try:
        hw_comparison_context = format_docs(
            _get_hw_comparison_chain().retriever.invoke("GPU/CPU 메모리 계층 간 동적 KV 캐시 관리 기법의 핵심 아이디어")
        )
    except Exception as e:  # 대조 기술 원문(InfiniGen) 누락 시 비교 문장 생략
        print(f"[WARN] HW 대조 기술 RAG 체인 호출 실패: {e}")
        hw_comparison_context = ""

    tech_research_sw = tech_research_chain.invoke(
        {
            "tech_name": tech_sw,
            "retrieved_chunks": sw_context,
            "comparison_chunks": sw_comparison_context,
        }
    )
    tech_research_hw = tech_research_chain.invoke(
        {
            "tech_name": tech_hw,
            "retrieved_chunks": hw_context,
            "comparison_chunks": hw_comparison_context,
        }
    )

    return {
        "tech_research_sw": tech_research_sw,
        "tech_research_hw": tech_research_hw,
        "messages": [("system", "[기술 조사 RAG] 완료")],
    }
