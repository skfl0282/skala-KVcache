"""
기술 조사 전문 워커 (tech_research)
원문 논문 RAG 및 대조 기술 RAG를 수행하여 기술성숙도(TRL 등급 및 근거)를 분석한다.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from agents.worker_utils import (
    WORKER_MODEL,
    join_errors,
    load_worker_prompt,
    run_with_retry,
    search_rag,
)
from graph.state import WorkerInput
from rag.pdf import (
    get_hw_comparison_retrieval_chain,
    get_sw_comparison_retrieval_chain,
    get_tech_retrieval_chain,
)

worker_llm = init_chat_model(WORKER_MODEL, model_provider="openai", temperature=0)
tech_research_chain = (
    load_worker_prompt("prompts/tech_research.txt") | worker_llm | StrOutputParser()
)


def _tech_research_once(state: WorkerInput):
    """원문 RAG + 대조 기술 RAG. 원문이 없으면 실패."""
    task = state["task"]
    is_sw = task.get("target") != "hw"
    tech_name = state["tech_sw"] if is_sw else state["tech_hw"]

    chunks, refs, error = search_rag(get_tech_retrieval_chain, tech_name)
    if not chunks:
        return None, [], error

    get_comparison = (
        get_sw_comparison_retrieval_chain if is_sw else get_hw_comparison_retrieval_chain
    )
    comparison, comparison_refs, comparison_error = search_rag(
        get_comparison, "핵심 아이디어와 구조적 접근 방식", label="대조 기술 RAG"
    )
    content = tech_research_chain.invoke(
        {
            "tech_name": tech_name,
            "retrieved_chunks": chunks,
            "comparison_chunks": comparison,
            "instruction": task["instruction"],
        }
    )
    return content, refs + comparison_refs, join_errors(error, comparison_error)


def tech_research(state: WorkerInput) -> dict:
    """그래프에 등록할 노드 함수 (Task.worker == 'tech_research')"""
    return run_with_retry(state, _tech_research_once, "tech_research")
