"""
이해관계자 평가 전문 워커 (stakeholder_eval)
원문 RAG와 웹검색을 결합하여 경쟁 기술 진영, 도입 기업/개발자, 투자 업계 시각을 분석한다.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from agents.worker_utils import (
    WORKER_MODEL,
    join_errors,
    load_worker_prompt,
    run_with_retry,
    search_rag,
    search_web,
)
from graph.state import WorkerInput
from rag.pdf import get_tech_retrieval_chain

worker_llm = init_chat_model(WORKER_MODEL, model_provider="openai", temperature=0)
stakeholder_chain = (
    load_worker_prompt("prompts/stakeholder_eval.txt") | worker_llm | StrOutputParser()
)


def _stakeholder_once(state: WorkerInput):
    """원문 RAG + 웹검색을 함께 사용하는 흐름. 둘 다 없으면 실패."""
    task = state["task"]
    chunks, rag_refs, rag_error = search_rag(
        get_tech_retrieval_chain, task["search_query"]
    )
    web, web_refs, web_error = search_web(task["search_query"])
    error = join_errors(rag_error, web_error)
    if not chunks and not web:
        return None, [], error

    content = stakeholder_chain.invoke(
        {
            "tech_sw": state["tech_sw"],
            "tech_hw": state["tech_hw"],
            "retrieved_chunks": chunks,
            "search_results": web,
            "supplement_search_results": "",
            "instruction": task["instruction"],
        }
    )
    return content, rag_refs + web_refs, error


def stakeholder_eval(state: WorkerInput) -> dict:
    """그래프에 등록할 노드 함수 (Task.worker == 'stakeholder_eval')"""
    return run_with_retry(state, _stakeholder_once, "stakeholder_eval")
