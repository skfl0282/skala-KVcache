"""
도메인 적용 평가 전문 워커 (domain_eval)
원문 RAG와 웹검색을 결합하여 대상 도메인 환경에서의 처리 규모, 비용(TCO), 에너지 효율, 제약을 분석한다.
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
domain_chain = load_worker_prompt("prompts/domain_eval.txt") | worker_llm | StrOutputParser()


def _domain_once(state: WorkerInput):
    """원문 RAG + 웹검색을 함께 사용하는 흐름. 둘 다 없으면 실패."""
    task = state["task"]
    chunks, rag_refs, rag_error = search_rag(
        get_tech_retrieval_chain, task["search_query"]
    )
    web, web_refs, web_error = search_web(task["search_query"])
    error = join_errors(rag_error, web_error)
    if not chunks and not web:
        return None, [], error

    content = domain_chain.invoke(
        {
            "tech_sw": state["tech_sw"],
            "tech_hw": state["tech_hw"],
            "domain": state["domain"],
            "retrieved_chunks": chunks,
            "search_results": web,
            "supplement_search_results": "",
            "instruction": task["instruction"],
        }
    )
    return content, rag_refs + web_refs, error


def domain_eval(state: WorkerInput) -> dict:
    """그래프에 등록할 노드 함수 (Task.worker == 'domain_eval')"""
    return run_with_retry(state, _domain_once, "domain_eval")
