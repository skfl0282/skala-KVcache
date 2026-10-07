"""
시장성 평가 전문 워커 (market_eval)
웹검색을 통해 시장 규모, 성장성, 상용화 및 표준화 동향을 객관적으로 분석한다.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from agents.worker_utils import (
    WORKER_MODEL,
    load_worker_prompt,
    run_with_retry,
    search_web,
)
from graph.state import WorkerInput

worker_llm = init_chat_model(WORKER_MODEL, model_provider="openai", temperature=0)
market_chain = load_worker_prompt("prompts/market_eval.txt") | worker_llm | StrOutputParser()


def _market_once(state: WorkerInput):
    """웹검색만 사용. 검색 결과가 없으면 실패."""
    task = state["task"]
    web, refs, error = search_web(task["search_query"])
    if not web:
        return None, [], error
    content = market_chain.invoke(
        {
            "tech_sw": state["tech_sw"],
            "tech_hw": state["tech_hw"],
            "search_results": web,
            "instruction": task["instruction"],
        }
    )
    return content, refs, error


def market_eval(state: WorkerInput) -> dict:
    """그래프에 등록할 노드 함수 (Task.worker == 'market_eval')"""
    return run_with_retry(state, _market_once, "market_eval")
