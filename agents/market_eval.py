"""
시장 평가 에이전트
담당: __________ (TODO: 담당자 배정)

설계서 A. Agent 정의 - "시장성, 채택 현황 조사" / RAG 미적용, 웹검색만 사용.
설계서 C. 평가 관점 - 시장 규모, CAGR, 도입 기업 수/발표 빈도 등을 기준으로 평가.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

from agents.prompt_utils import load_prompt
from graph.state import GraphState

MODEL_NAME = "gpt-4.1-mini"

web_search_tool = TavilySearch(max_results=5)
market_eval_prompt = load_prompt("prompts/market_eval.txt")
llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
market_eval_chain = market_eval_prompt | llm | StrOutputParser()


def market_eval(state: GraphState):
    """tech_sw / tech_hw의 시장성·채택 현황을 웹검색으로 조사해 market_eval을 write한다."""
    print("\n==== [MARKET EVAL] ====\n")
    tech_sw = state["tech_sw"]
    tech_hw = state["tech_hw"]

    search_results = web_search_tool.invoke({"query": f"{tech_sw} vs {tech_hw} 시장 규모 CAGR 채택 사례"})

    result = market_eval_chain.invoke(
        {"tech_sw": tech_sw, "tech_hw": tech_hw, "search_results": search_results}
    )

    return {
        "market_eval": result,
        "messages": [("system", "[시장 평가] 완료")],
    }
