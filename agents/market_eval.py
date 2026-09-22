"""
시장 평가 에이전트
담당: __________ (TODO: 담당자 배정)

설계서 A. Agent 정의 - "시장성, 채택 현황 조사" / RAG 미적용, 웹검색만 사용.
설계서 C. 평가 관점 - 시장 규모, CAGR, 도입 기업 수/발표 빈도 등을 기준으로 평가.

흐름: 웹검색 1차 평가 -> 충분성 판정 -> 불충분하면 웹검색 1회 보완 후
재생성·재판정 -> 그래도 부족하면 재시도 없이 data_limited에 기록해
최종 보고서 "한계점"에 반영한다.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from agents.prompt_utils import load_prompt, web_references
from graph.state import GraphState

MODEL_NAME = "gpt-4.1-mini"

web_search_tool = TavilySearch(max_results=5)
market_eval_prompt = load_prompt("prompts/market_eval.txt")
llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
market_eval_chain = market_eval_prompt | llm | StrOutputParser()


class GradeSufficiency(BaseModel):
    """시장 평가 결과가 충분한지 평가하는 이진 점수"""

    binary_score: str = Field(description="충분하면 'yes', 부족하면 'no'")


_grader_llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
_sufficiency_grader = _grader_llm.with_structured_output(GradeSufficiency)
_sufficiency_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 시장 평가 결과가 충분한지 평가하는 채점자입니다. "
            "시장 규모·CAGR, 도입 기업 수·발표 빈도, 생태계 지지가 구체적인 근거로 "
            "서술되어 있으면 충분('yes'), 그렇지 않으면 부족('no')으로 판정하세요.",
        ),
        ("human", "시장 평가 결과:\n\n{eval_text}"),
    ]
)


def _is_sufficient(eval_text: str) -> bool:
    score = (_sufficiency_prompt | _sufficiency_grader).invoke({"eval_text": eval_text})
    return score.binary_score == "yes"


def market_eval(state: GraphState):
    """tech_sw / tech_hw의 시장성·채택 현황을 웹검색으로 조사해 market_eval을 write한다."""
    print("\n==== [MARKET EVAL] ====\n")
    tech_sw = state["tech_sw"]
    tech_hw = state["tech_hw"]

    search_results = web_search_tool.invoke(
        {"query": f"{tech_sw} vs {tech_hw} 시장 규모 CAGR 채택 사례"}
    )
    references = web_references(search_results)
    result = market_eval_chain.invoke(
        {"tech_sw": tech_sw, "tech_hw": tech_hw, "search_results": search_results}
    )

    print("\n==== [CHECK MARKET EVAL SUFFICIENCY] ====\n")
    if _is_sufficient(result):
        print("==== [DECISION: SUFFICIENT] ====")
        return {
            "market_eval": result,
            "data_limited": [],
            "references": references,
            "messages": [("system", "[시장 평가] 완료")],
        }

    print("==== [DECISION: INSUFFICIENT -> WEB SEARCH 1회 보완] ====")
    extra_results = web_search_tool.invoke(
        {"query": f"{tech_sw} vs {tech_hw} 시장 점유율 생태계 표준화 컨소시엄 도입 사례"}
    )
    references.extend(web_references(extra_results))
    result = market_eval_chain.invoke(
        {
            "tech_sw": tech_sw,
            "tech_hw": tech_hw,
            "search_results": f"{search_results}\n\n{extra_results}",
        }
    )

    print("\n==== [RE-CHECK MARKET EVAL SUFFICIENCY] ====\n")
    # data_limited는 operator.add로 누적되는 채널이므로, 이 노드는 자신이
    # 새로 추가하는 항목만 담은 리스트를 반환한다 (전체 리스트를 재구성해서
    # 반환하면 병렬로 함께 쓰는 domain_eval의 항목과 합쳐질 때 중복된다).
    if _is_sufficient(result):
        print("==== [DECISION: SUFFICIENT AFTER SUPPLEMENT] ====")
        new_data_limited = []
    else:
        print("==== [DECISION: STILL INSUFFICIENT -> RECORD data_limited] ====")
        new_data_limited = ["market_eval"]

    return {
        "market_eval": result,
        "data_limited": new_data_limited,
        "references": list(dict.fromkeys(references)),  # 순서 유지 + 중복 제거
        "messages": [("system", "[시장 평가] 완료")],
    }
