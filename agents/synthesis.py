"""
평가 종합 에이전트
담당: 박종문

설계서 A. Agent 정의 - "관점별 의견 종합 및 비교" / RAG 미적용
설계서 C. 평가 관점 - 종합 의견: 우열 판정 없이 관점 간 일치/상충 지점을
                      병렬적으로 서술.

market_eval / stakeholder_eval / domain_eval 3개 노드가 모두 끝난 뒤
(fan-in) 실행된다.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser

from agents.prompt_utils import load_prompt
from graph.state import GraphState

MODEL_NAME = "gpt-5.6-terra"

synthesis_prompt = load_prompt("prompts/synthesis.txt")
llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
synthesis_chain = synthesis_prompt | llm | StrOutputParser()


def synthesis(state: GraphState):
    """market_eval, stakeholder_eval, domain_eval을 읽어 관점 간
    일치/상충 지점을 중심으로 synthesis를 write한다.
    """
    print("\n==== [SYNTHESIS] ====\n")

    result = synthesis_chain.invoke(
        {
            "tech_research_sw": state.get("tech_research_sw", ""),
            "tech_research_hw": state.get("tech_research_hw", ""),
            "market_eval": state.get("market_eval", ""),
            "stakeholder_eval": state.get("stakeholder_eval", ""),
            "domain_eval": state.get("domain_eval", ""),
        }
    )

    return {
        "synthesis": result,
        "messages": [("system", "[평가 종합] 완료")],
    }
