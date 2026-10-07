"""
이해관계자 평가 에이전트
담당: 정서영

설계서 A. Agent 정의 - "관계자별 반응 조사" / RAG 적용.
설계서 B. 설계 - RAG 적용 대상에 기술조사/도메인 평가 에이전트와 함께 포함.
설계서 C. 평가 관점 - 경쟁사 포지셔닝, 도입 기업/개발자 반응, 투자업계 시각.
설계서 D. Graph 흐름 설계(mermaid) - "이해관계자 평가: RAG + 웹검색 항상 병행".
"""

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from agents.prompt_utils import load_prompt, rag_references, web_references, web_search
from graph.state import GraphState
from rag.pdf import format_docs, get_tech_retrieval_chain

MODEL_NAME = "gpt-5.6-luna"

web_search_tool = TavilySearch(max_results=5)
stakeholder_eval_prompt = load_prompt("prompts/stakeholder_eval.txt")
llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
stakeholder_eval_chain = stakeholder_eval_prompt | llm | StrOutputParser()


class GradeSufficiency(BaseModel):
    """이해관계자 평가 결과가 충분한지 평가하는 이진 점수"""

    binary_score: Literal["yes", "no"] = Field(description="충분하면 'yes', 부족하면 'no'")


_grader_llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
_sufficiency_grader = _grader_llm.with_structured_output(GradeSufficiency)
_sufficiency_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 이해관계자 평가 결과가 충분한지 평가하는 채점자입니다. "
            "① 경쟁사의 유사 기술 발표 여부·포지셔닝, ② 실제 도입 사례 수, "
            "③ 업계 매체 보도량이 모두 구체적인 근거로 서술되어 있으면 "
            "충분('yes'), 그렇지 않으면 부족('no')으로 판정하세요.",
        ),
        ("human", "이해관계자 평가 결과:\n\n{eval_text}"),
    ]
)


def stakeholder_eval(state: GraphState):
    """경쟁사/개발자/투자업계 반응을 RAG(원문 자료)와 웹검색을 항상 병행하여
    조사하고, 결과가 부족하면 웹검색을 1회 보완한 뒤 재생성·재판정한다.
    보완 후에도 여전히 부족하면 그때만 data_limited에 "stakeholder_eval"을
    기록하고 stakeholder_eval을 write한다.
    """
    print("\n==== [STAKEHOLDER EVAL RAG + WEB SEARCH] ====\n")
    tech_sw = state["tech_sw"]
    tech_hw = state["tech_hw"]

    # data_limited는 operator.add로 누적되는 채널이므로, 이 노드는 자신이
    # 새로 추가하는 항목만 담은 리스트를 반환한다.
    new_data_limited: list[str] = []
    references: list[str] = []
    try:
        # 기술조사/도메인 평가 에이전트와 동일한 원문 RAG 체인을 공유한다.
        retriever = get_tech_retrieval_chain().retriever
        docs = retriever.invoke(f"{tech_sw} {tech_hw} 경쟁 기술 포지셔닝 한계")
        retrieved_chunks = format_docs(docs)
        references.extend(rag_references(docs))
    except Exception as e:
        print(f"[WARN] RAG 체인 호출 실패: {e}")
        retrieved_chunks = ""
        new_data_limited.append("stakeholder_eval (RAG 검색 실패)")

    search_results = web_search(
        web_search_tool, f"{tech_sw} vs {tech_hw} 경쟁사 반응 개발자 반응 투자 업계 시각"
    )
    references.extend(web_references(search_results))

    result = stakeholder_eval_chain.invoke(
        {
            "tech_sw": tech_sw,
            "tech_hw": tech_hw,
            "retrieved_chunks": retrieved_chunks,
            "search_results": search_results,
            "supplement_search_results": "",
        }
    )

    print("\n==== [CHECK STAKEHOLDER EVAL SUFFICIENCY] ====\n")
    score = (_sufficiency_prompt | _sufficiency_grader).invoke({"eval_text": result})
    if score.binary_score != "yes":
        print("==== [DECISION: INSUFFICIENT -> WEB SEARCH SUPPLEMENT] ====")
        supplement_search_results = web_search(
            web_search_tool, f"{tech_sw} {tech_hw} 경쟁사 대응 도입 사례 투자 동향"
        )
        references.extend(web_references(supplement_search_results))
        result = stakeholder_eval_chain.invoke(
            {
                "tech_sw": tech_sw,
                "tech_hw": tech_hw,
                "retrieved_chunks": retrieved_chunks,
                "search_results": search_results,
                "supplement_search_results": supplement_search_results,
            }
        )

        print("\n==== [RE-CHECK STAKEHOLDER EVAL SUFFICIENCY AFTER SUPPLEMENT] ====\n")
        score = (_sufficiency_prompt | _sufficiency_grader).invoke({"eval_text": result})
        if score.binary_score != "yes":
            print("==== [DECISION: STILL INSUFFICIENT AFTER SUPPLEMENT] ====")
            new_data_limited.append("stakeholder_eval")
        else:
            print("==== [DECISION: SUFFICIENT AFTER SUPPLEMENT] ====")
    else:
        print("==== [DECISION: SUFFICIENT] ====")

    return {
        "stakeholder_eval": result,
        "data_limited": new_data_limited,
        "references": list(dict.fromkeys(references)),  # 순서 유지 + 중복 제거
        "messages": [("system", "[이해관계자 평가 RAG + 웹검색] 완료")],
    }
