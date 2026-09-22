"""
도메인 평가 에이전트
담당: 정서영

설계서 A. Agent 정의 - "적용 도메인별 적합성" / RAG 적용
    (데이터센터, 클라우드에서의 평가)
설계서 C. 평가 관점 - ① 처리 가능한 규모, ② 비용 절감 효과(TCO, 에너지 효율 등)

설계서 원본 mermaid에는 "도메인 평가 RAG" 다음에 별도의 "충분한가?" 분기
노드가 그려져 있다. 그 분기 + 웹검색 보완 루프를 이 노드 안에서 순차적으로 처리한다.
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

from agents.prompt_utils import load_prompt, rag_references, web_references
from graph.state import GraphState
from rag.pdf import build_tech_retrieval_chain, format_docs

MODEL_NAME = "gpt-5.6-luna"

_domain_chain = None


def _get_domain_chain():
    global _domain_chain
    if _domain_chain is None:
        _domain_chain = build_tech_retrieval_chain()  # 기술조사와 동일한 원문 풀 재사용
    return _domain_chain


domain_eval_prompt = load_prompt("prompts/domain_eval.txt")
llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
domain_eval_chain = domain_eval_prompt | llm | StrOutputParser()

web_search_tool = TavilySearch(max_results=3)


class GradeSufficiency(BaseModel):
    """도메인 평가 결과가 충분한지 평가하는 이진 점수"""

    binary_score: str = Field(description="충분하면 'yes', 부족하면 'no'")


_grader_llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
_sufficiency_grader = _grader_llm.with_structured_output(GradeSufficiency)
_sufficiency_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 도메인 평가 결과가 충분한지 평가하는 채점자입니다. "
            "처리 가능 규모와 비용 절감 효과가 모두 구체적인 수치/근거로 서술되어 있으면 "
            "충분('yes'), 그렇지 않으면 부족('no')으로 판정하세요.",
        ),
        ("human", "도메인 평가 결과:\n\n{eval_text}"),
    ]
)


def domain_eval(state: GraphState):
    """도메인 평가를 RAG+웹검색으로 수행하고, 결과가 부족하면 웹검색을 1회
    보완한 뒤 재생성한다. 보완 후에도 여전히 부족하면(재판정 기준) 그때만
    data_limited에 "domain_eval"을 기록하고 domain_eval을 write한다.
    """
    print("\n==== [DOMAIN EVAL RAG] ====\n")
    tech_sw = state["tech_sw"]
    tech_hw = state["tech_hw"]
    domain = state["domain"]
    # RAG 검색
    references: list[str] = []
    try:
        retriever = _get_domain_chain().retriever
        rag_docs = retriever.invoke(f"{tech_sw} {tech_hw} {domain}")
        context = format_docs(rag_docs)
        references.extend(rag_references(rag_docs))
    except Exception as e:
        print(f"[WARN] RAG 체인이 아직 준비되지 않았습니다: {e}")
        context = ""

    # 웹검색은 항상 수행
    search_results = web_search_tool.invoke(
        {
            "query": (
                f"{tech_sw} {tech_hw} {domain} "
                "scalability throughput memory cost saving TCO energy efficiency"
            )
        }
    )
    references.extend(web_references(search_results))

    # 1차 평가 생성
    result = domain_eval_chain.invoke(
        {
            "tech_sw": tech_sw,
            "tech_hw": tech_hw,
            "domain": domain,
            "retrieved_chunks": context,
            "search_results": search_results,
            "supplement_search_results": "",
        }
    )
    print("\n==== [CHECK DOMAIN EVAL SUFFICIENCY] ====\n")
    score = (_sufficiency_prompt | _sufficiency_grader).invoke({"eval_text": result})
    # data_limited는 operator.add로 누적되는 채널이므로, 이 노드는 자신이
    # 새로 추가하는 항목만 담은 리스트를 반환한다 (전체 리스트를 재구성해서
    # 반환하면 병렬로 함께 쓰는 market_eval의 항목과 합쳐질 때 중복된다).
    new_data_limited: list[str] = []
    if score.binary_score != "yes":
        print("==== [DECISION: INSUFFICIENT -> WEB SEARCH SUPPLEMENT] ====")
        supplement_search_results = web_search_tool.invoke(
            {"query": f"{tech_sw} {tech_hw} {domain} 비용 절감 처리 규모"}
        )
        references.extend(web_references(supplement_search_results))
        result = domain_eval_chain.invoke(
            {
                "tech_sw": tech_sw,
                "tech_hw": tech_hw,
                "domain": domain,
                "retrieved_chunks": context,
                "search_results": search_results,
                "supplement_search_results": supplement_search_results,
            }
        )

        print("\n==== [RE-CHECK DOMAIN EVAL SUFFICIENCY AFTER SUPPLEMENT] ====\n")
        score = (_sufficiency_prompt | _sufficiency_grader).invoke({"eval_text": result})
        if score.binary_score != "yes":
            print("==== [DECISION: STILL INSUFFICIENT AFTER SUPPLEMENT] ====")
            new_data_limited = ["domain_eval"]
        else:
            print("==== [DECISION: SUFFICIENT AFTER SUPPLEMENT] ====")
    else:
        print("==== [DECISION: SUFFICIENT] ====")

    return {
        "domain_eval": result,
        "data_limited": new_data_limited,
        "references": list(dict.fromkeys(references)),  # 순서 유지 + 중복 제거
        "messages": [("system", "[도메인 평가 RAG] 완료")],
    }
