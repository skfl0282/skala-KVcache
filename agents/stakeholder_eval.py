"""
이해관계자 평가 에이전트
담당: __________ (TODO: 담당자 배정)

설계서 A. Agent 정의 - "관계자별 반응 조사" / RAG 적용.
설계서 B. 설계 - RAG 적용 대상에 기술조사/도메인 평가 에이전트와 함께 포함.
설계서 C. 평가 관점 - 경쟁사 포지셔닝, 도입 기업/개발자 반응, 투자업계 시각.
설계서 D. Graph 흐름 설계(mermaid) - "이해관계자 평가: RAG + 웹검색 항상 병행".
"""

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch

from agents.prompt_utils import load_prompt, rag_references, web_references
from graph.state import GraphState
from rag.pdf import build_tech_retrieval_chain, format_docs

MODEL_NAME = "gpt-4.1-mini"

# 기술조사/도메인 평가 에이전트와 동일한 원문(DeepSeek-V2, ITME) RAG 체인을 공유한다.
_stakeholder_chain = None


def _get_stakeholder_chain():
    global _stakeholder_chain
    if _stakeholder_chain is None:
        _stakeholder_chain = build_tech_retrieval_chain()
    return _stakeholder_chain


web_search_tool = TavilySearch(max_results=5)
stakeholder_eval_prompt = load_prompt("prompts/stakeholder_eval.txt")
llm = init_chat_model(MODEL_NAME, model_provider="openai", temperature=0)
stakeholder_eval_chain = stakeholder_eval_prompt | llm | StrOutputParser()


def stakeholder_eval(state: GraphState):
    """경쟁사/개발자/투자업계 반응을 RAG(원문 자료)와 웹검색을 항상 병행하여
    조사해 stakeholder_eval을 write한다.
    """
    print("\n==== [STAKEHOLDER EVAL RAG + WEB SEARCH] ====\n")
    tech_sw = state["tech_sw"]
    tech_hw = state["tech_hw"]

    references: list[str] = []
    try:
        retriever = _get_stakeholder_chain().retriever
        docs = retriever.invoke(f"{tech_sw} {tech_hw} 경쟁 기술 포지셔닝 한계")
        retrieved_chunks = format_docs(docs)
        references.extend(rag_references(docs))
    except Exception as e:  # TODO: data/raw/ 원문 PDF 준비 전까지의 임시 예외처리
        print(f"[WARN] RAG 체인이 아직 준비되지 않았습니다: {e}")
        retrieved_chunks = ""

    search_results = web_search_tool.invoke(
        {"query": f"{tech_sw} vs {tech_hw} 경쟁사 반응 개발자 반응 투자 업계 시각"}
    )
    references.extend(web_references(search_results))

    result = stakeholder_eval_chain.invoke(
        {
            "tech_sw": tech_sw,
            "tech_hw": tech_hw,
            "retrieved_chunks": retrieved_chunks,
            "search_results": search_results,
        }
    )

    return {
        "stakeholder_eval": result,
        "references": list(dict.fromkeys(references)),  # 순서 유지 + 중복 제거
        "messages": [("system", "[이해관계자 평가 RAG + 웹검색] 완료")],
    }
