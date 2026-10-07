"""
전문 워커들을 위한 공통 유틸리티 모듈.
RAG/웹 검색, 재시도 제어(run_with_retry), 프롬프트 로더 등을 제공한다.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch

from agents.prompt_utils import rag_references, web_references, web_search
from graph.state import Decision, WorkerInput
from rag.pdf import format_docs

WORKER_MODEL = os.getenv("WORKER_MODEL", os.getenv("MODEL_NAME", "gpt-5.6-luna"))
PLANNER_MODEL = os.getenv("PLANNER_MODEL", os.getenv("MODEL_NAME", "gpt-5.6-terra"))

MAX_ATTEMPTS = 2  # task별 시도 상한 (종료 보장)

_web_search_tool = None


def get_web_search_tool() -> TavilySearch:
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = TavilySearch(max_results=5)
    return _web_search_tool


INSTRUCTION_SUFFIX = """

[이번 조사에서 집중할 내용]
{instruction}

[작성 원칙]
- 두 기술의 우열을 판정하거나 한쪽을 추천하지 마세요. 각 기술의 특성을 근거와 함께 사실대로 정리하세요.
- 유리한 근거만 고르지 말고, 자료에 있는 한계와 반대 근거도 함께 쓰세요.
"""


def load_worker_prompt(path: str) -> PromptTemplate:
    """기존 프롬프트 파일에 Orchestrator 지시 자리를 덧붙인다."""
    return PromptTemplate.from_template(
        Path(path).read_text(encoding="utf-8") + INSTRUCTION_SUFFIX
    )


def search_rag(get_chain, query: str, label: str = "RAG"):
    """RAG 검색. (본문, 출처, 에러)를 반환하고 실패 시 본문은 빈 문자열이다."""
    try:
        docs = get_chain().retriever.invoke(query)
        return format_docs(docs), rag_references(docs), None
    except Exception as e:
        return "", [], f"{label} 검색 실패: {e}"


def search_web(query: str):
    """웹검색. 제목/URL/본문만 골라 정리한다. 결과가 없으면 본문은 빈 문자열이다."""
    try:
        tool = get_web_search_tool()
        found = web_search(tool, query[:300])  # Tavily 쿼리 길이 제한 대비
        if not found.get("results"):
            return "", [], "웹검색 결과 없음"
        text = "\n\n".join(
            f"<web><title>{item.get('title')}</title><url>{item.get('url')}</url>"
            f"<content>{item.get('content')}</content></web>"
            for item in found["results"]
        )
        return text, web_references(found), None
    except Exception as e:
        return "", [], f"웹검색 실패: {e}"


def join_errors(*errors) -> Optional[str]:
    return "; ".join(e for e in errors if e) or None


def run_with_retry(state: WorkerInput, run_once, node: str):
    """run_once(state) -> (content, references, error)를 MAX_ATTEMPTS까지 시도한다.
    content가 None이거나 예외가 나면 실패로 본다."""
    task_id = state["task"]["task_id"]
    decisions: list[Decision] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content, references, error = run_once(state)
        except Exception as e:
            content, references, error = None, [], f"{type(e).__name__}: {e}"
        if content is not None:
            return {
                "tasks": {
                    task_id: {
                        "status": "done",
                        "attempts": attempt,
                        "error": error,
                    }
                },
                "results": {
                    task_id: {
                        "content": content,
                        "references": list(dict.fromkeys(references)),
                    }
                },
                "decisions": decisions,
            }
        if attempt < MAX_ATTEMPTS:
            decisions.append(
                {
                    "node": node,
                    "action": "retry",
                    "task_ids": [task_id],
                    "reason": f"{attempt}회 실패({error}), 재시도",
                }
            )

    return {
        "tasks": {
            task_id: {
                "status": "failed",
                "attempts": MAX_ATTEMPTS,
                "error": error,
            }
        },
        "decisions": decisions,
    }
