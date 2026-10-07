"""prompts/*.txt 템플릿을 PromptTemplate으로 불러오는 공용 유틸리티.
"""

import threading
import time

from langchain_core.prompts import PromptTemplate

# Tavily는 요청이 한꺼번에 몰리면 429(요청 과다)로 막는다. 병렬 워커들의 웹검색을
# 한 번에 하나씩만 보내고, 그래도 429가 나오면 아래 간격(초)으로 기다렸다 다시 보낸다.
_search_lock = threading.Lock()
RATE_LIMIT_WAITS = (2, 5, 10)


def load_prompt(path: str) -> PromptTemplate:
    """prompts/ 폴더의 .txt 템플릿 파일을 읽어 PromptTemplate으로 변환합니다."""
    with open(path, encoding="utf-8") as f:
        template = f.read()
    return PromptTemplate.from_template(template)


def rag_references(docs) -> list[str]:
    """RAG로 실제 검색된 문서들의 출처(source/page)를 REFERENCE용 문자열로 뽑는다."""
    refs = []
    for doc in docs:
        source = doc.metadata.get("source")
        if not source:
            continue
        page = doc.metadata.get("page")
        refs.append(f"{source} (p.{page + 1})" if page is not None else source)
    return refs


def web_references(search_results) -> list[str]:
    """TavilySearch 응답에서 실제로 인용 가능한 출처(제목/URL)를 뽑는다."""
    refs = []
    if isinstance(search_results, dict):
        for item in search_results.get("results", []):
            url = item.get("url")
            if not url:
                continue
            title = item.get("title")
            refs.append(f"{title} - {url}" if title else url)
    return refs


def web_search(tool, query: str) -> dict:
    """TavilySearch를 호출하되, 실패해도 그래프가 죽지 않도록 빈 결과로 폴백한다.

    TavilySearch는 결과가 0건이면 ToolException을 던지고, API/네트워크 오류는
    예외 대신 {"error": ...} dict로 돌려준다. 두 경우 모두 빈 결과로 통일해서
    오류 메시지가 프롬프트에 그대로 들어가지 않게 한다. 자료 부족 여부는
    이후 충분성 판정이 data_limited로 기록한다.
    """
    for wait in (*RATE_LIMIT_WAITS, None):
        with _search_lock:
            try:
                results = tool.invoke({"query": query})
            except Exception as e:
                print(f"[WARN] 웹검색 실패 ({query}): {e}")
                return {"results": []}
            rate_limited = isinstance(results, dict) and "429" in str(results.get("error", ""))
            if not rate_limited or wait is None:
                break
            print(f"[WARN] 웹검색 요청 과다(429), {wait}초 후 재시도 ({query})")
            time.sleep(wait)
    if not isinstance(results, dict) or "error" in results:
        print(f"[WARN] 웹검색 실패 ({query}): {results}")
        return {"results": []}
    return results
