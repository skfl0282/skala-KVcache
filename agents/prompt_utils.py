"""prompts/*.txt 템플릿을 PromptTemplate으로 불러오는 공용 유틸리티.

담당: __________ (TODO: 담당자 배정)
"""

from langchain_core.prompts import PromptTemplate


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
