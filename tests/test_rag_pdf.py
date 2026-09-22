"""PDF RAG 검색 체인 및 파서 단위 테스트
"""

from pathlib import Path
from rag.pdf import (
    PDFRetrievalChain,
    format_docs,
    TECH_PAPER_PATHS,
    SW_COMPARISON_PAPER_PATHS,
    HW_COMPARISON_PAPER_PATHS,
)


def test_pdf_load_documents_and_format():
    """PDF 파서가 LangChain Document 객체를 올바르게 생성하고 포맷팅하는지 검증합니다."""
    sample_pdf = "data/raw/ITME.pdf"
    assert Path(sample_pdf).exists(), f"테스트용 PDF가 누락되었습니다: {sample_pdf}"

    chain = PDFRetrievalChain(source_uri=[sample_pdf])
    docs = chain.load_documents([sample_pdf])

    # 1. 문서 페이지 수 검증
    assert len(docs) == 13, f"ITME.pdf 페이지 수가 일치하지 않습니다 (기대: 13, 실제: {len(docs)})"

    # 2. 필수 메타데이터 검증
    for idx, doc in enumerate(docs):
        assert doc.metadata.get("source") == sample_pdf
        assert doc.metadata.get("page") == idx  # 0-indexed 확인
        assert "figures" in doc.metadata
        assert "tables" in doc.metadata
        assert len(doc.page_content.strip()) > 0

    # 3. format_docs 검증
    formatted = format_docs(docs[:2])
    assert "<document><content>" in formatted
    assert f"<source>{sample_pdf}</source>" in formatted
    assert "<page>1</page>" in formatted
    assert "<page>2</page>" in formatted

    # 4. TextSplitter 청킹 검증
    splitter = chain.create_text_splitter()
    split_chunks = splitter.split_documents(docs)
    assert len(split_chunks) > len(docs)
    assert split_chunks[0].metadata.get("source") == sample_pdf


if __name__ == "__main__":
    test_pdf_load_documents_and_format()
    print("OK: PDF RAG 파이프라인 단위 테스트를 통과했습니다.")
