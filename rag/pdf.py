"""PDF 문서 기반 RAG 검색 체인 구현 모듈입니다.

기술 조사 에이전트(agents/tech_research.py), 도메인 평가 에이전트
(agents/domain_eval.py), 이해관계자 평가 에이전트(agents/stakeholder_eval.py,
변경됨: RAG 적용)가 이 체인의 retriever를 공통으로 사용합니다.

담당: __________ (TODO: 담당자 배정)
"""

from typing import Annotated

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.base import RetrievalChain


def format_docs(docs: list[Document]) -> str:
    """검색된 문서 리스트를 출처(source)/페이지(page)가 보존된 형태로 포맷팅합니다.
    """
    return "\n\n".join(
        f"<document><content>{doc.page_content}</content>"
        f"<source>{doc.metadata.get('source')}</source>"
        f"<page>{doc.metadata.get('page', 0) + 1}</page></document>"
        
        for doc in docs
    )


class PDFRetrievalChain(RetrievalChain):
    def __init__(self, source_uri: Annotated[list[str], "Source URIs"], **kwargs):
        super().__init__(**kwargs)
        self.source_uri = source_uri

    def load_documents(self, source_uris: list[str]) -> list[Document]:
        docs = []
        for source_uri in source_uris:
            loader = PDFPlumberLoader(source_uri)
            docs.extend(loader.load())
        return docs

    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)


# --- 설계서 B. RAG 적용 대상: 기술조사/도메인 평가/이해관계자 평가 에이전트가 함께 쓰는 원문 자료 ---
TECH_PAPER_PATHS = [
    "data/raw/deepseek_v2.pdf",   # TODO: 실제 파일명으로 교체
    "data/raw/itme.pdf",          # TODO: 실제 파일명으로 교체
]


def build_tech_retrieval_chain(source_uri: list[str] = TECH_PAPER_PATHS) -> PDFRetrievalChain:
    """기술 조사 / 도메인 평가 에이전트가 공용으로 사용하는 RAG 체인을 생성합니다.
    반환된 인스턴스의 .retriever를 각 agent가 자신의 prompts/*.txt 체인에 연결해서 쓴다.

    TODO: data/raw/에 원문 PDF 2건을 넣은 뒤 사용하세요.
    """
    return PDFRetrievalChain(source_uri=source_uri).create_chain()
