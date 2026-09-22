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
from rag.pdf_parser import parse_pdf_to_documents


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
            try:
                parsed = parse_pdf_to_documents(source_uri)
                docs.extend(parsed)
            except Exception as e:
                print(f"[WARN] 알고리즘 PDF 파서 오류, PDFPlumberLoader로 폴백 ({source_uri}): {e}")
                loader = PDFPlumberLoader(source_uri)
                docs.extend(loader.load())
        return docs

    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)


# --- 설계서 B. RAG 적용 대상: 기술조사/도메인 평가/이해관계자 평가 에이전트가 함께 쓰는 원문 자료 ---
TECH_PAPER_PATHS = [
    "data/raw/A_Case_Against_CXL_Memory_Pooling_no-refs.pdf",
    "data/raw/DeepSeek-V2.pdf",
    "data/raw/DeepSeek_V4_in_vLLM_Efficient_Long_Context_Attention.pdf",
    "data/raw/Design_Tradeoffs_CXL_Memory_Pools_no-refs.pdf",
    "data/raw/GQA_Training_Generalized_Multi_Query_Transformer_Models_from_Multi_Head_Checkpoints.pdf",
    "data/raw/ITME.pdf",
    "data/raw/InfiniGen.pdf",
    "data/raw/KIVI_no-refs.pdf",
    "data/raw/LIMINAL_Efficient_LLM_Inference_no-refs.pdf",
    "data/raw/PIMCXL_no-refs.pdf",
    "data/raw/Rearchitecting_Datacenter_Lifecycle_for_AI_no-refs.pdf",
    "data/raw/SGLang_v0.3_Release_7x_ Faster_DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image_Video LLaVA-OneVision - LMSYS Org.pdf",
    "data/raw/Systematic_CXL_Memory_Characterization_at_Scale_no-refs.pdf",
    "data/raw/TransMLA_MLA_Is_All_You_Need.pdf",
    "data/raw/TurboQuant.pdf",
]


def build_tech_retrieval_chain(source_uri: list[str] = TECH_PAPER_PATHS) -> PDFRetrievalChain:
    """기술 조사 / 도메인 평가 에이전트가 공용으로 사용하는 RAG 체인을 생성합니다.
    반환된 인스턴스의 .retriever를 각 agent가 자신의 prompts/*.txt 체인에 연결해서 쓴다.
    """
    return PDFRetrievalChain(source_uri=source_uri).create_chain()


# --- 선정 과정에서 제외된 대조 기술 원문. tech_research()가 SW/HW 각각의
# "기술 개요"에 왜 이 대안을 안 골랐는지 한 줄 비교로 녹여 쓸 때만 참고한다.
# domain_eval / stakeholder_eval이 쓰는 TECH_PAPER_PATHS 풀과는 별도 Chroma
# collection으로 분리해서, 대조 기술 청크가 그쪽 평가 결과에 섞여 들어가지
# 않게 한다. SW(DeepSeek-V2) 대조군과 HW(ITME) 대조군도 서로 다른 논문이라
# collection을 나눠서 검색 시 서로 섞이지 않도록 한다. ---
SW_COMPARISON_PAPER_PATHS = [
    "data/raw/TurboQuant.pdf",
]
HW_COMPARISON_PAPER_PATHS = [
    "data/raw/InfiniGen.pdf",
]


def build_sw_comparison_retrieval_chain(
    source_uri: list[str] = SW_COMPARISON_PAPER_PATHS,
) -> PDFRetrievalChain:
    """SW(DeepSeek-V2) 쪽에서 제외된 대조 기술(TurboQuant) 원문 RAG 체인을 생성한다."""
    return PDFRetrievalChain(
        source_uri=source_uri, collection_name="kv_cache_comparison_sw", k=4
    ).create_chain()


def build_hw_comparison_retrieval_chain(
    source_uri: list[str] = HW_COMPARISON_PAPER_PATHS,
) -> PDFRetrievalChain:
    """HW(ITME) 쪽에서 제외된 대조 기술(InfiniGen) 원문 RAG 체인을 생성한다."""
    return PDFRetrievalChain(
        source_uri=source_uri, collection_name="kv_cache_comparison_hw", k=4
    ).create_chain()
