"""RAG 파이프라인의 기본 검색 체인 추상 클래스를 정의합니다.


이 클래스는 retriever(원문 로딩 -> 청킹 -> BGE-M3 임베딩 -> Chroma 인덱싱 ->
retriever 생성)까지만 책임진다. 실제 생성(LLM 호출)은 각 agents/*.py가
prompts/ 폴더의 템플릿을 직접 읽어서 자기 체인을 구성한다.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from langchain_chroma import Chroma

from rag.embeddings import create_bge_m3_embeddings


class RetrievalChain(ABC):
    def __init__(self, collection_name: str = "kv_cache_eval", k: int = 8):
        self.source_uri = None
        self.k = k
        self.index_dir = Path(".cache/chroma_index")
        self.collection_name = collection_name

    @abstractmethod
    def load_documents(self, source_uris):
        """loader를 사용하여 문서를 로드합니다."""
        pass

    @abstractmethod
    def create_text_splitter(self):
        """text splitter를 생성합니다."""
        pass

    def split_documents(self, docs, text_splitter):
        """text splitter를 사용하여 문서를 분할합니다."""
        return text_splitter.split_documents(docs)

    def create_embedding(self):
        """BGE-M3 임베딩 모델을 생성합니다. (설계서 B. 선정한 Embedding 모델)"""
        return create_bge_m3_embeddings()

    def _sanitize_metadata(self, doc):
        """Chroma는 메타데이터 값으로 str/int/float/bool만 허용한다 (빈 리스트도 거부).
        rag/pdf_parser.py가 붙이는 figures/tables 같은 리스트 메타데이터를
        문자열로 변환해서 upsert 시 ValueError가 나지 않게 한다.
        """
        for key, value in list(doc.metadata.items()):
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(str(v) for v in value)
        return doc

    def _open_vectorstore(self):
        """디스크의 Chroma 컬렉션을 연다 (없으면 빈 컬렉션이 만들어진다)."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.create_embedding(),
            persist_directory=str(self.index_dir),
        )

    def _indexed_sources(self, vectorstore) -> set:
        """컬렉션에 이미 인덱싱되어 있는 출처(source) 집합을 반환한다."""
        metadatas = vectorstore.get(include=["metadatas"])["metadatas"]
        return {metadata.get("source") for metadata in metadatas}

    def create_vectorstore(self, split_docs):
        """분할된 문서로부터 Chroma 벡터스토어를 생성합니다.

        같은 collection_name의 인덱스가 디스크에 이미 있으면 재사용한다.
        문서 단위(source)로 이미 인덱싱된 출처는 건너뛰고, source_uri에
        새로 추가된 파일만 임베딩/삽입한다. tech_research / domain_eval /
        stakeholder_eval이 각자 build_tech_retrieval_chain()을 호출할 때
        동일 문서가 중복 삽입되는 것을 막으면서도, TECH_PAPER_PATHS에
        논문이 새로 추가됐을 때 기존 컬렉션이 비어있지 않다는 이유로
        누락되지 않도록 한다. 반대로 source_uri에서 빠진 출처의 청크는
        컬렉션에서 삭제해, 목록에서 제외한 논문이 계속 검색되지 않게 한다.
        """
        split_docs = [self._sanitize_metadata(doc) for doc in split_docs]

        vectorstore = self._open_vectorstore()
        existing_sources = self._indexed_sources(vectorstore)
        stale_sources = existing_sources - set(self.source_uri or [])
        if stale_sources:
            vectorstore.delete(where={"source": {"$in": sorted(stale_sources)}})
        new_docs = [
            doc for doc in split_docs if doc.metadata.get("source") not in existing_sources
        ]
        if new_docs:
            vectorstore.add_documents(new_docs)
        return vectorstore

    def create_retriever(self, vectorstore):
        # Cosine Similarity 사용하여 검색을 수행하는 retriever를 생성합니다.
        dense_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": self.k}
        )
        return dense_retriever

    def create_chain(self):
        """원문 로딩부터 retriever 생성까지 수행하고 self.retriever를 채운다.
        생성(LLM) 단계는 이 클래스의 책임이 아니므로 여기서 끝난다.

        이미 인덱싱된 출처는 파싱/청킹부터 건너뛴다 (PDF 파싱이 실행 시간의
        대부분이라, 인덱스가 다 채워진 뒤에는 로딩 없이 retriever만 만든다).
        """
        indexed_sources = self._indexed_sources(self._open_vectorstore())
        pending_uris = [uri for uri in self.source_uri if uri not in indexed_sources]
        docs = self.load_documents(pending_uris) if pending_uris else []
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        return self
