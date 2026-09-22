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
    def __init__(self):
        self.source_uri = None
        self.k = 8
        self.index_dir = Path(".cache/chroma_index")

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

    def create_vectorstore(self, split_docs):
        """분할된 문서로부터 Chroma 벡터스토어를 생성합니다."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.create_embedding(),
            persist_directory=str(self.index_dir),
            collection_name="kv_cache_eval",
        )
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
        """
        docs = self.load_documents(self.source_uri)
        text_splitter = self.create_text_splitter()
        split_docs = self.split_documents(docs, text_splitter)
        self.vectorstore = self.create_vectorstore(split_docs)
        self.retriever = self.create_retriever(self.vectorstore)
        return self
