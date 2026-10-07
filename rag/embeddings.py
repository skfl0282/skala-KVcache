"""임베딩 모델 생성 함수.

담당: __________ (TODO: 담당자 배정)
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

BGE_M3_MODEL_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def create_bge_m3_embeddings() -> HuggingFaceEmbeddings:
    """BGE-M3 임베딩 모델을 생성합니다.

    설계서 B. 선정한 Embedding 모델 - 긴 문맥·다국어 지원, Dense/Sparse/
    Multi-vector Retrieval, MIT License 오픈소스.

    모델(약 2GB)을 컬렉션마다 다시 올리지 않도록 프로세스당 한 번만 생성한다.
    """
    return HuggingFaceEmbeddings(
        model_name=BGE_M3_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # TODO: GPU 사용 시 "cuda"로 변경
        encode_kwargs={"normalize_embeddings": True},
    )
