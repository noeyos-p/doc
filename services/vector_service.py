"""
벡터 DB 서비스 (ChromaDB + OpenAI 임베딩)
- 문서별 컬렉션 생성 (파일명 기반)
- 청크 저장 / 유사도 검색
"""
import os
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config.settings import settings
from models.schemas import DocumentChunk
from utils.exceptions import VectorStoreException
from utils.logger import get_logger

logger = get_logger(__name__)


def _safe_collection_name(filename: str) -> str:
    """ChromaDB 컬렉션명: ASCII 영숫자·하이픈·점·밑줄만, 3~512자"""
    import hashlib
    stem = Path(filename).stem
    # 영숫자·하이픈·밑줄만 남기고 나머지(한글 포함)는 제거
    ascii_part = re.sub(r"[^a-zA-Z0-9\-_]", "", stem)
    if len(ascii_part) < 3:
        # 한글 파일명 → 해시로 대체
        h = hashlib.md5(stem.encode()).hexdigest()[:12]
        ascii_part = f"doc_{h}"
    return ascii_part[:63]


def _get_client() -> chromadb.PersistentClient:
    persist_dir = settings.chroma_persist_dir
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def _get_embedding_fn() -> OpenAIEmbeddingFunction:
    return OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.openai_embedding_model,
    )


def add_chunks(chunks: list[DocumentChunk], source_file: str) -> str:
    """
    청크를 ChromaDB에 저장.
    Returns: 컬렉션명
    """
    if not chunks:
        raise VectorStoreException("저장할 청크가 없습니다.")

    collection_name = _safe_collection_name(source_file)

    try:
        client = _get_client()
        ef = _get_embedding_fn()

        # 기존 컬렉션 삭제 후 재생성 (동일 파일 재업로드 시)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                "chunk_type": c.chunk_type,
                "section_title": c.section_title,
                "page_number": c.page_number,
                "source_file": c.source_file,
                "report_date": c.metadata.get("report_date", ""),
            }
            for c in chunks
        ]

        # 100개씩 배치 처리 (API 제한 대응)
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[i : i + batch_size],
                documents=texts[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

        logger.info(f"벡터 DB 저장 완료: {len(chunks)}개 청크 → '{collection_name}'")
        return collection_name

    except VectorStoreException:
        raise
    except Exception as e:
        raise VectorStoreException(f"벡터 DB 저장 실패: {e}") from e


def search(
    query: str,
    collection_name: str,
    top_k: int = 6,
    chunk_type_filter: str | None = None,
) -> list[dict]:
    """
    유사도 검색.
    Returns: [{"text": ..., "metadata": ..., "distance": ...}, ...]
    """
    try:
        client = _get_client()
        ef = _get_embedding_fn()
        collection = client.get_collection(
            name=collection_name, embedding_function=ef
        )

        where = {"chunk_type": chunk_type_filter} if chunk_type_filter else None
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({"text": doc, "metadata": meta, "distance": dist})

        return hits

    except Exception as e:
        raise VectorStoreException(f"검색 실패: {e}") from e


def list_collections() -> list[str]:
    """저장된 컬렉션 목록 반환"""
    try:
        return [c.name for c in _get_client().list_collections()]
    except Exception:
        return []
