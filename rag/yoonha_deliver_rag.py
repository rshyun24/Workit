"""
retriever.py
------------
Workit — SW/SI 산출물 피드백 플랫폼
사용자가 업로드한 산출물 청크에 대해 Qdrant에서 양식 기준 청크를 검색하고,
section_title 기준 매핑 + 인접 청크 확장 + sLLM 프롬프트 컨텍스트를 구성한다.

설계 문서 4절(리트리버 설계) 및 5절(sLLM 프롬프트 컨텍스트 구성) 구현.

사용 예:
    from retriever import WorkitRetriever
    from yoonha_deliver_chunking import get_qdrant_client

    client = get_qdrant_client()
    retriever = WorkitRetriever(client)

    ctx = retriever.retrieve_context(
        user_text="3.1 영역별 결과 요약\n표 형식으로 ...",
        user_section_title="3.1 영역별 결과 요약",
        doc_type="테스트결과보고서",
    )
    print(ctx.prompt_context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from yoonha_deliver_chunking import embed_texts, COLLECTION, QDRANT_HOST, QDRANT_PORT


# ─────────────────────────────────────────────
# 반환 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """검색된 단일 청크 정보."""
    chunk_id: str
    doc_id: str
    doc_type: str
    section_title: str
    section_path: list[str]
    section_depth: int
    section_type: str
    is_required: bool
    guideline_reference: Optional[str]
    guideline_merged: bool
    chunk_index: int
    chunk_total: int
    text: str
    score: float = 0.0


@dataclass
class RetrievalContext:
    """
    단일 사용자 섹션에 대한 검색 결과 + sLLM 프롬프트 컨텍스트.
    """
    user_section_title: str
    user_text: str
    reference_chunk: Optional[RetrievedChunk]
    adjacent_chunks: list[RetrievedChunk]
    fallback_used: bool = False
    prompt_context: str = ""


# ─────────────────────────────────────────────
# 헬퍼 — Qdrant payload → RetrievedChunk
# ─────────────────────────────────────────────

def _payload_to_chunk(payload: dict, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=payload.get("chunk_id", ""),
        doc_id=payload.get("doc_id", ""),
        doc_type=payload.get("doc_type", ""),
        section_title=payload.get("section_title", ""),
        section_path=payload.get("section_path", []),
        section_depth=payload.get("section_depth", 1),
        section_type=payload.get("section_type", "서술"),
        is_required=payload.get("is_required", True),
        guideline_reference=payload.get("guideline_reference"),
        guideline_merged=payload.get("guideline_merged", False),
        chunk_index=payload.get("chunk_index", 0),
        chunk_total=payload.get("chunk_total", 1),
        text=payload.get("text", ""),
        score=score,
    )


# ─────────────────────────────────────────────
# 메인 리트리버 클래스
# ─────────────────────────────────────────────

class WorkitRetriever:
    """
    설계 문서 4절 리트리버 설계 구현체.

    Parameters
    ----------
    client : QdrantClient
        적재가 완료된 Qdrant 클라이언트 인스턴스
    model_name : str
        임베딩 모델명 (indexer와 동일한 모델 사용 필수)
    top_k : int
        벡터 검색 결과 수 (기본 5)
    """

    def __init__(
        self,
        client: QdrantClient,
        model_name: str = "BAAI/bge-m3",
        top_k: int = 5,
    ):
        self.client = client
        self.model_name = model_name
        self.top_k = top_k

    # ─────────────────────────────────────
    # 1. 벡터 검색 (설계 4.2절)
    # ─────────────────────────────────────

    def _vector_search(
        self,
        query_text: str,
        doc_type: str,
    ) -> list[RetrievedChunk]:
        """
        doc_type 필터 + 벡터 유사도 검색.
        반환: 유사도 내림차순 RetrievedChunk 리스트
        """
        query_vector = embed_texts([query_text], self.model_name)[0]

        results = self.client.query_points(       # ← search → query_points
            collection_name=COLLECTION,           # ← COLLECTION_NAME → COLLECTION
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_type",
                        match=MatchValue(value=doc_type),
                    )
                ]
            ),
            limit=self.top_k,
            with_payload=True,
        ).points

        return [_payload_to_chunk(r.payload, score=r.score) for r in results]

    # ─────────────────────────────────────
    # 2. section_title 매핑 (설계 4.3절)
    # ─────────────────────────────────────

    def _map_by_section_title(
        self,
        search_results: list[RetrievedChunk],
        user_section_title: str,
    ) -> tuple[RetrievedChunk, bool]:
        """
        검색 결과에서 section_title 일치 청크를 우선 선택.
        없으면 유사도 1위 fallback 사용.

        Returns
        -------
        (reference_chunk, fallback_used)
        """
        # 완전 일치
        matched = [
            c for c in search_results
            if c.section_title.strip() == user_section_title.strip()
        ]
        if matched:
            return matched[0], False

        # 부분 일치
        partial = [
            c for c in search_results
            if (user_section_title.strip() in c.section_title.strip()
                or c.section_title.strip() in user_section_title.strip())
        ]
        if partial:
            return partial[0], False

        # fallback: 유사도 1위
        return search_results[0], True

    # ─────────────────────────────────────
    # 3. 인접 청크 확장 (설계 4.4절)
    # ─────────────────────────────────────

    def _expand_adjacent_chunks(
        self,
        reference_chunk: RetrievedChunk,
    ) -> list[RetrievedChunk]:
        """
        같은 doc_id + section_title 의 모든 청크를 조회,
        chunk_index 기준 정렬하여 반환.
        """
        scroll_result, _ = self.client.scroll(
            collection_name=COLLECTION,           # ← COLLECTION_NAME → COLLECTION
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=reference_chunk.doc_id),
                    ),
                    FieldCondition(
                        key="section_title",
                        match=MatchValue(value=reference_chunk.section_title),
                    ),
                ]
            ),
            limit=100,
            with_payload=True,
        )

        chunks = [_payload_to_chunk(p.payload) for p in scroll_result]
        chunks.sort(key=lambda c: c.chunk_index)
        return chunks

    # ─────────────────────────────────────
    # 4. sLLM 프롬프트 컨텍스트 구성 (설계 5절)
    # ─────────────────────────────────────

    def _build_prompt_context(
        self,
        user_section_title: str,
        user_text: str,
        reference_chunk: RetrievedChunk,
        adjacent_chunks: list[RetrievedChunk],
    ) -> str:
        """
        설계 문서 5절 형식대로 sLLM 프롬프트 컨텍스트 문자열 생성.
        """
        if adjacent_chunks:
            full_reference_text = "\n".join(c.text for c in adjacent_chunks)
        else:
            full_reference_text = reference_chunk.text

        section_path_str = (
            " > ".join(reference_chunk.section_path)
            if reference_chunk.section_path
            else reference_chunk.section_title
        )

        guideline_block = ""
        if reference_chunk.guideline_reference:
            guideline_block = (
                f"\n\n[근거 조항]\n{reference_chunk.guideline_reference}"
            )

        context = (
            f"[섹션 위치]\n{section_path_str}"
            f"\n\n[양식 기준]\n{full_reference_text}"
            f"\n\n[필수 여부]\n{'필수 항목' if reference_chunk.is_required else '선택 항목'} "
            f"(섹션 유형: {reference_chunk.section_type})"
            f"{guideline_block}"
            f"\n\n[사용자 작성 내용]\n{user_text}"
        )
        return context.strip()

    # ─────────────────────────────────────
    # 5. 공개 인터페이스
    # ─────────────────────────────────────

    def retrieve_context(
        self,
        user_text: str,
        user_section_title: str,
        doc_type: str,
    ) -> RetrievalContext:
        """
        단일 사용자 섹션에 대해 전체 리트리버 파이프라인 수행.
        """
        search_results = self._vector_search(user_text, doc_type)
        if not search_results:
            return RetrievalContext(
                user_section_title=user_section_title,
                user_text=user_text,
                reference_chunk=None,
                adjacent_chunks=[],
                prompt_context="[오류] 해당 doc_type에 대한 양식 데이터가 없습니다.",
            )

        reference_chunk, fallback_used = self._map_by_section_title(
            search_results, user_section_title
        )
        adjacent_chunks = self._expand_adjacent_chunks(reference_chunk)
        prompt_ctx = self._build_prompt_context(
            user_section_title=user_section_title,
            user_text=user_text,
            reference_chunk=reference_chunk,
            adjacent_chunks=adjacent_chunks,
        )

        return RetrievalContext(
            user_section_title=user_section_title,
            user_text=user_text,
            reference_chunk=reference_chunk,
            adjacent_chunks=adjacent_chunks,
            fallback_used=fallback_used,
            prompt_context=prompt_ctx,
        )

    def retrieve_document(
        self,
        user_sections: list[dict],
        doc_type: str,
    ) -> list[RetrievalContext]:
        """
        사용자 산출물 전체 섹션 리스트에 대해 일괄 리트리버 수행.

        Parameters
        ----------
        user_sections : list[dict]
            [{"text": "...", "section_title": "..."}, ...] 형태
        doc_type : str
            산출물 종류
        """
        return [
            self.retrieve_context(
                user_text=sec["text"],
                user_section_title=sec["section_title"],
                doc_type=doc_type,
            )
            for sec in user_sections
        ]


# ─────────────────────────────────────────────
# 동작 확인 (직접 실행 시)
# 실제 평가는 rag/evaluation/yoonha_deliver_evaluation.py 에서 수행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from yoonha_deliver_chunking import get_qdrant_client

    client    = get_qdrant_client()
    retriever = WorkitRetriever(client)

    count = client.count(collection_name=COLLECTION)
    print(f"✅ workit_forms KB: {count.count}개 청크")
    print("실제 평가는 rag/evaluation/yoonha_deliver_evaluation.py 를 실행하세요.")