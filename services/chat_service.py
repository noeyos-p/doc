"""
챗봇 서비스 (RAG 기반 OpenAI 호출)
- 질문 → 벡터 검색 → 컨텍스트 구성 → GPT 응답 (스트리밍)
"""
from openai import OpenAI

from config.settings import settings
from services.vector_service import search
from utils.exceptions import ChatException
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """당신은 DS투자증권 시황분석 리포트(DS Daily)를 분석하는 전문 금융 어시스턴트입니다.

[규칙]
- 반드시 제공된 리포트 내용을 근거로만 답변합니다.
- 리포트에 없는 내용은 "해당 리포트에서 확인되지 않습니다"라고 명확히 밝힙니다.
- 숫자, 수익률, 종목명 등은 정확하게 인용합니다.
- 조건에 해당하는 항목이 여러 개이면 빠짐없이 모두 나열합니다. 절대 일부만 답변하지 마세요.
- 표 데이터를 인용할 때는 마크다운 표 형식을 유지합니다.
- "수익률 음(-)" 또는 "수익률 양(+)"은 이미지 처리로 추출한 방향 정보이며, 정확한 수치가 아닌 방향만 확인된 것입니다.
- "약 +3.5%" 같은 추정치는 이미지 기반 캘리브레이션 결과이며, 근사값임을 밝히세요.
- 출처(페이지, 섹션)를 답변 끝에 간략히 표시합니다.
- 한국어로 답변합니다."""


def _build_context(hits: list[dict]) -> str:
    """검색 결과를 컨텍스트 문자열로 조합"""
    parts = []
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        header = (
            f"[{i}] {meta.get('section_title', '')} | "
            f"페이지 {meta.get('page_number', '?')} | "
            f"{meta.get('chunk_type', 'text')}"
        )
        parts.append(f"{header}\n{hit['text']}")
    return "\n\n---\n\n".join(parts)


def _build_source_summary(hits: list[dict]) -> str:
    """출처 요약 문자열"""
    seen = set()
    sources = []
    for hit in hits:
        meta = hit["metadata"]
        key = (meta.get("section_title", ""), meta.get("page_number", "?"))
        if key not in seen:
            seen.add(key)
            sources.append(f"• {key[0]} (p.{key[1]})")
    return "\n".join(sources) if sources else ""


def chat_stream(
    question: str,
    collection_name: str,
    chat_history: list[dict],
):
    """
    RAG 기반 스트리밍 응답 제너레이터
    Yields: str (텍스트 조각) | dict (메타: {"sources": ...})
    """
    # ── 1. 관련 청크 검색 ─────────────────────────────────────────────
    # 1차: 소규모 검색으로 질문 유형 판별
    try:
        probe_hits = search(question, collection_name, top_k=10)
    except Exception as e:
        raise ChatException(f"검색 오류: {e}") from e

    if not probe_hits:
        yield "관련 내용을 찾을 수 없습니다. 질문을 다시 입력해 주세요."
        return

    # 거리 분포로 넓은 질문(요약/전체) vs 구체적 질문 자동 판별
    distances = [h["distance"] for h in probe_hits]
    spread = max(distances) - min(distances)

    # 스프레드가 작으면 = 모든 청크가 비슷한 거리 = 넓은 질문
    if spread < 0.10:
        hits = search(question, collection_name, top_k=15)
    else:
        hits = probe_hits[:6]

    context = _build_context(hits)
    sources = _build_source_summary(hits)

    # ── 2. 메시지 구성 ────────────────────────────────────────────────
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 이전 대화 (최근 6턴만 포함)
    for msg in chat_history[-12:]:
        messages.append(msg)

    messages.append(
        {
            "role": "user",
            "content": (
                f"[리포트 내용]\n{context}\n\n"
                f"[질문]\n{question}"
            ),
        }
    )

    # ── 3. OpenAI 스트리밍 호출 ───────────────────────────────────────
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        stream = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            stream=True,
            temperature=0.2,
            max_tokens=3000 if spread < 0.10 else 1500,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

        # 출처 메타데이터 마지막에 전달
        if sources:
            yield {"sources": sources}

    except Exception as e:
        raise ChatException(f"OpenAI API 오류: {e}") from e
