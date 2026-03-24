"""
청킹 서비스
- 섹션 기반 청킹 (자동 감지된 섹션 헤더 사용, 하드코딩 없음)
- 표는 독립 청크로 분리
- 긴 텍스트는 overlap 포함 분할
"""
import hashlib
from models.schemas import DocumentChunk, ParsedDocument, TableContent
from utils.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 1200       # 문자 기준 (차트 데이터 포함 시 잘림 방지)
CHUNK_OVERLAP = 200     # 겹침 문자 수


def _make_id(text: str, page: int, idx: int) -> str:
    h = hashlib.md5(f"{text[:80]}{page}{idx}".encode()).hexdigest()[:8]
    return f"p{page}_{idx}_{h}"


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    """긴 텍스트를 overlap 포함 분할"""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def _get_section_patterns(doc: ParsedDocument) -> list[str]:
    """문서에서 자동 감지된 섹션 헤더 목록 반환"""
    for page in doc.pages:
        headers = page.metadata.get("section_headers", [])
        if headers:
            return headers
    return []


def _split_by_section(text: str, section_patterns: list[str]) -> list[tuple[str, str]]:
    """
    텍스트를 섹션 기준으로 분리.
    Returns: [(section_title, section_text), ...]
    """
    result: list[tuple[str, str]] = []
    current_section = "기타"
    buffer: list[str] = []

    for line in text.split("\n"):
        detected = None
        for s in section_patterns:
            if s in line:
                detected = s
                break

        if detected:
            if buffer:
                result.append((current_section, "\n".join(buffer).strip()))
                buffer = []
            current_section = detected

        buffer.append(line)

    if buffer:
        result.append((current_section, "\n".join(buffer).strip()))

    return result


def chunk_document(doc: ParsedDocument) -> list[DocumentChunk]:
    """
    ParsedDocument → DocumentChunk 리스트
    순서: 섹션별 텍스트 청크 → 표 청크 (페이지 순)
    """
    section_patterns = _get_section_patterns(doc)
    if section_patterns:
        logger.info(f"청킹 섹션 패턴: {len(section_patterns)}개 사용")

    chunks: list[DocumentChunk] = []
    source = doc.metadata.filename
    chunk_idx = 0

    for page in doc.pages:
        page_num = page.page_number

        # ── 텍스트 청크 (섹션 단위 분리) ────────────────────────────────
        text = page.text.strip()
        sections: list[tuple[str, str]] = []
        if text:
            sections = _split_by_section(text, section_patterns)
            for section_title, section_text in sections:
                if len(section_text.strip()) < 30:
                    continue
                for fragment in _split_long_text(section_text, CHUNK_SIZE, CHUNK_OVERLAP):
                    if len(fragment.strip()) < 30:
                        continue
                    chunks.append(
                        DocumentChunk(
                            chunk_id=_make_id(fragment, page_num, chunk_idx),
                            text=fragment.strip(),
                            chunk_type="text",
                            section_title=section_title,
                            page_number=page_num + 1,
                            source_file=source,
                            metadata={"report_date": doc.metadata.report_date},
                        )
                    )
                    chunk_idx += 1

        # ── 표 청크 ─────────────────────────────────────────────────────
        last_section = sections[-1][0] if sections else "기타"
        for tbl in page.tables:
            table_text = _table_to_text(tbl)
            if len(table_text.strip()) < 10:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=_make_id(table_text, page_num, chunk_idx),
                    text=table_text,
                    chunk_type="table",
                    section_title=last_section,
                    page_number=page_num + 1,
                    source_file=source,
                    metadata={
                        "report_date": doc.metadata.report_date,
                        "accuracy": tbl.accuracy,
                    },
                )
            )
            chunk_idx += 1

    logger.info(
        f"청킹 완료: 총 {len(chunks)}개 "
        f"(텍스트 {sum(1 for c in chunks if c.chunk_type=='text')}, "
        f"표 {sum(1 for c in chunks if c.chunk_type=='table')})"
    )
    return chunks


def _table_to_text(tbl: TableContent) -> str:
    """표 → 검색 가능한 텍스트로 변환"""
    parts = []
    if tbl.caption:
        parts.append(f"[표] {tbl.caption}")
    parts.append(tbl.markdown)
    return "\n".join(parts)
