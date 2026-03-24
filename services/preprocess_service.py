"""
전처리 서비스
- 헤더/푸터/페이지번호/워터마크 제거
- DS Daily 섹션 구조 감지 (폰트 속성 기반 자동 감지, 하드코딩 없음)
- 텍스트 정제
"""
import re
from models.schemas import PageContent, ParsedDocument
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_section_patterns(doc: ParsedDocument) -> list[str]:
    """
    문서에서 자동 감지된 섹션 헤더 목록 반환.
    parser_service에서 폰트 속성(볼드 + 크기 ≥ 7pt)으로 추출하여
    PageContent.metadata["section_headers"]에 저장해둔 값을 사용.
    """
    for page in doc.pages:
        headers = page.metadata.get("section_headers", [])
        if headers:
            return headers
    return []


# 제거할 반복 노이즈 패턴
NOISE_PATTERNS = [
    r"DS\s*INVESTMENT\s*&\s*SECURITIES",
    r"DS\s*투자증권",
    r"DS\s*Daily",
    r"Compliance\s*Notice.*?(?=\n\n|\Z)",   # 면책조항 블록
    r"본\s*자료에\s*기재된.*?(?=\n\n|\Z)",
    r"(?<!\d)(\d{1,2})\s*/\s*(\d{1,2})(?!\d)(?=\s*$)",  # 페이지 번호 (줄 끝)
    r"^\s*\d+\s*$",                          # 단독 숫자 줄 (페이지 번호)
    r"guswl\d+@ds-sec\.co\.kr",            # 이메일
    r"\d{3}-\d{3,4}-\d{4}",               # 전화번호
    r"12MF\s+EPS\s+12MF\s+PER",           # 스파크라인 차트 헤더
]

# 섹션 헤더 인식 정규식 (마크다운 heading 또는 굵은 텍스트)
SECTION_HEADER_RE = re.compile(
    r"^#{1,3}\s+(.+)$|^\*\*(.+)\*\*$",
    re.MULTILINE,
)


def _remove_noise(text: str) -> str:
    """노이즈 패턴 제거"""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    # 연속 빈 줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_section(text: str, section_patterns: list[str]) -> str:
    """텍스트에서 가장 가까운 섹션 제목 감지"""
    text_upper = text[:300]
    for section in section_patterns:
        if section in text_upper:
            return section
    m = SECTION_HEADER_RE.search(text_upper)
    if m:
        return (m.group(1) or m.group(2) or "").strip()
    return "기타"


def preprocess_document(doc: ParsedDocument) -> tuple[ParsedDocument, list[str]]:
    """
    문서 전처리 수행
    Returns: (전처리된 문서, 처리 로그 리스트)
    """
    # 자동 감지된 섹션 패턴 사용
    section_patterns = _get_section_patterns(doc)
    if section_patterns:
        logger.info(f"자동 감지 섹션 패턴 사용: {len(section_patterns)}개")
    else:
        logger.warning("섹션 패턴 자동 감지 실패. 섹션 구분 없이 처리.")

    logs: list[str] = []
    processed_pages: list[PageContent] = []

    for page in doc.pages:
        original_len = len(page.text)
        cleaned = _remove_noise(page.text)
        removed = original_len - len(cleaned)

        if removed > 0:
            logs.append(f"  페이지 {page.page_number + 1}: 노이즈 {removed}자 제거")

        processed_pages.append(
            PageContent(
                page_number=page.page_number,
                text=cleaned,
                tables=page.tables,
                metadata={
                    **page.metadata,
                    "section": _detect_section(cleaned, section_patterns),
                },
            )
        )

    doc.pages = processed_pages
    logger.info(f"전처리 완료: {len(logs)}개 페이지에서 노이즈 제거")
    return doc, logs


def get_section_map(doc: ParsedDocument) -> dict[str, list[str]]:
    """섹션별 텍스트 묶음 반환 {섹션명: [텍스트, ...]}"""
    section_patterns = _get_section_patterns(doc)
    section_map: dict[str, list[str]] = {}
    current_section = "기타"

    for page in doc.pages:
        lines = page.text.split("\n")
        buffer: list[str] = []

        for line in lines:
            for section in section_patterns:
                if section in line:
                    if buffer:
                        section_map.setdefault(current_section, []).append(
                            "\n".join(buffer).strip()
                        )
                        buffer = []
                    current_section = section
                    break
            buffer.append(line)

        if buffer:
            section_map.setdefault(current_section, []).append(
                "\n".join(buffer).strip()
            )

    return section_map
