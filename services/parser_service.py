"""
PDF 파싱 서비스
- PyMuPDF fitz get_text("words"): 단어별 좌표 추출
  → Y 기준 행 그룹화 → 행 내 X 기준 좌→우 정렬 (룰베이스, 하드코딩 없음)
- camelot-py: 표 정밀 추출 (stream mode 기본)
"""
import os
import re
import shutil
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd

from models.schemas import DocumentMetadata, PageContent, ParsedDocument, TableContent
from utils.exceptions import ParserException
from utils.logger import get_logger

logger = get_logger(__name__)


# DS Daily 특성: 왼쪽 본문 칼럼 경계 (전체 width=595pt, 칼럼 분리 기준)
LEFT_COL_MAX_X = 330


def _is_ghostscript_available() -> bool:
    return shutil.which("gs") is not None or shutil.which("gswin64c") is not None


ROW_TOLERANCE = 4.0  # Y 좌표 차이가 이 값 이하면 같은 행으로 처리 (pt 단위)

# 섹션 헤더 자동 감지 기준값
SECTION_HEADER_MIN_SIZE = 7.0   # 이 크기 이상
SECTION_HEADER_BOLD = True      # 볼드체 필수


def extract_section_headers(page: fitz.Page) -> list[str]:
    """
    PDF 페이지에서 폰트 속성(크기 + 볼드)으로 섹션 헤더를 자동 감지.
    룰: 볼드 + 폰트 크기 >= SECTION_HEADER_MIN_SIZE → 섹션 헤더
    """
    blocks = page.get_text("dict")["blocks"]
    headers: list[str] = []
    seen: set[str] = set()

    for b in blocks:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            line_texts: list[str] = []
            is_header = False

            for span in line["spans"]:
                text = span["text"].strip()
                if not text:
                    continue
                size = span["size"]
                flags = span["flags"]
                bold = bool(flags & (1 << 4))

                if bold and size >= SECTION_HEADER_MIN_SIZE:
                    is_header = True
                line_texts.append(text)

            if is_header and line_texts:
                header = " ".join(line_texts)
                # 노이즈 필터: 너무 긴 것(본문), 단일 기호, 숫자만
                if len(header) > 30 or len(header) < 2:
                    continue
                if header.startswith("[") and header.endswith("]"):
                    continue  # 이슈 태그 [한국], [미국] 등 제외
                if header not in seen:
                    seen.add(header)
                    headers.append(header)

    return headers


def _extract_text_fitz(page: fitz.Page) -> str:
    """
    fitz words 기반 텍스트 추출 (룰베이스).

    규칙:
    1. 단어(word) 단위로 Y 좌표를 비교해 같은 행 묶기 (tolerance=ROW_TOLERANCE)
    2. 같은 행 내 단어는 X 좌표 오름차순(좌→우) 정렬
    3. X < LEFT_COL_MAX_X → 왼쪽 칼럼 / 이상 → 오른쪽 칼럼

    이 방식으로 아래 문제가 자동 해결됨:
    - 블록 단위 추출 시 PDF 스트림 순서로 인한 좌우 뒤바뀜
    - 두 칼럼 헤더가 같은 블록에 묶이는 문제 (업종별/테마별 등)
    하드코딩 없이 X·Y 좌표만으로 처리.
    """
    words = page.get_text("words", sort=True)

    left_rows: list[str] = []
    right_rows: list[str] = []

    current_y: float | None = None
    row_left: list[tuple[float, str]] = []
    row_right: list[tuple[float, str]] = []

    def _flush():
        if row_left:
            left_rows.append(" ".join(t for _, t in sorted(row_left)))
        if row_right:
            right_rows.append(" ".join(t for _, t in sorted(row_right)))

    for w in words:
        x0, y0, x1, y1, text, *_ = w
        text = text.strip()
        if not text:
            continue

        # 새 행 시작 판단
        if current_y is None or abs(y0 - current_y) > ROW_TOLERANCE:
            _flush()
            row_left = []
            row_right = []
            current_y = y0

        if x0 < LEFT_COL_MAX_X:
            row_left.append((x0, text))
        else:
            row_right.append((x0, text))

    _flush()

    left_text = "\n".join(left_rows)
    right_text = "\n".join(right_rows)

    combined = left_text
    if right_text:
        combined += "\n\n[우측 데이터]\n" + right_text

    return combined


def _extract_tables_camelot(file_path: str) -> dict[int, list[TableContent]]:
    """camelot stream 모드로 표 추출. 실패해도 빈 dict 반환."""
    try:
        import camelot
    except ImportError:
        logger.warning("camelot-py 미설치. 표 추출 건너뜀.")
        return {}

    tables_by_page: dict[int, list[TableContent]] = {}
    flavor = "lattice" if _is_ghostscript_available() else "stream"
    logger.info(f"camelot 모드: {flavor}")

    try:
        # stream 모드 파라미터
        kwargs: dict = {}
        if flavor == "stream":
            kwargs = {"edge_tol": 50, "row_tol": 5}
        else:
            kwargs = {"line_scale": 40}

        raw_tables = camelot.read_pdf(
            file_path,
            flavor=flavor,
            pages="all",
            suppress_stdout=True,
            **kwargs,
        )

        for tbl in raw_tables:
            page_idx = tbl.page - 1
            df: pd.DataFrame = tbl.df.copy()

            # 완전히 빈 표 제외
            non_empty_cells = df.map(lambda c: bool(str(c).strip())).sum().sum()
            if non_empty_cells < 4:
                continue

            # 첫 행 헤더 승격
            if df.shape[0] > 1:
                first_row = df.iloc[0].tolist()
                if any(str(c).strip() for c in first_row):
                    df.columns = [str(c).strip() for c in first_row]
                    df = df[1:].reset_index(drop=True)

            md = df.to_markdown(index=False)

            tables_by_page.setdefault(page_idx, []).append(
                TableContent(
                    page_number=page_idx,
                    markdown=md,
                    dataframe=df,
                    accuracy=round(tbl.accuracy, 2) if flavor == "lattice" else None,
                )
            )

        total = sum(len(v) for v in tables_by_page.values())
        logger.info(f"camelot 추출 표 수: {total}")

    except Exception as e:
        logger.warning(f"camelot 추출 실패 ({e}). 텍스트 데이터만 사용.")

    return tables_by_page


def _extract_report_date(text: str) -> str:
    m = re.search(r"(\d{4}\.\d{2}\.\d{2})", text)
    return m.group(1) if m else ""


def _extract_author(text: str) -> str:
    m = re.search(r"([가-힣]{2,4})\s*연구원", text)
    return m.group(1) if m else ""


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """DS Daily PDF 파싱 메인 함수"""
    file_path = str(file_path)

    if not os.path.exists(file_path):
        raise ParserException(f"파일을 찾을 수 없습니다: {file_path}")

    try:
        doc = fitz.open(file_path)
        page_count = len(doc)

        # ── 섹션 헤더 자동 감지 (폰트 속성 기반) ─────────────────────
        all_section_headers: list[str] = []
        for page in doc:
            headers = extract_section_headers(page)
            all_section_headers.extend(headers)
        all_section_headers = list(dict.fromkeys(all_section_headers))  # 중복 제거, 순서 유지
        logger.info(f"자동 감지 섹션 헤더: {all_section_headers}")

        # ── 텍스트 추출 (fitz words) ──────────────────────────────────
        pages: list[PageContent] = []
        full_text_parts = []

        for i, page in enumerate(doc):
            text = _extract_text_fitz(page)
            full_text_parts.append(text)
            pages.append(
                PageContent(
                    page_number=i,
                    text=text,
                    tables=[],
                    metadata={"section_headers": all_section_headers},
                )
            )

        # ── 차트 이미지 처리 (모든 차트) ──────────────────────────────
        try:
            from services.chart_service import extract_all_charts

            for i, page in enumerate(doc):
                chart_text = extract_all_charts(page)
                if chart_text:
                    pages[i].text += "\n\n" + chart_text
                    full_text_parts[i] += "\n\n" + chart_text
                    logger.info(f"페이지 {i}: 차트 이미지 추출 완료")
        except Exception as e:
            logger.warning(f"차트 추출 실패 (텍스트 데이터만 사용): {e}")

        doc.close()
        full_text = "\n\n".join(full_text_parts)

        # ── 메타데이터 ─────────────────────────────────────────────────
        metadata = DocumentMetadata(
            filename=Path(file_path).name,
            page_count=page_count,
            file_size=os.path.getsize(file_path),
            report_date=_extract_report_date(full_text),
            author=_extract_author(full_text),
        )

        # ── 표 추출 (camelot) ──────────────────────────────────────────
        tables_by_page = _extract_tables_camelot(file_path)
        for i, page in enumerate(pages):
            page.tables = tables_by_page.get(i, [])

        total_tables = sum(len(p.tables) for p in pages)
        text_lines = len([l for l in full_text.split("\n") if l.strip()])
        logger.info(
            f"파싱 완료 | {metadata.filename} | "
            f"페이지 {page_count} | 텍스트 {text_lines}줄 | 표 {total_tables}개"
        )
        return ParsedDocument(metadata=metadata, pages=pages)

    except ParserException:
        raise
    except Exception as e:
        raise ParserException(f"PDF 파싱 중 오류: {e}") from e
