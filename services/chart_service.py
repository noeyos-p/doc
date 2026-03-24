"""
차트 이미지 처리 서비스 (LLM 미사용, 순수 이미지 처리)
- 라인 차트: 색상별 라인 추적 → 축 라벨 기반 값 변환
- 바 차트: 색상 필터링 → 바 길이 측정 → 캘리브레이션
- 스파크라인: 추세 방향 판별

대상 차트:
1. KOSPI 12MF PER & EPS (라인)
2. KOSPI 목표주가 상∙하향 종목 수 (바+라인)
3. 신용융자잔고 & 고객예탁금 (라인)
4. 투자자별 52주 누적 순매수액 추이 (라인 3개)
5. 외국인/기관 순매수 상위 수익률(%) (바)
6. 업종별 12MF EPS/PER (스파크라인)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import fitz
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

DPI = 300
SCALE = DPI / 72.0


# ═══════════════════════════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class LineChartData:
    """라인 차트 추출 결과"""
    title: str
    series: dict[str, list[tuple[str, float]]]  # {시리즈명: [(시점, 값), ...]}
    latest: dict[str, float] = field(default_factory=dict)  # {시리즈명: 최신값}


@dataclass
class BarChartEntry:
    company: str
    amount: float | None = None
    return_pct: float | None = None
    return_direction: str = ""
    bar_pixel_len: int = 0
    calibrated: bool = False


@dataclass
class BarChartResult:
    table_name: str
    entries: list[BarChartEntry] = field(default_factory=list)
    calibration_source: str = ""


@dataclass
class SparklineTrend:
    sector: str
    eps_trend: str = "?"    # ↑/↓/→
    per_trend: str = "?"


# ═══════════════════════════════════════════════════════════════════════
# 색상 감지 함수
# ═══════════════════════════════════════════════════════════════════════
def _is_blue_line(r, g, b):
    """파란 라인 (PER, 신용융자, 기관)"""
    return int(b) > 130 and (int(b) - int(r)) > 30 and int(g) > 80 and int(r) < 160


def _is_black_line(r, g, b):
    """검은 라인 (EPS, 고객예탁금, 외국인)"""
    return int(r) < 70 and int(g) < 70 and int(b) < 70


def _is_gray_line(r, g, b):
    """회색 라인 (개인)"""
    return 100 < int(r) < 190 and 100 < int(g) < 190 and 100 < int(b) < 190 and abs(int(r) - int(g)) < 20 and abs(int(g) - int(b)) < 20


def _is_red_bar(r, g, b):
    """분홍/빨강 바 (양수)"""
    return int(r) > 200 and (int(r) - int(g)) > 20 and (int(r) - int(b)) > 20


def _is_blue_bar(r, g, b):
    """파랑 바 (음수)"""
    return int(b) > 140 and (int(b) - int(r)) > 25 and int(g) > int(r)


# ═══════════════════════════════════════════════════════════════════════
# 범용 라인 추적
# ═══════════════════════════════════════════════════════════════════════
def _trace_line(
    plot: np.ndarray,
    color_fn,
    y_top_val: float,
    y_bot_val: float,
    start_from_bottom: bool = False,
) -> dict[int, float]:
    """
    플롯 영역에서 특정 색상의 라인을 추적하여 값으로 변환.

    Args:
        plot: 크롭된 플롯 영역 이미지 (numpy)
        color_fn: 색상 판별 함수 (r, g, b) -> bool
        y_top_val: 플롯 상단의 Y축 값
        y_bot_val: 플롯 하단의 Y축 값
        start_from_bottom: True면 초기 탐색을 하단에서 시작

    Returns: {x_pixel: value}
    """
    h, w = plot.shape[:2]

    # 축선 제거: 전체 너비/높이 40% 이상을 차지하는 행/열은 축선
    col_mask = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            if color_fn(*plot[y, x]):
                col_mask[y, x] = True

    # 수평 축선 제거
    for y in range(h):
        if np.sum(col_mask[y, :]) > w * 0.4:
            col_mask[max(0, y - 2):min(h, y + 3), :] = False
    # 수직 축선 제거
    for x in range(w):
        if np.sum(col_mask[:, x]) > h * 0.4:
            col_mask[:, max(0, x - 2):min(w, x + 3)] = False

    # 연속성 추적
    line_points: dict[int, int] = {}
    prev_y = None

    for x in range(w):
        col_ys = np.where(col_mask[:, x])[0]
        if len(col_ys) == 0:
            continue

        # 클러스터링
        clusters: list[list[int]] = []
        cur = [int(col_ys[0])]
        for j in range(1, len(col_ys)):
            if col_ys[j] - col_ys[j - 1] <= 5:
                cur.append(int(col_ys[j]))
            else:
                clusters.append(cur)
                cur = [int(col_ys[j])]
        clusters.append(cur)

        if prev_y is None:
            # 첫 포인트: 시작 위치 전략
            if start_from_bottom:
                best = max(clusters, key=lambda c: np.median(c))
            else:
                # 가장 큰 클러스터
                best = max(clusters, key=len)
        else:
            # 이전과 가장 가까운 클러스터
            best = min(clusters, key=lambda c: abs(np.median(c) - prev_y))
            if abs(np.median(best) - prev_y) > 30:
                continue

        y_val = int(np.median(best))
        line_points[x] = y_val
        prev_y = y_val

    # 픽셀 → 값 변환
    result: dict[int, float] = {}
    for x, y_px in line_points.items():
        val = y_top_val - (y_px / h) * (y_top_val - y_bot_val)
        result[x] = round(val, 2)

    return result


def _sample_line(
    line_data: dict[int, float],
    total_width: int,
    dates: list[str],
) -> list[tuple[str, float]]:
    """라인 데이터를 날짜 포인트로 샘플링"""
    samples: list[tuple[str, float]] = []
    if not line_data:
        return samples

    for i, date in enumerate(dates):
        x = int(i / max(1, len(dates) - 1) * (total_width - 1))
        # 가장 가까운 x 찾기
        closest = min(line_data.keys(), key=lambda k: abs(k - x))
        if abs(closest - x) < 15:
            samples.append((date, line_data[closest]))

    return samples


# ═══════════════════════════════════════════════════════════════════════
# 1. KOSPI 12MF PER & EPS
# ═══════════════════════════════════════════════════════════════════════
def _extract_per_eps(page: fitz.Page, img: np.ndarray) -> LineChartData | None:
    """KOSPI 12MF PER & EPS 라인 차트 추출"""
    # 플롯 영역: x=344~436pt, y=170~245pt
    plot = img[int(170 * SCALE):int(245 * SCALE), int(344 * SCALE):int(436 * SCALE)]
    h, w = plot.shape[:2]

    if h < 10 or w < 10:
        return None

    # PER (파란선): Y축 8~12배
    per_raw = _trace_line(plot, _is_blue_line, 12.0, 8.0)
    # EPS (검은선): Y축 635~275원
    eps_raw = _trace_line(plot, _is_black_line, 635.0, 275.0, start_from_bottom=True)

    dates = ["25/01", "25/03", "25/05", "25/07", "25/09", "25/11", "26/01"]
    per_series = _sample_line(per_raw, w, dates)
    eps_series = _sample_line(eps_raw, w, dates)

    per_latest = per_raw[max(per_raw.keys())] if per_raw else 0
    eps_latest = eps_raw[max(eps_raw.keys())] if eps_raw else 0

    logger.info(f"PER/EPS 추출: PER={per_latest:.2f}배, EPS={eps_latest:.0f}원")

    return LineChartData(
        title="KOSPI 12MF PER & EPS",
        series={"P/E(Fwd.12M)(배)": per_series, "EPS(Fwd.12M)(원)": eps_series},
        latest={"P/E(Fwd.12M)(배)": per_latest, "EPS(Fwd.12M)(원)": eps_latest},
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. 신용융자잔고 & 고객예탁금
# ═══════════════════════════════════════════════════════════════════════
def _extract_credit_deposit(page: fitz.Page, img: np.ndarray) -> LineChartData | None:
    """신용융자잔고 & 고객예탁금 라인 차트 추출"""
    # 플롯 영역: x=348~432pt, y=280~345pt
    plot = img[int(280 * SCALE):int(345 * SCALE), int(348 * SCALE):int(432 * SCALE)]
    h, w = plot.shape[:2]

    if h < 10 or w < 10:
        return None

    # 신용융자잔고 (파란선): Y축 23000~8000 (십억원)
    credit_raw = _trace_line(plot, _is_blue_line, 23000, 8000)
    # 고객예탁금 (검은선): Y축 140000~50000 (십억원)
    deposit_raw = _trace_line(plot, _is_black_line, 140000, 50000, start_from_bottom=True)

    dates = ["25/01", "25/04", "25/07", "25/10", "26/01"]
    credit_series = _sample_line(credit_raw, w, dates)
    deposit_series = _sample_line(deposit_raw, w, dates)

    credit_latest = credit_raw[max(credit_raw.keys())] if credit_raw else 0
    deposit_latest = deposit_raw[max(deposit_raw.keys())] if deposit_raw else 0

    logger.info(f"신용융자/예탁금: 신용={credit_latest:.0f}, 예탁금={deposit_latest:.0f}")

    return LineChartData(
        title="신용융자잔고 & 고객예탁금",
        series={
            "신용융자잔고(십억원)": credit_series,
            "고객예탁금(십억원)": deposit_series,
        },
        latest={
            "신용융자잔고(십억원)": credit_latest,
            "고객예탁금(십억원)": deposit_latest,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. 투자자별 52주 누적 순매수액 추이
# ═══════════════════════════════════════════════════════════════════════
def _extract_investor_flow(page: fitz.Page, img: np.ndarray) -> LineChartData | None:
    """투자자별 52주 누적 순매수액 라인 차트 추출"""
    # 플롯 영역: x=477~570pt, y=280~345pt
    plot = img[int(280 * SCALE):int(343 * SCALE), int(477 * SCALE):int(570 * SCALE)]
    h, w = plot.shape[:2]

    if h < 10 or w < 10:
        return None

    # 기관 (파란선): Y축 30~-45 (조원)
    inst_raw = _trace_line(plot, _is_blue_line, 30, -45)
    # 외국인 (검은선): Y축 30~-45 (조원)
    foreign_raw = _trace_line(plot, _is_black_line, 30, -45, start_from_bottom=True)
    # 개인 (회색선): Y축 30~-45 (조원)
    retail_raw = _trace_line(plot, _is_gray_line, 30, -45)

    dates = ["25/01", "25/04", "25/07", "25/10", "26/01"]
    inst_series = _sample_line(inst_raw, w, dates)
    foreign_series = _sample_line(foreign_raw, w, dates)
    retail_series = _sample_line(retail_raw, w, dates)

    inst_latest = inst_raw[max(inst_raw.keys())] if inst_raw else 0
    foreign_latest = foreign_raw[max(foreign_raw.keys())] if foreign_raw else 0
    retail_latest = retail_raw[max(retail_raw.keys())] if retail_raw else 0

    logger.info(f"투자자별: 기관={inst_latest:.1f}, 외국인={foreign_latest:.1f}, 개인={retail_latest:.1f}")

    return LineChartData(
        title="투자자별 52주 누적 순매수액 추이",
        series={
            "기관(조원)": inst_series,
            "외국인(조원)": foreign_series,
            "개인(조원)": retail_series,
        },
        latest={
            "기관(조원)": inst_latest,
            "외국인(조원)": foreign_latest,
            "개인(조원)": retail_latest,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# 4. 목표주가 상∙하향 종목 수
# ═══════════════════════════════════════════════════════════════════════
def _extract_target_price(page: fitz.Page, img: np.ndarray) -> LineChartData | None:
    """KOSPI 목표주가 상∙하향 종목 수 (바+라인) 추출"""
    # 플롯 영역: x=477~560pt, y=170~240pt
    plot = img[int(172 * SCALE):int(240 * SCALE), int(477 * SCALE):int(560 * SCALE)]
    h, w = plot.shape[:2]

    if h < 10 or w < 10:
        return None

    # 상향-하향 차이 라인 (검은선): 우축 Y축 120~-40 (개)
    diff_raw = _trace_line(plot, _is_black_line, 120, -40)

    # 바 차트: 상향(분홍), 하향(파랑) 바 높이 측정
    # 0 기준선 찾기 (y축 값 0 → 픽셀 위치)
    # Y축: top=180개, bottom=-120개 (바 차트 좌축)
    # 0은 중간 쯤: 180/(180+120) = 0.6 from top → y_zero = h * 0.6
    # 하지만 좌축은 180, 120, 60, 0 ... 즉 top=180, bottom은 차트에서 보면 120이 음수 쪽
    # 좌축 스케일: 180(top) ~ 0(중간 어딘가)
    # 바 = 위에서 아래(하향), 아래에서 위(상향) 양쪽으로 나감

    # 분홍 바(상향)와 파랑 바(하향)의 높이를 날짜별로 측정
    up_heights: list[tuple[int, int]] = []  # (x, height_px)
    down_heights: list[tuple[int, int]] = []

    for x in range(w):
        red_ys = [y for y in range(h) if _is_red_bar(*plot[y, x])]
        blue_ys = [y for y in range(h) if _is_blue_bar(*plot[y, x])]
        if red_ys:
            up_heights.append((x, len(red_ys)))
        if blue_ys:
            down_heights.append((x, len(blue_ys)))

    # 좌축 스케일: 180(top) ~ 0 (0선은 차트 내 어딘가)
    # 바 높이를 값으로 변환: 좌축에서 눈금 간격 = 60개 단위
    # top=180, 그 다음 120, 60, 0
    # top -> y=0, 0 -> y = h * (180/180) 아니고...
    # 축 라벨: 180(top), 120, 60, 0 (중심), 60, 120 (bottom)
    # 이건 복잡하므로, 차이 라인(검은선) 값만 추출

    dates = ["25/01", "25/04", "25/07", "25/10", "26/01"]
    diff_series = _sample_line(diff_raw, w, dates)
    diff_latest = diff_raw[max(diff_raw.keys())] if diff_raw else 0

    logger.info(f"목표주가 상-하향 차이: 최신={diff_latest:.0f}개")

    return LineChartData(
        title="KOSPI 목표주가 상∙하향 종목 수",
        series={"목표주가상향-하향(개)": diff_series},
        latest={"목표주가상향-하향(개)": diff_latest},
    )


# ═══════════════════════════════════════════════════════════════════════
# 5. 외국인/기관 순매수 상위 수익률(%) 바 차트
# ═══════════════════════════════════════════════════════════════════════
BAR_TABLE_CONFIGS = [
    {
        "name": "외국인 순매수 상위",
        "name_x": (330, 395), "val_x": (390, 415),
        "bar_crop_x": (395, 460), "data_y": (373, 472),
    },
    {
        "name": "기관 순매수 상위",
        "name_x": (455, 510), "val_x": (510, 535),
        "bar_crop_x": (525, 590), "data_y": (373, 472),
    },
]


def _detect_bars(crop: np.ndarray) -> tuple[list[dict], int]:
    """크롭된 바 차트 영역에서 바 감지"""
    h, w = crop.shape[:2]

    col_dark = np.zeros(w)
    for x in range(w):
        col = crop[:, x]
        col_dark[x] = np.sum((col[:, 0] < 80) & (col[:, 1] < 80) & (col[:, 2] < 80))

    mid_region = col_dark[w // 4: 3 * w // 4]
    candidates = np.where(mid_region > h * 0.12)[0] + w // 4
    center_x = int(np.median(candidates)) if len(candidates) > 0 else w // 2

    bars: list[dict] = []
    current: dict | None = None

    for y in range(h):
        row = crop[y]
        reds = [x for x in range(w) if _is_red_bar(*row[x])]
        blues = [x for x in range(w) if _is_blue_bar(*row[x])]

        if len(reds) > 2 and len(reds) >= len(blues):
            color, xs = "RED", reds
        elif len(blues) > 2:
            color, xs = "BLUE", blues
        else:
            color, xs = None, []

        if color:
            if current and current["color"] == color and (y - current["ye"]) <= 3:
                current["ye"] = y
                current["xmin"] = min(current["xmin"], min(xs))
                current["xmax"] = max(current["xmax"], max(xs))
            else:
                if current:
                    bars.append(current)
                current = {"color": color, "ys": y, "ye": y,
                           "xmin": min(xs), "xmax": max(xs)}
        else:
            if current:
                bars.append(current)
                current = None
    if current:
        bars.append(current)

    bars = [b for b in bars if (b["ye"] - b["ys"]) >= 5]

    for b in bars:
        b["y_mid"] = (b["ys"] + b["ye"]) / 2
        if b["color"] == "RED":
            b["bar_len"] = max(0, b["xmax"] - center_x)
            b["direction"] = "+"
        else:
            b["bar_len"] = max(0, center_x - b["xmin"])
            b["direction"] = "-"

    return bars, center_x


def _extract_table_text(page, name_x, val_x, y_range):
    blocks = page.get_text("dict")["blocks"]
    skip_kw = {"기업명", "금액", "수익률", "외국인", "기관", "순매수", "상위", "(%)", "(십억원)"}
    names, vals = [], []

    for b in blocks:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                bbox = span["bbox"]
                text = span["text"].strip()
                if not text or text in skip_kw:
                    continue
                y_mid = (bbox[1] + bbox[3]) / 2
                if not (y_range[0] < y_mid < y_range[1]):
                    continue
                if name_x[0] < bbox[0] < name_x[1]:
                    try:
                        float(text)
                    except ValueError:
                        names.append((text, y_mid))
                if val_x[0] < bbox[0] < val_x[1]:
                    try:
                        vals.append((float(text), y_mid))
                    except ValueError:
                        pass

    names.sort(key=lambda x: x[1])
    vals.sort(key=lambda x: x[1])
    return names, vals


def _extract_daily_returns(page: fitz.Page) -> dict[str, float]:
    blocks = page.get_text("dict")["blocks"]
    entries, one_d_vals = [], []

    for b in blocks:
        if "lines" not in b:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                bbox = span["bbox"]
                text = span["text"].strip()
                if not text:
                    continue
                y_mid = (bbox[1] + bbox[3]) / 2
                if not (490 < y_mid < 585):
                    continue
                x0 = bbox[0]
                if (336 < x0 < 375) or (460 < x0 < 502):
                    try:
                        float(text)
                    except ValueError:
                        entries.append((text, y_mid))
                if (406 < x0 < 422) or (526 < x0 < 542):
                    try:
                        one_d_vals.append((float(text), y_mid))
                    except ValueError:
                        pass

    results = {}
    for name, name_y in entries:
        best_val, best_dist = None, 999
        for val, val_y in one_d_vals:
            d = abs(name_y - val_y)
            if d < best_dist and d < 5:
                best_dist, best_val = d, val
        if best_val is not None:
            results[name] = best_val
    return results


def _extract_bar_charts(page: fitz.Page, img: np.ndarray) -> list[BarChartResult]:
    daily_returns = _extract_daily_returns(page)
    results = []

    for cfg in BAR_TABLE_CONFIGS:
        names, vals = _extract_table_text(page, cfg["name_x"], cfg["val_x"], cfg["data_y"])
        if not names:
            continue

        y0, y1 = cfg["data_y"]
        x0, x1 = cfg["bar_crop_x"]
        crop = img[int(y0 * SCALE):int(y1 * SCALE), int(x0 * SCALE):int(x1 * SCALE)]

        bars, center_x = _detect_bars(crop)
        if not bars:
            continue

        entries = []
        for b in bars:
            bar_y_pt = y0 + b["y_mid"] / SCALE
            best_name, best_d = "?", 999.0
            for nm, yp in names:
                d = abs(bar_y_pt - yp)
                if d < best_d:
                    best_d, best_name = d, nm
            if best_d > 15:
                best_name = "?"

            amount, best_d = None, 999.0
            for v, yp in vals:
                d = abs(bar_y_pt - yp)
                if d < best_d:
                    best_d, amount = d, v
            if best_d > 15:
                amount = None

            entries.append(BarChartEntry(
                company=best_name, amount=amount,
                return_direction=b["direction"], bar_pixel_len=b["bar_len"],
            ))

        # 캘리브레이션
        cal_points = [(e.company, daily_returns[e.company], e.bar_pixel_len)
                      for e in entries if e.company in daily_returns and e.bar_pixel_len > 0]
        cal_source = ""
        if cal_points:
            scales = [p / px for _, p, px in cal_points if px > 0]
            avg_s = sum(scales) / len(scales)
            for e in entries:
                if e.bar_pixel_len > 0:
                    raw = e.bar_pixel_len * avg_s
                    e.return_pct = round(-raw if e.return_direction == "-" else raw, 1)
                    e.calibrated = True
            cal_source = ", ".join(n for n, _, _ in cal_points)

        results.append(BarChartResult(
            table_name=cfg["name"], entries=entries, calibration_source=cal_source,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════
# 6. 업종별 12MF EPS/PER 스파크라인
# ═══════════════════════════════════════════════════════════════════════
def _extract_sparklines(page: fitz.Page, img: np.ndarray) -> list[SparklineTrend]:
    """업종별 12MF EPS/PER 스파크라인 추세 추출"""
    words = page.get_text("words", sort=True)

    # 업종명 수집 (x<65, y=610~770)
    raw_sectors = []
    for w in words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        if 610 < y0 < 770 and x0 < 65:
            raw_sectors.append((text, y0, y1))

    # 같은 행 병합
    sectors = []
    i = 0
    while i < len(raw_sectors):
        name = raw_sectors[i][0]
        y_top = raw_sectors[i][1]
        y_bot = raw_sectors[i][2]
        while i + 1 < len(raw_sectors) and abs(raw_sectors[i + 1][1] - y_top) < 3:
            i += 1
            name += raw_sectors[i][0]
            y_bot = max(y_bot, raw_sectors[i][2])
        sectors.append((name, y_top - 1, y_bot + 1))
        i += 1

    def get_trend(x_start_pt, x_end_pt, y_start_pt, y_end_pt) -> str:
        crop = img[
            int(y_start_pt * SCALE):int(y_end_pt * SCALE),
            int(x_start_pt * SCALE):int(x_end_pt * SCALE),
        ]
        h, w = crop.shape[:2]
        if h < 3 or w < 3:
            return "?"
        dark = (crop[:, :, 0].astype(int) + crop[:, :, 1].astype(int) + crop[:, :, 2].astype(int)) < 400
        left_ys = np.where(dark[:, :w // 4].any(axis=1))[0]
        right_ys = np.where(dark[:, -w // 4:].any(axis=1))[0]
        if len(left_ys) < 2 or len(right_ys) < 2:
            return "?"
        diff = float(np.median(left_ys) - np.median(right_ys))
        threshold = h * 0.15
        if diff > threshold:
            return "↑"
        elif diff < -threshold:
            return "↓"
        return "→"

    results = []
    for name, y_top, y_bot in sectors:
        eps_t = get_trend(88, 130, y_top, y_bot)
        per_t = get_trend(140, 185, y_top, y_bot)
        results.append(SparklineTrend(sector=name, eps_trend=eps_t, per_trend=per_t))

    return results


# ═══════════════════════════════════════════════════════════════════════
# 텍스트 변환 (파싱 결과 삽입용)
# ═══════════════════════════════════════════════════════════════════════
def _line_chart_to_text(data: LineChartData) -> str:
    lines = [f"[{data.title}]"]

    # 최신값 먼저
    latest_parts = [f"{k}: {v:.2f}" if abs(v) < 100 else f"{k}: {v:,.0f}"
                    for k, v in data.latest.items()]
    lines.append(f"최신값: {', '.join(latest_parts)}")

    # 시계열
    for series_name, points in data.series.items():
        lines.append(f"  {series_name}:")
        for date, val in points:
            if abs(val) < 100:
                lines.append(f"    {date}: {val:.2f}")
            else:
                lines.append(f"    {date}: {val:,.0f}")

    return "\n".join(lines)


def _bar_charts_to_text(results: list[BarChartResult]) -> str:
    parts = []
    for r in results:
        lines = [f"[{r.table_name} 수익률(%)]"]
        if r.calibration_source:
            lines.append(f"(추정치, 캘리브레이션 기준: {r.calibration_source})")
        for e in r.entries:
            amt = f"순매수 {e.amount}십억원" if e.amount else ""
            if e.return_pct is not None:
                pct = f"수익률 약 {e.return_pct:+.1f}%"
            else:
                pct = f"수익률 {'양(+)' if e.return_direction == '+' else '음(-)'}"
            lines.append(f"  {e.company}: {amt}, {pct}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _sparklines_to_text(trends: list[SparklineTrend]) -> str:
    lines = [
        "[업종별 12MF EPS/PER 추세]",
        "(참고: 업종별 기간 수익률 테이블에서 12MF EPS, 12MF PER 컬럼은 스파크라인 차트임.",
        " 숫자 컬럼은 1D, 1W, 1M, 3M 수익률(%) 순서이며 12MF EPS/PER 값이 아님)",
    ]
    for t in trends:
        lines.append(f"  {t.sector}: 12MF EPS 추세 {t.eps_trend}, 12MF PER 추세 {t.per_trend}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 메인 진입점
# ═══════════════════════════════════════════════════════════════════════
def extract_all_charts(page: fitz.Page) -> str:
    """
    PDF 페이지에서 모든 차트를 이미지 처리로 추출 → 텍스트 반환.
    parser_service.py에서 호출.
    """
    mat = fitz.Matrix(SCALE, SCALE)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    text_parts: list[str] = []

    # 1. KOSPI 12MF PER & EPS
    try:
        per_eps = _extract_per_eps(page, img)
        if per_eps:
            text_parts.append(_line_chart_to_text(per_eps))
    except Exception as e:
        logger.warning(f"PER/EPS 추출 실패: {e}")

    # 2. 목표주가 상∙하향
    try:
        target = _extract_target_price(page, img)
        if target:
            text_parts.append(_line_chart_to_text(target))
    except Exception as e:
        logger.warning(f"목표주가 추출 실패: {e}")

    # 3. 신용융자잔고 & 고객예탁금
    try:
        credit = _extract_credit_deposit(page, img)
        if credit:
            text_parts.append(_line_chart_to_text(credit))
    except Exception as e:
        logger.warning(f"신용융자잔고 추출 실패: {e}")

    # 4. 투자자별 52주 순매수
    try:
        investor = _extract_investor_flow(page, img)
        if investor:
            text_parts.append(_line_chart_to_text(investor))
    except Exception as e:
        logger.warning(f"투자자별 순매수 추출 실패: {e}")

    # 5. 외국인/기관 순매수 수익률 바 차트
    try:
        bar_results = _extract_bar_charts(page, img)
        if bar_results:
            text_parts.append(_bar_charts_to_text(bar_results))
    except Exception as e:
        logger.warning(f"수익률 바 차트 추출 실패: {e}")

    # 6. 업종별 스파크라인
    try:
        sparklines = _extract_sparklines(page, img)
        if sparklines:
            text_parts.append(_sparklines_to_text(sparklines))
    except Exception as e:
        logger.warning(f"스파크라인 추출 실패: {e}")

    total = len(text_parts)
    if total:
        logger.info(f"차트 추출 완료: {total}개 차트")

    return "\n\n".join(text_parts)
