"""
DS Daily 챗봇 - Streamlit UI
실행: streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

# ── 페이지 설정 ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DS Daily 챗봇",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .source-box {
        background: #f0f4f8;
        border-left: 3px solid #4A90D9;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.82em;
        color: #555;
        margin-top: 6px;
    }
    .status-ok  { color: #28a745; font-weight: 600; }
    .status-err { color: #dc3545; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── 세션 초기화 ────────────────────────────────────────────────────────
def _init_session():
    defaults = {
        "messages": [],          # {"role": "user"|"assistant", "content": str}
        "sources": [],           # 마지막 답변의 출처
        "collection_name": None, # ChromaDB 컬렉션명
        "doc_info": {},          # 문서 메타데이터
        "processing_log": [],    # 처리 로그
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()


# ── 사이드바 ───────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 리포트 업로드")
    st.caption("DS투자증권 DS Daily PDF를 업로드하세요.")

    uploaded = st.file_uploader(
        "PDF 파일 선택",
        type=["pdf"],
        label_visibility="collapsed",
    )

    process_btn = st.button(
        "🔄 리포트 처리",
        use_container_width=True,
        disabled=(uploaded is None),
        type="primary",
    )

    if process_btn and uploaded:
        # ── 처리 파이프라인 ──────────────────────────────────────────
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        log = []

        with st.status("리포트 처리 중...", expanded=True) as status:
            # Step 1: 파싱
            st.write("📑 PDF 파싱 중...")
            try:
                from services.parser_service import parse_pdf
                parsed = parse_pdf(tmp_path)
                st.write(
                    f"✅ 파싱 완료 | {parsed.metadata.page_count}페이지 | "
                    f"표 {sum(len(p.tables) for p in parsed.pages)}개"
                )
                log.append("파싱 완료")
            except Exception as e:
                st.error(f"❌ 파싱 실패: {e}")
                status.update(label="처리 실패", state="error")
                st.stop()

            # Step 2: 전처리
            st.write("🧹 전처리 중...")
            try:
                from services.preprocess_service import preprocess_document
                parsed, noise_logs = preprocess_document(parsed)
                st.write(f"✅ 전처리 완료 | 노이즈 제거: {len(noise_logs)}건")
                log.extend(noise_logs)
            except Exception as e:
                st.error(f"❌ 전처리 실패: {e}")
                status.update(label="처리 실패", state="error")
                st.stop()

            # Step 3: 청킹
            st.write("✂️ 청킹 중...")
            try:
                from services.chunk_service import chunk_document
                chunks = chunk_document(parsed)
                text_n = sum(1 for c in chunks if c.chunk_type == "text")
                table_n = sum(1 for c in chunks if c.chunk_type == "table")
                st.write(f"✅ 청킹 완료 | 텍스트 {text_n}개 + 표 {table_n}개")
            except Exception as e:
                st.error(f"❌ 청킹 실패: {e}")
                status.update(label="처리 실패", state="error")
                st.stop()

            # Step 4: 벡터 DB
            st.write("💾 벡터 DB 저장 중...")
            try:
                from services.vector_service import add_chunks
                col_name = add_chunks(chunks, parsed.metadata.filename)
                st.write(f"✅ 벡터 DB 저장 완료 | 컬렉션: {col_name}")
            except Exception as e:
                st.error(f"❌ 벡터 DB 저장 실패: {e}")
                status.update(label="처리 실패", state="error")
                st.stop()

            status.update(label="✅ 처리 완료!", state="complete", expanded=False)

        # 세션 업데이트
        st.session_state.collection_name = col_name
        st.session_state.doc_info = {
            "파일명": parsed.metadata.filename,
            "날짜": parsed.metadata.report_date,
            "연구원": parsed.metadata.author,
            "페이지": parsed.metadata.page_count,
        }
        st.session_state.messages = []
        st.session_state.processing_log = log
        st.rerun()

    # ── 문서 정보 표시 ───────────────────────────────────────────────
    if st.session_state.doc_info:
        st.divider()
        st.subheader("📌 문서 정보")
        for k, v in st.session_state.doc_info.items():
            if v:
                st.write(f"**{k}:** {v}")

    # ── 대화 초기화 버튼 ─────────────────────────────────────────────
    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources = []
            st.rerun()


# ── 메인 챗 영역 ───────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📈 DS Daily 챗봇")
with col2:
    st.caption("DS투자증권 시황분석 리포트 Q&A")

st.divider()

# 리포트 미업로드 상태
if not st.session_state.collection_name:
    st.info(
        "👈 왼쪽 사이드바에서 DS Daily PDF를 업로드하고 **리포트 처리** 버튼을 눌러주세요.",
        icon="📂",
    )
    st.stop()

# ── 예시 질문 버튼 ────────────────────────────────────────────────────
example_questions = [
    "오늘 코스피 시황을 요약해줘",
    "외국인 순매수 상위 종목은?",
    "반도체 업종 이슈가 뭐야?",
    "오늘의 주요 이벤트는?",
    "업종별 수익률 알려줘",
]

with st.expander("💡 예시 질문", expanded=False):
    cols = st.columns(3)
    for i, q in enumerate(example_questions):
        if cols[i % 3].button(q, key=f"example_{i}", use_container_width=True):
            st.session_state["preset_question"] = q

# ── 대화 기록 출력 ────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 어시스턴트 마지막 메시지에 출처 표시
        if (
            msg["role"] == "assistant"
            and i == len(st.session_state.messages) - 1
            and st.session_state.sources
        ):
            st.markdown(
                f'<div class="source-box">📎 <b>출처</b><br>{st.session_state.sources}</div>',
                unsafe_allow_html=True,
            )

# ── 사용자 입력 ───────────────────────────────────────────────────────
preset = st.session_state.pop("preset_question", None)
user_input = st.chat_input("질문을 입력하세요... (예: 오늘 코스피 어때?)")
question = preset or user_input

if question:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 어시스턴트 응답 (스트리밍)
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_answer = ""
        sources_text = ""

        try:
            from services.chat_service import chat_stream

            # 이전 대화 기록 (role/content만 전달)
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            for token in chat_stream(
                question=question,
                collection_name=st.session_state.collection_name,
                chat_history=history,
            ):
                if isinstance(token, dict):
                    # 출처 메타데이터
                    sources_text = token.get("sources", "")
                else:
                    full_answer += token
                    answer_placeholder.markdown(full_answer + "▌")

            answer_placeholder.markdown(full_answer)

            if sources_text:
                st.markdown(
                    f'<div class="source-box">📎 <b>출처</b><br>{sources_text}</div>',
                    unsafe_allow_html=True,
                )

        except Exception as e:
            full_answer = f"⚠️ 오류가 발생했습니다: {e}"
            answer_placeholder.error(full_answer)

    # 대화 기록에 저장
    st.session_state.messages.append({"role": "assistant", "content": full_answer})
    st.session_state.sources = sources_text
