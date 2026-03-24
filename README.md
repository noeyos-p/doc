# Doc

그래프, 표, 이미지가 포함된 다양한 문서를 분석하고 질의응답할 수 있는 RAG 기반 AI 챗봇. PDF 문서를 업로드하면 텍스트, 테이블, 차트를 자동 추출하여 맥락 인식 Q&A를 제공합니다.

---

## 주요 기능

- **복합 문서 처리**: 텍스트, 테이블, 차트 이미지가 포함된 PDF 자동 파싱
- **RAG 기반 Q&A**: ChromaDB 시맨틱 검색 기반 맥락 인식 질의응답
- **스트리밍 응답**: OpenAI GPT-4o 실시간 토큰 스트리밍
- **출처 인용**: 답변 시 참조 문서 섹션 자동 표시
- **차트 분석**: 라인차트, 바차트, 스파크라인 이미지 분석
- **2단 레이아웃 처리**: 좌표 기반 좌/우 컬럼 텍스트 정확 추출
- **섹션 자동 감지**: 폰트 속성 기반 문서 구조 자동 인식

---

## 기술 스택

### Application
| 항목 | 기술 |
|------|------|
| UI Framework | Streamlit |
| Language | Python |
| LLM | OpenAI GPT-4o |
| Embeddings | text-embedding-3-small |
| Vector DB | ChromaDB |

### PDF Processing
| 항목 | 기술 |
|------|------|
| Text Extraction | PyMuPDF (fitz) |
| Table Extraction | Camelot |
| Data Processing | Pandas, NumPy |
| Configuration | Pydantic Settings |

---

## 프로젝트 구조

```
doc/
├── app.py                 # Streamlit 앱 엔트리포인트
├── requirements.txt       # Python 의존성
├── config/
│   └── settings.py        # 환경 설정 (API 키, DB 경로)
├── models/
│   └── schemas.py         # 데이터 클래스 (ParsedDocument, DocumentChunk 등)
├── services/
│   ├── parser_service.py  # PDF 파싱 (텍스트 + 테이블 + 차트)
│   ├── preprocess_service.py  # 노이즈 제거, 텍스트 정제
│   ├── chunk_service.py   # 섹션 기반 문서 청킹
│   ├── chart_service.py   # 차트 이미지 추출 및 분석
│   ├── vector_service.py  # ChromaDB 연동 (추가, 검색, 목록)
│   └── chat_service.py    # RAG 기반 스트리밍 채팅
└── utils/
    ├── logger.py          # 로깅 설정
    └── exceptions.py      # 커스텀 예외 클래스
```

---

## 로컬 개발 환경 설정

### 사전 요구사항
- Python 3.11+
- OpenAI API 키

### 1. 환경 설정

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 실행

```bash
streamlit run app.py
```

> Streamlit 앱이 `http://localhost:8501` 에서 실행됩니다.

### 처리 파이프라인
PDF 업로드 시 자동으로 실행됩니다:
1. PDF 파싱 (텍스트 + 테이블 + 차트)
2. 전처리 (노이즈/헤더/푸터 제거)
3. 섹션 기반 청킹
4. ChromaDB 벡터 저장
5. Q&A 준비 완료

---

## 환경 변수

### `.env`
| 변수 | 설명 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 키 |
| `OPENAI_MODEL` | LLM 모델 (기본: `gpt-4o`) |
| `OPENAI_EMBEDDING_MODEL` | 임베딩 모델 (기본: `text-embedding-3-small`) |
| `CHROMA_PERSIST_DIR` | 벡터 DB 경로 (기본: `./data/chroma`) |
| `LOG_LEVEL` | 로깅 레벨 (기본: `INFO`) |
