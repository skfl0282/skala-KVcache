# KV Cache 최적화 기술 다관점 평가 Agentic RAG

판교 9반 5조 · SKALA

# Subject
본 프로젝트는 KV cache 최적화 기술을 소프트웨어, 하드웨어 두 진영에서 선정하여,
시장·이해관계자·도메인 관점에서 평가하는 Agentic RAG를 개발하는 프로젝트임.

## Overview
- Objective : 하나의 기술을 복수 관점에서 비교 평가
- Method : Multi-Agent(Distributed) + Agentic RAG
- Tools : LangGraph, LangChain(`init_chat_model`), Chroma, TavilySearch

## Selected Technologies
- SW : **DeepSeek-V2 (MLA)** — 어텐션 메커니즘을 재설계해 별도 사후처리 없이
  생성 단계에서 KV Cache 크기를 93.3% 감축하는 구조적 특징을 가지며, 실제
  대규모 상용 서비스에 적용되어 기술 성숙도가 검증됨. 사후 압축 방식인
  양자화 계열(정밀도 손실·추가 연산 지연 한계) 대비 비교 평가 대상으로서
  타당성이 높다고 판단해 선정.
- HW : **ITME (Inference Tiered Memory Expansion with Disaggregated
  CXL-Hybrid Memories)** — SK hynix의 CXL 메모리 장치 등 실제 하드웨어로
  검증된 결과와 업계 공식 자료가 있어 신뢰도 높은 RAG 문서 풀을 구성할 수
  있다고 판단해 선정. (InfiniGen은 SW만 개선하는 기술이라 SW/HW 대비 구도에
  맞지 않아 제외, Scalable PNM은 시뮬레이션 기반 연구라 상용화/업계 자료
  확보가 어려워 제외)

## Features
- PDF 원문(DeepSeek-V2, ITME 논문) 기반 RAG 정보 추출 (기술 조사, 도메인 평가)
- 웹검색 기반 시장성 · 이해관계자 반응 조사 (RAG 미적용)
- 기술 조사 / 도메인 평가 단계에서 자료가 부족하면 `CorrectiveRAG`의
  `GradeDocuments` 그레이더 패턴으로 충분성을 판정하고, 부족하면 자동으로
  웹검색 보완 후 `data_limited`에 기록 → 최종 보고서 "6. 한계점"에 반영
- 관점별(시장성 / 이해관계자 / 도메인) Fan-out 병렬 평가 → 평가 종합 Fan-in
- 확증 편향 방지 전략 : 종합 단계에서 우열 판정 없이 관점 간 일치/상충
  지점을 병렬 서술하도록 프롬프트 설계, 보고서 REFERENCE에는 실제로
  활용한 자료만 기재하여 근거 추적 가능하도록 함

## Tech Stack
- Framework : LangGraph (`init_chat_model` 기반 LangChain 체인)
- LLM/Generator : `gpt-4.1-mini` (TODO: 팀 확정, 각 `agents/*.py`의 `MODEL_NAME` 수정)
- LLM/Judge : `gpt-4.1-mini` (충분성 그레이더용, TODO: 팀 확정)
- Retrieval : Chroma — `{Hit Rate@K}`, `{MRR}` (TODO: 측정 후 기재)
- Embedding : BGE-M3 (`rag/embeddings.py`) — 긴 문맥·다국어 지원, Dense/Sparse/
  Multi-vector Retrieval, MIT License 오픈소스 — 논문 중심 텍스트 검색
  적합성/구현 난이도/연산 자원/라이선스를 종합 고려해 선정

## Agents
설계서 A. Agent 정의 기준.

| Agent | 노드 함수 | RAG 여부 | 역할 |
|---|---|---|---|
| 기술 조사 에이전트 | `tech_research` | O | 원문 2건에서 기술 개요, 범위, 한계 추출 |
| 시장 평가 에이전트 | `market_eval` | X | 시장 규모, 상용화/채택 현황, 성장 전망 검색 |
| 이해관계자 평가 에이전트 | `stakeholder_eval` | X | 경쟁사 반응, 개발자 평가, 투자/업계 시각 검색 |
| 도메인 평가 에이전트 | `domain_eval` | O | 데이터센터·클라우드에서의 적용 적합성 평가 |
| 평가 종합 에이전트 | `synthesis` | X | 관점별 의견 종합 및 비교 (일치/상충 지점 서술) |
| 보고서 생성 에이전트 | `report_gen` | X | 단계별 내용을 연결해 평가 보고서 생성 |

## Architecture
```
START
  └─ select_technology (기술 선정)
       └─ tech_research (기술 조사 RAG)
            ├─[sufficient]──────────────────────────┐
            └─[insufficient]→ web_search_supplement_tech ┘
                                                        ├─ market_eval (웹검색)
                                                        ├─ stakeholder_eval (웹검색)
                                                        └─ domain_eval (RAG, 내부 보완 루프 포함)
                             ["market_eval","stakeholder_eval","domain_eval"] 조인
                                          └─ synthesis (fan-in)
                                               └─ report_gen
                                                    └─ END
```
원본 mermaid 설계도는 업로드된 설계산출물(`RAG-Design_판교-9반.pdf`) D절 참고.
`graph/build_graph.py` 상단 docstring에 설계서 대비 단순화한 지점
(도메인 평가의 충분성 체크를 별도 그래프 노드 대신
노드 내부 로직으로 처리한 이유)을 설명해 두었다.

## Directory Structure
```
├── data/
│   ├── raw/                # 원문 PDF (DeepSeek-V2, ITME 논문)
│   └── processed/          # Chroma 인덱스 캐시 (.cache/)
├── graph/
│   ├── state.py             # GraphState(TypedDict + Annotated 설명)
│   └── build_graph.py       # 노드/엣지 배선
├── agents/                  # Agent별 노드 함수 (담당자별로 분담)
│   ├── prompt_utils.py       # prompts/*.txt -> PromptTemplate 로더
│   ├── tech_selector.py      # select_technology
│   ├── tech_research.py      # tech_research, route_after_tech_research, web_search_supplement_tech
│   ├── market_eval.py        # market_eval
│   ├── stakeholder_eval.py   # stakeholder_eval
│   ├── domain_eval.py        # domain_eval
│   ├── synthesis.py          # synthesis
│   └── report_gen.py         # report_gen
├── rag/                     # RAG 공통 모듈 (실습자료 20-RAG/rag/ 구조를 따르되, retriever까지만 책임)
│   ├── embeddings.py          # create_bge_m3_embeddings()
│   ├── base.py                # RetrievalChain (ABC) - 원문 로딩~retriever 생성만 담당
│   └── pdf.py                 # PDFRetrievalChain, format_docs, build_tech_retrieval_chain
├── prompts/                 # 에이전트별 프롬프트 템플릿 (PromptTemplate.from_template)
├── outputs/                 # 평가 결과 저장 (report.md)
├── tests/
│   └── test_graph_smoke.py  # 그래프 배선 스모크 테스트
├── app.py                   # 실행 스크립트
├── requirements.txt
├── .env.example
└── README.md
```

## Usage
```bash
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY, TAVILY_API_KEY 채우기

# 원문 PDF 2건을 data/raw/deepseek_v2.pdf, data/raw/itme.pdf 로 저장
# (rag/pdf.py의 TECH_PAPER_PATHS와 파일명을 맞출 것)

# 전체 그래프 실행
python app.py

# 그래프 배선 확인 (실제 LLM/웹검색 호출 발생 — API 키 필요)
pytest tests/test_graph_smoke.py
```

## 분담 가이드
각 `agents/*.py`, `rag/*.py` 파일 상단에 `담당: __________` 와 함께 해당
노드의 역할·입출력·TODO가 docstring으로 적혀 있습니다. `graph/build_graph.py`는
전체 배선 담당(또는 팀 공통)이 관리하는 것을 권장합니다. 그래프 자체(노드
등록/엣지 연결)는 `python3 -c "from graph.build_graph import build_graph; build_graph()"`
로 API 키 없이도 구성 여부를 확인할 수 있습니다.

## Contributors
- 장나리 —
- —
- —
- —
