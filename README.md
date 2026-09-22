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
- SW : **DeepSeek-V2 (Multi-head Latent Attention, MLA)** — Key-Value를
  저랭크 latent 벡터로 공동 압축하는 어텐션 구조로, KV 캐시를 93.3%
  절감하고 8×H800 GPU 환경에서 생성 처리량 50K tokens/s 이상을 달성.
  실제 서비스 배포 및 실측 처리량 근거는 있으나 장기 상용 운영·SLA 근거는
  부족해 **추정 TRL 8**로 평가됨. 사후 압축 방식인 양자화 계열
  (**TurboQuant** 등, 정밀도 손실·추가 연산 지연 한계) 대비 비교 평가
  대상으로서 타당성이 높다고 판단해 선정.
- HW : **ITME (Inference Tiered Memory Expansion with Disaggregated
  CXL-Hybrid Memories)** — CXL 하이브리드 메모리와 SSD-backed 원격 메모리를
  GPU 서버가 RDMA로 접근하게 하는 계층형 메모리 확장 아키텍처. FPGA
  프로토타입 기준 약 18GB/s 프리페칭 처리량, CPU 오프로딩 대비 최대 35.7%
  처리량 향상을 실측했으나 상용 배포 사례는 없어 **추정 TRL 5**로 평가됨.
  GPU 내부에서 KV 캐시 자체를 줄이는 SW 접근(**InfiniGen** 등 동적 KV
  캐시 관리 기법)과 달리 GPU 외부 메모리 용량을 TB 단위로 확장하는
  방향이라 SW(DeepSeek-V2)와 상호 보완적 비교 대상으로 선정.
- 각 기술의 대조군(SW: TurboQuant, HW: InfiniGen) 원문도 RAG 인덱스에
  포함해, `tech_research`가 "왜 이 대안을 선택하지 않았는지"를 근거 기반
  으로 한 줄 비교하도록 함 (`rag/pdf.py`의 `SW_COMPARISON_PAPER_PATHS` /
  `HW_COMPARISON_PAPER_PATHS`).
- `TECH_PAPER_PATHS`에는 선정 기술 원문 2건 외에도 CXL 메모리 풀링/특성화,
  GQA, KIVI, LIMINAL, PIMCXL, 데이터센터 인프라, SGLang, TransMLA,
  vLLM 롱컨텍스트 등 관련 논문 총 15건을 함께 인덱싱해, `domain_eval`/
  `stakeholder_eval`이 더 넓은 근거로 평가할 수 있도록 함.

## Features
- PDF 원문(선정 기술 DeepSeek-V2·ITME, 대조 기술 TurboQuant·InfiniGen,
  관련 논문 11건 = 총 15건) 기반 RAG 정보 추출 (기술 조사, 도메인 평가,
  이해관계자 평가)
- Vision 딥러닝 모델 없이 PyMuPDF/pdfplumber 기하 분석만으로 Figure/Table
  캡션 추출, 2단 컬럼 논문의 읽기 순서 재정렬, 수식·헤딩 정규화, 하이픈
  결합을 수행하는 알고리즘 기반 PDF 파서 사용 (`rag/pdf_parser.py`,
  실패 시 `PDFPlumberLoader`로 폴백)
- 웹검색 기반 시장성 · 이해관계자 반응 조사
- 기술 조사 결과에 NASA TRL 9단계 척도 기준 **기술성숙도(TRL) 평가**를
  포함 (논문 게재/동료심사 여부, 실측 vs 시뮬레이션 검증 방식, 실제
  서비스·제품 적용 여부를 근거로 판단)
- 시장 / 이해관계자 / 도메인 평가 단계 모두 자료가 부족하면 `CorrectiveRAG`의
  `GradeDocuments` 그레이더 패턴으로 충분성을 판정하고, 부족하면 웹검색
  1회 보완 후 재생성·재판정 → 그래도 부족하면 재시도 없이 `data_limited`에
  기록 → 최종 보고서 "한계점"에 반영
- 관점별(시장성 / 이해관계자 / 도메인) Fan-out 병렬 평가 → 평가 종합 Fan-in
  (`data_limited`, `references`는 `operator.add`로 누적되는 채널이라 각
  노드가 자신의 항목만 반환)
- 확증 편향 방지 전략 : 종합 단계에서 우열 판정 없이 관점 간 일치/상충
  지점을 병렬 서술하도록 프롬프트 설계, 각 RAG/웹검색 노드가 실제로 인용한
  출처만 `references`에 담아 보고서 REFERENCE 절에 그대로 반영(LLM이 자료를
  지어내지 않도록 실제 출처 목록만 프롬프트에 전달)

## Tech Stack
- Framework : LangGraph (`init_chat_model` 기반 LangChain 체인)
- LLM/Generator : `market_eval`/`domain_eval`/`stakeholder_eval`/`tech_research`는
  `gpt-5.6-luna`, `synthesis`/`report_gen`은 `gpt-5.6-terra` (각 `agents/*.py`의
  `MODEL_NAME` 참고)
- LLM/Judge : `gpt-5.6-luna` (충분성 그레이더용, `market_eval`/`domain_eval`/
  `stakeholder_eval` 공통)
- Retrieval : Chroma — `{Hit Rate@K}`, `{MRR}` (TODO: 측정 후 기재)
- Embedding : BGE-M3 (`rag/embeddings.py`) — 긴 문맥·다국어 지원, Dense/Sparse/
  Multi-vector Retrieval, MIT License 오픈소스 — 논문 중심 텍스트 검색
  적합성/구현 난이도/연산 자원/라이선스를 종합 고려해 선정

## RAG 흐름
1. **PDF → 텍스트 추출** (`rag/pdf_parser.py`, 실패 시 `PDFPlumberLoader` 폴백)
   - 자체 파서를 1차로 쓰고, 실패하면 기본 PDF 로더로 폴백
   - Figure/Table 영역은 캡션과 함께 고해상도 이미지로 따로 저장하고, 본문에서는 그 영역을 마스킹해서 제외
   - 표는 셀 단위로 추출해 마크다운 표로 변환, 본문 하단에 첨부
   - 2단 컬럼 논문의 읽기 순서를 자동 판별해서 올바르게 재정렬
   - 수식은 KaTeX 형식으로 정규화, 줄바꿈에 끊긴 하이픈 단어도 복원
   - 결과: PDF 한 페이지가 문서 하나로 변환되고, 출처(파일명)·페이지 번호·표/그림 목록이 메타데이터로 붙음
2. **청킹** — 1200자 단위, 200자씩 겹치게 분할
3. **임베딩** — BGE-M3 모델 사용, 정규화된 벡터로 변환 (코사인 유사도 계산 전제)
4. **벡터 저장** (`rag/base.py`)
   - 로컬 Chroma에 영구 저장
   - 용도별로 컬렉션을 분리: 메인 원문(DeepSeek-V2, ITME + 관련 논문) / SW 대조 기술(TurboQuant) / HW 대조 기술(InfiniGen)
   - 같은 컬렉션에 이미 데이터가 있으면 재삽입하지 않고 기존 인덱스를 재사용 → 여러 에이전트가 같은 풀을 반복 호출해도 중복 임베딩 안 됨
5. **검색** — 코사인 유사도 기반, 메인 풀은 상위 8개, 대조 기술 풀은 상위 4개 청크를 가져옴. 각 에이전트가 자기 목적에 맞는 쿼리로 검색
6. **프롬프트 주입** — 검색된 청크를 출처/페이지 정보가 보존된 형태로 묶어서 각 에이전트 프롬프트의 컨텍스트 자리에 넣고, 그 위에서 LLM이 최종 평가/조사 텍스트를 생성

## Agents
설계서 A. Agent 정의 기준.

| Agent | 노드 함수 | RAG 여부 | 역할 |
|---|---|---|---|
| 기술 조사 에이전트 | `tech_research` | O | 원문 15건 풀에서 기술 개요·범위·한계·TRL 추출, SW/HW 대조 기술과 비교 |
| 시장 평가 에이전트 | `market_eval` | X | 시장 규모, 상용화/채택 현황, 성장 전망 검색 (충분성 판정 + 웹검색 1회 보완) |
| 이해관계자 평가 에이전트 | `stakeholder_eval` | O | 경쟁사 반응, 개발자 평가, 투자/업계 시각 (RAG+웹검색, 충분성 판정 + 웹검색 1회 보완) |
| 도메인 평가 에이전트 | `domain_eval` | O | 데이터센터·클라우드에서의 적용 적합성 평가 (충분성 판정 + 웹검색 1회 보완) |
| 평가 종합 에이전트 | `synthesis` | X | 기술 조사 + 관점별 의견 종합 및 비교 (일치/상충 지점 서술) |
| 보고서 생성 에이전트 | `report_gen` | X | 단계별 내용을 연결해 평가 보고서 생성 |

## Architecture
```
START
  └─ select_technology (기술 선정)
       └─ tech_research (기술 조사 RAG + SW/HW 대조 기술 비교 + TRL 평가)
            ├─ market_eval (웹검색, 내부 충분성 체크 + 보완 루프 포함)
            ├─ stakeholder_eval (RAG + 웹검색, 내부 충분성 체크 + 보완 루프 포함)
            └─ domain_eval (RAG + 웹검색, 내부 충분성 체크 + 보완 루프 포함)
                 ["market_eval","stakeholder_eval","domain_eval"] 조인
                              └─ synthesis (fan-in, 기술 조사 결과도 함께 입력)
                                   └─ report_gen
                                        └─ END
```
원본 mermaid 설계도는 업로드된 설계산출물(`RAG-Design_판교-9반.pdf`) D절 참고.
설계서 원안에는 `tech_research` 뒤에도 "충분한가?" 분기 + 웹검색 보완 노드가
그래프 레벨로 그려져 있었으나, 팀 논의 후 그래프에서 완전히 제거하고
`tech_research`가 RAG 조사 결과를 바로 3개 평가 노드로 fan-out하도록
단순화했다. 시장/이해관계자/도메인 평가의 충분성 체크 + 웹검색 보완
루프는 `graph/build_graph.py` 상단 docstring에 설명된 대로 그래프 레벨
분기 대신 각 노드(`market_eval.py`, `stakeholder_eval.py`,
`domain_eval.py`) 내부 로직으로 처리한다.

## Example Output (`python app.py` 실행 결과)
아래는 `outputs/report.md`에 실제로 생성된 결과 발췌다 (기획 의도가 실제로
구현·동작한다는 근거).

- **TRL 평가 구현 → 실제 결과**: "4.1 기술 성숙도 관점"에 등급, 검증
  근거, 미확인 사항까지 표로 정리되어 출력된다.
  > | 구분 | DeepSeek-V2 MLA | ITME |
  > |---|---|---|
  > | 추정 TRL | TRL 8 | TRL 5 |
  > | 검증 근거 | 실제 DeepSeek 서비스 환경에서 8개 H800 GPU 기준 처리량 측정 제시 | FPGA 프로토타입 및 LLM 추론 워크로드 기반 실험 |
  > | 주요 미확인 사항 | 장기간 대규모 상용 운용, SLA, 멀티테넌트 지연시간 | 상용 서비스, 제품화, 대규모 운영, 표준화 및 장기 안정성 |

- **SW/HW 결합 아키텍처 제안 → 실제 결과**: "4.4 도메인 적용 관점"에서
  두 기술을 GPU HBM부터 원격 스토리지까지 계층별로 배치하는 구체적인
  아키텍처까지 함께 제시된다.
  > | 계층 | 권장 데이터 배치 |
  > |---|---|
  > | T1: GPU HBM | 현재 실행 레이어, 활성 expert, hot KV cache |
  > | T3.5: ITME CXL 하이브리드 메모리 | 장기 KV/context, prefix cache, 저빈도 가중치 |
  > | T4: 원격 공유 스토리지 | cold archive, 낮은 접근 빈도의 상태 |

- **시장/이해관계자 평가에서 근거 없는 수치 추정 방지 → 실제 결과**:
  확인되지 않은 항목은 임의로 채우지 않고 "근거 부족"/"추가 검증 필요"로
  명시된다.
  > 두 기술 모두 독립적인 시장 규모 및 CAGR 자료가 없어 정량적 시장성
  > 판단에는 **근거 부족**이 있다.

- **`references` 실제 출처 수집 구현 → 실제 결과**: REFERENCE 절에
  생성형 LLM이 지어낸 설명이 아니라, RAG로 검색된 실제 원문 페이지와
  실제 웹검색 URL이 그대로 출력된다.
  ```
  - data/raw/DeepSeek-V2.pdf (p.1)
  - data/raw/ITME.pdf (p.2)
  - data/raw/TurboQuant.pdf (p.1)
  - data/raw/InfiniGen.pdf (p.9)
  - ITME: Inference Tiered Memory Expansion ... - https://arxiv.org/html/2606.12556
  - DeepSeek-V2: A Strong, Economical, and Efficient Mixture- ... - https://huggingface.co/papers/2405.04434
  ```
  (fan-out 노드들이 각자 검색한 출처를 병합만 하고 있어 위 목록에 중복이
  남아있는 것은 알려진 한계 — `report_gen`에서 전역 dedupe 필요)

## Directory Structure
```
├── data/
│   └── raw/                # 원문 PDF 15건 (선정 기술 2 + 대조 기술 2 + 관련 논문 11)
├── graph/
│   ├── state.py             # GraphState(TypedDict + Annotated 설명)
│   └── build_graph.py       # 노드/엣지 배선
├── agents/                  # Agent별 노드 함수 (담당자별로 분담)
│   ├── prompt_utils.py       # prompts/*.txt -> PromptTemplate 로더
│   ├── tech_selector.py      # select_technology
│   ├── tech_research.py      # tech_research (SW/HW 조사 + 대조 기술 비교 + TRL)
│   ├── market_eval.py        # market_eval
│   ├── stakeholder_eval.py   # stakeholder_eval
│   ├── domain_eval.py        # domain_eval
│   ├── synthesis.py          # synthesis
│   └── report_gen.py         # report_gen
├── rag/                     # RAG 공통 모듈 (실습자료 20-RAG/rag/ 구조를 따르되, retriever까지만 책임)
│   ├── embeddings.py          # create_bge_m3_embeddings()
│   ├── base.py                # RetrievalChain (ABC) - 원문 로딩~retriever 생성만 담당
│   ├── pdf.py                 # PDFRetrievalChain, format_docs, build_tech_retrieval_chain
│   └── pdf_parser.py          # 알고리즘 기반 학술 논문 PDF 파서 (DLA + 공간 마스킹 + 다단 정렬)
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

# data/raw/에 원문 PDF를 넣는다. 정확한 파일 목록/파일명은
# rag/pdf.py의 TECH_PAPER_PATHS(선정 기술 + 관련 논문 전체) /
# SW_COMPARISON_PAPER_PATHS / HW_COMPARISON_PAPER_PATHS 참고

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
- 박종문 — RAG수집, 정리, synthesis agent개발, 발표
- 장나리 — 기술 조사 에이전트 및 시장 조사 에이전트 개발
- 정서영 — 도메인 평가 에이전트, 이해관계자 평가 에이전트, 보고서 작성 에이전트 구현
- 한상현 — PDF Parsing