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

## Agents
설계서 A. Agent 정의 기준.

| Agent | 노드 함수 | RAG 여부 | 역할 |
|---|---|---|---|
| 기술 조사 에이전트 | `tech_research` | O | 원문 2건에서 기술 개요·범위·한계·TRL 추출, SW/HW 대조 기술과 비교 |
| 시장 평가 에이전트 | `market_eval` | X | 시장 규모, 상용화/채택 현황, 성장 전망 검색 (충분성 판정 + 웹검색 1회 보완) |
| 이해관계자 평가 에이전트 | `stakeholder_eval` | O | 경쟁사 반응, 개발자 평가, 투자/업계 시각 (RAG+웹검색, 충분성 판정 + 웹검색 1회 보완) |
| 도메인 평가 에이전트 | `domain_eval` | O | 데이터센터·클라우드에서의 적용 적합성 평가 (충분성 판정 + 웹검색 1회 보완) |
| 평가 종합 에이전트 | `synthesis` | X | 기술 조사 + 관점별 의견 종합 및 비교 (일치/상충 지점 서술) |
| 보고서 생성 에이전트 | `report_gen` | X | 단계별 내용을 연결해 평가 보고서 생성 |

## Architecture
<img width="286" height="692" alt="Technology Evaluation-2026-09-22-004218" src="https://github.com/user-attachments/assets/a33f6fc4-5a8d-4550-ac0b-40b7a9bab335" />

## Example Output (`python app.py` 실행 결과)
아래는 `outputs/report.md`에 실제로 생성된 결과 발췌다 (기획 의도가 실제로
구현·동작한다는 근거).

- **TRL 평가 구현 → 실제 결과**: `tech_research` 프롬프트에 TRL 평가
  지시를 추가한 결과, "4.1 기술 성숙도 평가"에 아래처럼 등급과 판단 근거,
  실측 수치가 표로 정리되어 출력된다.
  > | 구분 | DeepSeek-V2 MLA | ITME |
  > |---|---|---|
  > | 기술성숙도 | TRL 9 | 추정 TRL 5 |
  > | 주요 확인 수치 | KV 캐시 93.3% 감소, 128K 컨텍스트, 50K tokens/s 이상 생성 처리량 | 프리페칭 시 약 18GB/s, CPU 오프로딩 대비 최대 35.7% 처리량 향상 |

- **SW/HW 대조 기술 비교 구현 → 실제 결과**: `tech_research`가 검색한
  대조 기술(TurboQuant/InfiniGen) 발췌를 근거로, "기술 개요" 항목 안에
  실제로 비교 문장이 포함된다.
  > MLA는 TurboQuant와 구별된다. TurboQuant가 고차원 벡터를 저비트 정수로
  > 양자화하여 압축하는 방식이라면, MLA는 어텐션의 Key-Value 표현을
  > 저랭크 latent 구조로 공동 압축하는 아키텍처 수준의 방식이다.

- **`market_eval`/`stakeholder_eval`/`domain_eval` 충분성 판정 + 웹검색
  보완 구현 → 실제 결과**: 근거가 부족한 항목은 추정하지 않고 "확인되지
  않음"으로 명시되며, 실제 도입 사례 수 등은 표로 정리된다.
  > | 구분 | 확인 가능한 공개 도입 사례 |
  > |---|---:|
  > | DeepSeek-V2/MLA | 0건 |
  > | ITME | 0건 |
  >
  > 위 수치는 실제 도입이 없다는 의미가 아니라, 제공된 자료에서 검증
  > 가능한 공개 도입 사례가 없다는 의미다.

- **`references` 실제 출처 수집 구현 → 실제 결과**: REFERENCE 절에
  생성형 LLM이 지어낸 문장이 아니라, RAG로 검색된 실제 원문 파일과 각
  자료가 어디에 쓰였는지가 그대로 출력된다.
  ```
  1. DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model — data/raw/DeepSeek-V2.pdf
     - 활용 범위: MLA 구조, KV 캐시 감소, 컨텍스트 길이, 실제 서비스 배포 및 처리량 관련 근거
  2. ITME: Inference Tiered Memory Expansion ... — data/raw/ITME.pdf
     - 활용 범위: ITME 아키텍처, CXL 하이브리드 메모리, FPGA 프로토타입 및 성능 평가 근거
  3. TurboQuant — data/raw/TurboQuant.pdf
     - 활용 범위: MLA와 양자화 기반 벡터 압축 방식의 구조적 차이 비교
  4. InfiniGen — data/raw/InfiniGen.pdf
     - 활용 범위: CPU 메모리 KV 캐시 오프로딩·프리페칭 방식과 ITME의 차이 비교
  ```

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
