# 평가 보고서

## 보고서 목차(초안)
1. 분석 배경  
2. 기술 선정  
3. 기술 개요  
4. 관점별 평가  
5. 시사점  
6. 한계점  
7. REFERENCE  

## SUMMARY

본 보고서는 데이터센터/클라우드 서빙 환경에서 KV cache로 인한 메모리 용량·대역폭 병목에 대응하는 소프트웨어 기술인 **DeepSeek-V2의 Multi-Head Latent Attention(MLA)**와 하드웨어·시스템 기술인 **ITME(Inference Tiered Memory Expansion)**를 평가하였다. MLA는 어텐션 구조 내부에서 KV 정보를 저랭크 latent 표현으로 공동 압축하여 요청당 KV cache의 기본 크기를 줄이는 접근이다. DeepSeek-V2는 KV cache 93.3% 절감 및 최대 생성 처리량 5.76배 향상을 보고했으나, 해당 수치는 논문 조건에 기반하므로 일반적 운영 성과로 확대 해석할 수 없다. ITME는 GPU·호스트 메모리 바깥에 CXL 하이브리드 메모리, NVMe 및 RDMA 기반 계층을 구성하여 대규모 가중치와 장문 KV cache를 수용하는 접근이다. 특정 멀티턴 실험에서 CPU 오프로딩 대비 최대 35.7% 처리량 향상이 보고되었으나, 실제 클라우드 서비스 운영의 비용·지연시간 보장값은 아니다.

공개 정보 기준으로 MLA의 기술성숙도는 **TRL 5~6**, ITME는 **TRL 4~5**로 추정된다. MLA는 모델 적용과 추론 엔진 최적화 사례가 확인되는 반면, ITME는 FPGA 프로토타입 및 시스템 벤치마크가 제시되었으나 상용 서비스 배포 근거가 제한적이다. 두 기술은 경쟁 관계라기보다 KV cache 문제를 각각 “표현량 축소”와 “계층형 수용”으로 다루는 상호보완 가능성이 있는 접근으로 볼 수 있다. 다만 두 기술을 결합한 동일 조건의 성능, TCO, TTFT·TPOT, p95/p99 지연시간, 멀티테넌시 및 장애 복구 실증 자료는 확인되지 않아 최종 도입 효과는 추가 검증이 필요하다.

---

## 1. 분석 배경

대규모 언어 모델의 자동회귀 추론에서는 이전 토큰의 attention 계산 결과인 Key-Value(KV) cache를 유지해야 한다. KV cache는 생성 토큰 수, 컨텍스트 길이, 동시 요청 수, 레이어 수 및 어텐션 헤드 구성에 따라 증가한다. 따라서 장문 컨텍스트와 대규모 배치 환경에서는 모델 가중치보다 KV cache가 더 큰 메모리 부담이 될 수 있으며, GPU 메모리 용량과 메모리 대역폭이 추론 처리량을 제한할 수 있다. `KIVI_no-refs.pdf`는 540B PaLM 모델에서 배치 512, 컨텍스트 2,048 조건의 KV cache가 3TB에 이를 수 있는 사례를 제시한다. 다만 이는 DeepSeek-V2나 ITME의 실험 결과가 아니라 KV cache 병목의 일반적 규모를 보여주는 사례다. (`data/raw/KIVI_no-refs.pdf`, p.1)

데이터센터/클라우드 서빙에서는 KV cache 증가가 GPU HBM 증설, 호스트 메모리 오프로딩, 외부 스토리지 활용, 네트워크 전송량, 요청별 지연시간 및 운영 복잡도와 연결된다. 이에 따라 KV cache 자체의 크기를 줄이는 모델 구조와, GPU 외부 계층으로 KV cache를 확장·관리하는 메모리 인프라를 함께 검토할 필요가 있다.

본 분석은 이러한 관점에서 다음 두 접근을 비교·정리한다.

- **MLA:** 모델 내부의 KV 표현량을 줄이는 소프트웨어·모델 아키텍처 접근
- **ITME:** GPU·호스트 메모리의 용량 한계를 CXL·NVMe·RDMA 기반 계층으로 확장하는 하드웨어·시스템 접근

---

## 2. 기술 선정

### 2.1 SW 기술: DeepSeek-V2의 Multi-Head Latent Attention(MLA)

MLA는 DeepSeek-V2에 적용된 어텐션 구조로, Key와 Value를 저차원 latent 표현으로 공동 압축하여 KV cache 요구량을 줄이는 방식이다. 표준 MHA가 생성 과정에서 대규모 KV cache를 유지하는 것과 달리, MLA는 어텐션 내부 표현을 변경해 KV cache의 기본 저장량을 줄이는 데 초점을 둔다. DeepSeek-V2 논문은 DeepSeek 67B 대비 KV cache를 93.3% 줄이고 최대 생성 처리량을 5.76배 높였다고 보고한다. (`data/raw/DeepSeek-V2.pdf`, p.1)

선정 이유는 다음과 같다.

- KV cache 병목을 모델 구조 차원에서 직접 완화하는 접근이다.
- DeepSeek-V2 및 DeepSeek-V2-Lite에 실제 적용된 사례가 있다. (`data/raw/DeepSeek-V2.pdf`, p.20)
- SGLang, vLLM 및 TransMLA 등과 연결되는 구현·변환·커널 최적화 사례가 확인된다.
- 다만 기존 MHA/GQA 모델의 전환, 런타임 호환성 및 전용 커널 최적화가 필요할 수 있다.

### 2.2 HW 기술: ITME(Inference Tiered Memory Expansion)

ITME는 CXL 하이브리드 메모리, 내부 DRAM cache, NVMe SSD, RDMA 및 다계층 DMA 프리페칭을 결합해 추론용 메모리 용량을 확장하는 시스템 아키텍처다. GPU HBM(T1), 호스트 메모리(T2), 로컬 NVMe(T3), CXL 하이브리드 메모리(T3.5), 원격 공유 스토리지(T4)를 계층화하고, 예측 가능한 가중치와 장문 prefix KV cache를 원격 계층에 배치·프리페치한다. (`data/raw/ITME.pdf`, pp.1–3)

선정 이유는 다음과 같다.

- 장문·멀티턴·에이전트형 AI 워크로드에서 GPU 및 호스트 메모리를 넘어서는 KV cache 수요를 다룬다.
- CXL, RDMA, NVMe, DRAM cache를 결합한 메모리 확장 구조를 제시한다.
- FPGA 프로토타입 및 ShareGPT 기반 통합 벤치마크가 제시되어 있다.
- 다만 실제 데이터센터 상용 배포, 대규모 고객 채택, 장기 운영 안정성은 확인되지 않는다.

---

## 3. 기술 개요

### 3.1 DeepSeek-V2 MLA의 접근 방향과 한계

MLA는 Key와 Value를 저랭크 latent 벡터로 압축해 저장하고, 이를 attention 계산에 활용하는 구조다. MHA, MQA, GQA가 헤드 구성 변경을 통해 KV cache 크기를 조정하는 방식과 비교하면, MLA는 KV 정보를 latent 표현으로 공동 압축한다는 점에서 차이가 있다. DeepSeek-V2는 128K 컨텍스트를 지원하며, DeepSeek-V2-Lite에서도 MLA를 사용한다. Lite 모델은 KV 압축 차원 512를 사용하지만 query를 압축하지 않는 구성을 택한다. (`data/raw/DeepSeek-V2.pdf`, p.20)

SGLang v0.3은 MLA 구현을 위해 weight absorption, grouped decoding kernel, FP8 batched MatMul, FP8 KV-cache 양자화 등을 적용했다. H100 기반 특정 벤치마크에서는 기준 시스템 대비 3~7배 높은 처리량을 보고했으나, 이는 H100, BF16·FP8, tensor parallelism 및 ShareGPT 조건에 의존한다. (`data/raw/SGLang_v0.3_Release_7x_ Faster_DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image_Video LLaVA-OneVision - LMSYS Org.pdf`, p.1)

주요 한계는 다음과 같다.

- 기존 오픈소스 라이브러리에서 MLA 구조가 충분히 최적화되지 않았으며, 전용 커널과 메모리 최적화가 필요하다.
- 기존 GQA/MHA 모델을 MLA로 전환할 때 RoPE, query 투영, 저랭크 분해 등 추가 설계가 필요할 수 있다. (`data/raw/TransMLA_MLA_Is_All_You_Need.pdf`, p.2)
- MLA 적용 후 정확도, TTFT, TPOT, p95/p99 지연시간, 멀티노드 통신량을 동일 조건에서 비교한 데이터센터 운영 자료는 제한적이다.
- MLA의 KV cache 절감이 전체 서비스 비용 또는 전체 지연시간 감소로 동일 비율로 이어진다는 근거는 부족하다.

### 3.2 ITME의 접근 방향과 한계

ITME는 대규모 모델 가중치와 KV cache를 GPU HBM과 호스트 메모리에만 저장하지 않고, CXL 하이브리드 메모리를 중간 계층으로 활용해 용량을 확장한다. CXL 하이브리드 메모리는 SSD-backed 용량을 내부 DRAM cache와 결합해 바이트 주소 지정 가능한 확장 메모리처럼 제공하며, 하드웨어 컨트롤러가 NVMe 요청을 직접 발행한다. GPU 서버는 표준 RDMA 및 DMA 파이프라인을 통해 확장 계층의 데이터에 접근한다. (`data/raw/ITME.pdf`, pp.1–3)

ITME는 모델 가중치의 레이어 단위 순차 접근과 장문 prefix KV cache의 순차 restore 패턴을 활용해 프리페치를 수행한다. 반면 decode 단계의 working KV cache는 지연시간 민감도가 높고 접근 예측성이 낮으므로 GPU 또는 호스트 메모리에 우선 배치하는 구조다. (`data/raw/ITME.pdf`, p.3)

주요 한계는 다음과 같다.

- 원격 CXL·SSD 계층은 GPU HBM과 동일한 접근 지연시간을 제공하지 않는다.
- 프리페치 효과는 가중치 및 장문 prefix KV cache의 예측 가능한 접근 패턴에 의존한다.
- 원격 계층으로 축출이 발생하는 구간에서는 I/O 경합, stall, 처리량 변동 가능성이 있다.
- CXL, RDMA, NVMe, 호스트 메모리, GPU DMA를 함께 운영해야 하므로 시스템 통합과 장애 관리 복잡도가 높아질 수 있다.
- 실제 클라우드 서비스에서의 지연시간 분포, 멀티테넌트 공정성, 장애 복구 시간 및 TCO는 추가 검증이 필요하다.

---

## 4. 관점별 평가

### 4.1 기술성숙도 관점

#### DeepSeek-V2 MLA: TRL 5~6 — 공개 정보 기반 추정

MLA는 단순 이론 제안에 머물지 않고 DeepSeek-V2 및 DeepSeek-V2-Lite 모델에 통합되어 있다. DeepSeek-V2 논문은 128K 컨텍스트 모델에서 KV cache 93.3% 절감과 최대 생성 처리량 5.76배 향상을 보고한다. (`data/raw/DeepSeek-V2.pdf`, p.1) 또한 Lite 모델에도 MLA 구성과 압축 차원이 제시되어 모델 단위 구현과 실험이 수행된 것으로 볼 수 있다. (`data/raw/DeepSeek-V2.pdf`, p.20)

시스템 구현 측면에서는 SGLang v0.3이 MLA 전용 최적화와 H100 기반 처리량 벤치마크를 공개했다. 이는 실제 서버 실행 환경과 전용 커널 수준의 통합이 제시되었다는 근거다. (`data/raw/SGLang_v0.3_Release_7x_ Faster_DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image_Video LLaVA-OneVision - LMSYS Org.pdf`, pp.1, 4)

다만 이 등급은 **공개 정보 기반 추정**이다. 제공 자료에는 MLA가 특정 클라우드 사업자의 장기 상용 서비스에서 운영되었는지, 고객 도입과 제품 납품이 이루어졌는지, 어떤 수준의 서비스 안정성이 검증되었는지에 대한 충분한 근거가 없다. 논문 발표와 벤치마크 시점, 실제 서비스 채택 시점 사이에는 시차가 있을 수 있으므로 TRL 7~9로 단정하기는 어렵다.

#### ITME: TRL 4~5 — 공개 정보 기반 추정

ITME는 CXL 하이브리드 메모리, 내부 DRAM cache, NVMe 직접 요청, RDMA, DMA 프리페칭을 결합한 통합 아키텍처를 제시한다. FPGA 프로토타입에서 내부 DRAM cache에 프리페치된 경우 약 18GB/s에 가까운 처리량이 관찰되었다. (`data/raw/ITME.pdf`, p.10)

또한 ShareGPT 기반 멀티턴 조건에서 CPU 오프로딩 및 로컬 NVMe-oF 구성과 비교하는 엔드투엔드 평가가 제시되었다. 장기 대화 구간에서 CPU 오프로딩 대비 최대 35.7% 처리량 향상이 보고되었으며, 8B와 70B 모델 풋프린트에서 기준선에 가까운 성능이 제시된다. (`data/raw/ITME.pdf`, pp.9–10)

다만 ITME의 TRL 역시 **공개 정보 기반 추정**이다. FPGA 및 CXL 메모리 모듈 기반의 실증은 확인되지만, 실제 상용 데이터센터 또는 클라우드 환경에서의 장기 운영, 양산 적합성, 고객 도입, 서비스 수준 협약(SLA) 기반 검증은 자료에서 확인되지 않는다. 또한 논문 발표 시점의 프로토타입 성과가 이후 실제 채택으로 이어졌는지는 확인할 수 없다. 따라서 TRL 6 이상으로 판단하기에는 근거가 부족하다.

---

### 4.2 시장성 관점

#### DeepSeek-V2 MLA

**① 시장 규모·성장성**  
MLA 자체의 시장 규모, 매출, 시장점유율 또는 CAGR은 **근거 부족**이다. 다만 MLA가 속한 상위 AI 추론 시장은 2025년 1,061.5억 달러에서 2030년 2,549억 달러로 성장하고 CAGR 19.2%가 전망된다는 자료가 있다. 그러나 이 수치는 MLA의 직접 시장 규모가 아니라 AI 추론 시장 전체의 지표다.  
(How CXL Transforms RAG and KV Cache Performance - https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance)

**② 상용화·채택 현황**  
MLA는 DeepSeek-V2뿐 아니라 DeepSeek-V3 및 R1 계열에도 사용되는 구조로 소개되며, vLLM의 MLA 및 FP8 커널 최적화 사례가 제시된다.  
(Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM - https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm)

다만 실제 클라우드 서비스 제공자, 유료 API 운영 사례, 고객 프로덕션 배포, 도입 기업 수에 관한 직접 근거는 부족하다. 모델 적용 및 오픈소스 런타임 지원 사례를 실제 상용 고객 채택과 동일하게 해석할 수는 없다.

**③ 생태계 지지**  
SGLang에서 FlashInfer 기반 MLA 커널 사용 사례가 있고, vLLM의 MLA 지원 및 FP8 최적화 사례도 확인된다.  
(Part 3 — Implementation/Engine-Level: Choosing the Runtime That Gives You These for Free - https://pub.towardsai.net/part-3-implementation-engine-level-choosing-the-runtime-that-gives-you-these-for-free-b0e9081205b0)  
(Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM - https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm)

반면 주요 범용 프레임워크가 MHA·GQA·MQA·MLA를 하나의 통합 메모리 관리 체계로 충분히 지원하지 못한다는 연구 지적도 있다. MLA KV cache 형식의 표준화, 엔진 간 상호운용성, 공식 안정 지원 범위는 추가 검증이 필요하다.  
(Predictive Multi-Tier Memory Management for KV Cachein Large-Scale GPU Inference - https://arxiv.org/html/2604.26968v2)

#### ITME

**① 시장 규모·성장성**  
ITME 자체의 시장 규모와 CAGR은 **근거 부족**이다. 인접 시장으로 KV-cache offloading infrastructure 시장이 2024년 18.7억 달러에서 2033년 149.9억 달러로 성장하고 CAGR 23.6%가 전망된다는 자료가 있다. 그러나 이는 ITME가 아닌 KV cache 오프로딩 인프라 전반의 시장 추정치다.  
(KV-Cache Offloading Infrastructure Market Research Report 2033 - https://growthmarketreports.com/report/kv-cache-offloading-infrastructure-market)

**② 상용화·채택 현황**  
ITME는 FPGA 프로토타입, CXL Memory Module, NVMe SSD를 활용한 실증 플랫폼이 제시된 연구 기술이다.  
(ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories - https://arxiv.org/html/2606.12556v2)

CXL 기반 KV cache 인프라에서는 Penguin Solutions가 최대 11TB의 CXL 기반 메모리를 제공하는 “production-ready” KV cache 서버를 발표한 사례가 있다. 다만 이는 ITME 알고리즘 자체의 상용 제품화 또는 ITME의 고객 도입 사례는 아니다.  
(Penguin Solutions - Penguin Solutions Introduces Industry's First Production-Ready CXL-Based KV Cache Server - https://ir.penguinsolutions.com/news/news-details/2026/Penguin-Solutions-Introduces-Industrys-First-Production-Ready-CXL-Based-KV-Cache-Server/default.aspx)

**③ 생태계 지지**  
CXL 기반 KV cache storage, shared memory KV cache, CXL switch, RDMA, processing-near-memory 등 관련 인프라와 연구 생태계는 확인된다.  
(An Internet for the KV Cache: Rethinking Classical Infrastructure Boundaries in the LLM Inference Age - https://arxiv.org/html/2608.01526v1)  
(Using CXL Fabric-Attached Memory to Enable Shared KV Caches Across GPUs for Faster LLM Inference | SNIA | Experts on Data - https://www.snia.org/sniadeveloper/session/19665)

그러나 ITME 전용 SDK, 추론 엔진 공식 플러그인, CXL 컨소시엄 표준 채택, 클라우드 제공자의 직접 지원 여부는 **근거 부족**이다. 관련 하드웨어·프로토콜 기반은 존재하지만, ITME 자체의 표준화·프레임워크 통합 수준은 확인되지 않는다.

---

### 4.3 이해관계자 관점

#### ① 경쟁 기술 진영

**MLA 관련 경쟁·대응 기술**  
MLA의 비교 대상은 GQA, MQA, KV cache 양자화, 프리픽스 캐싱, 페이징 기반 KV cache 관리 등이다. GQA와 MQA는 KV cache를 줄이는 방법이지만, 제공 자료에서는 품질·성능·메모리 사이의 절충이 존재하는 방식으로 설명된다. MLA는 저랭크 latent 표현을 통해 MHA 수준의 성능과 작은 KV cache를 함께 지향하는 구조로 제시된다.  
(`data/raw/DeepSeek-V2.pdf`, pp.2, 4)

기존 GQA 모델을 MLA로 전환하려는 TransMLA 연구도 제시된다. 이는 기존 가중치와 사전학습 투자를 활용하면서 MLA 구조로 전환하려는 대응 접근이다. 다만 Meta, Qwen 등 특정 GQA 모델 개발 주체가 MLA에 대해 공식적으로 어떤 대응을 했는지는 **근거 부족**이다.  
(`data/raw/TransMLA_MLA_Is_All_You_Need.pdf`, pp.2, 11)

**ITME 관련 경쟁·대응 기술**  
ITME의 대체 또는 보완 접근에는 GPU HBM 증설, 호스트 메모리 오프로딩, 로컬 NVMe, 원격 공유 스토리지, CXL 기반 메모리 확장, PNM(processing-near-memory) 기반 KV 관리 등이 있다. PIM-CXL은 GPU 메모리만으로 장문 KV cache를 수용하기 어렵다는 문제를 제기하며 CXL 기반 메모리 확장 및 KV cache 관리를 다룬다.  
(`data/raw/PIMCXL_no-refs.pdf`, pp.2–5)

또한 NVIDIA CMX와 개방형 CXL을 서로 다른 메모리 인프라 접근으로 비교하는 산업 분석이 제시된다. 다만 CMX, CXL, PNM-KV와 ITME를 동일 조건에서 비교한 자료는 없으며, 특정 기업이 ITME를 직접 겨냥해 대응 제품을 출시했다는 근거도 부족하다.  
(Nvidia, CXL, and the Battle to Improve AI Inference Economics - I/O Fund - https://io-fund.com/ai-stocks/nvidia-cxl-ai-inference-economics)

#### ② 도입 기업·개발자

**MLA의 도입 유인과 장벽**  
MLA는 KV cache 절감과 장문 컨텍스트·고동시성 추론 효율 개선 가능성 때문에 개발자에게 도입 유인을 제공한다. SGLang은 MLA 전용 커널과 FP8 최적화를 제시했고, TransMLA는 기존 MHA/GQA 모델을 MLA로 변환하는 연구를 제안했다. TransMLA는 특정 실험에서 KV cache를 92.97% 줄인 MLA 모델을 여러 소비자용 AI 가속기에서 평가했다.  
(`data/raw/TransMLA_MLA_Is_All_You_Need.pdf`, p.11)

그러나 기존 모델 전환에는 RoPE 처리, 저랭크 분해, query 구성, 런타임·커널 통합 등의 복잡성이 따른다. 또한 MLA의 실제 이점은 모델 구조뿐 아니라 저장 KV 벡터 수 감소와 하드웨어별 커널 최적화에 의존한다.  
(`data/raw/TransMLA_MLA_Is_All_You_Need.pdf`, pp.2, 5)

DriveNets는 GQA 모델의 MLA 전환 사례와 AMD MI355X 기반 동시성 증가를 주장하지만, 이는 해당 업체 블로그의 사례이며 독립 검증 여부는 확인되지 않는다.  
(Upgrade GQA models to MLA for massive KV Cache Savings - https://drivenets.com/blog/how-to-upgrade-gqa-models-to-mla-for-massive-kv-cache-savings)

**ITME의 도입 유인과 장벽**  
ITME는 에이전트형 AI, 장문 컨텍스트, 멀티턴 세션처럼 KV cache가 지속적으로 누적되는 환경에서 GPU·호스트 메모리 부족을 완화하려는 접근이다. ShareGPT 기반 128개 동시 대화 및 장기 멀티턴 평가가 제시되었으며, KV cache가 약 40GB 수준에 이르는 구간을 분석했다.  
(`data/raw/ITME.pdf`, p.9)

반면 ITME 도입에는 CXL 하이브리드 메모리, RDMA, NVMe, 호스트 메모리, GPU DMA 및 계층별 프리페치 정책을 통합해야 하는 부담이 있다. 원격 계층 접근이 시작되면 성능 격차와 I/O 경합이 발생할 수 있고, prefetch 정확도와 데이터 배치 정책이 성능에 영향을 준다. (`data/raw/ITME.pdf`, pp.2, 7, 9)

실제 클라우드 사업자의 ITME 도입, 장비 조달성, 호환성, 운영 인력 부담, 장애 대응 경험에 대한 직접 사례는 **근거 부족**이다.

#### ③ 투자 업계

MLA에 대한 직접 투자 규모, 금융기관 평가, 애널리스트 의견, 기업가치 영향 자료는 **근거 부족**이다. MLA의 KV cache 절감과 장문 컨텍스트 효율성에 대한 긍정적 기술 논조는 존재하지만, 이는 기술 블로그나 해설 자료 중심이며 투자 업계 전체의 평가로 일반화할 수 없다.  
(KV Cache Optimization for LLMs 2026: Engineering Guide - https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide)

ITME에 대해서도 직접 투자, 수익성, 시장점유율, 애널리스트 전망 자료는 부족하다. 다만 CXL 기반 메모리 확장과 KV cache 서버에 대해서는 산업계의 사업 기회 및 제품화 신호가 일부 존재한다. Astera Labs와 Penguin Solutions의 자료는 CXL 메모리 확장에 우호적인 산업 논조 및 제품 사례를 제시하지만, ITME 자체에 대한 독립적 투자 평가로 볼 수는 없다.  
(How CXL Transforms RAG and KV Cache Performance - https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance)  
(Penguin Solutions - Penguin Solutions Introduces Industry's First Production-Ready CXL-Based KV Cache Server - https://ir.penguinsolutions.com/news/news-details/2026/Penguin-Solutions-Introduces-Industrys-First-Production-Ready-CXL-Based-KV-Cache-Server/default.aspx)

---

### 4.4 도메인 적용 관점

본 절의 평가는 **데이터센터/클라우드 서빙 환경에 한정**된다. On-device AI, 단일 사용자 로컬 추론, 특정 장문 컨텍스트 애플리케이션에서는 민감한 제약과 평가 기준이 달라질 수 있다.

#### DeepSeek-V2 MLA

**처리 가능 규모**  
MLA는 요청당 KV cache의 표현량을 줄이는 방식이므로, 동일 GPU 메모리에서 더 긴 컨텍스트 또는 더 많은 동시 요청을 수용할 가능성이 있다. DeepSeek-V2는 128K 컨텍스트를 지원하며, 논문은 DeepSeek 67B 대비 KV cache 93.3% 절감 및 최대 생성 처리량 5.76배 향상을 보고한다. (`data/raw/DeepSeek-V2.pdf`, p.1)

다만 데이터센터 운영 관점에서 필요한 GPU별 동시 사용자 수, 배치 크기별 처리량, 멀티노드 확장성, GPU 간 통신량, TTFT·TPOT, p95/p99 지연시간에 대한 DeepSeek-V2 MLA의 직접적이고 일반화 가능한 수치는 부족하다. 따라서 실제 처리 가능 규모는 **추가 검증 필요**하다.

**비용 절감 효과**  
KV cache가 차지하는 HBM 용량이 감소하면 GPU당 수용 가능한 요청 수가 증가하고, 고용량 GPU 또는 추가 GPU 인스턴스 수요를 늦출 가능성이 있다. 그러나 MLA 단독의 실제 클라우드 TCO 절감률, GPU 수 감소량, 전력 절감량, 요청당 비용은 **근거 부족**이다.

일부 웹 자료는 MLA 변환과 다른 최적화를 결합한 비용·동시성 개선 사례를 제시하지만, 해당 수치는 실험 조건과 모델·하드웨어 구성이 충분히 공개되지 않았거나 DeepSeek-V2 MLA 단독 결과가 아니다. 따라서 일반적인 데이터센터 비용 절감률로 사용할 수 없다.  
(KV Cache Optimization for LLMs 2026: Engineering Guide - https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide)  
(Upgrade GQA models to MLA for massive KV Cache Savings - https://drivenets.com/blog/how-to-upgrade-gqa-models-to-mla-for-massive-kv-cache-savings)

**제약**  
MLA는 모델 구조와 런타임 커널에 의존한다. 기존 MHA/GQA 모델에서 전환할 경우 모델 변환, 정확도 검증, 커널 최적화, 프레임워크 호환성 검증이 필요할 수 있다. 또한 KV cache를 줄여도 모델 가중치, prefill 연산, 네트워크, 스케줄링, GPU 연산 병목이 모두 사라지는 것은 아니다.

#### ITME

**처리 가능 규모**  
ITME는 GPU HBM과 호스트 메모리만으로 수용하기 어려운 장문·멀티턴 KV cache 및 대규모 모델 가중치를 CXL 하이브리드 메모리와 NVMe 기반 계층으로 확장한다. 활성화와 working KV cache는 GPU 또는 호스트 메모리에 두고, 장문 prefix KV cache와 가중치는 원격 확장 계층에 배치·프리페치한다. (`data/raw/ITME.pdf`, pp.2–3)

ITME의 멀티턴 실험에서는 35턴, 256개 대화 조건에서 호스트 DRAM이 포화된 이후 CPU offload 대비 최대 35.7% 처리량 향상이 보고됐다. 그러나 이는 특정 모델·워크로드·하드웨어 구성에서의 결과다. 데이터센터 전체 환경에서의 동시 요청 수, 최대 컨텍스트 길이, 노드 수별 확장성, TTFT·TPOT 및 p95/p99는 **근거 부족**이다.  
(`data/raw/ITME.pdf`, p.9)

**비용 절감 효과**  
ITME는 모든 가중치와 KV cache를 고가의 GPU HBM에 두는 대신, CXL 메모리·NVMe·RDMA 기반 계층을 활용해 용량을 확장하는 구조다. 이론적으로 GPU 메모리 부족으로 인한 요청 축출, 재계산, 추가 GPU 증설 압력을 낮출 가능성이 있다.

그러나 CXL 장비 가격, RDMA 네트워크 비용, NVMe 비용, 전력·냉각 비용, 운영 인력 비용을 포함한 TCO 비교는 제공 자료에 없다. 따라서 ITME가 HBM 또는 DRAM보다 비용 우위에 있다고 단정할 수 없으며, 실제 비용 절감 효과는 **추가 검증 필요**하다.

**제약**  
ITME의 성능은 데이터 접근 예측성과 프리페치 정확도에 크게 의존한다. working KV cache와 같이 토큰 생성 과정에서 지연시간에 민감한 데이터가 원격 계층에서 회수되어야 하면 TPOT 및 꼬리 지연시간이 악화될 수 있다. 원격 계층 접근과 I/O 경합으로 인한 성능 변동도 확인된다. (`data/raw/ITME.pdf`, pp.3, 9)

또한 CXL·RDMA·NVMe·호스트 메모리·GPU DMA를 함께 관리해야 하므로, 멀티테넌시 환경에서의 대역폭 경합, 캐시 오염, 테넌트별 공정성, 장애 격리 및 복구가 중요한 운영 과제가 된다. 이 항목들에 대한 실제 운영 검증 결과는 근거 부족이다.

---

### 4.5 종합의견

두 기술은 모두 KV cache로 인한 메모리 용량 및 추론 효율 병목을 해결하려 하지만, 개입 지점이 다르다.

- **MLA**는 모델 내부의 KV 표현을 압축해 요청당 메모리 수요 자체를 줄이는 접근이다.
- **ITME**는 줄어들지 않은 가중치와 KV cache를 GPU·호스트 메모리 밖의 계층으로 확장·수용하는 접근이다.

기술성숙도 관점에서는 MLA가 DeepSeek 모델 적용, SGLang 커널 최적화, vLLM 관련 지원 사례를 바탕으로 ITME보다 높은 **TRL 5~6** 수준으로 추정된다. 반면 ITME는 FPGA 프로토타입과 엔드투엔드 벤치마크는 있으나 실제 상용 서비스 실증 근거가 제한되어 **TRL 4~5**로 추정된다.

그러나 시장성과 도메인 적용 관점에서는 단순한 성숙도 차이만으로 우열을 판단할 수 없다. MLA는 모델·추론 엔진 업데이트를 통해 채택될 수 있는 소프트웨어 중심 경로가 확인되는 반면, 기존 모델 변환과 런타임 호환성이라는 장벽이 있다. ITME는 CXL·RDMA·NVMe를 포함하는 인프라 투자와 통합이 필요하지만, GPU 메모리만으로 감당하기 어려운 장문·멀티턴 상태를 계층형으로 수용하려는 경로를 제공한다.

관점 간 상충도 명시적으로 존재한다.

- **어느 관점에서는 MLA의 생태계 지원이 vLLM·SGLang 구현 사례로 확인되지만**, 다른 관점에서는 MLA가 범용 프레임워크 전반에서 표준화되거나 통합 메모리 관리 체계로 안정 지원된다고 보기 어렵다고 평가한다.
- **어느 관점에서는 MLA의 KV cache 절감 및 처리량 개선 수치가 제시되지만**, 데이터센터 적용 관점에서는 동시 사용자 수, TTFT·TPOT, p95/p99, 멀티노드 확장성과 TCO가 동일 조건에서 검증되지 않았다고 본다.
- **어느 관점에서는 ITME가 FPGA·CXL 메모리 모듈 기반으로 시스템 수준 실증을 수행한 것으로 평가되지만**, 다른 관점에서는 CXL 3.0 기반 운영급 전체 스택 및 실제 클라우드 환경 검증이 향후 과제로 남아 있다고 본다.
- **어느 관점에서는 CXL 기반 KV cache 서버의 제품화 사례가 산업 신호로 제시되지만**, 다른 관점에서는 이를 ITME 자체의 상용화·고객 채택으로 해석할 수 없다고 구분한다.

결론적으로 MLA와 ITME는 배타적 기술이라기보다, 각각 KV cache의 기본 표현량과 저장 계층을 다루는 상이한 접근이다. 두 기술의 결합 효과, 실제 클라우드 비용, 지연시간, 멀티테넌시, 장애 대응 및 장기 운영성은 제공 자료만으로 판단하기 어렵다.

---

## 5. 시사점

1. **KV cache 병목은 모델과 인프라 양쪽에서 다뤄야 하는 문제다.**  
   MLA는 요청당 KV cache 표현량을 줄이는 방향이며, ITME는 GPU·호스트 메모리 한계를 넘어 저장 용량을 확장하는 방향이다. 장문 컨텍스트와 멀티턴 워크로드에서는 두 접근이 상호보완적으로 검토될 여지가 있다.

2. **MLA는 모델·커널·프레임워크의 통합성이 핵심 검증 항목이다.**  
   DeepSeek 모델 적용과 SGLang·vLLM 관련 사례는 확인되지만, 기존 GQA/MHA 모델 변환, 정확도 유지, 커널 성숙도, KV cache 형식 상호운용성은 서비스 도입 전 검증이 필요하다.

3. **ITME는 메모리 용량 확장과 지연시간 변동의 절충을 관리해야 한다.**  
   CXL·NVMe·RDMA 계층은 대규모 KV cache 수용 가능성을 제공하지만, 원격 접근과 프리페치 실패가 decode 단계의 지연시간에 미칠 영향을 별도로 검증해야 한다.

4. **데이터센터 도입 판단에는 동일 조건의 운영 지표가 필요하다.**  
   MLA와 ITME를 평가할 때는 동일 모델, 동일 GPU 구성, 동일 컨텍스트 길이, 동일 동시성, 동일 요청 분포에서 다음 항목을 함께 측정할 필요가 있다.  
   - GPU HBM 및 호스트 메모리 사용량  
   - TTFT, TPOT, p95/p99 지연시간  
   - 처리량 및 GPU 활용률  
   - 네트워크·CXL·NVMe 대역폭  
   - 멀티테넌시 공정성  
   - 장애 복구 시간  
   - 전력과 총소유비용(TCO)

5. **인접 시장 성장 수치와 개별 기술의 시장성을 구분해야 한다.**  
   AI 추론, KV cache 오프로딩, CXL 인프라 시장의 성장 전망은 두 기술의 잠재 수요 배경이 될 수 있으나, MLA 또는 ITME 자체의 시장 규모·매출·채택률로 해석할 수는 없다.

---

## 6. 한계점

본 보고서는 제공된 논문 발췌와 웹 자료에 기반한 **공개 정보 기반 추정**이다. 따라서 논문 발표 시점 이후의 실제 제품화, 고객 도입, 장기 서비스 운영, 성능 재현성, 장애율 및 수익성은 충분히 확인되지 않을 수 있다.

특히 다음 사항은 제한적이다.

- MLA와 ITME 자체의 직접 시장 규모, CAGR, 매출, 시장점유율
- 실제 클라우드 사업자 및 고객사의 장기 프로덕션 도입 사례
- 동일 조건에서의 MLA와 ITME 직접 비교 성능
- TTFT, TPOT, p95/p99, 멀티테넌시, 장애 복구, 전력, TCO의 운영급 데이터
- MLA 전환 후 정확도 보존, 재학습 필요성 및 전환 비용
- ITME의 CXL·RDMA·NVMe 장비 조달성, 호환성, 서비스 운영 복잡도

확증편향을 줄이기 위해 다음 원칙을 적용했다.

1. 각 기술의 성능 수치는 원문 또는 특정 벤치마크 조건에 한정해 해석했다.  
2. CXL 기반 제품 사례를 ITME 자체의 상용화 사례로 확대 해석하지 않았다.  
3. vLLM·SGLang의 MLA 지원 사례를 전 프레임워크의 표준 지원으로 일반화하지 않았다.  
4. 웹 자료의 비용 절감·동시성 개선 주장은 모델, GPU, 워크로드 조건이 불충분한 경우 일반적 운영 성과로 사용하지 않았다.  
5. 두 기술의 우열이나 도입 권고를 제시하지 않고, 해결 범위·성숙도·운영 조건의 차이로 구분했다.  

---

## REFERENCE

- data/raw/DeepSeek-V2.pdf (p.1)  
- data/raw/DeepSeek-V2.pdf (p.2)  
- data/raw/DeepSeek-V2.pdf (p.4)  
- data/raw/DeepSeek-V2.pdf (p.20)  
- data/raw/TransMLA_MLA_Is_All_You_Need.pdf (p.2)  
- data/raw/TransMLA_MLA_Is_All_You_Need.pdf (p.5)  
- data/raw/TransMLA_MLA_Is_All_You_Need.pdf (p.11)  
- data/raw/SGLang_v0.3_Release_7x_ Faster_DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image_Video LLaVA-OneVision - LMSYS Org.pdf (p.1)  
- data/raw/SGLang_v0.3_Release_7x_ Faster_DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image_Video LLaVA-OneVision - LMSYS Org.pdf (p.4)  
- data/raw/ITME.pdf (p.1)  
- data/raw/ITME.pdf (p.2)  
- data/raw/ITME.pdf (p.3)  
- data/raw/ITME.pdf (p.7)  
- data/raw/ITME.pdf (p.9)  
- data/raw/ITME.pdf (p.10)  
- data/raw/ITME.pdf (p.11)  
- data/raw/PIMCXL_no-refs.pdf (p.2)  
- data/raw/PIMCXL_no-refs.pdf (p.3)  
- data/raw/PIMCXL_no-refs.pdf (p.4)  
- data/raw/PIMCXL_no-refs.pdf (p.5)  
- data/raw/KIVI_no-refs.pdf (p.1)  
- How CXL Transforms RAG and KV Cache Performance - https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance  
- An Internet for the KV Cache: Rethinking Classical Infrastructure Boundaries in the LLM Inference Age - https://arxiv.org/html/2608.01526v1  
- KV-Cache Offloading Infrastructure Market Research Report 2033 - https://growthmarketreports.com/report/kv-cache-offloading-infrastructure-market  
- ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories - https://arxiv.org/html/2606.12556v2  
- Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM - https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm  
- Predictive Multi-Tier Memory Management for KV Cachein Large-Scale GPU Inference - https://arxiv.org/html/2604.26968v2  
- Part 3 — Implementation/Engine-Level: Choosing the Runtime That Gives You These for Free - https://pub.towardsai.net/part-3-implementation-engine-level-choosing-the-runtime-that-gives-you-these-for-free-b0e9081205b0  
- KV Cache Optimization for LLMs 2026: Engineering Guide - https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide  
- Upgrade GQA models to MLA for massive KV Cache Savings - https://drivenets.com/blog/how-to-upgrade-gqa-models-to-mla-for-massive-kv-cache-savings  
- Nvidia, CXL, and the Battle to Improve AI Inference Economics - I/O Fund - https://io-fund.com/ai-stocks/nvidia-cxl-ai-inference-economics  
- Penguin Solutions - Penguin Solutions Introduces Industry's First Production-Ready CXL-Based KV Cache Server - https://ir.penguinsolutions.com/news/news-details/2026/Penguin-Solutions-Introduces-Industrys-First-Production-Ready-CXL-Based-KV-Cache-Server/default.aspx  
- Using CXL Fabric-Attached Memory to Enable Shared KV Caches Across GPUs for Faster LLM Inference | SNIA | Experts on Data - https://www.snia.org/sniadeveloper/session/19665