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

본 보고서는 데이터센터/클라우드 서빙 환경에서 증가하는 장문 컨텍스트·멀티턴·에이전트형 추론의 KV cache 메모리 병목을 대상으로, DeepSeek-V2의 Multi-head Latent Attention(MLA)과 ITME(Inference Tiered Memory Expansion)를 비교가 아닌 병렬 관점에서 평가했다. MLA는 KV 표현 자체를 저랭크 latent 표현으로 압축하는 모델·소프트웨어 계층 기술이며, ITME는 CXL 하이브리드 메모리·DRAM cache·NVMe·RDMA를 이용해 GPU HBM 밖의 메모리 계층을 확장하는 하드웨어·시스템 계층 기술이다.

공개 자료 기준으로 MLA는 DeepSeek-V2 모델 및 SGLang·vLLM 관련 통합·최적화 사례가 확인되어 TRL 5~6으로, ITME는 FPGA 프로토타입과 ShareGPT 기반 평가가 확인되어 TRL 4~5로 추정된다. 다만 두 등급 모두 실제 상용 서비스의 안정성·장기 운영·고객 배포를 입증하는 등급은 아니다. MLA는 요청별 KV cache 크기를 줄이는 방향, ITME는 HBM을 넘는 가중치와 KV 상태를 계층형 메모리에 수용하는 방향으로 작동하므로 결합 가능성은 있으나, 결합 시 성능·비용·운영 복잡도는 추가 검증이 필요하다.

두 기술 모두 독립 시장 규모, 실제 도입 기업 수, TCO 절감액, P95/P99 지연시간, 장애 복구 및 멀티테넌시 운영 결과에 대한 직접 근거가 부족하다. 따라서 본 평가는 특정 기술의 우열이나 도입 권고가 아니라, 데이터센터/클라우드 서빙에서 검증해야 할 기술적·운영적 조건을 정리한 공개 정보 기반 평가다.

---

## 1. 분석 배경

대규모 언어모델의 자동회귀 추론에서는 이전 토큰의 Key와 Value를 저장하는 KV cache가 필요하다. 생성 과정에서 이전 문맥을 반복 계산하지 않고 attention 연산에 활용하기 위해서다. 그러나 컨텍스트 길이, 동시 요청 수, 멀티턴 세션 수가 증가할수록 KV cache는 GPU 메모리 용량과 메모리 대역폭을 압박한다.

데이터센터/클라우드 서빙에서는 이 문제가 단일 모델의 메모리 효율을 넘어 다음 운영 요소와 연결된다.

- 장문 컨텍스트 및 멀티턴 세션의 상태 유지
- 다수 동시 요청의 GPU HBM 수용량
- GPU 증설, 서버 수, 메모리 계층 구성에 따른 비용
- 프리픽스 cache 재사용 및 데이터 이동 지연
- 처리량과 평균·꼬리 지연시간의 상충
- 멀티테넌시, 장애 복구, 데이터 격리와 같은 운영 문제

DeepSeek-V2는 KV 표현을 구조적으로 압축하는 MLA를 적용했고, ITME는 GPU HBM·호스트 메모리·NVMe·원격 저장소 사이에 CXL 하이브리드 메모리 계층을 추가해 모델 가중치와 KV cache를 확장하는 방식을 제안한다. 본 보고서는 두 기술이 KV cache 병목의 어느 위치를 완화하는지와 데이터센터/클라우드 서빙 적용 시 검증해야 할 조건을 분석한다.

---

## 2. 기술 선정

### SW 기술: DeepSeek-V2의 Multi-head Latent Attention(MLA)

MLA는 Key와 Value를 저랭크 latent 표현으로 공동 압축해 KV cache 저장량을 줄이는 attention 구조다. DeepSeek-V2는 MLA와 희소 MoE 구조를 결합해 최대 128K 컨텍스트를 지원하도록 제시됐다. 모델 구조에서 KV cache 자체의 표현 크기를 줄인다는 점에서, 장문 컨텍스트와 동시 요청이 많은 클라우드 추론 환경의 메모리 병목 분석 대상으로 선정했다.  
근거: `data/raw/DeepSeek-V2.pdf` (p.1, p.2, p.3, p.4, p.20)

### HW 기술: ITME(Inference Tiered Memory Expansion)

ITME는 CXL 하이브리드 메모리, 내부 DRAM cache, NVMe SSD, RDMA 및 DMA 기반 프리페칭을 결합해 GPU가 원격의 SSD-backed 용량을 확장 메모리 계층처럼 활용하도록 하는 아키텍처다. GPU HBM만으로 수용하기 어려운 대형 모델 가중치와 장기 KV cache를 다계층 메모리에 배치하는 접근이므로, 데이터센터 인프라 관점의 대표 기술로 선정했다.  
근거: `data/raw/ITME.pdf` (p.1, p.2, p.3, p.7, p.9, p.10, p.11)

---

## 3. 기술 개요

### 3.1 DeepSeek-V2 MLA

MLA는 기존 MHA처럼 토큰별 Key·Value 전체를 저장하는 방식과 달리, Key와 Value를 저차원 latent vector로 압축해 저장하는 구조다. 일부 query·key 성분에는 RoPE를 적용하며, KV cache 용량을 줄이면서 attention 품질을 유지하려는 목적을 가진다.

DeepSeek-V2 논문 초록은 DeepSeek 67B 대비 KV cache를 93.3% 줄이고 최대 생성 처리량을 5.76배 높였으며 학습 비용을 42.5% 절감했다고 제시한다. 다만 이는 MLA만의 단독 효과가 아니라 DeepSeek-V2의 모델·시스템 설계와 특정 비교 조건을 포함한 결과로 해석해야 한다.  
근거: `data/raw/DeepSeek-V2.pdf` (p.1, p.2)

SGLang v0.3은 MLA를 위해 weight absorption, grouped decoding kernel, FP8 batched MatMul, FP8 KV cache quantization 등의 최적화를 구현했으며, H100 및 ShareGPT 조건에서 기준 시스템 대비 3~7배의 출력 처리량을 보고했다. 그러나 해당 수치는 특정 하드웨어, 데이터셋, 정밀도 및 구현 조건의 결과다.  
근거: `data/raw/SGLang_v0.3_Release_7x_ Faster_DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image_Video LLaVA-OneVision - LMSYS Org.pdf` (p.1, p.4)

주요 한계는 다음과 같다.

- MLA는 모델 attention 구조에 의존하므로 기존 MHA/GQA 모델에 무변경으로 적용되는 기술이 아니다.
- GQA 기반 모델을 MLA로 전환하려면 모델 구조·가중치·RoPE 처리·추론 런타임의 호환성 검증이 필요하다.
- MLA 전용 kernel과 런타임 최적화 수준에 따라 실제 효과가 달라질 수 있다.
- KV cache를 줄이더라도 prefill 연산, attention 연산량, MoE 라우팅, GPU 간 통신 병목은 별도 문제로 남을 수 있다.
- 실제 서비스의 P95/P99 지연시간, TCO, 장애 대응에 대한 직접 근거는 부족하다.

### 3.2 ITME

ITME는 GPU HBM(T1), 호스트 메모리(T2), 단일 서버 NVMe SSD(T3), 클러스터 원격 공유 스토리지(T4) 사이에 CXL 하이브리드 메모리 기반의 T3.5 계층을 추가하는 방식이다. CXL 하이브리드 메모리는 내부 DRAM cache와 SSD-backed 용량을 결합하며, GPU 서버는 RDMA를 통해 이를 바이트 주소 지정 가능한 확장 메모리처럼 접근한다.  
근거: `data/raw/ITME.pdf` (p.1, p.2, p.3)

ITME는 모델 가중치의 layer-wise 접근 패턴과 장문·멀티턴 KV cache의 append-write 및 sequential-restore 특성을 이용해 프리페칭을 수행한다. GPU 계산과 데이터 이동을 겹쳐 원격 메모리·스토리지 지연을 숨기는 것이 핵심이다.

FPGA 프로토타입 평가에서 내부 DRAM cache로 프리페칭된 CXL 하이브리드 메모리는 약 18GB/s에 가까운 처리량을 기록했다. ShareGPT 기반, 128개 동시 대화 및 최대 5턴 조건에서는 긴 대화 구간에서 CPU 오프로딩 기준선 대비 최대 35.7% 처리량 향상이 제시됐다. 다만 초기 턴에서는 작업 집합이 호스트 메모리에 머물러 CPU 오프로딩과 유사한 성능을 보였으며, 성능 이점은 원격 계층으로의 축출과 프리페칭이 유효한 조건에서 두드러진다.  
근거: `data/raw/ITME.pdf` (p.3, p.9, p.10)

주요 한계는 다음과 같다.

- GPU HBM보다 낮은 대역폭 및 높은 지연의 계층을 사용하므로, 프리페칭 정확도와 연산·통신 중첩 가능성에 의존한다.
- CXL 하이브리드 메모리, DRAM cache, NVMe, RNIC, RDMA, 호스트 staging buffer, DMA 파이프라인을 함께 구성해야 한다.
- I/O contention, 네트워크 변동, 원격 계층 장애가 발생할 경우 성능 변동과 운영 복잡성이 커질 수 있다.
- 실제 클라우드 환경의 장기 운영, 멀티테넌시, 보안 격리, 장애 복구, 랙 단위 확장성은 추가 검증이 필요하다.

---

## 4. 관점별 평가

### 4.1 기술성숙도 관점

#### DeepSeek-V2 MLA: TRL 5~6 — 공개 정보 기반 추정

MLA는 DeepSeek-V2 및 DeepSeek-V2-Lite 계열에 통합돼 공개됐고, DeepSeek-V2는 최대 128K 컨텍스트 지원 모델로 제시됐다. 논문에는 DeepSeek 67B 대비 KV cache 93.3% 감소와 최대 생성 처리량 5.76배 향상 등 정량적 시스템 결과가 포함돼 있다.  
근거: `data/raw/DeepSeek-V2.pdf` (p.1, p.2)

또한 SGLang v0.3은 MLA 전용 kernel, weight absorption, FP8 관련 최적화 및 실행 방법을 공개했다. H100 기반 ShareGPT 벤치마크에서 기준 대비 3~7배 높은 출력 처리량을 보고한 점은 이론적 제안 수준을 넘어 실제 추론 프레임워크에 통합된 시연 근거로 볼 수 있다.  
근거: `data/raw/SGLang_v0.3_Release_7x_ Faster_DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image_Video LLaVA-OneVision - LMSYS Org.pdf` (p.1, p.4)

다만 이 등급은 **공개 정보 기반 추정**이다. 제공 자료만으로는 논문의 동료심사 여부, 특정 상용 서비스의 공식 채택, 고객 배포, 장기 운영 안정성, 다양한 GPU 및 워크로드에서의 재현성을 확인할 수 없다. 논문 발표 시점의 구현·벤치마크 성과와 실제 서비스 채택 사이에는 시차가 있을 수 있으므로, TRL 6 이상 또는 상용화 단계로 단정할 근거는 부족하다.

#### ITME: TRL 4~5 — 공개 정보 기반 추정

ITME는 CXL 하이브리드 메모리, DRAM cache, 하드웨어 컨트롤러, RDMA/RNIC, GPU·호스트·원격 메모리 간 DMA 프리페칭을 결합한 통합 아키텍처를 제안했다. FPGA 프로토타입에서 약 18GB/s 처리량을 측정했으며, ShareGPT 기반 다중 대화 workload에서 CPU 오프로딩 기준선과 비교 평가를 수행했다.  
근거: `data/raw/ITME.pdf` (p.2, p.3, p.9, p.10, p.11)

장기 대화 구간에서 CPU 오프로딩 대비 최대 35.7% 처리량 향상이 제시된 점은 구성요소 수준의 개념 검증을 넘어 통합 시스템의 실험실 또는 유사 환경 시연 근거로 볼 수 있다.  
근거: `data/raw/ITME.pdf` (p.10)

다만 ITME도 **공개 정보 기반 추정**이며, 연구 논문과 FPGA 기반 평가에 근거한다. 상용 서버 제품, 클라우드 서비스 배포, 고객 운영, 장기간 장애 대응, 대규모 클러스터 확장, 전력·비용·보안 검증에 대한 공개 근거는 부족하다. 논문 발표 시점과 실제 채택 시점의 차이를 고려할 때 TRL 6 이상으로 판단하기는 어렵다.

---

### 4.2 시장성 관점

#### DeepSeek-V2 MLA

**① 시장 규모·성장성**  
MLA 자체의 시장 규모, 매출, CAGR, 설치 기반 또는 도입 기업 수는 근거 부족이다. AI 추론 시장이 2025년 1,061.5억 달러에서 2030년 2,549억 달러로 성장하고 CAGR 19.2%라는 수치는 MLA 시장이 아니라 상위 인접 시장의 전망이다.  
근거: How CXL Transforms RAG and KV Cache Performance - https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance

MLA는 KV cache 감소를 통해 장문 컨텍스트와 동시 요청 수용력을 높일 수 있는 구조로 평가되지만, 해당 기술의 독립적 시장 수요나 매출을 직접 입증하는 자료는 없다.

**② 상용화·채택 현황**  
vLLM은 MLA 모델의 대규모 서빙과 분산 배포를 다루며, Red Hat은 vLLM 환경에서 MLA·FP8 최적화 사례를 제시한다. 이는 추론 소프트웨어 생태계에서 MLA 지원 및 벤치마크 사례가 있음을 의미한다.  
근거:  
- vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP | vLLM Blog - https://vllm.ai/blog/2025-12-17-large-scale-serving  
- Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM - https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm  

그러나 MLA 자체의 상용 제품 출시, 실제 고객 수, 운영 중인 클라우드 서비스 수, 고객별 도입 효과는 근거 부족이다. AWS SageMaker의 DeepSeek-R1 증류 모델 배포 사례는 확인되지만, DeepSeek-V2 MLA 적용 사례로 볼 근거는 없다.  
근거: Deploy DeepSeek-R1 distilled models on Amazon SageMaker using a Large Model Inference container | Artificial Intelligence - https://aws.amazon.com/blogs/machine-learning/deploy-deepseek-r1-distilled-models-on-amazon-sagemaker-using-a-large-model-inference-container

**③ 생태계 지지**  
vLLM 및 SGLang의 MLA 관련 통합·최적화 사례는 확인된다. 다만 MLA 전용의 산업 표준, 공통 API, 컨소시엄 또는 모든 프레임워크에서의 범용 지원은 근거 부족이다. 현재 확인되는 생태계는 모델 구조와 특정 추론 엔진 간 통합 중심으로 해석하는 것이 적절하다.

#### ITME

**① 시장 규모·성장성**  
ITME 자체의 시장 규모, TAM/SAM, 매출, CAGR, 설치 기반은 근거 부족이다. AI 추론 시장의 성장 전망은 ITME가 겨냥할 수 있는 상위 시장의 신호일 뿐, ITME 시장 규모를 의미하지 않는다.  
근거: How CXL Transforms RAG and KV Cache Performance - https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance

장문 컨텍스트에서 단일 사용자 KV cache가 20~50GB에 이를 수 있고, 다수 동시 에이전트가 HBM 용량을 빠르게 초과할 수 있다는 인접 자료는 메모리 계층화 수요를 뒷받침한다. 그러나 ITME의 직접 시장 수치는 아니다.  
근거: Accelerating Inference with KV Cache, AMD Instinct, and VAST Data - https://www.vastdata.com/blog/beyond-hbm-limits-accelerating-inference-with-vast-amd

**② 상용화·채택 현황**  
ITME는 SK hynix CMM, KIOXIA PCIe Gen5 NVMe SSD 등을 포함한 연구 평가 플랫폼에서 검증된 아키텍처로 제시된다. 그러나 ITME라는 명칭의 상용 제품, 클라우드 서비스, 실제 고객 도입, 배포 규모는 확인되지 않는다.  
근거: ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories - https://arxiv.org/html/2606.12556v2

CXL 기반 메모리 확장, NVIDIA Dynamo, LMCache, KV cache offload와 같은 인접 솔루션과 프레임워크는 확인되지만, 이들이 ITME 자체의 상용 채택을 증명하지는 않는다.  
근거:  
- A High-Performance KV Cache Platform for Large-Scale AI ... - https://www.redbooks.ibm.com/docs/MD260021/MD260021.html  
- KV Cache Tiering: Why GPU Memory Alone Won't Scale | Aerospike - https://aerospike.com/blog/kv-cache-tiering-gpu-memory-limits  

**③ 생태계 지지**  
ITME는 vLLM의 PagedAttention 및 KV cache 관리와 연결되는 구조로 설명되며, CXL, RDMA, NVMe, SPDK 등 관련 기술 생태계와 접점을 갖는다.  
근거: ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories | alphaXiv - https://www.alphaxiv.org/abs/2606.12556

그러나 ITME 자체를 지원하는 표준 API, 산업 컨소시엄, 주요 클라우드 사업자의 관리형 서비스, 특정 서버 플랫폼의 공식 지원은 근거 부족이다.

---

### 4.3 이해관계자 관점

#### ① 경쟁 기술 진영

MLA의 인접 대안으로는 GQA, MQA, KV cache 양자화, paged attention, prefix caching 등이 제시된다. GQA와 MQA는 KV head 공유로 저장량을 줄이는 방식이며, MLA는 KV 표현 자체를 저차원 latent 구조로 압축한다는 점에서 접근이 다르다.  
근거: Multi-Head Latent Attention (MLA) | Sebastian Raschka, PhD - https://sebastianraschka.com/llm-architecture-gallery/mla

MLA는 다른 KV cache 최적화 기술과 완전한 대체 관계라기보다 결합 가능한 관계일 수 있다. 예를 들어 FP8 KV 양자화, paged attention, prefix caching은 MLA와 함께 적용될 수 있는 시스템 수준 최적화로 제시된다. 다만 특정 경쟁사가 MLA에 직접 대응한 공식 제품·시장 점유율 자료는 근거 부족이다.  
근거: KV Cache Optimization for LLMs 2026: Engineering Guide - https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide

ITME의 인접 대안에는 GPU 메모리 증설, CPU 메모리·NVMe 오프로딩, CXL 기반 메모리 확장, 계층형 메모리, NVIDIA CMX, NVMe/RDMA 기반 추론 스토리지가 포함된다. 이들은 GPU 외부의 메모리·스토리지 계층을 이용해 KV cache 병목을 줄이려는 접근이라는 점에서 ITME와 유사한 문제를 다룬다.  
근거:  
- Nvidia, CXL, and the Battle to Improve AI Inference Economics - I/O Fund - https://io-fund.com/ai-stocks/nvidia-cxl-ai-inference-economics  
- Less DRAM, Same KV Performance: Persistent KV Caching with CXL-Attached AI SSDs | SNIA | Experts on Data - https://www.snia.org/sniadeveloper/session/19721  
- `data/raw/PIMCXL_no-refs.pdf` (p.2, p.4, p.5, p.11)

그러나 이들 사례는 ITME에 대한 특정 경쟁사의 직접 대응으로 단정할 수 없으며, 동일 조건의 공개 비교 벤치마크도 근거 부족이다.

#### ② 도입 기업·개발자

MLA는 동일 GPU 메모리에서 더 긴 컨텍스트 또는 더 많은 활성 KV 상태를 유지하려는 개발자에게 관심 대상이 될 수 있다. TransMLA는 기존 MHA/GQA 모델을 MLA로 변환하고 KV cache를 92.97% 축소한 실험을 제시했다. 다만 이는 변환 모델과 소비자용 AI 가속기 조건의 결과이며, 데이터센터 운영 성과로 일반화할 수 없다.  
근거: `data/raw/TransMLA_MLA_Is_All_You_Need.pdf` (p.11)

SGLang은 MLA 전용 최적화를 적용해 H100 기반 조건에서 3~7배 처리량 향상을 보고했지만, SGLang 측도 DeepSeek 논문 성과를 완전히 재현하기 위해 추가 최적화가 필요하다고 밝혔다.  
근거: `data/raw/SGLang_v0.3_Release_7x_ Faster_DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image_Video LLaVA-OneVision - LMSYS Org.pdf` (p.1, p.4)

MLA 도입 장벽은 기존 MHA/GQA 모델의 구조적 호환성, 가중치 계승, 재학습·변환 검증, RoPE 처리, kernel 및 런타임 통합이다. 모델 품질 변화와 장기 운영 비용에 대한 직접 자료는 추가 검증이 필요하다.

ITME는 장문·멀티턴·에이전트형 워크로드에서 대규모 KV cache와 모델 가중치를 더 오래 유지하려는 인프라 운영자에게 의미가 있다. 모델 가중치의 layer-wise 접근과 KV cache의 순차적 복원 패턴을 활용해 프리페칭과 GPU 계산을 겹치는 구조가 제시된다.  
근거: `data/raw/ITME.pdf` (p.2, p.3, p.7)

그러나 ITME는 CXL 하이브리드 메모리, DRAM cache, NVMe, RDMA 네트워크, 호스트 메모리 staging, DMA 프리페칭을 함께 관리해야 한다. 따라서 일반적인 CPU/NVMe 오프로딩보다 조달·통합·운영·장애 처리 복잡성이 높을 가능성이 있다. 실제 클라우드 사업자나 데이터센터 운영자의 장기 도입 경험, 조달 비용, SLA 자료는 근거 부족이다.

#### ③ 투자 업계

MLA 자체에 대한 투자금, 인수·합병, 증권사 분석, 투자기관의 직접 평가 자료는 근거 부족이다. 기술 해설 자료는 MLA를 장문 컨텍스트와 KV cache 병목에 대한 대응으로 설명하지만, 이는 투자 업계의 합의나 정량적 투자 판단을 의미하지 않는다.  
근거: The DeepSeek Series: A Technical Overview - https://martinfowler.com/articles/deepseek-papers.html

DeepSeek 계열 모델의 광범위한 채택에는 규제, 데이터 프라이버시, 보안, 지식재산권 이슈가 장애가 될 수 있다는 분석도 존재한다. 다만 이는 MLA 자체보다 DeepSeek 모델·서비스 배포의 리스크와 관련된 평가다.  
근거: Navigating DeepSeek’s disruption: Opportunities and challenges in AI advancement | State Street - https://www.statestreet.com/in/en/insights/deepseek-disruption-ai-advancement

ITME 자체의 투자 동향, 기업별 투자 계획, 인수 사례, 애널리스트 평가도 근거 부족이다. CXL·KV cache 인프라에 대한 산업계 관심은 확인되지만, 이를 ITME의 사업성 또는 투자 우위로 해석할 수는 없다.  
근거:  
- Inference Tokenomics: How CXL Memory Expansion Improves AI Economics - https://www.asteralabs.com/resources/blog/inference-tokenomics-how-cxl-memory-expansion-improves-ai-economics  
- Nvidia, CXL, and the Battle to Improve AI Inference Economics - I/O Fund - https://io-fund.com/ai-stocks/nvidia-cxl-ai-inference-economics  

---

### 4.4 도메인 적용 관점

본 절의 평가는 **데이터센터/클라우드 서빙 환경에 한정**된다. 온디바이스 환경이나 특정 장문 컨텍스트 애플리케이션에서는 전력, 물리적 메모리, 지연시간, 네트워크 조건의 중요도가 달라질 수 있다.

#### DeepSeek-V2 MLA

**처리 가능 규모**  
DeepSeek-V2는 총 236B 파라미터, 토큰당 21B 활성 파라미터, 최대 128K 컨텍스트를 지원하는 모델로 제시됐다. 이 수치는 모델 구조의 설명이며, 특정 클라우드 배포의 실제 동시 사용자 수를 의미하지 않는다.  
근거: `data/raw/DeepSeek-V2.pdf` (p.2)

MLA는 요청별 KV 표현을 줄이므로 동일 GPU 메모리에서 더 많은 세션 또는 더 긴 문맥을 수용하는 방향으로 작동한다. 그러나 실제 동시 요청 수 증가율, 멀티테넌시 수용량, GPU당 세션 수는 모델 크기, 정밀도, 배치 정책, GPU 종류, 서빙 엔진에 좌우되므로 근거 부족이다.

TransMLA는 KV cache를 92.97% 줄인 변환 실험을 제시했지만, 해당 실험은 24GB·40GB·64GB 소비자용 AI 가속기와 vLLM 조건에 기반한다. 따라서 데이터센터 GPU 또는 DeepSeek-V2 운영 성과로 직접 환산할 수 없다.  
근거: `data/raw/TransMLA_MLA_Is_All_You_Need.pdf` (p.11)

**비용 절감 효과**  
MLA는 KV cache가 GPU HBM을 점유하는 비중을 낮춰 동일 동시성에서 GPU 메모리 압박을 완화하거나, 더 긴 컨텍스트를 위한 GPU 증설 시점을 늦출 가능성이 있다. 이는 GPU 임대, 서버 수, 전력, 냉각 비용에 긍정적일 수 있는 방향성이다.

그러나 DeepSeek-V2 MLA 적용으로 GPU 수, 서버 수, 전력, 냉각, TCO가 얼마나 감소하는지에 대한 직접 측정값은 없다. KV cache 감소율을 TCO 절감률 또는 GPU 감축률로 해석하는 것은 근거 부족이다.

**제약**  
MLA는 모델 구조·추론 kernel·프레임워크의 통합이 전제된다. KV cache가 감소해도 모델 가중치, MoE 라우팅, GPU 간 통신, prefill 계산, attention 계산이 새로운 병목이 될 수 있다. 또한 처리량을 위해 배치와 동시성을 높이면 꼬리 지연시간이 증가할 수 있으나, MLA의 P95/P99 지연시간에 대한 직접 근거는 부족하다.

#### ITME

**처리 가능 규모**  
ITME는 GPU HBM, 호스트 메모리, NVMe, 원격 공유 스토리지에 CXL 하이브리드 메모리 계층을 추가해 GPU 밖으로 모델 가중치와 KV cache를 확장하는 구조다. OPT-175B FP16 가중치 325GB를 CXL 하이브리드 메모리에 오프로딩하는 예시가 제시됐다.  
근거: `data/raw/ITME.pdf` (p.1, p.2, p.3)

FPGA 프로토타입은 내부 DRAM cache에 프리페치된 조건에서 약 18GB/s 처리량을 기록했다. 이는 특정 프로토타입과 전송 조건의 결과이며, 상용 GPU 서버·랙 단위 운영의 처리량이나 P99 지연시간으로 해석할 수 없다.  
근거: `data/raw/ITME.pdf` (p.3)

ITME는 장기 세션과 prefix KV cache를 GPU HBM 밖에 보존하고 필요 시 복원하는 방향으로 확장성을 제공한다. 그러나 서버당 동시 세션 수, 최대 TB 용량, 노드·랙 단위 선형 확장성은 근거 부족이다.

**비용 절감 효과**  
ITME는 모든 가중치와 KV cache를 고가 GPU HBM에 상주시킬 필요를 줄이고, CXL·DRAM·NVMe·원격 저장소를 용량 계층으로 활용하려는 접근이다. GPU HBM 부족으로 추가 GPU를 배치하거나 세션 상태를 재계산해야 하는 압박을 낮출 가능성이 있다.

반면 CXL 하이브리드 메모리, DRAM cache, NVMe, RDMA 네트워크, 하드웨어·소프트웨어 프리페칭 스택이 추가된다. 따라서 GPU 증설 회피 효과와 추가 인프라 비용을 함께 비교해야 한다. ITME의 순 TCO, 전력, 냉각, GPU 대체 규모, 데이터 이동 비용은 근거 부족이다.

ITME의 최대 35.7% 처리량 향상은 실험 조건에서의 성능 결과이며, 비용 35.7% 절감 또는 GPU 35.7% 감소를 의미하지 않는다.  
근거: `data/raw/ITME.pdf` (p.10)

**제약**  
ITME의 성능은 프리페칭 정확도, 데이터 접근 예측성, GPU 계산과 통신의 중첩, 원격 계층으로의 축출 여부, I/O contention에 영향을 받는다. 초기 턴에서는 작업 집합이 호스트 메모리에 있어 CPU 오프로딩 기준선과 유사한 성능이 관찰됐으며, 장기 대화에서 원격 계층 사용이 증가할 때 차이가 나타났다.  
근거: `data/raw/ITME.pdf` (p.9, p.10)

또한 GPU–호스트–CXL/DRAM–NVMe–원격 저장소로 데이터 경로가 확장되므로 장애 지점도 증가한다. 장애 복구, 세션 상태 일관성, 멀티테넌시 보안·격리, 네트워크 변동 시 SLA, 실제 클라우드 환경의 P95/P99 지연시간은 추가 검증이 필요하다.

---

### 4.5 종합의견

MLA와 ITME는 모두 장문 컨텍스트, 멀티턴 대화, 에이전트형 추론에서 증가하는 KV cache 문제를 대상으로 하지만, 병목을 완화하는 계층이 다르다. MLA는 KV 표현 자체를 줄이는 모델·소프트웨어 접근이고, ITME는 GPU 밖에 추가 메모리 계층을 구성하는 하드웨어·시스템 접근이다.

기술성숙도 관점에서는 MLA가 DeepSeek-V2 모델 통합과 SGLang 기반 최적화 사례를 근거로 TRL 5~6 수준으로 추정되는 반면, ITME는 FPGA 프로토타입과 연구 평가 중심이므로 TRL 4~5 수준으로 추정된다. 그러나 이는 공개 정보 기반 판단이며, 어느 기술도 실제 대규모 상용 서비스의 장기 안정성·운영성·고객 배포를 충분히 입증했다고 보기 어렵다.

시장성 관점에서는 MLA가 vLLM·SGLang과 연결된 소프트웨어 통합 사례를 보이는 반면, ITME는 CXL·RDMA·NVMe·계층형 KV cache 인프라 수요와 연결된다. 다만 두 기술 모두 독립적 시장 규모, 매출, 도입 기업 수, 상용 제품 채택 규모는 근거 부족이다.

관점 간 상충 또는 긴장 관계도 존재한다.

- **MLA 생태계에 대해**, 시장성 관점의 일부 자료는 vLLM·SGLang 기반 통합 사례를 제시하지만, 다른 자료는 MHA·GQA·MQA·MLA를 포괄하는 범용 메모리 관리 체계가 부족하다고 본다. 즉, 특정 프레임워크의 지원은 확인되지만 범용적 표준화나 일관된 지원 수준은 별도로 검증해야 한다.
- **MLA의 효율성에 대해**, 기술성숙도 관점에서는 KV cache 감소와 처리량 향상 수치가 통합 시연 근거가 되지만, 도메인 적용 관점에서는 그 수치를 데이터센터의 동시 사용자 수, GPU 감축률, TCO, P95/P99 지연시간으로 직접 환산할 수 없다고 본다.
- **ITME의 성능에 대해**, 기술성숙도 관점에서는 FPGA 및 ShareGPT 평가가 시스템 시연 근거가 되지만, 도메인 적용 관점에서는 성능 이점이 원격 축출, 장기 대화, 프리페칭 성공 등 특정 조건에 의존한다고 본다.
- **ITME의 상용화에 대해**, 인접 CXL 부품과 메모리 확장 솔루션의 시연·제품 활동은 확인되지만, 이것이 ITME 아키텍처 자체의 상용 제품화나 고객 도입을 의미하지는 않는다.

따라서 두 기술 가운데 어느 하나가 데이터센터/클라우드 서빙에서 더 높은 동시성, 더 낮은 지연시간, 더 낮은 비용 또는 더 높은 상용성을 제공한다고 판정할 직접 비교 근거는 없다. 두 기술의 결합 가능성은 구조적으로 존재하지만, 실제 결합 환경에서의 처리량, tail latency, TCO, 운영 복잡도는 추가 검증이 필요하다.

---

## 5. 시사점

1. **KV cache 최적화는 단일 계층의 문제가 아니다.**  
   MLA는 저장해야 할 KV 데이터량을 줄이고, ITME는 남은 상태와 가중치를 외부 메모리 계층으로 확장한다. 데이터센터 설계에서는 모델 구조, GPU HBM, 호스트 메모리, CXL, NVMe, RDMA, 스케줄링을 함께 고려해야 한다.

2. **처리량 수치와 운영 효과를 구분해야 한다.**  
   MLA의 KV cache 감소율이나 SGLang의 처리량 향상, ITME의 최대 35.7% 처리량 향상은 각각 특정 구현·하드웨어·워크로드의 결과다. 이를 GPU 감축, TCO 절감, 전력 절감 또는 SLA 개선으로 바로 전환해서는 안 된다.

3. **MLA는 모델·런타임 호환성이 핵심 검증 항목이다.**  
   기존 MHA/GQA 모델 전환, 가중치 계승, RoPE 처리, 정확도 변화, kernel 및 프레임워크 지원 여부가 실제 채택의 주요 변수다.

4. **ITME는 인프라 통합성과 운영 복잡성이 핵심 검증 항목이다.**  
   CXL 하이브리드 메모리, DRAM cache, NVMe, RNIC, RDMA, 호스트 staging, 프리페칭 정책을 함께 운영해야 하므로 단순 메모리 증설과 다른 검증 체계가 필요하다.

5. **데이터센터 도입 검증은 워크로드별로 수행해야 한다.**  
   장문 컨텍스트, 멀티턴 대화, 프리픽스 재사용, 높은 동시성, 낮은 QPS, 불규칙한 세션 재접속은 서로 다른 메모리 접근 패턴을 만든다. MLA와 ITME 모두 특정 패턴에서는 이점이 있을 수 있으나, 모든 워크로드에서 동일한 결과를 보장하는 근거는 없다.

6. **결합 효과는 별도 실험이 필요하다.**  
   MLA로 KV cache 크기를 줄인 뒤 ITME 계층으로 장기 상태를 확장하는 조합은 개념적으로 가능하다. 그러나 조합 시 프리페칭 이득, 데이터 이동량, tail latency, 하드웨어 비용, 운영 복잡도는 제공 자료만으로 판단할 수 없다.

---

## 6. 한계점

본 보고서는 제공된 논문, 기술 문서, 블로그 및 웹 자료에만 근거한 **공개 정보 기반 추정**이다. 논문 발표 또는 프로토타입 시점의 성능 결과가 실제 상용 채택, 고객 운영, 장기 안정성, 제품 양산 수준을 의미하지 않을 수 있다.

특히 다음 항목은 직접 근거가 부족하거나 추가 검증이 필요하다.

- MLA와 ITME 각각의 독립 시장 규모, 매출, CAGR, 도입 기업 수
- DeepSeek-V2 MLA의 실제 GPU·서버·전력·냉각 비용 절감량
- ITME의 CXL·DRAM·NVMe·RDMA 장비 비용을 포함한 순 TCO
- 동일 모델·동일 GPU·동일 동시성·동일 컨텍스트·동일 서빙 엔진 기준의 직접 비교
- 평균 지연시간, P95/P99 지연시간, 장애 복구 시간, 보안 격리 성능
- ITME의 랙·클러스터 확장성 및 실제 클라우드 운영 결과
- MLA와 ITME 결합 구성의 성능과 비용 효과

확증편향을 줄이기 위해 다음 조치를 취했다.

- 각 기술의 긍정적 성능 결과와 함께 실험 조건·재현성 한계·운영 리스크를 병기했다.
- 특정 공급자 또는 기업 자료의 성능·비용 주장을 업계 전체의 일반 결과로 확대 해석하지 않았다.
- AI 추론 시장 규모, CXL 사례, 인접 솔루션의 성능을 MLA 또는 ITME 자체의 시장성·성능으로 전환하지 않았다.
- MLA의 특정 프레임워크 지원 사례와 범용적 표준화·상용 채택을 구분했다.
- ITME 관련 부품·인접 CXL 솔루션의 상용 활동과 ITME 아키텍처 자체의 상용화·고객 도입을 구분했다.
- 직접 근거가 없는 항목은 “근거 부족” 또는 “추가 검증 필요”로 표기했다.

---

## REFERENCE

- data/raw/DeepSeek-V2.pdf (p.1)  
- data/raw/DeepSeek-V2.pdf (p.2)  
- data/raw/DeepSeek-V2.pdf (p.3)  
- data/raw/DeepSeek-V2.pdf (p.4)  
- data/raw/DeepSeek-V2.pdf (p.20)  
- data/raw/TransMLA_MLA_Is_All_You_Need.pdf (p.2)  
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
- data/raw/PIMCXL_no-refs.pdf (p.4)  
- data/raw/PIMCXL_no-refs.pdf (p.5)  
- data/raw/PIMCXL_no-refs.pdf (p.11)  
- How CXL Transforms RAG and KV Cache Performance - https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance  
- Inference Tokenomics: How CXL Memory Expansion Improves AI Economics - https://www.asteralabs.com/resources/blog/inference-tokenomics-how-cxl-memory-expansion-improves-ai-economics  
- A High-Performance KV Cache Platform for Large-Scale AI ... - https://www.redbooks.ibm.com/docs/MD260021/MD260021.html  
- Accelerating Inference with KV Cache, AMD Instinct, and VAST Data - https://www.vastdata.com/blog/beyond-hbm-limits-accelerating-inference-with-vast-amd  
- KV Cache Tiering: Why GPU Memory Alone Won't Scale | Aerospike - https://aerospike.com/blog/kv-cache-tiering-gpu-memory-limits  
- ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories | alphaXiv - https://www.alphaxiv.org/abs/2606.12556  
- ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories - https://arxiv.org/html/2606.12556v2  
- vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP | vLLM Blog - https://vllm.ai/blog/2025-12-17-large-scale-serving  
- Deploy DeepSeek-R1 distilled models on Amazon SageMaker using a Large Model Inference container | Artificial Intelligence - https://aws.amazon.com/blogs/machine-learning/deploy-deepseek-r1-distilled-models-on-amazon-sagemaker-using-a-large-model-inference-container  
- Enhancing DeepSeek models with MLA and FP8 optimizations in vLLM - https://www.redhat.com/en/blog/enhancing-deepseek-models-mla-and-fp8-optimizations-vllm  
- Multi-Head Latent Attention (MLA) | Sebastian Raschka, PhD - https://sebastianraschka.com/llm-architecture-gallery/mla  
- KV Cache Optimization for LLMs 2026: Engineering Guide - https://www.digitalapplied.com/blog/kv-cache-optimization-techniques-2026-engineering-guide  
- Nvidia, CXL, and the Battle to Improve AI Inference Economics - I/O Fund - https://io-fund.com/ai-stocks/nvidia-cxl-ai-inference-economics  
- Less DRAM, Same KV Performance: Persistent KV Caching with CXL-Attached AI SSDs | SNIA | Experts on Data - https://www.snia.org/sniadeveloper/session/19721  
- The DeepSeek Series: A Technical Overview - https://martinfowler.com/articles/deepseek-papers.html  
- Navigating DeepSeek’s disruption: Opportunities and challenges in AI advancement | State Street - https://www.statestreet.com/in/en/insights/deepseek-disruption-ai-advancement