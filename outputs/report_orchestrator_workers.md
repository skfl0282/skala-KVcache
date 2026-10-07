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

본 보고서는 데이터센터/클라우드 서빙 환경에서 증가하는 장문 컨텍스트 및 다중 동시 요청에 따른 KV cache 메모리 문제를 대상으로, 소프트웨어 기술인 **DeepSeek-V2의 Multi-Head Latent Attention(MLA)**와 하드웨어·시스템 기술인 **ITME(Inference Tiered Memory Expansion)**를 비교·평가하였다. 두 기술은 동일한 문제를 다루지만, MLA는 모델 내부의 KV 표현을 압축하는 접근이고 ITME는 GPU·호스트 메모리를 넘어 원격 CXL-hybrid memory 계층까지 활용하는 접근이라는 차이가 있다.

제공 자료 기준으로 MLA는 DeepSeek-V2 논문에서 KV cache 감소 및 생성 처리량 관련 실험 결과가 제시되고, SGLang·Triton 관련 구현 및 일부 모델 아키텍처 채택 사례도 확인된다. 다만 실제 클라우드 운영 환경에서의 동시 사용자 수, GPU 수 감소, TCO 절감, 표준화 수준 및 광범위한 상용 도입은 근거 부족이다. ITME는 CXL-hybrid memory와 RDMA를 이용한 계층형 메모리 확장 프로토타입으로서 특정 실험에서 처리량 및 TTFT 개선 결과가 보고되었으나, 상용 제품화, 고객 도입, 다중 노드·다중 랙 운용, TCO 및 장애 대응 측면은 추가 검증이 필요하다.

따라서 두 기술의 우열이나 시장 성공 여부를 판단하기보다는, MLA는 모델·추론 소프트웨어 계층의 메모리 효율화 기술, ITME는 데이터센터 메모리 계층 확장 기술로 구분하여 검토할 필요가 있다.

---

## 1. 분석 배경

대규모 언어모델(LLM)의 추론 과정에서는 이전 토큰의 attention 계산 결과를 재사용하기 위해 Key와 Value 정보를 저장하는 **KV cache**가 필요하다. KV cache는 응답 생성 과정에서 반복 계산을 줄이는 역할을 하지만, 컨텍스트 길이와 동시 요청 수가 늘어날수록 GPU 메모리 사용량을 크게 증가시키는 요인이 될 수 있다.

특히 데이터센터/클라우드 서빙 환경에서는 장문 컨텍스트, 다중 사용자 동시 요청, 에이전트형 AI의 지속 상태 관리, 검색증강생성(RAG) 기반 응답 생성 등이 확산될수록 KV cache의 용량과 데이터 이동이 운영 병목으로 작용할 가능성이 있다. 관련 자료는 장문 컨텍스트 및 에이전트형 AI가 KV cache, 그래프 데이터, 벡터 데이터베이스, 지속적 에이전트 상태에 대한 메모리 수요를 높일 수 있다고 설명한다.  
근거: [AI Data Center CXL Memory Expansion and Pooling Infrastructure Market: 2032](https://www.knowledge-sourcing.com/report/ai-data-center-cxl-memory-expansion-and-pooling-infrastructure-market), [How CXL Transforms RAG and KV Cache Performance](https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance)

이에 따라 KV cache 문제는 크게 두 방향에서 분석할 수 있다.

- **모델·소프트웨어 계층:** 저장해야 하는 KV 표현 자체를 줄이는 방식
- **하드웨어·시스템 계층:** GPU 및 호스트 메모리 밖으로 KV cache 수용 용량을 확장하는 방식

본 보고서는 이 두 방향을 대표하는 MLA와 ITME를 대상으로, 데이터센터/클라우드 서빙 적용 시의 기술적 방향성, 시장성, 이해관계자 관점 및 운영상 제약을 검토한다.

---

## 2. 기술 선정

### 2.1 SW 기술: DeepSeek-V2의 Multi-Head Latent Attention(MLA)

MLA는 DeepSeek-V2에서 제시된 attention 구조로, 기존 Multi-Head Attention(MHA) 또는 Grouped-Query Attention(GQA) 방식과 달리 Key·Value 정보를 저차원 latent 표현으로 압축하여 KV cache 사용량을 줄이는 접근이다. 장문 컨텍스트와 높은 동시성이 중요한 추론 환경에서 GPU 메모리 부담을 낮출 수 있다는 점에서 분석 대상으로 선정하였다.  
근거: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5), [A Gentle Introduction to Multi-Head Latent Attention (MLA)](https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla)

### 2.2 HW 기술: ITME(Inference Tiered Memory Expansion)

ITME는 GPU 또는 호스트 메모리에 모두 수용하기 어려운 대규모 KV cache를 대상으로, CPU staging buffer와 원격 CXL-hybrid memory를 추가 계층으로 활용하는 추론 메모리 확장 접근이다. CXL 및 RDMA 기반의 계층형 메모리 구조가 데이터센터 추론 환경의 메모리 용량 제약을 완화할 수 있는지 검토하기 위해 선정하였다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2), [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories | alphaXiv](https://www.alphaxiv.org/abs/2606.12556)

두 기술은 직접 경쟁 제품이라기보다, 동일한 KV cache 문제를 서로 다른 계층에서 다루는 기술로 볼 수 있다.

---

## 3. 기술 개요

### 3.1 DeepSeek-V2의 MLA

MLA는 KV cache에 Key와 Value를 고차원 형태 그대로 저장하는 대신, 이를 저차원 latent vector로 압축하는 attention 구조다. 이에 따라 긴 컨텍스트 또는 동시 세션이 많은 환경에서 KV cache가 GPU 메모리에 차지하는 비중을 줄이는 방향으로 작동한다.  
근거: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5)

DeepSeek-V2 원 논문은 DeepSeek 67B와의 비교 조건에서 KV cache가 93.3% 감소하고 최대 생성 처리량이 5.76배 향상되었다고 보고한다. 다만 이 수치는 특정 모델, 비교 대상, 실험 환경 및 측정 조건에 따른 결과이므로 일반적인 클라우드 서빙 환경의 성능 보장으로 해석할 수 없다.  
근거: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5)

MLA의 한계는 기존 MHA/GQA 모델과 구조적으로 다르다는 점이다. 기존 모델을 MLA 구조로 전환하려면 별도 변환, 파인튜닝 또는 재학습과 같은 추가 작업이 필요할 수 있으며, 기존 커널·런타임·최적화 체계와의 호환성도 검토 대상이다. MHA2MLA 및 TransMLA 연구는 이러한 전환 장벽을 낮추기 위한 접근을 제시하지만, 광범위한 상용 전환이 확인된 것은 아니다.  
근거: [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://www.alphaxiv.org/abs/2502.14837), [TransMLA: Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup](https://arxiv.org/html/2502.07864v4)

### 3.2 ITME

ITME는 KV cache가 GPU 또는 호스트 메모리 범위를 초과하는 상황에서 CPU offloading에 더해 원격 CXL-hybrid memory를 활용하는 계층형 메모리 확장 구조다. 자료상 ITME는 원격 메모리를 바이트 주소 지정 가능한 메모리로 구성하고, 표준 RDMA 프로토콜을 통해 GPU 서버가 모델 가중치와 KV cache에 접근하는 구조를 제안한다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2)

ITME 논문은 CPU-offload 대비 최대 35.7% 처리량 향상과, Llama-3.1 8B 및 70B 조건에서 특정 구간의 TTFT 1.81배 개선을 보고한다. 그러나 이 결과는 프로토타입과 특정 모델·워크로드의 실험 조건에 한정된다. 전체 서버 구성, CXL 링크 구성, 동시 사용자 수, 컨텍스트 길이 분포, 다중 노드 확장성과 같은 실제 데이터센터 운영 조건은 충분히 확인되지 않았다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2), [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories | alphaXiv](https://www.alphaxiv.org/abs/2606.12556)

ITME의 주요 한계는 계층형 메모리 운영 복잡성이다. CPU staging buffer, 원격 CXL-hybrid memory, 데이터 prefetching, 읽기 우선순위 스케줄링, 데이터 배치 정책, 원격 접근 지연시간, 장애 격리 등이 운영 성능과 안정성에 영향을 줄 수 있다. 특히 초기 1~9턴과 같이 working set이 호스트 메모리에 수용되는 구간에서는 CPU-offload와 성능이 유사하게 보고되어, 모든 상황에서 동일한 이점이 나타나는 것은 아니다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2)

---

## 4. 관점별 평가

### 4.1 기술성숙도 관점

기술성숙도 평가는 공개 정보 기반 추정을 전제로 하지만, 본 평가에서는 MLA와 ITME 모두 기술성숙도 조사 결과가 RAG 검색 실패로 제외되었다. 따라서 정식 TRL(Technology Readiness Level) 9단계 중 특정 단계 또는 범위를 신뢰성 있게 추정할 직접 근거가 부족하다.

| 기술 | TRL 추정 | 판단 근거 |
|---|---|---|
| DeepSeek-V2 MLA | 추정 불가 | 기술성숙도 평가 결과가 “근거 부족”으로 제시됨 |
| ITME | 추정 불가 | 기술성숙도 평가 결과가 “근거 부족”으로 제시됨 |

MLA는 논문, 공개 구현, SGLang·Triton 관련 최적화 언급, 기존 모델을 MLA로 전환하려는 연구가 존재한다는 점에서 연구 및 구현 활동은 확인된다. 그러나 논문 게재 여부, 공개 구현의 존재, 일부 모델 채택 사례만으로 실제 서비스 환경에서의 신뢰성, 통합 수준, 반복 검증 및 상용 준비도를 포함한 TRL을 특정 단계로 판단하기는 어렵다.  
근거: [DeepSeek + SGLang: Multi-Head Latent Attention](https://verda.com/blog/deepseek-sglang-multi-head-latent-attention), [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://www.alphaxiv.org/abs/2502.14837)

ITME는 논문에서 FPGA 기반 프로토타입과 실험 결과가 제시되지만, 상용 서버·CXL 메모리 풀·다중 노드 또는 다중 랙 환경에서의 반복 검증, 고객 도입, 상용 제품 출시가 확인되지 않았다. 따라서 프로토타입 성과와 상용 준비도를 동일시할 수 없다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories | alphaXiv](https://www.alphaxiv.org/abs/2606.12556)

또한 논문 발표 시점과 실제 상용 채택 사이에는 시차가 존재할 수 있으며, 공개 정보만으로는 해당 기간 동안의 제품화 진척, 장기 장애율, 고객 환경에서의 검증 수준을 확인하기 어렵다. 따라서 두 기술의 기술성숙도는 **추가 검증 필요**로 판단한다.

### 4.2 시장성 관점

#### DeepSeek-V2 MLA

**① 시장 규모·성장성**  
MLA 자체, KV cache 최적화 기술 자체 또는 DeepSeek-V2 MLA의 시장 규모·매출·CAGR에 대한 직접 근거는 부족하다. 일부 자료는 MLA가 KV cache를 줄여 장문 컨텍스트와 대규모 추론에서 메모리 효율을 높이는 방향성을 제시하지만, 이는 시장 규모를 의미하지 않는다.  
근거: [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://www.alphaxiv.org/abs/2502.14837), [A Gentle Introduction to Multi-Head Latent Attention (MLA)](https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla)

**② 상용화·채택 현황**  
MLA는 DeepSeek-V2의 attention 아키텍처로 소개되며, Raschka의 기술 정리 자료에는 Kimi K2, GLM-5, Ling 2.5 및 Sarvam의 일부 모델이 MLA 또는 MLA 결합 구조를 활용한 사례로 언급된다. 다만 이는 모델 아키텍처 채택 사례에 관한 기술 정리이며, 실제 클라우드 서비스 도입 규모, 상용 매출, 고객 수 또는 대규모 운영 성과를 의미하지는 않는다.  
근거: [Multi-Head Latent Attention (MLA) | Sebastian Raschka, PhD](https://sebastianraschka.com/llm-architecture-gallery/mla)

**③ 생태계 지지**  
SGLang 관련 자료에서는 DeepSeek-V2 MLA의 추론 최적화 및 Triton 기반 MLA 지원이 언급된다. 또한 공개 GitHub 구현과 MHA/GQA 모델을 MLA로 전환하려는 연구도 존재한다. 이는 일부 오픈소스·연구 생태계의 구현 활동을 보여준다.  
근거: [DeepSeek + SGLang: Multi-Head Latent Attention](https://verda.com/blog/deepseek-sglang-multi-head-latent-attention), [GitHub - junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention](https://github.com/junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention), [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://www.alphaxiv.org/abs/2502.14837)

다만 주요 추론 프레임워크 전반의 공식 지원 범위, 산업 표준, 컨소시엄 규격, 범용 인터페이스 표준화 수준은 근거 부족이다.

#### ITME

**① 시장 규모·성장성**  
ITME 자체의 시장 규모, 매출, 성장률 또는 CAGR은 근거 부족이다. 다만 인접 시장인 AI 데이터센터 CXL 메모리 확장·풀링 인프라 시장은 2026년 12억 달러에서 2032년 68.3억 달러로 성장하고, 같은 기간 연평균성장률은 33.6%로 전망된다는 자료가 있다. 이 수치는 ITME 자체가 아니라 CXL 메모리 확장·풀링 인프라 시장 전체에 대한 전망이다.  
근거: [AI Data Center CXL Memory Expansion and Pooling Infrastructure Market: 2032](https://www.knowledge-sourcing.com/report/ai-data-center-cxl-memory-expansion-and-pooling-infrastructure-market)

**② 상용화·채택 현황**  
ITME는 제공 자료상 연구 논문에서 제안된 아키텍처 및 프로토타입 성격의 기술이다. ITME라는 명칭의 상용 제품 출시, 실제 고객 도입, 데이터센터 운영 사례, 고객 수 또는 매출에 대한 근거는 부족하다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2)

다만 관련 CXL 생태계에서는 SK hynix의 풀드·컴퓨트 지원 CXL 메모리 시연, Samsung Electronics 및 Micron의 CXL 모듈 관련 활동, Primemas의 대용량 CXL 메모리 시스템 구축, LIQID의 랙 규모 메모리 풀링 관련 활동이 언급된다. 이는 ITME 도입 사례가 아니라 ITME가 활용할 수 있는 인접 인프라 생태계의 움직임으로 해석해야 한다.  
근거: [AI Data Center CXL Memory Expansion and Pooling Infrastructure Market: 2032](https://www.knowledge-sourcing.com/report/ai-data-center-cxl-memory-expansion-and-pooling-infrastructure-market)

**③ 생태계 지지**  
ITME는 CXL, RDMA, 메모리 풀링과 연결되는 구조를 제안한다. CXL 및 RDMA는 기존 데이터센터 하드웨어·네트워크 생태계와 연계될 수 있는 기반 기술이다. 그러나 ITME 전용 추론 엔진, 라이브러리, 운영체제 기능, 컴파일러 지원, 표준화 수준은 근거 부족이다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2)

### 4.3 이해관계자 관점

#### DeepSeek-V2 MLA

**① 경쟁 기술 진영**  
MLA의 대안 또는 비교 대상으로 MHA, MQA, GQA가 언급된다. GQA는 MHA와 MQA 사이의 절충안으로 설명되며, KV cache 효율과 품질의 균형을 추구하는 구조로 포지셔닝된다. TransMLA는 동일한 KV cache 오버헤드에서 MLA가 GQA보다 높은 표현력을 제공할 수 있다고 주장하지만, 이는 연구 결과이며 업계 전반의 합의로 해석할 수 없다.  
근거: [MHA vs MQA vs GQA vs MLA](https://medium.com/@zaiinn440/mha-vs-mqa-vs-gqa-vs-mla-c6cf8285bbec), [TransMLA: Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup](https://arxiv.org/html/2502.07864v4)

특정 경쟁사의 공식 대응 발표, MLA에 대한 직접적인 경쟁 전략 또는 공식 비판은 제공 자료에서 확인되지 않는다.

**② 도입 기업·개발자**  
MLA는 KV cache를 줄이고 장문 컨텍스트 및 대규모 추론의 메모리 부담을 낮출 가능성이 있는 기술로 평가된다. 일부 모델의 MLA 채택 사례도 기술 정리 자료에서 언급된다.  
근거: [Multi-Head Latent Attention (MLA) | Sebastian Raschka, PhD](https://sebastianraschka.com/llm-architecture-gallery/mla)

반면 기존 GQA 기반 모델과 런타임, 커널, 추론 최적화 체계에 이미 투자한 조직은 MLA 전환 과정에서 구조적 호환성 및 추가 엔지니어링 부담을 겪을 수 있다. 특히 재학습·파인튜닝·변환 비용의 정량 자료는 근거 부족이다.  
근거: [TransMLA: Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup](https://arxiv.org/html/2502.07864v4)

**③ 투자 업계**  
MLA와 관련된 투자금액, 투자 라운드, 기업가치, 금융 애널리스트의 공식 전망 또는 투자 수익성 분석은 제공 자료에서 확인되지 않는다. 기술 미디어와 연구 자료는 MLA를 KV cache 효율과 모델 품질의 균형을 추구하는 구조로 평가하지만, 이를 투자 업계의 공식 판단으로 일반화할 수 없다.  
근거: [DeepSeek-V3 Explained 1: Multi-head Latent Attention](https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4), [Multi-Head Latent Attention (MLA) | Sebastian Raschka, PhD](https://sebastianraschka.com/llm-architecture-gallery/mla)

#### ITME

**① 경쟁 기술 진영**  
ITME에 대한 특정 경쟁사의 공식 대응 또는 ITME를 직접 경쟁 대상으로 지목한 발표는 근거 부족이다. 다만 CXL 기반 메모리 확장·풀링, fabric-attached memory, HBM 확장, SSD·스토리지 오프로딩은 대규모 KV cache 및 메모리 용량 문제를 다루는 유사 또는 대체 방향으로 언급된다.  
근거: [Scaling the Memory Wall: HBM, CXL, and the New GPU Playbook](https://www.datacenterknowledge.com/data-center-hardware/scaling-the-memory-wall-hbm-cxl-and-the-new-gpu-playbook), [How CXL Transforms RAG and KV Cache Performance](https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance)

**② 도입 기업·개발자**  
장문 컨텍스트 및 에이전트형 AI 수요 증가는 더 큰 KV cache 수용 공간을 요구할 수 있으며, 이는 계층형 메모리 구조에 대한 수요 배경으로 제시된다. ITME는 소프트웨어 prefetching 및 읽기 우선순위 스케줄링을 통해 I/O 병목과 메모리 용량 문제를 완화하는 방식을 제안한다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2)

그러나 실제 도입 시에는 CXL의 지연시간, 플랫폼 상호운용성, 소프트웨어 성숙도, CXL 모듈과 인프라 비용, 공급망, 패브릭 관리, 장애 격리 등 여러 제약이 존재할 수 있다. 특정 기업의 ITME 상용 도입 사례는 확인되지 않는다.  
근거: [CXL Memory Market Size, Share & Forecast 2035](https://www.datamintelligence.com/research-report/cxl-memory-market), [AI Data Center CXL Memory Expansion and Pooling Infrastructure Market: 2032](https://www.knowledge-sourcing.com/report/ai-data-center-cxl-memory-expansion-and-pooling-infrastructure-market)

**③ 투자 업계**  
ITME 자체에 대한 투자 유치, 시장가치, 투자 라운드 또는 금융 분석은 근거 부족이다. 다만 CXL 기반 메모리 확장 시장과 AI 데이터센터의 메모리 월 문제에 대한 산업적 관심은 자료에서 확인된다. 이는 ITME의 직접적인 투자 매력도나 상용 성공 가능성을 의미하지는 않는다.  
근거: [Scaling the Memory Wall: HBM, CXL, and the New GPU Playbook](https://www.datacenterknowledge.com/data-center-hardware/scaling-the-memory-wall-hbm-cxl-and-the-new-gpu-playbook), [CXL Memory Market Size, Share & Forecast 2035](https://www.datamintelligence.com/research-report/cxl-memory-market)

### 4.4 도메인 적용 관점

본 절의 평가는 **데이터센터/클라우드 서빙 환경에 한정**된다. 온디바이스 AI, 단일 사용자 환경 또는 다른 장문 컨텍스트 애플리케이션에서는 전력 제약, 장치 메모리 구조, 네트워크 환경 및 컨텍스트 길이 조건이 다르므로 평가 결과가 달라질 수 있다.

#### DeepSeek-V2 MLA

**처리 가능한 규모**  
DeepSeek-V2는 총 236B 파라미터 중 토큰당 21B 파라미터를 활성화하는 MoE 구조와 기본 128K 컨텍스트를 사용한다고 소개된다. MLA는 KV cache를 latent vector로 압축해 장문 컨텍스트와 다수 동시 세션에서 GPU 메모리 부담을 줄이는 방향의 기술이다.  
근거: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5), [From 390 KB to 890 Bytes — DeepSeek's KV Cache Optimization from V1 to V4.1-Flash](https://local-ai-zone.github.io/blog/deepseek-kv-cache-optimization-research-paper.html)

원 논문에서는 DeepSeek 67B 대비 KV cache 93.3% 감소 및 최대 생성 처리량 5.76배 향상이 보고되었다. 다만 이는 논문 비교 조건의 결과이며, 실제 데이터센터의 GPU 종류, 동시 요청 수, 배치 크기, 입력·출력 길이, 서비스 지연시간 목표에 따른 성능으로 일반화할 수 없다. 실제 동시 요청 수와 GPU별 최대 처리량은 근거 부족이다.  
근거: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5)

**비용 절감 효과**  
MLA는 긴 컨텍스트와 높은 동시성 조건에서 KV cache가 차지하는 GPU 메모리 비중을 줄여 동일 GPU에서 더 많은 세션을 처리하거나 GPU 메모리 증설을 지연시킬 가능성이 있다. 그러나 실제 GPU 수 감소율, 전력 절감량, 냉각비 변화, 클라우드 비용 절감률, TCO 절감률은 제공 자료에서 근거 부족이다.

DeepSeek-V2 논문에 제시된 학습 비용 절감 수치는 H800 GPU 기반 학습 조건에 관한 내용으로, 데이터센터 추론 서빙의 비용 절감률로 해석할 수 없다.  
근거: [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/html/2405.04434v5)

**제약**  
MLA는 기존 모델에 단순 메모리 옵션으로 적용하는 기술이 아니라 모델 attention 구조와 결합된 방식이다. 따라서 MHA/GQA 기반 기존 모델의 전환 시 모델 구조, 가중치, 추론 커널, 런타임 호환성을 검토해야 한다. 재학습 필요 여부, 변환 비용, GPU 세대별 성능, 주요 추론 엔진 호환성은 추가 검증이 필요하다.  
근거: [Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs](https://www.alphaxiv.org/abs/2502.14837), [TransMLA: Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup](https://arxiv.org/html/2502.07864v4)

#### ITME

**처리 가능한 규모**  
ITME는 GPU 또는 호스트 메모리에 KV cache가 모두 들어가지 않는 상황에서 CPU offloading과 원격 CXL-hybrid memory를 함께 활용해 대규모 KV cache를 수용하는 방식을 제안한다. 자료에서는 CXL-hybrid memory를 TB급 T3.5 계층으로 사용하고, 호스트 메모리 계층 이후에도 요청 처리를 지속하는 구조가 제시된다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2), [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories | alphaXiv](https://www.alphaxiv.org/abs/2606.12556)

ITME는 CPU-offload 대비 최대 35.7% 처리량 향상을, Llama-3.1 8B 및 70B 실험에서는 5번째 대화 턴까지 TTFT 1.81배 개선을 보고했다. 다만 이는 프로토타입 및 특정 실험 조건에 한정되며, 지원 가능한 동시 사용자 수, 최대 컨텍스트 길이, 노드·랙 단위 확장 한계는 근거 부족이다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2), [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories | alphaXiv](https://www.alphaxiv.org/abs/2606.12556)

**비용 절감 효과**  
ITME는 GPU 메모리 부족으로 인해 KV cache를 재계산하는 상황을 줄이고, CXL-hybrid memory를 통해 추가 용량을 활용하는 방향의 기술이다. 특정 실험에서 TTFT 및 처리량 개선이 보고된 점은 재계산 부담을 낮출 가능성을 보여준다.

그러나 CXL 메모리, 컨트롤러, 네트워크, FPGA 프로토타입, 운영 소프트웨어를 포함한 전체 비용이 제공되지 않았기 때문에, GPU 증설 대비 비용 절감, 데이터센터 TCO, 전력·냉각비 효과를 정량적으로 판단할 수 없다. 따라서 ITME의 비용 절감 효과는 **추가 검증 필요**다.

**제약**  
ITME는 CPU staging buffer, 원격 CXL-hybrid memory, NVMe 등을 포함한 다계층 메모리 경로와 제어 소프트웨어를 요구한다. 소프트웨어 prefetching과 읽기 우선순위 스케줄링이 필요하며, 계층 간 데이터 이동과 캐시 배치가 성능에 영향을 미친다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2)

초기 1~9턴과 같이 working set이 호스트 메모리에 수용되는 구간에서는 CPU-offload와 성능이 유사하게 나타난 것으로 보고되었다. 이는 ITME의 이점이 메모리 압박이 커지는 구간에서 주로 나타날 수 있음을 시사한다.  
근거: [ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories](https://arxiv.org/html/2606.12556v2)

또한 실제 클라우드 운영에서는 계층별 cache hit, burst 트래픽, tail latency, CXL 링크 장애, 원격 계층 장애 시 fallback, 다중 테넌트 격리, 장애 전파 방지 등을 검토해야 한다. ITME 자체의 tail latency 분포, 장애 대응 시간, 다중 테넌트 환경 호환성은 근거 부족이다.  
근거: [Supercharging Inference for AI Factories: KV Cache Offload as a Memory-Hierarchy Problem](https://www.min.io/blog/supercharging-inference-for-ai-factories-kv-cache-offload-as-a-memory-hierarchy-problem), [Inference Optimization: Practical Techniques for Faster, Cost-Effective AI](https://www.weka.io/learn/ai-ml/inference-optimization)

### 4.5 종합의견

MLA와 ITME는 모두 KV cache와 추론 메모리 문제에 대응하지만, 해결 계층이 다르다. MLA는 모델 attention 구조와 KV 표현을 변경해 저장해야 하는 데이터를 줄이는 소프트웨어·모델 아키텍처 접근이다. ITME는 CPU offloading과 원격 CXL-hybrid memory를 활용해 KV cache를 수용할 수 있는 메모리 계층을 확장하는 하드웨어·시스템 접근이다.

기술성숙도 관점에서는 두 기술 모두 RAG 검색 실패로 인해 정식 성숙도 단계와 상용 준비도를 판단할 근거가 부족하다. MLA에는 논문·구현·일부 채택 사례가, ITME에는 프로토타입 실험 결과가 존재하지만, 이를 TRL 또는 상용화 수준으로 직접 환산할 수는 없다.

시장성 관점에서는 MLA에 일부 오픈소스 구현, SGLang·Triton 관련 최적화, 모델 전환 연구가 확인된다. 반면 MLA 자체의 시장 규모, 매출, 표준화, 광범위한 상용 도입은 근거 부족이다. ITME는 CXL 메모리 확장·풀링 인프라 시장과 연결되는 산업적 방향성을 보이지만, ITME 자체의 제품 출시, 고객 도입, 전용 소프트웨어 생태계는 확인되지 않는다.

관점 간 상충도 존재한다. **시장성 관점-SW에서는** SGLang·Triton·GitHub 구현과 MHA2MLA 연구를 근거로 MLA의 일부 생태계 활동을 확인한다. 반면 **시장성 관점-HW에서는** 공식적이고 범용적인 추론 엔진·라이브러리·컴파일러 지원이 확인되지 않았다고 평가한다. 이는 연구·구현 활동을 생태계 지지로 볼 것인지, 공식적이고 광범위한 산업 지원을 요구할 것인지에 따른 평가 기준 차이로 해석된다.

MLA의 채택 사례도 유사한 차이를 보인다. **이해관계자 관점-SW에서는** 일부 모델의 MLA 또는 MLA 결합 구조 채택 사례가 언급된다. 반면 **이해관계자 관점-HW에서는** 독립적으로 확인된 실제 도입 기업, 개발자 의견, 비용 및 운영 경험은 근거 부족으로 본다. 즉 모델 구조 채택과 실제 상용 운영 도입은 구분할 필요가 있다.

도메인 적용 관점에서도 MLA의 논문 기반 성능과 실제 운영 성능은 구분해야 한다. **도메인 적용 관점-SW에서는** DeepSeek-V2 논문의 KV cache 93.3% 감소 및 최대 생성 처리량 5.76배 결과를 제시한다. 반면 **도메인 적용 관점-HW에서는** 실제 데이터센터의 동시 사용자 수, GPU 구성, 지연시간 목표, 비용 효과로 확장할 수 있는 근거가 부족하다고 평가한다.

ITME 역시 일부 관점에서는 기술 정의와 프로토타입 성능이 충분히 확인되지 않은 것으로 평가되지만, 다른 관점에서는 CXL-hybrid memory, RDMA, CPU staging buffer, prefetching, read-priority scheduling 및 실험 결과가 제시된다. 이는 관점별로 참조한 자료 범위가 달라 발생한 정보 비대칭으로 보아야 하며, 모든 평가에서 동일 수준으로 검증된 사실로 확대 해석해서는 안 된다.

---

## 5. 시사점

1. **KV cache 문제는 단일 기술로만 해결하기 어려운 복합 과제다.**  
   MLA는 저장해야 하는 KV 표현을 줄이는 방향이며, ITME는 수용 가능한 메모리 계층을 확장하는 방향이다. 두 기술은 같은 문제를 다른 계층에서 다루므로 직접적인 대체 관계로 단정하기 어렵다.

2. **MLA의 효과는 모델 구조와 추론 소프트웨어 호환성에 좌우될 수 있다.**  
   MLA는 기존 MHA/GQA 모델에 단순히 적용할 수 있는 범용 옵션이라기보다 모델 구조, 가중치, 커널, 런타임과의 결합을 고려해야 하는 접근이다. 기존 모델의 전환 비용과 재학습 필요성은 추가 검증이 필요하다.

3. **ITME의 효과는 메모리 압박이 큰 조건에서 더 중요할 가능성이 있다.**  
   ITME 실험에서는 working set이 호스트 메모리에 수용되는 초기 구간에서 CPU-offload와 성능이 유사했다. 따라서 ITME의 적용 가치는 모든 추론 요청이 아니라 GPU·호스트 메모리 용량을 넘는 대규모 KV cache 조건에서 검토할 필요가 있다.

4. **성능 수치와 운영 경제성은 구분해야 한다.**  
   MLA의 KV cache 감소 및 처리량 결과, ITME의 처리량 및 TTFT 개선 결과는 각각 특정 실험 조건에 근거한다. 이를 실제 GPU 수 감소, 전력비 절감, TCO 절감 또는 클라우드 서비스 비용 절감률로 환산하려면 별도 운영 데이터가 필요하다.

5. **CXL 관련 산업 활동은 ITME의 상용화 근거와 구분해야 한다.**  
   CXL 메모리 확장·풀링 생태계와 관련 기업 활동은 ITME의 기술적 방향성과 인접 시장 수요를 뒷받침할 수 있다. 그러나 이는 ITME 자체의 제품화, 고객 도입 또는 상용 성공을 직접 입증하지는 않는다.

---

## 6. 한계점

본 보고서는 제공된 공개 자료와 Agentic RAG 평가 결과에 한정하여 작성되었다. 따라서 실제 데이터센터 운영 로그, 고객사 도입 계약, 장기 장애율, 하드웨어 조달비, 에너지 비용, GPU 배치 구성, 클라우드 서비스별 SLA 및 보안·격리 요구사항은 반영되지 않았다.

특히 기술성숙도 관점은 관련 RAG 검색 실패로 인해 충분한 근거가 확보되지 않았다. 이에 따라 MLA와 ITME의 TRL, 상용 준비도, 신뢰성, 반복 검증 수준을 특정 단계로 판정하지 않고 “추정 불가” 또는 “근거 부족”으로 제시하였다.

확증편향을 방지하기 위해 다음 원칙을 적용하였다.

- MLA의 KV cache 절감 및 처리량 관련 논문 결과를 실제 클라우드 운영 성과로 일반화하지 않았다.
- ITME의 프로토타입 성능 개선 결과를 상용 시스템의 TCO 또는 전력 절감으로 환산하지 않았다.
- MLA의 오픈소스 구현 및 일부 모델 채택 사례를 광범위한 상용 채택 또는 산업 표준화로 해석하지 않았다.
- CXL 메모리 확장·풀링 시장의 전망 및 관련 기업 활동을 ITME 자체의 시장 규모·매출·도입 사례로 해석하지 않았다.
- 관점별로 상충하는 평가가 존재하는 경우 이를 하나의 결론으로 통합하지 않고, 근거 범위와 판단 기준의 차이를 병렬적으로 제시하였다.
- 입력 자료에 없는 수치, 기업명, 성능 비교, 비용 절감률은 추가하지 않았다.

---

## REFERENCE

- Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs | alphaXiv - https://www.alphaxiv.org/abs/2502.14837  
- DeepSeek + SGLang: Multi-Head Latent Attention - https://verda.com/blog/deepseek-sglang-multi-head-latent-attention  
- A Gentle Introduction to Multi-Head Latent Attention (MLA) - MachineLearningMastery.com - https://machinelearningmastery.com/a-gentle-introduction-to-multi-head-latent-attention-mla  
- GitHub - junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention - https://github.com/junfanz1/MiniGPT-and-DeepSeek-MLA-Multi-Head-Latent-Attention  
- ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories - https://arxiv.org/html/2606.12556v2  
- AI Data Center CXL Memory Expansion and Pooling Infrastructure Market: 2032 - https://www.knowledge-sourcing.com/report/ai-data-center-cxl-memory-expansion-and-pooling-infrastructure-market  
- How CXL Transforms RAG and KV Cache Performance - https://www.asteralabs.com/resources/blog/breaking-through-the-memory-wall-how-cxl-transforms-rag-and-kv-cache-performance  
- ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories | alphaXiv - https://www.alphaxiv.org/abs/2606.12556  
- TransMLA: Migrating GQA Models to MLA with Full DeepSeek Compatibility and Speedup - https://arxiv.org/html/2502.07864v4  
- MHA vs MQA vs GQA vs MLA. Comparison of Deepseek’s new… | by Zain ul Abideen | Medium - https://medium.com/@zaiinn440/mha-vs-mqa-vs-gqa-vs-mla-c6cf8285bbec  
- Multi-Head Latent Attention (MLA) | Sebastian Raschka, PhD - https://sebastianraschka.com/llm-architecture-gallery/mla  
- DeepSeek-V3 Explained 1: Multi-head Latent Attention | Towards Data Science - https://towardsdatascience.com/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4  
- CXL Memory Market Size, Share & Forecast 2035 - https://www.datamintelligence.com/research-report/cxl-memory-market  
- Scaling the Memory Wall: HBM, CXL, and the New GPU Playbook - https://www.datacenterknowledge.com/data-center-hardware/scaling-the-memory-wall-hbm-cxl-and-the-new-gpu-playbook  
- From 390 KB to 890 Bytes — DeepSeek's KV Cache Optimization from V1 to V4.1-Flash - https://local-ai-zone.github.io/blog/deepseek-kv-cache-optimization-research-paper.html  
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model - https://arxiv.org/html/2405.04434v5  
- Supercharging Inference for AI Factories: KV Cache Offload as a Memory-Hierarchy Problem - https://www.min.io/blog/supercharging-inference-for-ai-factories-kv-cache-offload-as-a-memory-hierarchy-problem  
- Inference Optimization: Practical Techniques for Faster, Cost-Effective AI - WEKA - https://www.weka.io/learn/ai-ml/inference-optimization