"""
Orchestrator 에이전트

흐름: 실행 전에 한 번 LLM이 전체 평가를 task N개로 나누고, task마다 담당 전문 워커
(tech_research / market_eval / stakeholder_eval / domain_eval)를 지정한다.
-> 계획이 네 관점 각각에서 SW와 HW를 모두 다루는지 코드가 확인하고, 빠진 부분은 기본 task로 보완
-> task_id, status 같은 제어 값은 LLM이 아니라 코드가 붙인다
-> assign_workers가 task마다 해당 워커로 Send (워커로 가는 고정 엣지 없음, Dynamic Fan-out)

계획 생성 자체가 실패하면 DEFAULT_PLAN으로 대체하고, 그 사실을 decisions에 기록한다.
평가 기준은 워커 프롬프트(prompts/*.txt)에 있으므로 Orchestrator는 분할 방식과 집중할 내용만 정한다.
"""

import os
from typing import List, Literal, Optional

from langchain.chat_models import init_chat_model
from langgraph.types import Send
from pydantic import BaseModel, Field

from graph.state import GraphState, WorkerName

PLANNER_MODEL = os.getenv("PLANNER_MODEL", os.getenv("MODEL_NAME", "gpt-5.6-terra"))

# 워커와 보고서 평가 관점의 대응. 종합의견은 워커가 아니라 Synthesizer가 맡는다
VIEWPOINTS = {
    "tech_research": "기술성숙도 관점",
    "market_eval": "시장성 관점",
    "stakeholder_eval": "이해관계자 관점",
    "domain_eval": "도메인 적용 관점",
}


class PlannedTask(BaseModel):
    worker: WorkerName = Field(description="이 task를 처리할 워커")
    perspective: str = Field(description="task를 구분할 짧은 이름 (예: '시장성 - HW', '도메인 - 비용')")
    instruction: str = Field(description="이번 조사에서 특히 집중할 내용 1~3문장")
    search_query: str = Field(description="검색에 쓸 핵심 키워드. 기술명을 포함해 100자 이내")
    target: Optional[Literal["sw", "hw"]] = Field(
        default=None,
        description="한쪽 기술만 다루면 'sw' 또는 'hw', 두 기술을 함께 다루면 null. "
        "tech_research는 반드시 지정",
    )


class Plan(BaseModel):
    tasks: List[PlannedTask] = Field(description="서로 독립적으로 병렬 수행 가능한 task 목록")
    reason: str = Field(description="이렇게 계획한 이유 1~2문장")


PLAN_PROMPT = """당신은 KV Cache 최적화 기술 평가의 계획을 세우는 수석 아키텍트입니다.
이번에 평가할 두 기술과 대상 도메인을 바탕으로, 최종 보고서의 네 평가 관점을 균형 있게
조사하기 위한 하위 task 목록을 계획해 주세요.

[평가 대상]
- SW 기술: {tech_sw}
- HW 기술: {tech_hw}
- 적용 도메인: {domain}

[보고서의 네 평가 관점과 담당 워커]
1. 기술성숙도 관점 ← tech_research 워커 (원문 논문/스펙 RAG)
   SW 기술과 HW 기술 각각의 성숙도(TRL 9단계 기준 추정), 기술 원리, 적용 범위, 기술적 한계를 조사.
   한 번에 한 기술만 처리(target 필수)
2. 시장성 관점 ← market_eval 워커 (웹검색)
   ① 시장 규모·성장성(시장 리포트, 산업 뉴스) ② 상용화·채택 현황(실제 도입 사례, 제품 출시 발표)
   ③ 생태계 지지(지원 프레임워크, 표준화 동향)
3. 이해관계자 관점 ← stakeholder_eval 워커 (원문 + 웹검색)
   ① 경쟁 기술 진영(경쟁사 반응, 대응 기술) ② 도입 기업·개발자(도입 의견, 채택 시 장벽)
   ③ 투자 업계(투자 동향, 애널리스트/미디어 평가)
4. 도메인 적용 관점 ← domain_eval 워커 (원문 + 웹검색)
   같은 기술도 적용 환경에 따라 평가가 달라짐. 이번에는 "{domain}" 도메인을 선택해,
   이 도메인이 민감한 요소(규모, 비용 등)를 기준으로 처리 가능 규모와 비용 절감 효과를 조사
5. 종합 의견 ← 워커 없음 (계획하지 마세요)
   1~4번 관점의 의견을 종합하되 관점 간 상충 지점이 명시적으로 드러나도록 객관적·중립적으로 작성.

[규칙]
- 네 관점을 하나도 빠뜨리지 말고, 관점마다 SW와 HW가 모두 조사되게 하세요
  (두 기술을 함께 다루는 task 하나, 또는 target을 나눈 task 두 개).
- 한 관점을 기술별로 나눴다면 SW와 HW에 같은 기준을 적용하세요.
- 관점 안에서 범위가 넓으면 위 ①②③ 같은 세부 기준 단위로 task를 더 나눠도 됩니다.
  단, 한 관점의 세부 기준이 어느 task에서도 다뤄지지 않는 일이 없게 하세요.
- perspective는 "관점 이름 - 세부"로 쓰세요 (예: "시장성 관점 - HW", "도메인 적용 관점 - 비용").
- instruction에는 두 기술의 성격을 고려해 그 관점에서 특히 확인해야 할 점을 쓰세요.
- task는 최소 10개 최대 20개, 다른 task의 결과에 의존하지 않아야 합니다."""

# 계획 생성이 실패했을 때 쓰는 기본 계획
DEFAULT_PLAN = [
    {
        "worker": "tech_research",
        "perspective": "기술성숙도 관점 - SW",
        "target": "sw",
        "instruction": "기술 개요, 적용 범위, 한계를 정리하고 TRL 9단계 척도로 성숙도를 추정한다.",
        "search_query": "{tech_sw}",
    },
    {
        "worker": "tech_research",
        "perspective": "기술성숙도 관점 - HW",
        "target": "hw",
        "instruction": "기술 개요, 적용 범위, 한계를 정리하고 TRL 9단계 척도로 성숙도를 추정한다.",
        "search_query": "{tech_hw}",
    },
    {
        "worker": "market_eval",
        "perspective": "시장성 관점",
        "target": None,
        "instruction": "두 기술 각각의 시장 규모·성장성, 상용화·채택 현황, 생태계 지지를 우열 판정 없이 정리한다.",
        "search_query": "{tech_sw} {tech_hw} 시장 규모 CAGR 채택 사례",
    },
    {
        "worker": "stakeholder_eval",
        "perspective": "이해관계자 관점",
        "target": None,
        "instruction": "경쟁 기술 진영, 도입 기업·개발자(채택 장벽 포함), 투자 업계의 시각을 정리한다.",
        "search_query": "{tech_sw} {tech_hw} 경쟁사 대응 도입 사례 투자 동향",
    },
    {
        "worker": "domain_eval",
        "perspective": "도메인 적용 관점",
        "target": None,
        "instruction": "{domain} 도메인이 민감한 요소를 기준으로 처리 가능한 규모, 비용 절감 효과, 제약을 정리한다.",
        "search_query": "{tech_sw} {tech_hw} {domain} scalability TCO energy efficiency",
    },
]


def _split_tech_research(planned: list[dict]) -> list[dict]:
    """tech_research는 한 번에 한 기술만 처리하므로, target이 없으면 SW/HW 두 task로 나눈다."""
    normalized = []
    for task in planned:
        if task["worker"] == "tech_research" and task.get("target") is None:
            for target in ("sw", "hw"):
                normalized.append(
                    {
                        **task,
                        "target": target,
                        "perspective": f"{task['perspective']} - {target.upper()}",
                    }
                )
        else:
            normalized.append(task)
    return normalized


def _ensure_coverage(planned: list[dict], names: dict) -> tuple[list[dict], list[str]]:
    """네 관점 각각에서 SW와 HW가 모두 조사되는지 확인하고, 빠진 부분은 기본 task로 보완한다.
    보완한 내용을 사유로 돌려준다 (decisions에 기록)."""
    added: list[str] = []
    for worker, viewpoint in VIEWPOINTS.items():
        targets = {task.get("target") for task in planned if task["worker"] == worker}
        if None in targets or {"sw", "hw"} <= targets:
            continue
        base = next(task for task in DEFAULT_PLAN if task["worker"] == worker)
        if not targets and worker != "tech_research":
            missing = [None]  # 관점이 통째로 빠짐 -> 두 기술을 함께 다루는 task 하나
        else:
            missing = [t for t in ("sw", "hw") if t not in targets]
        for target in missing:
            label = f"{viewpoint} - {target.upper()}" if target else viewpoint
            planned.append(
                {
                    **base,
                    "target": target,
                    "perspective": label,
                    "instruction": base["instruction"].format(**names),
                    "search_query": base["search_query"].format(**names),
                }
            )
            added.append(label)
    return planned, added


def _label(task: dict) -> str:
    """task 이름이 어느 평가 관점에 속하는지 항상 드러나게 한다."""
    viewpoint = VIEWPOINTS[task["worker"]]
    name = task["perspective"]
    return name if name.replace(" ", "").startswith(viewpoint.replace(" ", "")) else f"{viewpoint} - {name}"


def orchestrator(state: GraphState) -> dict:
    """tech_sw / tech_hw / domain을 읽어 task 계획을 세우고 tasks, decisions를 write한다."""
    print("\n==== [ORCHESTRATOR] ====\n")
    names = {"tech_sw": state["tech_sw"], "tech_hw": state["tech_hw"], "domain": state["domain"]}
    try:
        planner_llm = init_chat_model(PLANNER_MODEL, model_provider="openai", temperature=0)
        planner = planner_llm.with_structured_output(Plan)
        plan = planner.invoke(PLAN_PROMPT.format(**names))
        planned = [task.model_dump() for task in plan.tasks]
        if not planned:
            raise ValueError("빈 계획")
        reason = plan.reason
    except Exception as e:
        planned = [
            {
                **task,
                "instruction": task["instruction"].format(**names),
                "search_query": task["search_query"].format(**names),
            }
            for task in DEFAULT_PLAN
        ]
        reason = f"계획 생성 실패({e})로 기본 계획 사용"

    # task_id, status 같은 제어 값은 코드가 붙인다. task_id는 "워커명-순번"
    planned, added = _ensure_coverage(_split_tech_research(planned), names)
    if added:
        reason += f" / 계획에 빠져 있어 보완한 관점: {', '.join(added)}"

    tasks: dict = {}
    counts: dict = {}
    for task in planned:
        task = {**task, "perspective": _label(task)}
        counts[task["worker"]] = counts.get(task["worker"], 0) + 1
        task_id = f"{task['worker']}-{counts[task['worker']]}"
        tasks[task_id] = {
            **task,
            "task_id": task_id,
            "status": "pending",
            "attempts": 0,
            "error": None,
        }

    print(f"🧩 계획된 task {len(tasks)}개: {reason}")
    for task_id, task in tasks.items():
        print(f"   {task_id}: {task['perspective']} (target={task['target']})")

    return {
        "tasks": tasks,
        "decisions": [
            {
                "node": "orchestrator",
                "action": "plan",
                "task_ids": list(tasks),
                "reason": reason,
            }
        ],
    }


def assign_workers(state: GraphState):
    """계획된 task마다 task에 적힌 워커로 Send한다 (orchestrator 뒤 조건부 엣지의 라우팅 함수).
    각 워커에는 전체 State가 아니라 자기 task와 입력값만 전달된다."""
    return [
        Send(
            task["worker"],
            {
                "task": task,
                "tech_sw": state["tech_sw"],
                "tech_hw": state["tech_hw"],
                "domain": state["domain"],
            },
        )
        for task in state["tasks"].values()
    ]
