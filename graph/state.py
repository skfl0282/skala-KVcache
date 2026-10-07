"""
설계산출물 D. Graph 설계 - State 테이블을 코드로 옮긴 모듈.
"""

import operator
from typing import Annotated, List

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated, List, Literal, Optional, TypedDict

WorkerName = Literal["tech_research", "market_eval", "stakeholder_eval", "domain_eval"]
WORKERS = ["tech_research", "market_eval", "stakeholder_eval", "domain_eval"]


class Task(TypedDict, total=False):
    """제어 메타: 분배와 실패 대응 판단에 쓰는 정보."""

    task_id: str
    worker: WorkerName                        # 이 task를 처리할 전문 워커
    perspective: str                          # 화면/보고서에 쓸 짧은 이름
    instruction: str                          # 이번 조사에서 집중할 내용
    search_query: str                         # 검색에 쓸 짧은 키워드
    target: Optional[Literal["sw", "hw"]]     # 특정 기술만 다루면 지정
    status: Literal["pending", "done", "failed", "excluded"]
    attempts: int
    error: Optional[str]                      # 실패/부분 실패 사유


class TaskResult(TypedDict):
    """페이로드: 워커 결과물."""

    content: str
    references: List[str]


class Decision(TypedDict):
    """결정 1건과 그 사유 (관측성)."""

    node: str                                 # 결정을 내린 노드
    action: Literal["plan", "retry", "continue", "exclude"]
    task_ids: List[str]
    reason: str


def merge_tasks(left: dict, right: dict) -> dict:
    merged = dict(left)
    for task_id, patch in right.items():
        merged[task_id] = {**merged.get(task_id, {}), **patch}
    return merged


def merge_results(left: dict, right: dict) -> dict:
    return {**left, **right}


class GraphState(TypedDict):
    # 입력
    tech_sw: str
    tech_hw: str
    domain: str
    # 제어
    run_id: str                                               # 체크포인트 thread_id / 트레이스 태그와 공유
    tasks: Annotated[dict, merge_tasks]                       # task_id -> Task
    decisions: Annotated[List[Decision], operator.add]
    # 결과
    results: Annotated[dict, merge_results]                   # task_id -> TaskResult
    synthesis: str


class WorkerInput(TypedDict):
    # Send로 각 워커에 전달되는 개별 입력 (전체 State가 아님)
    task: Task
    tech_sw: str
    tech_hw: str
    domain: str